"""
Model training and evaluation pipeline.

7 binary classifiers (Q1-Q3, S1-S4), one per metric.
- LightGBM (binary cross-entropy)
- LOSO-CV (GroupKFold by subject_id, n_splits=10)
- Q1-Q3: per-subject z-score features preferred
- S1-S4: absolute features preferred; both families share the same feature set
"""

import pickle
import numpy as np
import pandas as pd
from pathlib import Path

import lightgbm as lgb
from sklearn.metrics import f1_score, balanced_accuracy_score, roc_auc_score, log_loss
from sklearn.model_selection import GroupKFold

from src.config import METRICS, OUTPUT_DIR, RANDOM_SEED
from src.data_loader import load_labels
from src.feature_engineering import FEATURES_DIR

MODEL_DIR = OUTPUT_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

Q_METRICS = ["Q1", "Q2", "Q3"]
S_METRICS = ["S1", "S2", "S3", "S4"]

LGBM_PARAMS = {
    "objective":        "binary",
    "metric":           "binary_logloss",
    "n_estimators":     500,
    "learning_rate":    0.05,
    "max_depth":        6,
    "num_leaves":       31,
    "min_child_samples": 10,
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "reg_alpha":        0.1,
    "reg_lambda":       0.1,
    "random_state":     RANDOM_SEED,
    "verbosity":        -1,
    "n_jobs":           -1,
}


# ── feature selection ─────────────────────────────────────────────────────────

def get_feature_cols(features_df: pd.DataFrame) -> list[str]:
    """All feature columns (exclude id/date/target columns)."""
    drop = {"subject_id", "lifelog_date"} | set(METRICS)
    return [c for c in features_df.columns if c not in drop]


# ── data preparation ──────────────────────────────────────────────────────────

def prepare_xy(
    features_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    metric: str,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Join features with labels for one metric.
    Returns (X, y, groups) where groups=subject_id for GroupKFold.
    """
    merged = labels_df[["subject_id", "lifelog_date", metric]].merge(
        features_df, on=["subject_id", "lifelog_date"], how="left"
    )
    feat_cols = get_feature_cols(features_df)
    X = merged[feat_cols].copy()
    X["subject_id_enc"] = pd.Categorical(merged["subject_id"]).codes
    y = merged[metric].astype(int)
    groups = merged["subject_id"]
    return X, y, groups


# ── cross-validation ──────────────────────────────────────────────────────────

def cross_validate_metric(
    features_df: pd.DataFrame,
    train_df: pd.DataFrame,
    metric: str,
    n_splits: int = 10,
) -> dict:
    """LOSO-CV for a single metric. Returns oof predictions + scores."""
    X, y, groups = prepare_xy(features_df, train_df, metric)

    gkf = GroupKFold(n_splits=n_splits)
    oof_preds = np.zeros(len(y), dtype=int)
    oof_probs = np.zeros(len(y))

    for train_idx, val_idx in gkf.split(X, y, groups):
        X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

        pos_w = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)
        model = lgb.LGBMClassifier(**{**LGBM_PARAMS, "scale_pos_weight": pos_w})
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            callbacks=[
                lgb.early_stopping(50, verbose=False),
                lgb.log_evaluation(-1),
            ],
        )
        oof_probs[val_idx] = model.predict_proba(X_val)[:, 1]
        oof_preds[val_idx] = model.predict(X_val)

    return {
        "metric":     metric,
        "f1":         f1_score(y, oof_preds, zero_division=0),
        "bacc":       balanced_accuracy_score(y, oof_preds),
        "auc":        roc_auc_score(y, oof_probs),
        "oof_preds":  oof_preds,
        "oof_probs":  oof_probs,
    }


def temporal_cv_metric(
    features_df: pd.DataFrame,
    train_df: pd.DataFrame,
    metric: str,
    val_ratio: float = 0.3,
) -> dict:
    """
    Within-subject temporal CV: for each subject, hold out the last val_ratio
    fraction of days as validation, train on earlier days.
    This mirrors the actual temporal train→submission split structure.
    """
    merged = train_df[["subject_id", "lifelog_date", metric]].merge(
        features_df, on=["subject_id", "lifelog_date"], how="left"
    ).sort_values(["subject_id", "lifelog_date"])

    feat_cols = get_feature_cols(features_df)

    oof_preds = np.zeros(len(merged), dtype=int)
    oof_probs = np.zeros(len(merged))
    oof_mask  = np.zeros(len(merged), dtype=bool)

    for sid in merged["subject_id"].unique():
        idx = merged[merged["subject_id"] == sid].index
        n_val = max(1, int(len(idx) * val_ratio))
        tr_idx  = idx[:-n_val]
        val_idx = idx[-n_val:]

        X_tr  = merged.loc[tr_idx, feat_cols].copy()
        y_tr  = merged.loc[tr_idx, metric].astype(int)
        X_val = merged.loc[val_idx, feat_cols].copy()
        y_val = merged.loc[val_idx, metric].astype(int)

        # subject_id_enc
        enc = pd.Categorical(merged["subject_id"]).codes
        X_tr["subject_id_enc"]  = enc[tr_idx - merged.index[0]]
        X_val["subject_id_enc"] = enc[val_idx - merged.index[0]]

        pos_w = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)
        model = lgb.LGBMClassifier(**{**LGBM_PARAMS, "scale_pos_weight": pos_w})
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            callbacks=[
                lgb.early_stopping(50, verbose=False),
                lgb.log_evaluation(-1),
            ],
        )
        oof_probs[val_idx - merged.index[0]] = model.predict_proba(X_val)[:, 1]
        oof_preds[val_idx - merged.index[0]] = model.predict(X_val)
        oof_mask[val_idx - merged.index[0]]  = True

    y_all = merged[metric].astype(int).values
    y_val_true = y_all[oof_mask]
    y_val_pred = oof_preds[oof_mask]
    y_val_prob = oof_probs[oof_mask]
    # clip for log_loss stability
    y_val_prob_clip = np.clip(y_val_prob, 1e-7, 1 - 1e-7)

    return {
        "metric":   metric,
        "log_loss": log_loss(y_val_true, y_val_prob_clip),
        "auc":      roc_auc_score(y_val_true, y_val_prob),
        "bacc":     balanced_accuracy_score(y_val_true, y_val_pred),
        "n_val":    oof_mask.sum(),
    }


def run_temporal_cv(imputation_version: str = "llm", val_ratio: float = 0.3) -> pd.DataFrame:
    """Within-subject temporal CV (last 30% per subject as validation)."""
    features_df = pd.read_parquet(FEATURES_DIR / f"features_{imputation_version}.parquet")
    train_df, _ = load_labels()

    print(f"\n[temporal CV] version={imputation_version}  val_ratio={val_ratio}")
    print(f"{'metric':>8}  {'LogLoss':>8}  {'AUC':>6}  {'bAcc':>6}  {'n_val':>6}")
    print("-" * 48)

    rows = []
    for metric in METRICS:
        result = temporal_cv_metric(features_df, train_df, metric, val_ratio)
        print(f"{metric:>8}  {result['log_loss']:>8.4f}  {result['auc']:.4f}  {result['bacc']:.4f}  {result['n_val']:>6}")
        rows.append({k: v for k, v in result.items()})

    df = pd.DataFrame(rows)
    print("-" * 48)
    print(f"{'mean':>8}  {df['log_loss'].mean():>8.4f}  {df['auc'].mean():.4f}  {df['bacc'].mean():.4f}")
    return df


def run_cv(imputation_version: str = "llm", n_splits: int = 10) -> pd.DataFrame:
    """Run LOSO-CV for all 7 metrics. Prints and returns a results DataFrame."""
    features_df = pd.read_parquet(FEATURES_DIR / f"features_{imputation_version}.parquet")
    train_df, _ = load_labels()

    print(f"\n[CV] version={imputation_version}  folds={n_splits}")
    print(f"{'metric':>8}  {'F1':>6}  {'bAcc':>6}  {'AUC':>6}")
    print("-" * 35)

    rows = []
    for metric in METRICS:
        result = cross_validate_metric(features_df, train_df, metric, n_splits)
        print(f"{metric:>8}  {result['f1']:.4f}  {result['bacc']:.4f}  {result['auc']:.4f}")
        rows.append({k: v for k, v in result.items() if k not in ("oof_preds", "oof_probs")})

    df = pd.DataFrame(rows)
    print("-" * 35)
    print(f"{'mean':>8}  {df['f1'].mean():.4f}  {df['bacc'].mean():.4f}  {df['auc'].mean():.4f}")
    return df


# ── final training ────────────────────────────────────────────────────────────

def train_final(
    features_df: pd.DataFrame,
    train_df: pd.DataFrame,
    metric: str,
) -> lgb.LGBMClassifier:
    """Train final model on all training data (no val split)."""
    X, y, _ = prepare_xy(features_df, train_df, metric)
    pos_w = (y == 0).sum() / max((y == 1).sum(), 1)
    model = lgb.LGBMClassifier(**{**LGBM_PARAMS, "scale_pos_weight": pos_w})
    model.fit(X, y)
    return model


def train_all(
    imputation_version: str = "llm",
) -> dict[str, lgb.LGBMClassifier]:
    """Train final models for all metrics. Saves to MODEL_DIR."""
    features_df = pd.read_parquet(FEATURES_DIR / f"features_{imputation_version}.parquet")
    train_df, _ = load_labels()

    models = {}
    for metric in METRICS:
        print(f"  {metric} 학습...", end="", flush=True)
        model = train_final(features_df, train_df, metric)
        models[metric] = model
        path = MODEL_DIR / f"lgbm_{metric}_{imputation_version}.pkl"
        with open(path, "wb") as f:
            pickle.dump(model, f)
        print(f"  저장: {path.name}")

    return models


def load_models(imputation_version: str = "llm") -> dict[str, lgb.LGBMClassifier]:
    models = {}
    for metric in METRICS:
        path = MODEL_DIR / f"lgbm_{metric}_{imputation_version}.pkl"
        with open(path, "rb") as f:
            models[metric] = pickle.load(f)
    return models


# ── prediction ────────────────────────────────────────────────────────────────

def predict(
    models: dict[str, lgb.LGBMClassifier],
    features_df: pd.DataFrame,
    proba: bool = False,
) -> pd.DataFrame:
    """
    Generate predictions for all metrics.
    proba=True → return class-1 probabilities (for log loss submission).
    proba=False → return binary labels.
    """
    feat_cols = get_feature_cols(features_df)
    X = features_df[feat_cols].copy()
    X["subject_id_enc"] = pd.Categorical(features_df["subject_id"]).codes

    out = features_df[["subject_id", "lifelog_date"]].copy()
    for metric, model in models.items():
        if proba:
            out[metric] = model.predict_proba(X)[:, 1]
        else:
            out[metric] = model.predict(X)

    return out
