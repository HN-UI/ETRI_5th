"""
Submission generation.

Evaluation metric: log loss → submit probabilities (0~1), not binary 0/1.
Probabilities clipped to [1e-5, 1-1e-5] to avoid infinite log loss.
"""

import numpy as np
import pandas as pd

from src.config import METRICS, OUTPUT_DIR
from src.data_loader import load_labels
from src.feature_engineering import FEATURES_DIR
from src.model import get_feature_cols, load_models

SUBMISSION_DIR = OUTPUT_DIR / "submissions"
SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)

PROB_CLIP = 1e-5  # avoid log(0) = -inf on server side


def predict_proba_all(
    models: dict,
    features_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Predict class-1 probability for all rows in features_df.
    Returns DataFrame with subject_id, lifelog_date, Q1..S4 as floats.
    """
    feat_cols = get_feature_cols(features_df)
    X = features_df[feat_cols].copy()
    X["subject_id_enc"] = pd.Categorical(features_df["subject_id"]).codes

    probs = features_df[["subject_id", "lifelog_date"]].copy()
    for metric, model in models.items():
        p = model.predict_proba(X)[:, 1]
        probs[metric] = np.clip(p, PROB_CLIP, 1 - PROB_CLIP)

    return probs


def make_submission(
    imputation_version: str = "llm",
    output_name: str | None = None,
) -> pd.DataFrame:
    """
    Generate submission CSV with probability predictions (for log loss evaluation).
    """
    _, submission_template = load_labels()
    features_df = pd.read_parquet(FEATURES_DIR / f"features_{imputation_version}.parquet")
    models = load_models(imputation_version)

    all_probs = predict_proba_all(models, features_df)

    sub_keys = submission_template[["subject_id", "lifelog_date"]].copy()
    merged = sub_keys.merge(all_probs, on=["subject_id", "lifelog_date"], how="left")

    # fallback: use prior (training positive rate) for any unmatched rows
    for m in METRICS:
        merged[m] = merged[m].fillna(0.5)

    result = submission_template[["subject_id", "sleep_date", "lifelog_date"]].copy()
    for m in METRICS:
        result[m] = merged[m].values.round(6)

    if output_name is None:
        output_name = f"submission_{imputation_version}.csv"
    out_path = SUBMISSION_DIR / output_name
    result.to_csv(out_path, index=False)
    print(f"제출 파일 저장: {out_path}  ({len(result)}행, 확률값)")
    return result
