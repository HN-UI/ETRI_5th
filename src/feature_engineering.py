"""
Feature engineering pipeline.

Input:
  - preprocessed imputed parquet (output/preprocessed/imputed_{version}.parquet)
  - raw sensor data via load_all_sensors()

Output:
  - feature DataFrame (700 rows × N features)
  - saved to output/features/features_{version}.parquet
"""

import numpy as np
import pandas as pd
from pathlib import Path

from src.config import OUTPUT_DIR
from src.data_loader import load_labels, load_all_sensors

PREPROCESSED_DIR = OUTPUT_DIR / "preprocessed"
FEATURES_DIR = OUTPUT_DIR / "features"
FEATURES_DIR.mkdir(parents=True, exist_ok=True)

# ── constants ─────────────────────────────────────────────────────────────────

# Google Activity Recognition API (README confirmed): 0=IN_VEHICLE, 1=ON_BICYCLE,
# 3=STILL, 4=UNKNOWN, 7=WALKING, 8=RUNNING
ACTIVITY_LABELS = {0: "vehicle", 1: "bike", 3: "still", 4: "unknown", 7: "walking", 8: "running"}

AMBIENCE_TARGETS = {
    "music":   ["music"],
    "speech":  ["speech"],
    "vehicle": ["vehicle"],
    "outside": ["outside"],
    "silence": ["silence"],
}

# Continuous features to add per-subject z-score versions for (Q1-Q3 modeling)
NORM_FEATURES = [
    "hr_mean", "hr_std", "hr_min", "hr_max", "hr_range",
    "pedo_step", "pedo_distance", "pedo_speed", "pedo_calories",
    "light_w_mean", "light_w_max",
    "gps_count", "ble_count", "wifi_count",
    "act_still_min", "act_walking_min", "act_running_min", "act_bike_min", "act_active_min",
    "screen_on_total_min", "screen_on_ratio",
    "charging_total_min",
    "total_usage_time_ms",
]


# ── 1. activity features ──────────────────────────────────────────────────────

def _agg_activity(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["label"] = df["m_activity"].map(ACTIVITY_LABELS).fillna("unknown")
    pivot = (
        df.groupby(["subject_id", "date", "label"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    pivot.columns = ["subject_id", "date"] + [f"act_{c}_min" for c in pivot.columns[2:]]

    for lbl in ACTIVITY_LABELS.values():
        col = f"act_{lbl}_min"
        if col not in pivot.columns:
            pivot[col] = 0

    pivot["act_active_min"] = (
        pivot["act_walking_min"] + pivot["act_running_min"] + pivot["act_bike_min"]
    )
    pivot["act_total_min"] = sum(pivot[f"act_{lbl}_min"] for lbl in ACTIVITY_LABELS.values())
    pivot["act_sedentary_ratio"] = (
        (pivot["act_still_min"] + pivot["act_vehicle_min"]) /
        pivot["act_total_min"].replace(0, np.nan)
    ).fillna(0.0)

    return pivot.rename(columns={"date": "lifelog_date"})


# ── 2. screen status features (sleep onset proxy) ─────────────────────────────

def _agg_screen(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["subject_id", "date", "timestamp"]).copy()

    daily = df.groupby(["subject_id", "date"]).agg(
        screen_on_total_min=("m_screen_use", "sum"),
        _total_samples=("m_screen_use", "count"),
    ).reset_index()
    daily["screen_on_ratio"] = (
        daily["screen_on_total_min"] / daily["_total_samples"].replace(0, np.nan)
    ).fillna(0.0)
    daily = daily.drop(columns=["_total_samples"])

    # last screen-on hour — proxy for sleep onset (useful for S3)
    screen_on = df[df["m_screen_use"] == 1].copy()
    screen_on["hour"] = screen_on["timestamp"].dt.hour
    last_hour = (
        screen_on.groupby(["subject_id", "date"])["hour"]
        .max().reset_index()
        .rename(columns={"hour": "last_screen_on_hour"})
    )

    # screen wake events (0→1 transitions → phone pickup count)
    df["_prev"] = df.groupby(["subject_id", "date"])["m_screen_use"].shift(1).fillna(0)
    events = df[(df["m_screen_use"] == 1) & (df["_prev"] == 0)]
    event_count = (
        events.groupby(["subject_id", "date"]).size()
        .reset_index(name="screen_event_count")
    )

    result = daily.merge(last_hour, on=["subject_id", "date"], how="left")
    result = result.merge(event_count, on=["subject_id", "date"], how="left")
    result["last_screen_on_hour"] = result["last_screen_on_hour"].fillna(-1).astype(int)
    result["screen_event_count"]  = result["screen_event_count"].fillna(0).astype(int)
    return result.rename(columns={"date": "lifelog_date"})


# ── 3. charging / AC status features (sleep onset proxy for S3) ───────────────

def _agg_ac(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["subject_id", "date", "timestamp"]).copy()

    daily = df.groupby(["subject_id", "date"]).agg(
        charging_total_min=("m_charging", "sum"),
        _total_samples=("m_charging", "count"),
    ).reset_index()
    daily["charging_ratio"] = (
        daily["charging_total_min"] / daily["_total_samples"].replace(0, np.nan)
    ).fillna(0.0)
    daily = daily.drop(columns=["_total_samples"])

    # first charge of day
    charging = df[df["m_charging"] == 1].copy()
    charging["hour"] = charging["timestamp"].dt.hour
    first_charge = (
        charging.groupby(["subject_id", "date"])["hour"]
        .min().reset_index()
        .rename(columns={"hour": "first_charge_hour"})
    )

    # first charge at/after 21:00 — best proxy for "put phone on charger to sleep"
    night = charging[charging["hour"] >= 21]
    night_charge = (
        night.groupby(["subject_id", "date"])["hour"]
        .min().reset_index()
        .rename(columns={"hour": "night_charge_hour"})
    )

    result = daily.merge(first_charge, on=["subject_id", "date"], how="left")
    result = result.merge(night_charge, on=["subject_id", "date"], how="left")
    result["first_charge_hour"] = result["first_charge_hour"].fillna(-1).astype(int)
    result["night_charge_hour"] = result["night_charge_hour"].fillna(-1).astype(int)
    return result.rename(columns={"date": "lifelog_date"})


# ── 4. app usage features ─────────────────────────────────────────────────────

_SOCIAL_KW    = ["카카오", "kakao", "문자", "sms", "메시지", "message",
                 "instagram", "facebook", "twitter", "틱톡", "tiktok", "라인", "line"]
_PHONE_KW     = ["전화", "통화", "phone", "call", "dialer"]
_ENTERTAIN_KW = ["youtube", "유튜브", "netflix", "넷플릭스", "게임", "game",
                 "멜론", "spotify", "tiktok", "틱톡"]


def _agg_usage(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in df.iterrows():
        items = r["m_usage_stats"]
        if not isinstance(items, (list, np.ndarray)):
            continue
        for item in items:
            if not isinstance(item, dict):
                continue
            app = str(item.get("app_name", "")).strip()
            t   = int(item.get("total_time", 0) or 0)
            al  = app.lower()
            rows.append({
                "subject_id":  r["subject_id"],
                "date":        r["date"],
                "app":         app,
                "total_time":  t,
                "is_social":   int(any(kw in al for kw in _SOCIAL_KW)),
                "is_phone":    int(any(kw in al for kw in _PHONE_KW)),
                "is_entertain": int(any(kw in al for kw in _ENTERTAIN_KW)),
            })

    if not rows:
        cols = ["subject_id", "lifelog_date", "total_usage_time_ms", "unique_app_count",
                "social_time_ms", "phone_call_time_ms", "entertainment_time_ms"]
        return pd.DataFrame(columns=cols)

    flat = pd.DataFrame(rows)
    flat["social_time"]   = flat["total_time"] * flat["is_social"]
    flat["phone_time"]    = flat["total_time"] * flat["is_phone"]
    flat["entertain_time"] = flat["total_time"] * flat["is_entertain"]

    agg = flat.groupby(["subject_id", "date"]).agg(
        total_usage_time_ms   =("total_time",    "sum"),
        unique_app_count      =("app",           "nunique"),
        social_time_ms        =("social_time",   "sum"),
        phone_call_time_ms    =("phone_time",    "sum"),
        entertainment_time_ms =("entertain_time","sum"),
    ).reset_index()
    return agg.rename(columns={"date": "lifelog_date"})


# ── 5. ambience features ──────────────────────────────────────────────────────

def _agg_ambience(df: pd.DataFrame) -> pd.DataFrame:
    def _parse(amb_arr):
        out = []
        for item in amb_arr:
            try:
                out.append((str(item[0]).lower(), float(item[1])))
            except Exception:
                pass
        return out

    df = df.copy()
    df["parsed"] = df["m_ambience"].apply(_parse)
    exploded = df[["subject_id", "date"]].copy()
    exploded["pairs"] = df["parsed"]
    exploded = exploded.explode("pairs").dropna(subset=["pairs"])
    exploded[["amb_label", "amb_prob"]] = pd.DataFrame(
        exploded["pairs"].tolist(), index=exploded.index
    )

    parts = []
    for cat, keywords in AMBIENCE_TARGETS.items():
        mask = exploded["amb_label"].apply(lambda l: any(kw in l for kw in keywords))
        sub = exploded[mask].groupby(["subject_id", "date"])["amb_prob"].mean().reset_index()
        sub = sub.rename(columns={"amb_prob": f"amb_{cat}_mean"})
        parts.append(sub.set_index(["subject_id", "date"]))

    if not parts:
        return pd.DataFrame(columns=["subject_id", "lifelog_date"])

    result = pd.concat(parts, axis=1).fillna(0.0).reset_index()
    return result.rename(columns={"date": "lifelog_date"})


# ── 6. per-subject z-score normalization (for Q1-Q3 head) ────────────────────

def _add_subject_zscores(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in NORM_FEATURES:
        if col not in df.columns:
            continue
        subj_mean = df.groupby("subject_id")[col].transform("mean")
        subj_std  = df.groupby("subject_id")[col].transform("std").replace(0, np.nan)
        df[f"{col}_z"] = ((df[col] - subj_mean) / subj_std).fillna(0.0)
    return df


# ── 7. main pipeline ──────────────────────────────────────────────────────────

def build_features(imputation_version: str = "llm") -> pd.DataFrame:
    """
    Build feature matrix for all 700 (subject, date) pairs.

    Returns a DataFrame with subject_id, lifelog_date, all features.
    Call save_features() to persist to parquet.
    """
    print(f"[feature_engineering] version={imputation_version}")

    # base: preprocessed imputed sensor stats
    base = pd.read_parquet(PREPROCESSED_DIR / f"imputed_{imputation_version}.parquet")
    base["lifelog_date"] = pd.to_datetime(base["lifelog_date"]).dt.date
    # always-zero per README → drop
    base = base.drop(columns=["pedo_running", "pedo_walking"], errors="ignore")
    base["hr_range"] = base["hr_max"] - base["hr_min"]
    base["hr_cv"]    = (base["hr_std"] / base["hr_mean"].replace(0, np.nan)).fillna(0.0)

    print("  raw sensors 로드...")
    raw = load_all_sensors()

    print("  activity 집계...")
    act_df = _agg_activity(raw["activity"])
    act_df["lifelog_date"] = pd.to_datetime(act_df["lifelog_date"]).dt.date

    print("  screen 집계...")
    screen_df = _agg_screen(raw["screen"])
    screen_df["lifelog_date"] = pd.to_datetime(screen_df["lifelog_date"]).dt.date

    print("  ac_status 집계...")
    ac_df = _agg_ac(raw["ac_status"])
    ac_df["lifelog_date"] = pd.to_datetime(ac_df["lifelog_date"]).dt.date

    print("  usage_stats 집계...")
    usage_df = _agg_usage(raw["usage_stats"])
    usage_df["lifelog_date"] = pd.to_datetime(usage_df["lifelog_date"]).dt.date

    print("  ambience 집계...")
    amb_df = _agg_ambience(raw["ambience"])
    amb_df["lifelog_date"] = pd.to_datetime(amb_df["lifelog_date"]).dt.date

    # merge all
    print("  피처 병합...")
    features = base.copy()
    for df in [act_df, screen_df, ac_df, usage_df, amb_df]:
        new_cols = [c for c in df.columns if c not in ["subject_id", "lifelog_date"]]
        if not new_cols:
            continue
        features = features.merge(
            df[["subject_id", "lifelog_date"] + new_cols],
            on=["subject_id", "lifelog_date"],
            how="left",
        )

    # temporal features
    dt = pd.to_datetime(features["lifelog_date"])
    features["day_of_week"] = dt.dt.dayofweek
    features["is_weekend"]  = (features["day_of_week"] >= 5).astype(int)
    features["study_day"]   = features.groupby("subject_id")["lifelog_date"].transform(
        lambda x: (pd.to_datetime(x) - pd.to_datetime(x.min())).dt.days
    )

    # fill remaining NaN from sensors without full coverage (treated as "sensor off")
    act_cols   = [c for c in features.columns if c.startswith("act_")]
    screen_cols = ["screen_on_total_min", "screen_on_ratio", "screen_event_count"]
    ac_cols    = ["charging_total_min", "charging_ratio"]
    usage_cols = ["total_usage_time_ms", "unique_app_count",
                  "social_time_ms", "phone_call_time_ms", "entertainment_time_ms"]
    amb_cols   = [c for c in features.columns if c.startswith("amb_")]
    for col in act_cols + screen_cols + ac_cols + usage_cols + amb_cols:
        if col in features.columns:
            features[col] = features[col].fillna(0.0)
    for col in ["last_screen_on_hour", "first_charge_hour", "night_charge_hour"]:
        if col in features.columns:
            features[col] = features[col].fillna(-1).astype(int)

    # per-subject z-score features (normalized deviation from personal baseline)
    print("  subject z-score 정규화...")
    features = _add_subject_zscores(features)

    print(f"  완료: {features.shape[0]}행 × {features.shape[1]}열")
    return features


def save_features(imputation_version: str = "llm") -> pd.DataFrame:
    feats = build_features(imputation_version)
    out_path = FEATURES_DIR / f"features_{imputation_version}.parquet"
    feats.to_parquet(out_path, index=False)
    print(f"저장: {out_path}")
    return feats
