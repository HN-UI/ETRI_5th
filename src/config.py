from pathlib import Path

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
ITEMS_DIR = DATA_DIR / "ch2025_data_items"
OUTPUT_DIR = ROOT / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

TRAIN_CSV = DATA_DIR / "ch2026_metrics_train.csv"
SUBMISSION_CSV = DATA_DIR / "ch2026_submission_sample.csv"

METRICS = ["Q1", "Q2", "Q3", "S1", "S2", "S3", "S4"]

PARQUET_FILES = {
    "ambience":    ITEMS_DIR / "ch2025_mAmbience.parquet",
    "ac_status":   ITEMS_DIR / "ch2025_mACStatus.parquet",
    "pedo":        ITEMS_DIR / "ch2025_wPedo.parquet",
    "light_w":     ITEMS_DIR / "ch2025_wLight.parquet",
    "light_m":     ITEMS_DIR / "ch2025_mLight.parquet",
    "gps":         ITEMS_DIR / "ch2025_mGps.parquet",
    "activity":    ITEMS_DIR / "ch2025_mActivity.parquet",
    "wifi":        ITEMS_DIR / "ch2025_mWifi.parquet",
    "usage_stats": ITEMS_DIR / "ch2025_mUsageStats.parquet",
    "hr":          ITEMS_DIR / "ch2025_wHr.parquet",
    "ble":         ITEMS_DIR / "ch2025_mBle.parquet",
    "screen":      ITEMS_DIR / "ch2025_mScreenStatus.parquet",
}

RANDOM_SEED = 42
