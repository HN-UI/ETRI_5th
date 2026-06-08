import pandas as pd
from src.config import TRAIN_CSV, SUBMISSION_CSV, PARQUET_FILES


def load_labels() -> tuple[pd.DataFrame, pd.DataFrame]:
    """train/submission CSV 로드. lifelog_date를 date 타입으로 변환."""
    train = pd.read_csv(TRAIN_CSV, parse_dates=["sleep_date", "lifelog_date"])
    submission = pd.read_csv(SUBMISSION_CSV, parse_dates=["sleep_date", "lifelog_date"])

    train["lifelog_date"] = train["lifelog_date"].dt.date
    submission["lifelog_date"] = submission["lifelog_date"].dt.date

    return train, submission


def load_sensor(name: str) -> pd.DataFrame:
    """단일 센서 parquet 로드. date 컬럼 추가."""
    path = PARQUET_FILES[name]
    df = pd.read_parquet(path)
    df["date"] = df["timestamp"].dt.date
    return df


def load_all_sensors() -> dict[str, pd.DataFrame]:
    """모든 센서 데이터를 딕셔너리로 반환."""
    return {name: load_sensor(name) for name in PARQUET_FILES}


def summary(raw_data: dict[str, pd.DataFrame]) -> None:
    """로드된 센서 데이터 요약 출력."""
    print(f"{'센서':<15} {'rows':>10} {'subjects':>10} {'기간 시작':>12} {'기간 끝':>12}")
    print("-" * 65)
    for name, df in raw_data.items():
        n_subjects = df["subject_id"].nunique()
        date_min = df["timestamp"].min().date()
        date_max = df["timestamp"].max().date()
        print(f"{name:<15} {len(df):>10,} {n_subjects:>10} {str(date_min):>12} {str(date_max):>12}")
