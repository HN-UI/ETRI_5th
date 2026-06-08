import argparse
from src.data_loader import load_labels, load_all_sensors, summary
from src.config import METRICS


def main(
    run_preprocessing: bool = False,
    run_features: bool = False,
    run_cv: bool = False,
    run_train: bool = False,
    run_predict: bool = False,
    imputation_version: str = "llm",
):
    print("=" * 65)
    print("[1] Label 데이터 로드")
    print("=" * 65)
    train, submission = load_labels()

    print(f"Train    : {train.shape}  ({train['lifelog_date'].min()} ~ {train['lifelog_date'].max()})")
    print(f"Submission: {submission.shape}  ({submission['lifelog_date'].min()} ~ {submission['lifelog_date'].max()})")
    print(f"Subjects : {sorted(train['subject_id'].unique())}")
    print(f"Metrics  : {METRICS}")
    print()
    print("Train 샘플:")
    print(train.head(3).to_string(index=False))
    print()

    print("=" * 65)
    print("[2] 센서 데이터 로드")
    print("=" * 65)
    raw_data = load_all_sensors()
    summary(raw_data)
    print()

    print("=" * 65)
    print("[3] 데이터 정합성 확인")
    print("=" * 65)
    train_subjects = set(train["subject_id"].unique())
    for name, df in raw_data.items():
        sensor_subjects = set(df["subject_id"].unique())
        missing = train_subjects - sensor_subjects
        if missing:
            print(f"[WARNING] {name}: train에는 있으나 센서에 없는 subject → {missing}")
        else:
            print(f"[OK] {name}: 모든 subject 존재")

    print()

    if run_preprocessing:
        print("=" * 65)
        print("[4] 결측치 보간 (4가지 전략 전체 실행)")
        print("=" * 65)
        from src.preprocessing import run_all_strategies
        run_all_strategies()

        print()
        print("=" * 65)
        print("[5] 보간 결과 확인")
        print("=" * 65)
        import pandas as pd
        from src.config import OUTPUT_DIR
        preprocessed_dir = OUTPUT_DIR / "preprocessed"
        for f in sorted(preprocessed_dir.glob("imputed_*.parquet")):
            df = pd.read_parquet(f)
            flag_cols = [c for c in df.columns if c.endswith("_was_missing")]
            remaining_nan = df.drop(columns=flag_cols).isna().sum().sum()
            print(f"  {f.name}: {df.shape}  NaN 잔존={remaining_nan}")
    else:
        print("[INFO] 보간 실행하려면: python main.py --preprocess")

    if run_features:
        print()
        print("=" * 65)
        print("[6] 피처 엔지니어링")
        print("=" * 65)
        from src.feature_engineering import save_features
        for version in ["zero", "subj_median", "dow_median", "llm"]:
            save_features(version)
    else:
        print("[INFO] 피처 빌드하려면: python main.py --features")

    if run_cv:
        print()
        print("=" * 65)
        print(f"[7] Temporal CV 평가  (version={imputation_version})")
        print("=" * 65)
        from src.model import run_temporal_cv
        results = run_temporal_cv(imputation_version)
        from src.config import OUTPUT_DIR
        results_path = OUTPUT_DIR / f"cv_results_{imputation_version}.csv"
        results.to_csv(results_path, index=False)
        print(f"\nCV 결과 저장: {results_path}")

    if run_train:
        print()
        print("=" * 65)
        print(f"[8] 최종 모델 학습  (version={imputation_version})")
        print("=" * 65)
        from src.model import train_all
        train_all(imputation_version)

    if run_predict:
        print()
        print("=" * 65)
        print(f"[9] 제출 파일 생성  (version={imputation_version})")
        print("=" * 65)
        from src.predict import make_submission
        sub = make_submission(imputation_version)
        print(sub[["subject_id", "lifelog_date"] + __import__("src.config", fromlist=["METRICS"]).METRICS].head(5).to_string(index=False))

    print("\n완료.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--preprocess", action="store_true", help="결측치 보간 4가지 전략 실행")
    parser.add_argument("--features",   action="store_true", help="피처 엔지니어링 (4가지 버전)")
    parser.add_argument("--cv",         action="store_true", help="LOSO-CV 평가")
    parser.add_argument("--train",      action="store_true", help="최종 모델 학습")
    parser.add_argument("--predict",    action="store_true", help="제출 파일 생성")
    parser.add_argument("--version",    default="llm",      help="imputation 버전 (zero/subj_median/dow_median/llm)")
    args = parser.parse_args()
    main(
        run_preprocessing=args.preprocess,
        run_features=args.features,
        run_cv=args.cv,
        run_train=args.train,
        run_predict=args.predict,
        imputation_version=args.version,
    )
