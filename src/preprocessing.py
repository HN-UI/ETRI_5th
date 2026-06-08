import json
import os
import re

# TensorFlow/JAX와 PyTorch 충돌 방지 (Qwen3 로딩 시 segfault 방어)
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_JAX", "0")

import pandas as pd
import numpy as np
from pathlib import Path

from src.config import OUTPUT_DIR
from src.data_loader import load_labels, load_all_sensors

PREPROCESSED_DIR = OUTPUT_DIR / "preprocessed"
PREPROCESSED_DIR.mkdir(parents=True, exist_ok=True)

STRATEGIES = ["zero", "subj_median", "dow_median", "llm"]

# 결측이 있는 센서만 보간 대상
IMPUTE_SENSORS = ["hr", "pedo", "light_w", "gps", "ble", "wifi", "usage_stats"]

# 전략별 센서별 보간 방법
STRATEGY_MAP = {
    "zero":        {s: "zero"       for s in IMPUTE_SENSORS},
    "subj_median": {**{s: "zero"    for s in IMPUTE_SENSORS}, "hr": "subj_median", "light_w": "subj_median"},
    "dow_median":  {**{s: "zero"    for s in IMPUTE_SENSORS}, "hr": "dow_median",  "light_w": "subj_median"},
    "llm":         {**{s: "zero"    for s in IMPUTE_SENSORS}, "hr": "llm",         "light_w": "subj_median"},
}

# activity 코드 매핑 (Google Activity Recognition API 기준)
ACTIVITY_LABELS = {0: "VEHICLE", 1: "BIKE", 3: "STILL", 4: "UNKNOWN", 7: "WALKING", 8: "RUNNING"}

# ── 1. 센서별 일별 집계 ──────────────────────────────────────────────────────

def _agg_hr(df: pd.DataFrame) -> pd.DataFrame:
    # heart_rate 컬럼이 배열(리스트) 형태 → explode 후 집계
    df = df.copy()
    df = df.explode("heart_rate")
    df["heart_rate"] = pd.to_numeric(df["heart_rate"], errors="coerce")
    daily = df.groupby(["subject_id", "date"]).agg(
        hr_mean=("heart_rate", "mean"),
        hr_std=("heart_rate", "std"),
        hr_min=("heart_rate", "min"),
        hr_max=("heart_rate", "max"),
    ).reset_index()
    daily["hr_std"] = daily["hr_std"].fillna(0)
    return daily

def _agg_pedo(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby(["subject_id", "date"]).agg(
        pedo_step=("step", "sum"),
        pedo_running=("running_step", "sum"),
        pedo_walking=("walking_step", "sum"),
        pedo_distance=("distance", "sum"),
        pedo_speed=("speed", "mean"),
        pedo_calories=("burned_calories", "sum"),
    ).reset_index()

def _agg_light_w(df: pd.DataFrame) -> pd.DataFrame:
    daily = df.groupby(["subject_id", "date"]).agg(
        light_w_mean=("w_light", "mean"),
        light_w_std=("w_light", "std"),
        light_w_max=("w_light", "max"),
    ).reset_index()
    daily["light_w_std"] = daily["light_w_std"].fillna(0)
    return daily

def _agg_count(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """타임스탬프 행 수 집계. dict/list 컬럼은 길이도 합산."""
    col = [c for c in df.columns if c not in ["subject_id", "date", "timestamp"]][0]
    if df[col].dtype == object and df[col].iloc[0] is not None and hasattr(df[col].iloc[0], "__len__"):
        df = df.copy()
        df[f"{name}_items"] = df[col].apply(lambda x: len(x) if isinstance(x, (list, np.ndarray)) else 1)
        return df.groupby(["subject_id", "date"]).agg(
            **{f"{name}_count": (f"{name}_items", "sum")}
        ).reset_index()
    return df.groupby(["subject_id", "date"]).size().reset_index(name=f"{name}_count")

def _agg_activity(df: pd.DataFrame) -> pd.DataFrame:
    """LLM 컨텍스트용 - 활동 타입별 분 수 집계."""
    df = df.copy()
    df["activity_label"] = df["m_activity"].map(ACTIVITY_LABELS).fillna("UNKNOWN")
    pivot = (
        df.groupby(["subject_id", "date", "activity_label"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    pivot.columns = ["subject_id", "date"] + [
        f"act_{c.lower()}_min" for c in pivot.columns[2:]
    ]
    return pivot

AGG_FUNCS = {
    "hr":          _agg_hr,
    "pedo":        _agg_pedo,
    "light_w":     _agg_light_w,
    "gps":         lambda df: _agg_count(df, "gps"),
    "ble":         lambda df: _agg_count(df, "ble"),
    "wifi":        lambda df: _agg_count(df, "wifi"),
    "usage_stats": lambda df: _agg_count(df, "usage_stats"),
    "activity":    _agg_activity,   # LLM 컨텍스트 전용
}

def aggregate_all(raw_data: dict) -> dict[str, pd.DataFrame]:
    daily = {}
    for name, func in AGG_FUNCS.items():
        if name in raw_data:
            daily[name] = func(raw_data[name])
    return daily


# ── 2. 전체 (subject, date) 그리드 생성 ──────────────────────────────────────

def build_full_grid(
    all_labels: pd.DataFrame, daily: dict[str, pd.DataFrame]
) -> dict[str, pd.DataFrame]:
    base = all_labels[["subject_id", "lifelog_date"]].copy()
    grids = {}
    for name, df in daily.items():
        merged = base.merge(
            df.rename(columns={"date": "lifelog_date"}),
            on=["subject_id", "lifelog_date"],
            how="left",
        )
        feat_cols = [c for c in merged.columns if c not in ["subject_id", "lifelog_date"]]
        if feat_cols:
            merged[f"{name}_was_missing"] = merged[feat_cols[0]].isna().astype(int)
        grids[name] = merged
    return grids


# ── 3. 보간 전략 ─────────────────────────────────────────────────────────────

ZERO_DEFAULTS: dict[str, dict] = {
    "hr":          {"hr_mean": 0.0, "hr_std": 0.0, "hr_min": 0.0, "hr_max": 0.0},
    "pedo":        {"pedo_step": 0, "pedo_running": 0, "pedo_walking": 0,
                    "pedo_distance": 0.0, "pedo_speed": 0.0, "pedo_calories": 0.0},
    "light_w":     {"light_w_mean": 0.0, "light_w_std": 0.0, "light_w_max": 0.0},
    "gps":         {"gps_count": 0},
    "ble":         {"ble_count": 0},
    "wifi":        {"wifi_count": 0},
    "usage_stats": {"usage_stats_count": 0},
}

def _feat_cols(sensor: str) -> list[str]:
    return list(ZERO_DEFAULTS[sensor].keys())


def impute_zero(df: pd.DataFrame, sensor: str) -> pd.DataFrame:
    df = df.copy()
    for col, val in ZERO_DEFAULTS[sensor].items():
        df[col] = df[col].fillna(val)
    return df


def impute_subj_median(df: pd.DataFrame, sensor: str) -> pd.DataFrame:
    df = df.copy()
    global_med = df[_feat_cols(sensor)].median()
    for col in _feat_cols(sensor):
        subj_med = df.groupby("subject_id")[col].transform("median")
        df[col] = df[col].fillna(subj_med).fillna(global_med[col])
    return df


def impute_dow_median(df: pd.DataFrame, sensor: str) -> pd.DataFrame:
    df = df.copy()
    df["_dow"] = pd.to_datetime(df["lifelog_date"]).dt.dayofweek
    global_med = df[_feat_cols(sensor)].median()
    for col in _feat_cols(sensor):
        dow_med  = df.groupby(["subject_id", "_dow"])[col].transform("median")
        subj_med = df.groupby("subject_id")[col].transform("median")
        df[col]  = df[col].fillna(dow_med).fillna(subj_med).fillna(global_med[col])
    return df.drop(columns=["_dow"])


# ── LLM (Qwen3-8B) 보간 ──────────────────────────────────────────────────────

_qwen_model = None
_qwen_tokenizer = None

def _load_qwen(gpu_id: int = 1):
    # Qwen3-8B bfloat16 ~16GB → GPU 1 단독으로는 부족, GPU 0+1 분산 로드
    global _qwen_model, _qwen_tokenizer
    if _qwen_model is None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        print(f"[LLM] Qwen/Qwen3-8B 로드 중... (GPU 0+1 분산, bfloat16)")
        _qwen_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
        _qwen_model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen3-8B",
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        print("[LLM] 모델 로드 완료.")
    return _qwen_model, _qwen_tokenizer


def _build_hr_prompt(row: pd.Series, subject_stats: dict, act_row: pd.Series | None) -> str:
    dow_names = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    dow = pd.to_datetime(row["lifelog_date"]).dayofweek
    st = subject_stats[row["subject_id"]]
    dow_hr = st["dow"].get(dow, None)

    act_info = "N/A"
    if act_row is not None:
        act_cols = {c: int(act_row[c]) for c in act_row.index if c.startswith("act_") and act_row[c] > 0}
        if act_cols:
            act_info = ", ".join(f"{k.replace('act_','').replace('_min','')}={v}min" for k, v in act_cols.items())

    return (
        f"Impute missing daily heart rate stats for a health lifelog study.\n\n"
        f"Subject: {row['subject_id']} | Date: {row['lifelog_date']} ({dow_names[dow]})\n\n"
        f"Subject baseline (from available days):\n"
        f"  hr_mean={st['mean']:.1f}, hr_std={st['std']:.1f}, "
        f"hr_min={st['min']:.1f}, hr_max={st['max']:.1f}\n"
        f"  {dow_names[dow]} typical hr_mean: "
        f"{f'{dow_hr:.1f}' if dow_hr is not None else 'N/A'}\n\n"
        f"Activity this day: {act_info}\n\n"
        f"Return ONLY valid JSON (no explanation):\n"
        f"{{\"hr_mean\":X.X,\"hr_std\":X.X,\"hr_min\":X.X,\"hr_max\":X.X}}"
    )


def _run_qwen_batch(prompts: list[str], model, tokenizer, batch_size: int = 8) -> list[str]:
    import torch
    # generation 배치는 left-padding 필수 (right-padding이면 패딩 뒤 출력이 꼬임)
    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    results = []
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i : i + batch_size]
        messages_batch = [[{"role": "user", "content": p}] for p in batch]
        texts = [
            tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
            for msgs in messages_batch
        ]
        inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(model.device)
        input_len = inputs["input_ids"].shape[1]
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=150,
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
                pad_token_id=tokenizer.eos_token_id,
            )
        for out in outputs:
            gen = tokenizer.decode(out[input_len:], skip_special_tokens=True)
            results.append(gen.strip())
        print(f"  배치 {i//batch_size + 1}/{(len(prompts)-1)//batch_size + 1} 완료")
    tokenizer.padding_side = original_padding_side
    return results


def impute_llm(df: pd.DataFrame, activity_grid: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    missing_mask = df["hr_was_missing"] == 1
    if not missing_mask.any():
        return df

    # subject별 통계 (present 데이터 기준)
    present = df[~missing_mask].copy()
    subject_stats = {}
    for sid in df["subject_id"].unique():
        s = present[present["subject_id"] == sid].copy()
        if len(s) == 0:
            subject_stats[sid] = {"mean": 70.0, "std": 10.0, "min": 55.0, "max": 110.0, "dow": {}}
            continue
        s["_dow"] = pd.to_datetime(s["lifelog_date"]).dt.dayofweek
        subject_stats[sid] = {
            "mean": s["hr_mean"].mean(),
            "std":  s["hr_std"].mean(),
            "min":  s["hr_min"].mean(),
            "max":  s["hr_max"].mean(),
            "dow":  s.groupby("_dow")["hr_mean"].mean().to_dict(),
        }

    # activity 컨텍스트 준비
    act_lookup = activity_grid.set_index(["subject_id", "lifelog_date"])

    # 프롬프트 일괄 생성
    missing_rows = df[missing_mask].reset_index()
    prompts, fallback_stats = [], []
    for _, row in missing_rows.iterrows():
        key = (row["subject_id"], row["lifelog_date"])
        act_row = act_lookup.loc[key] if key in act_lookup.index else None
        prompts.append(_build_hr_prompt(row, subject_stats, act_row))
        fallback_stats.append(subject_stats[row["subject_id"]])

    print(f"[LLM] 총 {len(prompts)}개 결측 hr 보간 시작")
    model, tokenizer = _load_qwen()
    responses = _run_qwen_batch(prompts, model, tokenizer)

    # 결과 파싱 및 채우기
    for i, (_, row) in enumerate(missing_rows.iterrows()):
        original_idx = row["index"]
        st = fallback_stats[i]
        try:
            m = re.search(r'\{[^}]+\}', responses[i])
            if not m:
                raise ValueError("JSON not found")
            vals = json.loads(m.group())
            df.loc[original_idx, "hr_mean"] = float(vals.get("hr_mean", st["mean"]))
            df.loc[original_idx, "hr_std"]  = float(vals.get("hr_std",  st["std"]))
            df.loc[original_idx, "hr_min"]  = float(vals.get("hr_min",  st["min"]))
            df.loc[original_idx, "hr_max"]  = float(vals.get("hr_max",  st["max"]))
        except Exception as e:
            print(f"  [fallback] {row['subject_id']} {row['lifelog_date']}: {e}")
            df.loc[original_idx, "hr_mean"] = st["mean"]
            df.loc[original_idx, "hr_std"]  = st["std"]
            df.loc[original_idx, "hr_min"]  = st["min"]
            df.loc[original_idx, "hr_max"]  = st["max"]

    return df


# ── 4. 전략 적용 및 센서 병합 ────────────────────────────────────────────────

def _apply_strategy(grids: dict[str, pd.DataFrame], strategy_name: str) -> pd.DataFrame:
    sensor_map = STRATEGY_MAP[strategy_name]
    result = None

    for sensor, method in sensor_map.items():
        df = grids[sensor].copy()

        if method == "zero":
            df = impute_zero(df, sensor)
        elif method == "subj_median":
            df = impute_subj_median(df, sensor)
        elif method == "dow_median":
            df = impute_dow_median(df, sensor)
        elif method == "llm":
            df = impute_llm(df, grids["activity"])

        if result is None:
            result = df
        else:
            new_cols = [c for c in df.columns if c not in ["subject_id", "lifelog_date"]]
            result = result.merge(
                df[["subject_id", "lifelog_date"] + new_cols],
                on=["subject_id", "lifelog_date"],
                how="left",
            )

    return result


# ── 5. 전략 전체 실행 ────────────────────────────────────────────────────────

def run_all_strategies() -> None:
    print("데이터 로드 중...")
    train, submission = load_labels()
    raw_data = load_all_sensors()

    all_labels = pd.concat([
        train[["subject_id", "lifelog_date"]],
        submission[["subject_id", "lifelog_date"]],
    ]).drop_duplicates().reset_index(drop=True)

    print("일별 집계 중...")
    daily = aggregate_all(raw_data)

    print("전체 그리드 생성 중...")
    grids = build_full_grid(all_labels, daily)

    print()
    summary_rows = []
    for strategy in STRATEGIES:
        print(f"{'='*50}")
        print(f"[{strategy}] 보간 실행 중...")
        result = _apply_strategy(grids, strategy)
        out_path = PREPROCESSED_DIR / f"imputed_{strategy}.parquet"
        result.to_parquet(out_path, index=False)

        # 결측 플래그 집계 (보간 전 기준)
        flag_cols = [c for c in result.columns if c.endswith("_was_missing")]
        missing_summary = {c.replace("_was_missing", ""): int(result[c].sum()) for c in flag_cols}
        summary_rows.append({"strategy": strategy, **missing_summary})
        print(f"  저장: {out_path}  ({len(result)}행 × {len(result.columns)}열)")

    print()
    print("=== 보간 전 결측 수 요약 ===")
    print(pd.DataFrame(summary_rows).set_index("strategy").to_string())
    print("\n완료.")
