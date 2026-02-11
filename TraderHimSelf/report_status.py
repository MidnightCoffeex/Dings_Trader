#!/usr/bin/env python3
"""Compact training status report for copy/paste diagnosis."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DATA_RAW = BASE_DIR / "data_raw"
DATA_PROCESSED = BASE_DIR / "data_processed"
MODELS = BASE_DIR / "models"
RUNS = BASE_DIR / "runs"
CHECKPOINTS = BASE_DIR / "checkpoints"

LOOKBACK = 512
FORECAST_HORIZON = 192
EXPECTED_MIN_ROWS = LOOKBACK + FORECAST_HORIZON + 1

# Keep this in sync with feature_engine.FEATURE_COLUMNS (single source of truth).
try:
    from feature_engine import FEATURE_COLUMNS as CORE_FEATURE_COLUMNS  # type: ignore
except Exception:
    CORE_FEATURE_COLUMNS = [
        "ret_1",
        "ret_4",
        "ret_16",
        "ret_48",
        "hl_range_pct",
        "oc_range_pct",
        "vol_16",
        "vol_96",
        "vol_672",
        "atr_14",
        "ema_20_dist",
        "ema_50_dist",
        "ema_200_dist",
        "ema_20_slope",
        "ema_50_slope",
        "adx_14",
        "rsi_14",
        "macd",
        "macd_signal",
        "macd_hist",
        "vol_log",
        "vol_z_96",
        "hour_sin",
        "hour_cos",
        "dow_sin",
        "dow_cos",
        "funding_rate_now",
        "time_to_next_funding_steps",
    ]
FORECAST_COLUMNS = [f"forecast_{i}" for i in range(35)]


@dataclass
class FileInfo:
    path: Path
    exists: bool
    size_bytes: int = 0


@dataclass
class DataSummary:
    rows: Optional[int]
    start: Optional[str]
    end: Optional[str]
    cols: Optional[int]
    nan_ratio: Optional[float] = None


def human_size(num: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    n = float(num)
    for u in units:
        if n < 1024 or u == units[-1]:
            return f"{n:.1f}{u}"
        n /= 1024
    return f"{num}B"


def file_info(path: Path) -> FileInfo:
    if path.exists():
        return FileInfo(path=path, exists=True, size_bytes=path.stat().st_size)
    return FileInfo(path=path, exists=False, size_bytes=0)


def _date_range_from_df(df: pd.DataFrame) -> Tuple[Optional[str], Optional[str]]:
    try:
        if isinstance(df.index, pd.DatetimeIndex) and len(df.index) > 0:
            return str(df.index.min()), str(df.index.max())
        if "open_time_ms" in df.columns and len(df) > 0:
            ts = pd.to_datetime(df["open_time_ms"], unit="ms", utc=True, errors="coerce")
            return str(ts.min()), str(ts.max())
    except Exception:
        pass
    return None, None


def summarize_parquet(path: Path, nan_cols: Optional[List[str]] = None) -> DataSummary:
    if not path.exists():
        return DataSummary(rows=None, start=None, end=None, cols=None, nan_ratio=None)
    try:
        df = pd.read_parquet(path)
        rows = len(df)
        cols = len(df.columns)
        start, end = _date_range_from_df(df)
        nan_ratio = None
        if nan_cols:
            target_cols = [c for c in nan_cols if c in df.columns]
            if target_cols:
                nan_ratio = float(df[target_cols].isna().mean().mean())
        return DataSummary(rows=rows, start=start, end=end, cols=cols, nan_ratio=nan_ratio)
    except Exception:
        return DataSummary(rows=None, start=None, end=None, cols=None, nan_ratio=None)


def first_file_match(directory: Path, pattern: str) -> Optional[Path]:
    matches = sorted(directory.glob(pattern))
    return matches[0] if matches else None


def line(s: str = ""):
    print(s)


def main():
    now = datetime.now(timezone.utc).astimezone().isoformat()

    raw_15m = first_file_match(DATA_RAW, "*15m.parquet")
    raw_3m = first_file_match(DATA_RAW, "*3m.parquet")
    raw_funding = first_file_match(DATA_RAW, "*funding*.parquet")

    p_aligned_15m = DATA_PROCESSED / "aligned_15m.parquet"
    p_aligned_3m = DATA_PROCESSED / "aligned_3m.parquet"
    p_features = DATA_PROCESSED / "features.parquet"
    p_forecast = DATA_PROCESSED / "forecast_features.parquet"
    p_scaler = DATA_PROCESSED / "scaler.pkl"
    p_forecast_model = MODELS / "forecast_model.pt"

    files = [
        ("raw_15m", raw_15m),
        ("raw_3m", raw_3m),
        ("raw_funding", raw_funding),
        ("aligned_15m", p_aligned_15m),
        ("aligned_3m", p_aligned_3m),
        ("features", p_features),
        ("forecast_features", p_forecast),
        ("scaler", p_scaler),
        ("forecast_model", p_forecast_model),
    ]

    file_infos: Dict[str, FileInfo] = {}
    for name, path in files:
        if path is None:
            file_infos[name] = FileInfo(path=Path("<missing>"), exists=False, size_bytes=0)
        else:
            file_infos[name] = file_info(path)

    sum_aligned_15m = summarize_parquet(p_aligned_15m)
    sum_aligned_3m = summarize_parquet(p_aligned_3m)
    sum_features = summarize_parquet(p_features, nan_cols=CORE_FEATURE_COLUMNS)
    sum_forecast = summarize_parquet(p_forecast, nan_cols=FORECAST_COLUMNS)

    missing_core = []
    missing_fc = []
    if p_features.exists():
        try:
            feat_cols = list(pd.read_parquet(p_features).columns)
            missing_core = [c for c in CORE_FEATURE_COLUMNS if c not in feat_cols]
        except Exception:
            missing_core = ["<read_error>"]
    if p_forecast.exists():
        try:
            fc_cols = list(pd.read_parquet(p_forecast).columns)
            missing_fc = [c for c in FORECAST_COLUMNS if c not in fc_cols]
        except Exception:
            missing_fc = ["<read_error>"]

    fail: List[str] = []
    warn: List[str] = []

    required = ["aligned_15m", "aligned_3m", "features", "scaler", "forecast_model", "forecast_features"]
    for name in required:
        if not file_infos[name].exists:
            fail.append(f"missing artifact: {name}")

    if sum_aligned_15m.rows is not None and sum_aligned_15m.rows < EXPECTED_MIN_ROWS:
        fail.append(f"aligned_15m too short ({sum_aligned_15m.rows} < {EXPECTED_MIN_ROWS})")

    if missing_core:
        fail.append(f"features missing core columns ({len(missing_core)})")
    if missing_fc:
        fail.append(f"forecast_features missing forecast columns ({len(missing_fc)})")

    if sum_features.nan_ratio is not None and sum_features.nan_ratio > 0.15:
        warn.append(f"high NaN ratio in features ({sum_features.nan_ratio:.2%})")
    if sum_forecast.nan_ratio is not None and sum_forecast.nan_ratio > 0.25:
        warn.append(f"high NaN ratio in forecast features ({sum_forecast.nan_ratio:.2%})")

    ppo_candidates = list(RUNS.glob("**/*.zip")) + list(CHECKPOINTS.glob("**/*.zip"))
    if not ppo_candidates:
        warn.append("no PPO zip/checkpoint artifact found yet")

    verdict = "FAIL" if fail else "WARN" if warn else "PASS"

    line("DINGS_TRADER_REPORT v1")
    line(f"generated_at={now}")
    line("")

    line("[FILES]")
    for name, info in file_infos.items():
        if info.exists:
            line(f"- {name}: OK ({human_size(info.size_bytes)}) :: {info.path}")
        else:
            line(f"- {name}: MISSING :: {info.path}")

    line("")
    line("[DATA_SUMMARY]")
    line(f"- aligned_15m: rows={sum_aligned_15m.rows} cols={sum_aligned_15m.cols} start={sum_aligned_15m.start} end={sum_aligned_15m.end} nan_ratio={sum_aligned_15m.nan_ratio}")
    line(f"- aligned_3m: rows={sum_aligned_3m.rows} cols={sum_aligned_3m.cols} start={sum_aligned_3m.start} end={sum_aligned_3m.end} nan_ratio={sum_aligned_3m.nan_ratio}")
    line(f"- features: rows={sum_features.rows} cols={sum_features.cols} start={sum_features.start} end={sum_features.end} nan_ratio={sum_features.nan_ratio}")
    line(f"- forecast_features: rows={sum_forecast.rows} cols={sum_forecast.cols} start={sum_forecast.start} end={sum_forecast.end} nan_ratio={sum_forecast.nan_ratio}")

    line("")
    line("[COLUMN_CHECKS]")
    line(f"- core_features_expected={len(CORE_FEATURE_COLUMNS)} missing={len(missing_core)}")
    if missing_core:
        line(f"  missing_core_sample={missing_core[:8]}")
    line(f"- forecast_features_expected={len(FORECAST_COLUMNS)} missing={len(missing_fc)}")
    if missing_fc:
        line(f"  missing_forecast_sample={missing_fc[:8]}")

    line("")
    line("[RECOMMENDATION]")
    line(f"- verdict={verdict}")
    for r in fail:
        line(f"- FAIL_REASON: {r}")
    for r in warn:
        line(f"- WARN_REASON: {r}")


if __name__ == "__main__":
    main()
