"""Training-only Feature Engine (30D core).

Goal:
- Keep live/inference feature vector stable (28D; see feature_engine.FEATURE_COLUMNS)
- Provide a *separate* training feature set with +2 time features (30D total)
  without touching the live pipeline.

Adds (derived from candle timestamp in UTC, no leakage):
- hour_utc  (0..23)
- dow_utc   (0..6, Mon..Sun)

Outputs are written into a separate folder so existing artifacts stay intact:
- data_processed/train30/features.parquet
- data_processed/train30/scaler.pkl

Usage:
    cd TraderHimSelf
    venv/bin/python feature_engine_train30.py build

Note:
- This module intentionally does NOT change the live FEATURE_COLUMNS.
- Training code must explicitly opt into the train30 artifacts.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Reuse the proven indicator + funding merge logic from the live engine.
from feature_engine import (
    FeatureEngineConfig,
    compute_core_features as compute_core_features_live28,
    _ensure_datetime_index,  # noqa: SLF001 (internal reuse is fine here)
)


# -----------------------------------------------------------------------------
# Feature order (FIX!) — training only
# -----------------------------------------------------------------------------

# Start with live 28D ordering, then append two explicit time features.
from feature_engine import FEATURE_COLUMNS as FEATURE_COLUMNS_LIVE28

FEATURE_COLUMNS_TRAIN30: List[str] = list(FEATURE_COLUMNS_LIVE28) + [
    "hour_utc",
    "dow_utc",
]


# -----------------------------------------------------------------------------
# Paths / defaults (separate folder)
# -----------------------------------------------------------------------------

DEFAULT_INPUT_15M = Path("data_processed/aligned_15m.parquet")
DEFAULT_INPUT_FUNDING = Path("data_processed/funding.parquet")
DEFAULT_OUTPUT_DIR = Path("data_processed/train30")
DEFAULT_OUTPUT_FEATURES = DEFAULT_OUTPUT_DIR / "features.parquet"
DEFAULT_OUTPUT_SCALER = DEFAULT_OUTPUT_DIR / "scaler.pkl"


def _compute_time_extras(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Compute (hour_utc, dow_utc) from a DatetimeIndex."""

    if not isinstance(index, pd.DatetimeIndex):
        raise TypeError("index must be a DatetimeIndex")

    ts_utc = index.tz_localize("UTC") if index.tz is None else index.tz_convert("UTC")
    return pd.DataFrame(
        {
            "hour_utc": ts_utc.hour.astype(float),
            "dow_utc": ts_utc.dayofweek.astype(float),
        },
        index=index,
    )


def compute_core_features_train30(
    buf_15m: pd.DataFrame,
    funding_df: Optional[pd.DataFrame] = None,
    *,
    cfg: FeatureEngineConfig = FeatureEngineConfig(),
) -> pd.DataFrame:
    """Compute the training-only 30D feature matrix."""

    base = compute_core_features_live28(buf_15m, funding_df, cfg=cfg)
    extras = _compute_time_extras(base.index)

    out = base.join(extras, how="left")

    # Enforce fixed column order
    return out[FEATURE_COLUMNS_TRAIN30]


def fit_scaler_train30(features: pd.DataFrame, *, cfg: FeatureEngineConfig = FeatureEngineConfig()) -> StandardScaler:
    """Fit a StandardScaler on the train split only (2019-2023)."""

    train_start = pd.Timestamp(cfg.train_start)
    train_end = pd.Timestamp(cfg.train_end)
    if train_start.tz is None:
        train_start = train_start.tz_localize("UTC")
    if train_end.tz is None:
        train_end = train_end.tz_localize("UTC")

    mask = (features.index >= train_start) & (features.index <= train_end)
    train = features.loc[mask].dropna()

    if len(train) == 0:
        # Keep the same dev/CI fallback semantics as the live engine.
        train = features.dropna()
        if len(train) == 0:
            print(
                "[feature_engine_train30] WARNING: No non-NaN rows available to fit scaler. "
                "Falling back to a dummy scaler fitted on zeros (mock data)."
            )
            train = pd.DataFrame(np.zeros((1, len(FEATURE_COLUMNS_TRAIN30))), columns=FEATURE_COLUMNS_TRAIN30)
        print(
            "[feature_engine_train30] WARNING: No rows in the configured train window (2019-2023). "
            "Falling back to fitting scaler on all available non-NaN rows (likely mock/short data)."
        )

    scaler = StandardScaler()
    scaler.fit(train.values)
    return scaler


def apply_scaler_train30(features: pd.DataFrame, scaler: StandardScaler) -> pd.DataFrame:
    """Apply scaler to rows without NaNs; keep NaN-rows unchanged."""

    out = features.copy()
    valid = ~features.isna().any(axis=1)
    if valid.any():
        out.loc[valid, FEATURE_COLUMNS_TRAIN30] = scaler.transform(features.loc[valid, FEATURE_COLUMNS_TRAIN30].values)
    return out


def build(
    *,
    input_15m: Path = DEFAULT_INPUT_15M,
    input_funding: Path = DEFAULT_INPUT_FUNDING,
    out_features: Path = DEFAULT_OUTPUT_FEATURES,
    out_scaler: Path = DEFAULT_OUTPUT_SCALER,
    cfg: FeatureEngineConfig = FeatureEngineConfig(),
) -> None:
    """Build + scale train30 features and write artifacts."""

    candles = pd.read_parquet(input_15m)
    funding = pd.read_parquet(input_funding) if input_funding.exists() else None

    candles = _ensure_datetime_index(candles)
    if funding is not None:
        funding = _ensure_datetime_index(funding, time_col_candidates=("time_ms",))

    feats = compute_core_features_train30(candles, funding, cfg=cfg)

    scaler = fit_scaler_train30(feats, cfg=cfg)
    scaled = apply_scaler_train30(feats, scaler)

    out_features.parent.mkdir(parents=True, exist_ok=True)
    scaled.to_parquet(out_features)
    joblib.dump(scaler, out_scaler)

    print(f"[feature_engine_train30] wrote: {out_features}  shape={scaled.shape}")
    print(f"[feature_engine_train30] wrote: {out_scaler}")


def _cmd_build(args: argparse.Namespace) -> None:
    build(
        input_15m=Path(args.input_15m),
        input_funding=Path(args.input_funding),
        out_features=Path(args.out_features),
        out_scaler=Path(args.out_scaler),
        cfg=FeatureEngineConfig(),
    )


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="Build train30 features + scaler into data_processed/train30/")
    b.add_argument("--input-15m", default=str(DEFAULT_INPUT_15M))
    b.add_argument("--input-funding", default=str(DEFAULT_INPUT_FUNDING))
    b.add_argument("--out-features", default=str(DEFAULT_OUTPUT_FEATURES))
    b.add_argument("--out-scaler", default=str(DEFAULT_OUTPUT_SCALER))
    b.set_defaults(func=_cmd_build)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
