"""Time series splitting utilities (walk-forward).

Goal: mimic deployment and avoid leakage when labels look ahead.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from pandas.api.types import is_datetime64_any_dtype


@dataclass
class WalkForwardConfig:
    train_days: int = 180
    val_days: int = 30
    step_days: int = 30
    embargo_bars: int = 0  # typically horizon
    time_col: str = "close_time"


def _ensure_dt(s: pd.Series) -> pd.Series:
    # pandas tz-aware dtype trips up numpy. use pandas type checks.
    if is_datetime64_any_dtype(s.dtype):
        return s
    # assume ms timestamps (Binance open/close_time)
    return pd.to_datetime(s, unit="ms", utc=True)


def walk_forward_splits(df: pd.DataFrame, cfg: WalkForwardConfig):
    """Yield (train_idx, val_idx) index arrays.

    Uses rolling time windows.
    Purges the last `embargo_bars` from training to avoid overlap.
    """
    if cfg.time_col not in df.columns:
        raise ValueError(f"missing time column {cfg.time_col!r}")
    if cfg.train_days <= 0 or cfg.val_days <= 0 or cfg.step_days <= 0:
        raise ValueError("train_days/val_days/step_days must be > 0")

    t = _ensure_dt(df[cfg.time_col])

    start = t.min()
    end = t.max()

    train_td = pd.Timedelta(days=cfg.train_days)
    val_td = pd.Timedelta(days=cfg.val_days)
    step_td = pd.Timedelta(days=cfg.step_days)

    cursor = start + train_td
    while True:
        train_start = cursor - train_td
        train_end = cursor
        val_start = cursor
        val_end = cursor + val_td

        if val_end > end:
            break

        train_mask = (t >= train_start) & (t < train_end)
        val_mask = (t >= val_start) & (t < val_end)

        train_idx = np.flatnonzero(train_mask.to_numpy())
        val_idx = np.flatnonzero(val_mask.to_numpy())

        if len(train_idx) == 0 or len(val_idx) == 0:
            cursor = cursor + step_td
            continue

        if cfg.embargo_bars > 0 and len(train_idx) > cfg.embargo_bars:
            train_idx = train_idx[: -cfg.embargo_bars]

        yield train_idx, val_idx

        cursor = cursor + step_td
