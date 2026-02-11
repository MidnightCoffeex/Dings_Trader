"""Dataset builder (Step 4).

Builds aligned datasets for offline training/backtest.

Inputs (from data_raw/):
- btcusdt_15m.parquet: OHLCV candles (15m)
- btcusdt_3m.parquet: OHLCV candles (3m)
- btcusdt_funding.parquet: funding events (typically 8h)

Outputs (to data_processed/):
- aligned_15m.parquet: strict 15m grid, gaps kept as NaNs and marked via flags
- aligned_3m.parquet: strict 3m grid covering the 15m span, gaps kept as NaNs
- funding.parquet: funding_rate mapped step-wise (ffill) onto 15m grid

Design goals:
- No time-travel: only backward-looking joins/ffill.
- UTC everywhere.
- Each 15m slot has exactly 5 expected 3m subbars (T, T+3, T+6, T+9, T+12).
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Paths:
    base_dir: str

    @property
    def raw_dir(self) -> str:
        return os.path.join(self.base_dir, "data_raw")

    @property
    def processed_dir(self) -> str:
        return os.path.join(self.base_dir, "data_processed")


DEFAULT_BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def load_parquet(raw_dir: str, filename: str) -> pd.DataFrame:
    """Load a parquet file from raw_dir."""
    path = os.path.join(raw_dir, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found")
    return pd.read_parquet(path)


def _to_utc_index_from_ms(df: pd.DataFrame, ms_col: str) -> pd.DataFrame:
    """Return df indexed by UTC timestamp created from millisecond epoch column."""
    out = df.copy()
    out["time"] = pd.to_datetime(out[ms_col], unit="ms", utc=True)
    out = out.set_index("time").sort_index()
    out = out[~out.index.duplicated(keep="first")]
    return out


def prepare_candles(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare candle records.

    Expected columns: open_time_ms, open, high, low, close, volume
    Returns DataFrame indexed by UTC time.
    """
    if "open_time_ms" not in df.columns:
        raise ValueError("Candle dataframe must contain open_time_ms")
    return _to_utc_index_from_ms(df, "open_time_ms")


def prepare_funding(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare funding records.

    Supported input schemas (we see both in the wild):

    1) Contract schema (preferred): columns `time_ms`, `funding_rate`
    2) Binance export schema: index `fundingTime` (datetime) + column `fundingRate`
       (sometimes also `symbol`, `markPrice`)

    Returns a DataFrame indexed by UTC time with a single column `funding_rate`.
    """

    out = df.copy()

    # Case A: contract schema with epoch ms
    if "time_ms" in out.columns:
        if "funding_rate" not in out.columns:
            # common variant naming
            if "fundingRate" in out.columns:
                out = out.rename(columns={"fundingRate": "funding_rate"})
            else:
                raise ValueError("Funding dataframe must contain funding_rate (or fundingRate)")
        return _to_utc_index_from_ms(out, "time_ms")[["funding_rate"]]

    # Case B: already indexed by datetime (e.g. fundingTime index)
    if isinstance(out.index, pd.DatetimeIndex):
        idx = out.index
        # Normalize tz to UTC
        if idx.tz is None:
            idx = idx.tz_localize("UTC")
        else:
            idx = idx.tz_convert("UTC")
        out.index = idx

        if "funding_rate" not in out.columns:
            if "fundingRate" in out.columns:
                out = out.rename(columns={"fundingRate": "funding_rate"})
            else:
                raise ValueError("Funding dataframe must contain funding_rate (or fundingRate)")

        return out[["funding_rate"]].sort_index()

    # Case C: fundingTime column as datetime
    if "fundingTime" in out.columns:
        out = out.rename(columns={"fundingRate": "funding_rate"}) if "fundingRate" in out.columns and "funding_rate" not in out.columns else out
        if "funding_rate" not in out.columns:
            raise ValueError("Funding dataframe must contain funding_rate (or fundingRate)")
        out["time"] = pd.to_datetime(out["fundingTime"], utc=True, errors="coerce")
        if out["time"].isna().any():
            raise ValueError("Funding dataframe contains unparsable fundingTime values")
        out = out.set_index("time").sort_index()
        return out[["funding_rate"]]

    raise ValueError("Funding dataframe must contain time_ms or be indexed by datetime (fundingTime).")

def make_strict_grid(start: pd.Timestamp, end: pd.Timestamp, freq: str) -> pd.DatetimeIndex:
    """Create a strict UTC DatetimeIndex [start, end] with given frequency."""
    if start.tz is None or end.tz is None:
        raise ValueError("start/end must be tz-aware (UTC)")
    return pd.date_range(start=start, end=end, freq=freq, tz="UTC")


def align_15m_3m(
    candles_15m: pd.DataFrame,
    candles_3m: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Align 15m and 3m candles.

    Rules:
    - 15m timeline is the master grid.
    - For each 15m start time T, the expected 3m subbars are:
      T + {0,3,6,9,12} minutes.
    - Missing candles are kept as NaNs and marked with boolean flags.

    Returns:
        (aligned_15m, aligned_3m)
    """
    if candles_15m.empty:
        raise ValueError("15m candles dataframe is empty")

    start_time = candles_15m.index.min()
    end_time = candles_15m.index.max()

    # Strict 15m grid (inclusive end)
    idx_15m = make_strict_grid(start_time, end_time, "15min")
    aligned_15m = candles_15m.reindex(idx_15m)

    # Strict 3m grid that covers every 15m candle's 5 expected subbars.
    # If 15m grid ends at end_time, last 15m candle expects a subbar at end_time+12m.
    idx_3m = make_strict_grid(start_time, end_time + pd.Timedelta(minutes=12), "3min")
    aligned_3m = candles_3m.reindex(idx_3m)

    # Missing flags
    aligned_15m["is_missing"] = aligned_15m["close"].isna()
    aligned_3m["is_missing"] = aligned_3m["close"].isna()

    # Also store the parent 15m slot start for each 3m bar (useful for grouping)
    aligned_3m["slot_15m"] = aligned_3m.index.floor("15min")

    return aligned_15m, aligned_3m


def subbar_counts_per_15m(aligned_3m: pd.DataFrame, idx_15m: pd.DatetimeIndex) -> pd.Series:
    """Count present 3m subbars per 15m slot, aligned to idx_15m."""
    present = (~aligned_3m["is_missing"]).astype("int64")
    counts = present.resample("15min", origin="start").sum()
    return counts.reindex(idx_15m)

def map_funding_stepwise(
    funding_events: pd.DataFrame,
    target_index: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Map funding events onto a target timeline step-wise.

    Uses forward-fill (ffill) so each point uses the latest known funding rate
    at or before that timestamp (no future leakage).

    Note: If the timeline starts before the first funding event, the initial
    rows will be NaN. This is intentional (no backfilling from the future).
    """
    aligned = funding_events.reindex(target_index, method="ffill")
    return aligned[["funding_rate"]]


def add_time_ms_column(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """Add a millisecond epoch column from UTC DatetimeIndex."""
    out = df.copy()
    out[name] = out.index.view("int64") // 10**6
    return out


def validate_no_time_travel_funding(
    funding_events: pd.DataFrame,
    mapped: pd.DataFrame,
) -> None:
    """Basic sanity checks to prevent time-travel in funding mapping."""
    # For mapped points, the funding value must equal the last event at or before t.
    # We verify by recomputing with merge_asof.
    events = funding_events.reset_index().rename(columns={funding_events.index.name or "index": "event_time"})
    target = mapped.reset_index().rename(columns={mapped.index.name or "index": "target_time"})

    # pandas merge_asof requires sorted keys
    events = events.sort_values("event_time")
    target = target.sort_values("target_time")

    merged = pd.merge_asof(
        target,
        events[["event_time", "funding_rate"]],
        left_on="target_time",
        right_on="event_time",
        direction="backward",
        allow_exact_matches=True,
    )

    # Compare (NaNs allowed)
    a = merged["funding_rate_x"]
    b = merged["funding_rate_y"]
    ok = (a.isna() & b.isna()) | (a == b)
    if not bool(ok.all()):
        bad = merged.loc[~ok, ["target_time", "funding_rate_x", "funding_rate_y", "event_time"]].head(10)
        raise AssertionError(f"Funding time-travel / mismatch detected. Examples:\n{bad}")


def validate_alignment(aligned_15m: pd.DataFrame, aligned_3m: pd.DataFrame) -> None:
    """Validate that 3m grid is consistent with 15m grid expectations."""
    # 15m index must be strict 15-min spaced
    diffs = aligned_15m.index.to_series().diff().dropna().unique()
    if len(diffs) > 1 or (len(diffs) == 1 and diffs[0] != pd.Timedelta(minutes=15)):
        raise AssertionError("15m index is not a strict 15-minute grid")

    counts = subbar_counts_per_15m(aligned_3m, aligned_15m.index)
    # counts must be between 0 and 5 (because grid has 5 expected subbars)
    if not ((counts >= 0) & (counts <= 5)).all():
        raise AssertionError("Found 15m slots with >5 present 3m subbars (grid mismatch)")

    # Additional strict check: every 3m timestamp should map into a 15m slot within range
    if aligned_3m["slot_15m"].min() < aligned_15m.index.min() or aligned_3m["slot_15m"].max() > aligned_15m.index.max() + pd.Timedelta(minutes=15):
        # allow the last slot+? but this shouldn't happen with our grid
        raise AssertionError("3m slot_15m out of expected range")


def run(paths: Paths) -> None:
    """Main entry: build and save aligned datasets."""
    os.makedirs(paths.processed_dir, exist_ok=True)

    # 1) Load
    candles_15m_raw = load_parquet(paths.raw_dir, "btcusdt_15m.parquet")
    candles_3m_raw = load_parquet(paths.raw_dir, "btcusdt_3m.parquet")
    funding_raw = load_parquet(paths.raw_dir, "btcusdt_funding.parquet")

    # 2) Prepare (UTC index)
    candles_15m = prepare_candles(candles_15m_raw)
    candles_3m = prepare_candles(candles_3m_raw)
    funding_events = prepare_funding(funding_raw)

    # 3) Align
    aligned_15m, aligned_3m = align_15m_3m(candles_15m, candles_3m)
    validate_alignment(aligned_15m, aligned_3m)

    counts = subbar_counts_per_15m(aligned_3m, aligned_15m.index)
    aligned_15m["subbar_count_3m"] = counts.astype("Int64")

    # 4) Funding mapping (onto 15m timeline)
    funding_mapped = map_funding_stepwise(funding_events, aligned_15m.index)
    validate_no_time_travel_funding(funding_events, funding_mapped)

    # 5) Add ms columns and write parquet
    aligned_15m_out = add_time_ms_column(aligned_15m, "open_time_ms")
    aligned_3m_out = add_time_ms_column(aligned_3m, "open_time_ms")
    funding_out = add_time_ms_column(funding_mapped, "time_ms")

    aligned_15m_out.to_parquet(os.path.join(paths.processed_dir, "aligned_15m.parquet"), index=True)
    aligned_3m_out.to_parquet(os.path.join(paths.processed_dir, "aligned_3m.parquet"), index=True)
    funding_out.to_parquet(os.path.join(paths.processed_dir, "funding.parquet"), index=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build aligned 15m/3m/funding datasets")
    parser.add_argument("--base-dir", default=DEFAULT_BASE_DIR, help="TraderHimSelf base dir")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(Paths(base_dir=args.base_dir))


if __name__ == "__main__":
    main()
