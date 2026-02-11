"""projects/dings-trader/TraderHimSelf/feature_engine.py

Schritt 5 — Feature Engine

- Berechnet den *fixen* Core-Feature-Vektor (aktuell 33D) (Reihenfolge gemäß TRAINING_ROADMAP.md)
- Fit StandardScaler nur auf Train (2019–2023) und speichert ihn als scaler.pkl
- Wendet Scaler identisch auf Val/Test/Live an (nie live fitten)
- Schreibt Output: data_processed/features.parquet
- Enthält Parity Unit-Test: Offline-Batch vs. Live-Stream Simulation

Nutzung (lokal):
    cd TraderHimSelf
    venv/bin/python feature_engine.py build
    venv/bin/python feature_engine.py parity

Hinweis:
    In diesem Repo sind aktuell nur Beispiel/Mock-Daten (2024, wenige Tage) vorhanden.
    Der Scaler-Fit auf 2019–2023 funktioniert daher erst, wenn echte Historie geladen ist.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler


# -----------------------------------------------------------------------------
# Feature order (FIX!)
# -----------------------------------------------------------------------------

FEATURE_COLUMNS: List[str] = [
    # A) Returns & Range
    "ret_1",
    "ret_4",
    "ret_16",
    "ret_48",
    "hl_range_pct",
    "oc_range_pct",
    # B) Volatility & ATR
    "vol_16",
    "vol_96",
    "vol_672",
    "atr_14",
    # C) Trend / Mean Reversion
    "ema_20_dist",
    "ema_50_dist",
    "ema_200_dist",
    "ema_20_slope",
    "ema_50_slope",
    "adx_14",
    # D) Momentum
    "rsi_14",
    "macd",
    "macd_signal",
    "macd_hist",
    # E) Volume
    "vol_log",
    "vol_z_96",
    # F) Time (UTC, cyclic + session flags + optional seasonality)
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "session_asia",
    "session_europe",
    "session_us",
    "woy_sin",
    "woy_cos",
    # G) Funding
    "funding_rate_now",
    "time_to_next_funding_steps",
]


# -----------------------------------------------------------------------------
# Paths / defaults
# -----------------------------------------------------------------------------

DEFAULT_INPUT_15M = Path("data_processed/aligned_15m.parquet")
DEFAULT_INPUT_FUNDING = Path("data_processed/funding.parquet")
DEFAULT_OUTPUT_FEATURES = Path("data_processed/features.parquet")
DEFAULT_OUTPUT_SCALER = Path("data_processed/scaler.pkl")


@dataclass(frozen=True)
class FeatureEngineConfig:
    """Configuration for building and scaling core features."""

    train_start: str = "2019-01-01 00:00:00"
    train_end: str = "2023-12-31 23:59:59"
    funding_cycle_hours: int = 8
    tz_aware_ok: bool = True


# -----------------------------------------------------------------------------
# Indicator helpers (pandas/numpy, deterministic)
# -----------------------------------------------------------------------------


def ema(series: pd.Series, span: int) -> pd.Series:
    """Exponential moving average (adjust=False for deterministic streaming parity)."""

    return series.ewm(span=span, adjust=False).mean()


def atr_wilder(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    """ATR using Wilder's smoothing (EMA with alpha=1/window)."""

    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / window, adjust=False).mean()


def rsi_wilder(close: pd.Series, window: int = 14) -> pd.Series:
    """RSI using Wilder's smoothing."""

    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1 / window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / window, adjust=False).mean()

    avg_loss = avg_loss.replace(0.0, np.nan)
    rs = avg_gain / avg_loss
    out = 100.0 - (100.0 / (1.0 + rs))
    return out


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """MACD line, signal line, histogram."""

    macd_line = ema(close, fast) - ema(close, slow)
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def adx_wilder(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    """ADX using Wilder's smoothing."""

    up_move = high.diff()
    down_move = -low.diff()  # (prev_low - low)

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    atr = tr.ewm(alpha=1 / window, adjust=False).mean()
    atr = atr.replace(0.0, np.nan)

    plus_dm_s = pd.Series(plus_dm, index=close.index).ewm(alpha=1 / window, adjust=False).mean()
    minus_dm_s = pd.Series(minus_dm, index=close.index).ewm(alpha=1 / window, adjust=False).mean()

    plus_di = 100.0 * (plus_dm_s / atr)
    minus_di = 100.0 * (minus_dm_s / atr)

    denom = (plus_di + minus_di).replace(0.0, np.nan)
    dx = 100.0 * (plus_di - minus_di).abs() / denom

    return dx.ewm(alpha=1 / window, adjust=False).mean()


def rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    """Rolling z-score with ddof=0 (population std) for determinism."""

    r = series.rolling(window=window)
    mean = r.mean()
    std = r.std(ddof=0).replace(0.0, np.nan)
    return (series - mean) / std


# -----------------------------------------------------------------------------
# IO / alignment helpers
# -----------------------------------------------------------------------------


def _ensure_datetime_index(df: pd.DataFrame, *, time_col_candidates: Sequence[str] = ("open_time_ms", "time_ms")) -> pd.DataFrame:
    """Ensure df has a DatetimeIndex; if not, try converting known timestamp columns."""

    if isinstance(df.index, pd.DatetimeIndex):
        return df

    for c in time_col_candidates:
        if c in df.columns:
            df = df.copy()
            df.index = pd.to_datetime(df[c], unit="ms", utc=True)
            return df

    raise ValueError("DataFrame has no DatetimeIndex and no known time column to convert.")


def _merge_funding_asof(candles: pd.DataFrame, funding: pd.DataFrame) -> pd.DataFrame:
    """Merge funding rate into candles by asof backward match (step-wise funding)."""

    candles = candles.sort_index().copy()
    funding = funding.sort_index().copy()

    candles["__ts"] = candles.index
    funding["__ts"] = funding.index

    merged = pd.merge_asof(
        candles,
        funding[["__ts", "funding_rate"]],
        on="__ts",
        direction="backward",
    )
    merged.index = candles.index
    merged = merged.drop(columns=["__ts"])

    return merged


# -----------------------------------------------------------------------------
# Core feature computation (batch)
# -----------------------------------------------------------------------------


def compute_core_features(buf_15m: pd.DataFrame, funding_df: Optional[pd.DataFrame] = None, *, cfg: FeatureEngineConfig = FeatureEngineConfig()) -> pd.DataFrame:
    """Compute the core features (see FEATURE_COLUMNS) on a buffer of 15m candles.

    This function is designed to be used *identically* for offline batch computation
    and for live/streaming computation (input = rolling buffer).

    Args:
        buf_15m: DataFrame with columns open, high, low, close, volume.
                Index must be DatetimeIndex (UTC recommended).
        funding_df: Optional funding series (DataFrame) containing funding_rate.
                    Index must be DatetimeIndex.
        cfg: Configuration.

    Returns:
        DataFrame with FEATURE_COLUMNS.

    Notes:
        The first rows will contain NaNs due to rolling windows (up to 672).
    """

    df = _ensure_datetime_index(buf_15m)

    required = ["open", "high", "low", "close", "volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.copy()
    for c in required:
        df[c] = df[c].astype(float)

    # Funding merge
    if "funding_rate" not in df.columns:
        if funding_df is not None:
            funding_df = _ensure_datetime_index(funding_df, time_col_candidates=("time_ms",))
            if "funding_rate" not in funding_df.columns:
                raise ValueError("funding_df must contain 'funding_rate' column")
            df = _merge_funding_asof(df, funding_df)
        else:
            df["funding_rate"] = 0.0

    df["funding_rate"] = df["funding_rate"].fillna(0.0)

    log_close = np.log(df["close"].replace(0.0, np.nan))

    # A) Returns & range
    out = pd.DataFrame(index=df.index)
    out["ret_1"] = log_close.diff(1)
    out["ret_4"] = log_close.diff(4)
    out["ret_16"] = log_close.diff(16)
    out["ret_48"] = log_close.diff(48)
    out["hl_range_pct"] = (df["high"] - df["low"]) / df["close"].replace(0.0, np.nan)
    out["oc_range_pct"] = (df["close"] - df["open"]) / df["open"].replace(0.0, np.nan)

    # B) Volatility & ATR
    out["vol_16"] = out["ret_1"].rolling(16).std(ddof=0)
    out["vol_96"] = out["ret_1"].rolling(96).std(ddof=0)
    out["vol_672"] = out["ret_1"].rolling(672).std(ddof=0)
    out["atr_14"] = atr_wilder(df["high"], df["low"], df["close"], 14)

    # C) Trend / mean reversion
    ema20 = ema(df["close"], 20)
    ema50 = ema(df["close"], 50)
    ema200 = ema(df["close"], 200)

    out["ema_20_dist"] = (df["close"] - ema20) / ema20.replace(0.0, np.nan)
    out["ema_50_dist"] = (df["close"] - ema50) / ema50.replace(0.0, np.nan)
    out["ema_200_dist"] = (df["close"] - ema200) / ema200.replace(0.0, np.nan)

    out["ema_20_slope"] = (ema20 - ema20.shift(1)) / ema20.shift(1).replace(0.0, np.nan)
    out["ema_50_slope"] = (ema50 - ema50.shift(1)) / ema50.shift(1).replace(0.0, np.nan)

    out["adx_14"] = adx_wilder(df["high"], df["low"], df["close"], 14)

    # D) Momentum
    out["rsi_14"] = rsi_wilder(df["close"], 14)
    macd_line, macd_sig, macd_hist = macd(df["close"], 12, 26, 9)
    out["macd"] = macd_line
    out["macd_signal"] = macd_sig
    out["macd_hist"] = macd_hist

    # E) Volume
    out["vol_log"] = np.log1p(df["volume"].clip(lower=0.0))
    out["vol_z_96"] = rolling_zscore(out["vol_log"], 96)

    # F) Time (UTC)
    # NOTE: These features use only the candle timestamp (no future data) => no leakage.
    ts = out.index
    hours = ts.hour  # 0..23 (UTC)
    dows = ts.dayofweek  # 0..6 (Mon..Sun)

    # Cyclic encodings
    out["hour_sin"] = np.sin(2 * np.pi * hours / 24.0)
    out["hour_cos"] = np.cos(2 * np.pi * hours / 24.0)
    out["dow_sin"] = np.sin(2 * np.pi * dows / 7.0)
    out["dow_cos"] = np.cos(2 * np.pi * dows / 7.0)

    # Session flags (deterministic from UTC hour; overlapping by design)
    # Asia:   00:00–08:59 UTC
    # Europe: 07:00–15:59 UTC
    # US:     13:00–21:59 UTC
    out["session_asia"] = ((hours >= 0) & (hours < 9)).astype(float)
    out["session_europe"] = ((hours >= 7) & (hours < 16)).astype(float)
    out["session_us"] = ((hours >= 13) & (hours < 22)).astype(float)

    # Week-of-year seasonality (ISO week). Encode cyclic to avoid discontinuity at year boundaries.
    # ISO week can be 53 for some years; we wrap it into 0..51 for stable 52-week cyclic encoding.
    woy = (ts.isocalendar().week.astype(int) - 1) % 52
    out["woy_sin"] = np.sin(2 * np.pi * woy / 52.0)
    out["woy_cos"] = np.cos(2 * np.pi * woy / 52.0)

    # G) Funding
    out["funding_rate_now"] = df["funding_rate"].astype(float)

    # Time to next funding in 15m steps, capped 0..32.
    # Funding at 00:00, 08:00, 16:00 UTC -> 8h cycle -> 32 steps.
    cycle = cfg.funding_cycle_hours
    steps_total = int((cycle * 60) / 15)
    hour_mod = (hours % cycle).astype(int)
    minute_steps = (ts.minute // 15).astype(int)
    steps_in_cycle = hour_mod * 4 + minute_steps
    steps_left = pd.Series(steps_total - steps_in_cycle, index=out.index).clip(lower=0, upper=steps_total)
    out["time_to_next_funding_steps"] = steps_left.astype(float)

    # Enforce fixed column order
    return out[FEATURE_COLUMNS]


# -----------------------------------------------------------------------------
# Scaling
# -----------------------------------------------------------------------------


def fit_scaler(features: pd.DataFrame, *, cfg: FeatureEngineConfig = FeatureEngineConfig()) -> StandardScaler:
    """Fit a StandardScaler on the train split only (2019-2023)."""

    # Ensure train window timestamps are tz-aware (UTC) to match our UTC DatetimeIndex.
    train_start = pd.Timestamp(cfg.train_start)
    train_end = pd.Timestamp(cfg.train_end)
    if train_start.tz is None:
        train_start = train_start.tz_localize("UTC")
    if train_end.tz is None:
        train_end = train_end.tz_localize("UTC")

    mask = (features.index >= train_start) & (features.index <= train_end)
    train = features.loc[mask].dropna()

    if len(train) == 0:
        # Dev/CI friendliness: allow running with limited/mock data.
        # Roadmap default is still "fit on 2019-2023" — if that window is absent,
        # we fallback to "fit on all available non-NaN rows" with a loud warning.
        train = features.dropna()
        if len(train) == 0:
            print(
                "[feature_engine] WARNING: No non-NaN rows available to fit scaler. "
                "Falling back to a dummy scaler fitted on zeros (mock data)."
            )
            train = pd.DataFrame(np.zeros((1, len(FEATURE_COLUMNS))), columns=FEATURE_COLUMNS)
        print(
            "[feature_engine] WARNING: No rows in the configured train window (2019-2023). "
            "Falling back to fitting scaler on all available non-NaN rows (likely mock/short data)."
        )

    scaler = StandardScaler()
    scaler.fit(train.values)
    return scaler


def apply_scaler(features: pd.DataFrame, scaler: StandardScaler) -> pd.DataFrame:
    """Apply scaler to rows without NaNs; keep NaN-rows unchanged."""

    out = features.copy()
    valid = ~features.isna().any(axis=1)
    if valid.any():
        out.loc[valid, FEATURE_COLUMNS] = scaler.transform(features.loc[valid, FEATURE_COLUMNS].values)
    return out


# -----------------------------------------------------------------------------
# Parity test
# -----------------------------------------------------------------------------


def parity_test(
    candles_15m: pd.DataFrame,
    funding: Optional[pd.DataFrame],
    *,
    buffer_size: int = 800,
    sample_points: int = 50,
    atol: float = 1e-8,
) -> None:
    """Parity unit test: offline features vs. streaming buffer computation.

    We compute offline features once on the full series.
    Then we simulate live by feeding a rolling buffer into compute_core_features.
    For chosen timestamps, the last row of buffer features must equal offline row.

    Raises:
        AssertionError if mismatch.
    """

    candles_15m = _ensure_datetime_index(candles_15m)
    if funding is not None:
        funding = _ensure_datetime_index(funding, time_col_candidates=("time_ms",))

    offline = compute_core_features(candles_15m, funding)

    valid = ~offline.isna().any(axis=1)
    valid_idx = offline.index[valid]
    if len(valid_idx) < sample_points:
        raise ValueError("Not enough valid rows for parity test. Need more history (>=672+).")

    test_ts = valid_idx[-sample_points:]

    for ts in test_ts:
        pos = candles_15m.index.get_loc(ts)
        start = max(0, pos - buffer_size + 1)
        buf = candles_15m.iloc[start : pos + 1]

        stream = compute_core_features(buf, funding)
        v_stream = stream.iloc[-1].to_numpy(dtype=float)
        v_off = offline.loc[ts].to_numpy(dtype=float)

        if not np.allclose(v_stream, v_off, atol=atol, equal_nan=True):
            diff = np.nanmax(np.abs(v_stream - v_off))
            raise AssertionError(f"Parity mismatch at {ts}: max_abs_diff={diff}")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def build(
    *,
    in_15m: Path = DEFAULT_INPUT_15M,
    in_funding: Path = DEFAULT_INPUT_FUNDING,
    out_features: Path = DEFAULT_OUTPUT_FEATURES,
    out_scaler: Path = DEFAULT_OUTPUT_SCALER,
    cfg: FeatureEngineConfig = FeatureEngineConfig(),
) -> None:
    """Build features + fit scaler + write artifacts."""

    candles = pd.read_parquet(in_15m)
    funding = pd.read_parquet(in_funding) if in_funding.exists() else None

    candles = _ensure_datetime_index(candles)
    if funding is not None:
        funding = _ensure_datetime_index(funding, time_col_candidates=("time_ms",))

    feats = compute_core_features(candles, funding, cfg=cfg)

    scaler = fit_scaler(feats, cfg=cfg)
    out_scaler.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, out_scaler)

    scaled = apply_scaler(feats, scaler)

    out_features.parent.mkdir(parents=True, exist_ok=True)
    scaled.to_parquet(out_features)


def _usage() -> str:
    return (
        "Usage:\n"
        "  python feature_engine.py build   # compute features + fit scaler (2019-2023)\n"
        "  python feature_engine.py parity  # run parity unit test\n"
    )


def main(argv: Optional[Sequence[str]] = None) -> None:
    import sys

    args = list(sys.argv[1:] if argv is None else argv)
    cmd = args[0] if args else "build"

    if cmd == "build":
        build()
        print(f"Wrote {DEFAULT_OUTPUT_FEATURES} and {DEFAULT_OUTPUT_SCALER}")
        return

    if cmd == "parity":
        candles = pd.read_parquet(DEFAULT_INPUT_15M)
        funding = pd.read_parquet(DEFAULT_INPUT_FUNDING) if DEFAULT_INPUT_FUNDING.exists() else None
        parity_test(candles, funding)
        print("✅ Parity test passed")
        return

    raise SystemExit(_usage())


if __name__ == "__main__":
    main()
