"""Labeling utilities for BTC candle datasets.

We want labels that reflect *tradable* direction with realistic constraints.

v1: directional triple-barrier using first passage of up/down barriers.
- label +1: up barrier hit first
- label -1: down barrier hit first
- label  0: neither barrier hit within horizon

Notes
- Uses next bar open as entry to avoid same-bar lookahead.
- Barrier widths can be ATR%-scaled (recommended).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class TripleBarrierConfig:
    horizon: int = 12  # bars
    atr_k: float = 1.5  # barrier multiplier
    fee_bps: float = 6.0  # per side
    slippage_bps: float = 2.0  # per side
    use_atr: bool = True


def _cost_buffer_return(cfg: TripleBarrierConfig) -> float:
    # approximate round-trip costs in return space
    rt_bps = 2.0 * (cfg.fee_bps + cfg.slippage_bps)
    return rt_bps / 10_000.0


def triple_barrier_labels(df: pd.DataFrame, cfg: TripleBarrierConfig) -> pd.Series:
    """Compute {-1,0,+1} labels via first passage of up/down barriers.

    Requires columns: open, high, low, close, atr_pct (if use_atr).
    """
    if cfg.horizon < 1:
        raise ValueError("horizon must be >= 1")

    required = {"open", "high", "low", "close"}
    if cfg.use_atr:
        required.add("atr_pct")
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")

    n = len(df)
    y = np.zeros(n, dtype=np.int8)

    open_ = df["open"].to_numpy(dtype=float)
    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)

    if cfg.use_atr:
        atr_pct = df["atr_pct"].to_numpy(dtype=float)
        base_mult = cfg.atr_k
    else:
        atr_pct = np.ones(n, dtype=float)
        base_mult = cfg.atr_k

    cost_buf = _cost_buffer_return(cfg)

    # last index we can label with next-open entry and horizon lookahead
    last_i = n - cfg.horizon - 2
    for i in range(0, max(-1, last_i) + 1):
        entry = open_[i + 1]  # next open
        if not np.isfinite(entry) or entry <= 0:
            y[i] = 0
            continue

        # volatility scaled barrier in return
        base = base_mult * atr_pct[i]
        if not np.isfinite(base) or base <= 0:
            y[i] = 0
            continue

        b = float(base + cost_buf)

        up_level = entry * (1.0 + b)
        dn_level = entry * (1.0 - b)

        up_t = None
        dn_t = None

        # scan forward
        for j in range(1, cfg.horizon + 1):
            idx = i + 1 + j
            if up_t is None and high[idx] >= up_level:
                up_t = j
            if dn_t is None and low[idx] <= dn_level:
                dn_t = j
            if up_t is not None or dn_t is not None:
                break

        if up_t is None and dn_t is None:
            y[i] = 0
        elif dn_t is None:
            y[i] = 1
        elif up_t is None:
            y[i] = -1
        else:
            y[i] = 1 if up_t < dn_t else -1

    return pd.Series(y, index=df.index, name="y")


def sanity_test_labels() -> None:
    # tiny deterministic test
    df = pd.DataFrame(
        {
            "open": [100, 100, 100, 100],
            "high": [100, 110, 100, 100],
            "low": [100, 100, 90, 100],
            "close": [100, 100, 100, 100],
            "atr_pct": [0.05, 0.05, 0.05, 0.05],
        }
    )
    cfg = TripleBarrierConfig(horizon=2, atr_k=1.0, fee_bps=0.0, slippage_bps=0.0)
    y = triple_barrier_labels(df, cfg)
    # for i=0 entry at bar1 open=100; up barrier=105 hit at bar1 high=110 => long
    assert int(y.iloc[0]) == 1


if __name__ == "__main__":
    sanity_test_labels()
    print("ok")
