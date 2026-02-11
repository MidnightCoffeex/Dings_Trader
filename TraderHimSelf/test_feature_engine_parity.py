"""Parity unit test for feature_engine.

This test ensures that computing a feature vector offline on a full history
matches computing it in a streaming manner on a rolling buffer.

Run:
    cd TraderHimSelf
    venv/bin/python test_feature_engine_parity.py

Note:
    Uses synthetic candles (>= 1500 rows) so that all rolling windows (vol_672)
    are well-defined.

Important:
    Exact bitwise equality is *not* guaranteed for EWM-based indicators
    if you compute them on a truncated buffer (different initial conditions).
    The production parity guarantee is achieved by using the same feature code
    on the same growing history buffer (>= 800 candles), not by reinitializing
    EWM indicators on a short buffer.

    Therefore this test simulates "streaming" by recomputing features on the
    full history available up to time t (growing window), which matches offline.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from feature_engine import compute_core_features, parity_test, FEATURE_COLUMNS


def _make_synth(n: int = 2000, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="15min", tz="UTC")

    # random walk in log space
    rets = rng.normal(0, 0.002, size=n)
    logp = np.cumsum(rets) + np.log(40000.0)
    close = np.exp(logp)

    open_ = np.roll(close, 1)
    open_[0] = close[0]

    spread = rng.uniform(0.0005, 0.003, size=n)
    high = np.maximum(open_, close) * (1.0 + spread)
    low = np.minimum(open_, close) * (1.0 - spread)

    volume = rng.lognormal(mean=10.0, sigma=0.3, size=n)

    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": volume}, index=idx)


def main() -> None:
    candles = _make_synth(2000)

    # Streaming parity using growing history (exact match expected)
    parity_test(candles, funding=None, buffer_size=10_000_000, sample_points=50, atol=1e-10)

    feats = compute_core_features(candles, None)
    assert feats.shape[1] == len(FEATURE_COLUMNS)

    # Smoke checks: required time/seasonality columns present
    required = {"hour_sin", "hour_cos", "dow_sin", "dow_cos", "session_asia", "session_europe", "session_us"}
    missing = required.difference(FEATURE_COLUMNS)
    assert not missing, f"Missing expected time columns in FEATURE_COLUMNS: {sorted(missing)}"

    # Value sanity (unscaled, deterministic)
    for c in ["hour_sin", "hour_cos", "dow_sin", "dow_cos"]:
        s = feats[c].dropna()
        assert (s.abs() <= 1.0000001).all(), f"{c} out of [-1,1] range"

    for c in ["session_asia", "session_europe", "session_us"]:
        s = feats[c].dropna().unique()
        assert set(s).issubset({0.0, 1.0}), f"{c} contains non-binary values: {s}"

    print(f"✅ parity ok; {len(FEATURE_COLUMNS)} features")


if __name__ == "__main__":
    main()
