"""Simple cost-aware backtest for long/short/flat signals.

This is intentionally minimal (v0):
- single position at a time
- entry on next open
- SL/TP fixed as ATR%-scaled barrier (same spirit as labels)
- fees+slippage
- leverage cap

NOT production trading code.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np
import pandas as pd

from features import make_features


@dataclass
class BacktestConfig:
    # trade management
    horizon: int = 12  # max bars to hold (force exit at next open)
    atr_k: float = 1.5  # TP/SL distance = atr_k * atr_pct (symmetric barriers)

    # costs
    fee_bps: float = 6.0
    slippage_bps: float = 2.0

    # signal/positioning
    entry_threshold: float = 0.20  # threshold on (p_long - p_short)
    leverage: float = 3.0
    max_leverage: float = 5.0
    max_daily_trades: int = 5

    # evaluation cadence: only allow NEW entries when a trigger fires
    # (position management always runs every bar)
    trigger: str = "none"  # none|ma_cross|vol_spike|regime_shift
    ma_fast: int = 12
    ma_slow: int = 48
    vol_window: int = 96
    vol_mult: float = 1.5


def _compute_trigger(df: pd.DataFrame, cfg: BacktestConfig) -> pd.Series:
    """Return a boolean series where True means "allow model to (re)decide when flat".

    This is a simple event-driven cadence to avoid evaluating/entering every candle.
    """

    if cfg.trigger in (None, "none"):
        # default behaviour: evaluate every bar (legacy)
        return pd.Series(True, index=df.index)

    close = df["close"].astype(float)

    if cfg.trigger == "ma_cross":
        fast = close.ewm(span=int(cfg.ma_fast), adjust=False).mean()
        slow = close.ewm(span=int(cfg.ma_slow), adjust=False).mean()
        spread = fast - slow
        # Cross event = sign change of spread
        cross = (spread.shift(1) <= 0) & (spread > 0) | ((spread.shift(1) >= 0) & (spread < 0))
        return cross.fillna(False)

    if cfg.trigger == "vol_spike":
        w = int(cfg.vol_window)
        # Use atr_pct (computed by features) as vol proxy
        base = df["atr_pct"].astype(float).rolling(w, min_periods=max(10, w // 4)).median()
        trig = df["atr_pct"].astype(float) > (base * float(cfg.vol_mult))
        return trig.fillna(False)

    if cfg.trigger == "regime_shift":
        # crude regime change: rolling stdev of log returns increases/decreases sharply
        w = int(cfg.vol_window)
        r = np.log(close).diff()
        vol = r.rolling(w, min_periods=max(10, w // 4)).std()
        vol_prev = vol.shift(w)
        trig = (vol_prev > 0) & ((vol / vol_prev) > float(cfg.vol_mult))
        return trig.fillna(False)

    raise ValueError(f"unknown trigger: {cfg.trigger}")


def run_backtest(candles: pd.DataFrame, preds: pd.DataFrame | None, cfg: BacktestConfig) -> pd.DataFrame:
    df = candles.copy()
    if "atr_pct" not in df.columns:
        df = make_features(df)

    time_col = "close_time" if "close_time" in df.columns else "open_time"

    # merge preds by time (if provided)
    if preds is not None:
        p = preds[[time_col, "p_short", "p_flat", "p_long"]].copy()
        df = df.merge(p, on=time_col, how="left")
    else:
        # baseline modes can run without preds
        df["p_short"] = np.nan
        df["p_flat"] = np.nan
        df["p_long"] = np.nan

    # need atr and price columns
    df = df.dropna(subset=["atr_pct", "open", "high", "low", "close"]).reset_index(drop=True)

    trigger = _compute_trigger(df, cfg)
    df["trigger"] = trigger.astype(bool)

    slip = cfg.slippage_bps / 10_000.0
    fee = cfg.fee_bps / 10_000.0
    lev = min(cfg.leverage, cfg.max_leverage)

    equity = 1.0
    eq: list[float] = []

    # diagnostics
    entries = 0
    exits = 0
    round_trips = 0
    wins = 0
    losses = 0
    trade_rets: list[float] = []  # per round-trip, BEFORE leverage & fees (price return)
    trade_bars: list[int] = []

    pos = 0  # -1 short, 0 flat, +1 long
    entry: float | None = None  # executed entry price incl slippage
    tp: float | None = None
    sl: float | None = None
    entry_i: int | None = None

    # track daily trades
    current_day = None
    daily_trades = 0

    for i in range(len(df) - 2):
        # track day for trade limiting
        ts = df.loc[i, time_col]
        day = ts.date() if hasattr(ts, 'date') else pd.to_datetime(ts).date()
        if day != current_day:
            current_day = day
            daily_trades = 0

        # model edge (if preds are present)
        p_long = df.loc[i, "p_long"]
        p_short = df.loc[i, "p_short"]
        edge = float(p_long - p_short) if (pd.notna(p_long) and pd.notna(p_short)) else 0.0

        # decide desired position when flat
        if pos == 0:
            desired = 0

            # event-driven evaluation: only allow new entries on trigger bars
            # AND respect daily trade limit
            if bool(df.loc[i, "trigger"]) and daily_trades < cfg.max_daily_trades:
                if edge > cfg.entry_threshold:
                    desired = 1
                elif edge < -cfg.entry_threshold:
                    desired = -1

            if desired != 0:
                # enter at next open
                entry_price = float(df.loc[i + 1, "open"])
                b = float(cfg.atr_k * df.loc[i, "atr_pct"])
                if desired == 1:
                    entry_exec = entry_price * (1.0 + slip)
                    tp = entry_price * (1.0 + b)
                    sl = entry_price * (1.0 - b)
                else:
                    entry_exec = entry_price * (1.0 - slip)
                    tp = entry_price * (1.0 - b)
                    sl = entry_price * (1.0 + b)

                pos = desired
                entry = entry_exec
                entry_i = i + 1
                daily_trades += 1

                # pay entry fee (fee scales with notional, approx leverage)
                equity *= (1.0 - fee * lev)
                entries += 1

        # manage open position
        if pos != 0 and entry is not None and entry_i is not None:
            # if held too long, force exit at next open
            if (i + 1) - entry_i >= cfg.horizon:
                exit_price = float(df.loc[i + 1, "open"])
                if pos == 1:
                    exit_exec = exit_price * (1.0 - slip)
                    ret = (exit_exec / entry) - 1.0
                else:
                    exit_exec = exit_price * (1.0 + slip)
                    ret = (entry / exit_exec) - 1.0

                equity *= (1.0 + lev * ret)
                equity *= (1.0 - fee * lev)

                exits += 1
                round_trips += 1
                trade_rets.append(float(ret))
                trade_bars.append(int((i + 1) - entry_i))
                if ret >= 0:
                    wins += 1
                else:
                    losses += 1

                pos = 0
                entry = tp = sl = None
                entry_i = None
            else:
                hi = float(df.loc[i + 1, "high"])
                lo = float(df.loc[i + 1, "low"])
                hit = None

                if pos == 1:
                    # conservative: if both hit in same bar, assume SL first
                    if lo <= float(sl):
                        exit_exec = float(sl) * (1.0 - slip)
                        ret = (exit_exec / float(entry)) - 1.0
                        hit = (exit_exec, ret)
                    elif hi >= float(tp):
                        exit_exec = float(tp) * (1.0 - slip)
                        ret = (exit_exec / float(entry)) - 1.0
                        hit = (exit_exec, ret)
                else:
                    if hi >= float(sl):
                        exit_exec = float(sl) * (1.0 + slip)
                        ret = (float(entry) / exit_exec) - 1.0
                        hit = (exit_exec, ret)
                    elif lo <= float(tp):
                        exit_exec = float(tp) * (1.0 + slip)
                        ret = (float(entry) / exit_exec) - 1.0
                        hit = (exit_exec, ret)

                if hit is not None:
                    _, ret = hit
                    equity *= (1.0 + lev * ret)
                    equity *= (1.0 - fee * lev)

                    exits += 1
                    round_trips += 1
                    trade_rets.append(float(ret))
                    trade_bars.append(int((i + 1) - entry_i))
                    if ret >= 0:
                        wins += 1
                    else:
                        losses += 1

                    pos = 0
                    entry = tp = sl = None
                    entry_i = None

        eq.append(equity)

    out = df.iloc[: len(eq)].copy()
    out["equity"] = eq
    out["edge"] = (out["p_long"] - out["p_short"]).astype(float)

    out.attrs["entries"] = int(entries)
    out.attrs["exits"] = int(exits)
    out.attrs["round_trips"] = int(round_trips)
    out.attrs["wins"] = int(wins)
    out.attrs["losses"] = int(losses)
    out.attrs["win_rate"] = float(wins / round_trips) if round_trips else 0.0
    out.attrs["avg_trade_ret"] = float(np.mean(trade_rets)) if trade_rets else 0.0
    out.attrs["med_trade_ret"] = float(np.median(trade_rets)) if trade_rets else 0.0
    out.attrs["avg_trade_bars"] = float(np.mean(trade_bars)) if trade_bars else 0.0

    # Estimated total cost paid as fraction of equity at the time of payment.
    out.attrs["fee_bps"] = float(cfg.fee_bps)
    out.attrs["slippage_bps"] = float(cfg.slippage_bps)
    out.attrs["leverage"] = float(lev)
    out.attrs["trigger"] = str(cfg.trigger)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candles", default="../data/btcusdt_5m.parquet")
    ap.add_argument("--preds", default="../data/preds_lgbm_5m.parquet")
    ap.add_argument("--out", default="../data/backtest_equity.parquet")

    ap.add_argument("--horizon", type=int, default=12)
    ap.add_argument("--atr-k", type=float, default=1.5)
    ap.add_argument("--fee-bps", type=float, default=6.0)
    ap.add_argument("--slip-bps", type=float, default=2.0)
    ap.add_argument("--threshold", type=float, default=0.20)
    ap.add_argument("--leverage", type=float, default=3.0)
    ap.add_argument("--max-daily-trades", type=int, default=5)

    ap.add_argument(
        "--mode",
        choices=["model", "always_flat", "always_long", "always_short"],
        default="model",
        help="sanity baselines: always_flat/always_long/always_short",
    )
    ap.add_argument(
        "--trigger",
        choices=["none", "ma_cross", "vol_spike", "regime_shift"],
        default="none",
        help="event-driven evaluation: only allow NEW entries on trigger bars",
    )
    ap.add_argument("--ma-fast", type=int, default=12)
    ap.add_argument("--ma-slow", type=int, default=48)
    ap.add_argument("--vol-window", type=int, default=96)
    ap.add_argument("--vol-mult", type=float, default=1.5)

    args = ap.parse_args()

    candles = pd.read_parquet(args.candles)

    # Load or synthesize preds depending on mode
    if args.mode == "model":
        preds = pd.read_parquet(args.preds)
    else:
        time_col = "close_time" if "close_time" in candles.columns else "open_time"
        preds = candles[[time_col]].copy()
        if args.mode == "always_flat":
            preds["p_short"], preds["p_flat"], preds["p_long"] = 0.0, 1.0, 0.0
        elif args.mode == "always_long":
            preds["p_short"], preds["p_flat"], preds["p_long"] = 0.0, 0.0, 1.0
        elif args.mode == "always_short":
            preds["p_short"], preds["p_flat"], preds["p_long"] = 1.0, 0.0, 0.0
        else:
            raise ValueError(args.mode)

    cfg = BacktestConfig(
        horizon=args.horizon,
        atr_k=args.atr_k,
        fee_bps=args.fee_bps,
        slippage_bps=args.slip_bps,
        entry_threshold=args.threshold,
        leverage=args.leverage,
        max_daily_trades=args.max_daily_trades,
        trigger=args.trigger,
        ma_fast=args.ma_fast,
        ma_slow=args.ma_slow,
        vol_window=args.vol_window,
        vol_mult=args.vol_mult,
    )

    eq = run_backtest(candles, preds, cfg)
    eq.to_parquet(args.out, index=False)

    final = float(eq["equity"].iloc[-1]) if len(eq) else 1.0
    dd = (eq["equity"] / eq["equity"].cummax() - 1.0).min() if len(eq) else 0.0
    entries = int(eq.attrs.get("entries", 0))
    exits = int(eq.attrs.get("exits", 0))
    round_trips = int(eq.attrs.get("round_trips", 0))
    wins = int(eq.attrs.get("wins", 0))
    losses = int(eq.attrs.get("losses", 0))
    wr = float(eq.attrs.get("win_rate", 0.0))
    avg_ret = float(eq.attrs.get("avg_trade_ret", 0.0))
    avg_bars = float(eq.attrs.get("avg_trade_bars", 0.0))
    trig = str(eq.attrs.get("trigger", "none"))

    print(
        "final equity: "
        f"{final:.6f} | max drawdown: {dd:.3%} | rows: {len(eq)} | "
        f"entries: {entries} | exits: {exits} | round_trips: {round_trips} | "
        f"wins: {wins} | losses: {losses} | win_rate: {wr:.1%} | "
        f"avg_trade_ret: {avg_ret:.4%} | avg_hold_bars: {avg_bars:.1f} | trigger: {trig}"
    )
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
