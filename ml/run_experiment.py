"""End-to-end experiment runner (v0).

Pipeline:
- load candles
- features
- labels (triple barrier)
- walk-forward train LightGBM
- write predictions parquet
- run backtest and write equity curve
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd

from backtest import BacktestConfig, run_backtest
from features import make_features
from labels import TripleBarrierConfig, triple_barrier_labels
from train_lgbm import main as train_lgbm_main


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="../data/btcusdt_5m.parquet")
    ap.add_argument("--horizon", type=int, default=12)
    ap.add_argument("--atr-k", type=float, default=1.5)
    ap.add_argument("--fee-bps", type=float, default=6.0)
    ap.add_argument("--slip-bps", type=float, default=2.0)
    ap.add_argument("--threshold", type=float, default=0.20)
    ap.add_argument("--leverage", type=float, default=3.0)
    args = ap.parse_args()

    stamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    out_dir = Path("runs") / stamp
    out_dir.mkdir(parents=True, exist_ok=True)

    preds_path = out_dir / "preds.parquet"
    equity_path = out_dir / "equity.parquet"

    # train_lgbm is currently a CLI script; we call it by reusing its code path via subprocess
    # but for simplicity keep it as CLI use in docs. Here we implement minimal in-process logic.
    #
    # For now: just re-run the same logic directly here.
    df = pd.read_parquet(args.data)
    df = make_features(df)

    lcfg = TripleBarrierConfig(horizon=args.horizon, atr_k=args.atr_k, fee_bps=args.fee_bps, slippage_bps=args.slip_bps)
    df["y"] = triple_barrier_labels(df, lcfg)

    # We reuse train_lgbm.py by importing it as a module would require refactor; keep simple:
    from train_lgbm import _class_map  # type: ignore
    from lightgbm import LGBMClassifier
    from split import WalkForwardConfig, walk_forward_splits

    feature_cols = [c for c in df.columns if c not in {"open_time", "close_time", "ignore", "y"}]
    X = df[feature_cols]
    y = df["y"]
    mask = X.notna().all(axis=1) & y.notna()
    X = X[mask]
    y = y[mask]
    df2 = df.loc[mask].copy()

    y_map = _class_map(y)
    time_col = "close_time" if "close_time" in df2.columns else "open_time"
    wcfg = WalkForwardConfig(train_days=180, val_days=30, step_days=30, embargo_bars=args.horizon, time_col=time_col)

    preds = []
    for fold, (tr_idx, va_idx) in enumerate(walk_forward_splits(df2, wcfg), 1):
        clf = LGBMClassifier(
            objective="multiclass",
            num_class=3,
            n_estimators=600,
            learning_rate=0.03,
            num_leaves=63,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            random_state=42,
            class_weight="balanced",
        )
        clf.fit(X.iloc[tr_idx], y_map.iloc[tr_idx])
        proba = clf.predict_proba(X.iloc[va_idx])

        out = df2.iloc[va_idx][[time_col, "open", "high", "low", "close", "volume", "atr_pct"]].copy()
        out["fold"] = fold
        out["y"] = y.iloc[va_idx].astype(int).to_numpy()
        out["p_short"] = proba[:, 0]
        out["p_flat"] = proba[:, 1]
        out["p_long"] = proba[:, 2]
        preds.append(out)

    pred_df = pd.concat(preds, ignore_index=True).sort_values(time_col)
    pred_df.to_parquet(preds_path, index=False)

    btcfg = BacktestConfig(
        horizon=args.horizon,
        atr_k=args.atr_k,
        fee_bps=args.fee_bps,
        slippage_bps=args.slip_bps,
        entry_threshold=args.threshold,
        leverage=args.leverage,
    )
    equity = run_backtest(df, pred_df, btcfg)
    equity.to_parquet(equity_path, index=False)

    final = float(equity["equity"].iloc[-1]) if len(equity) else 1.0
    dd = (equity["equity"] / equity["equity"].cummax() - 1.0).min() if len(equity) else 0.0

    print(f"run: {out_dir}")
    print(f"preds: {preds_path}")
    print(f"equity: {equity_path}")
    print(f"final equity: {final:.3f} | max drawdown: {dd:.3%}")


if __name__ == "__main__":
    main()
