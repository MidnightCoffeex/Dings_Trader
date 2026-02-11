"""Train a LightGBM multiclass baseline on engineered features.

Outputs out-of-sample probabilities per walk-forward fold.

Classes:
  -1 (short), 0 (flat), +1 (long)

We map to 0..2 as: short=0, flat=1, long=2.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report

from lightgbm import LGBMClassifier

from features import make_features
from labels import TripleBarrierConfig, triple_barrier_labels
from split import WalkForwardConfig, walk_forward_splits


def _class_map(y: pd.Series) -> pd.Series:
    return y.map({-1: 0, 0: 1, 1: 2}).astype(int)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="../data/btcusdt_5m.parquet")
    ap.add_argument("--out", default="../data/preds_lgbm_5m.parquet")

    ap.add_argument("--horizon", type=int, default=12)
    ap.add_argument("--atr-k", type=float, default=1.5)
    ap.add_argument("--fee-bps", type=float, default=6.0)
    ap.add_argument("--slip-bps", type=float, default=2.0)

    ap.add_argument("--train-days", type=int, default=180)
    ap.add_argument("--val-days", type=int, default=30)
    ap.add_argument("--step-days", type=int, default=30)

    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    df = pd.read_parquet(args.data)
    df = make_features(df)

    lcfg = TripleBarrierConfig(
        horizon=args.horizon,
        atr_k=args.atr_k,
        fee_bps=args.fee_bps,
        slippage_bps=args.slip_bps,
        use_atr=True,
    )
    df["y"] = triple_barrier_labels(df, lcfg)

    feature_cols = [c for c in df.columns if c not in {"open_time", "close_time", "ignore", "y"}]
    X = df[feature_cols]
    y = df["y"]

    # drop warmup NaNs
    mask = X.notna().all(axis=1) & y.notna()
    X = X[mask]
    y = y[mask]
    df2 = df.loc[mask].copy()

    y_map = _class_map(y)

    wcfg = WalkForwardConfig(
        train_days=args.train_days,
        val_days=args.val_days,
        step_days=args.step_days,
        embargo_bars=args.horizon,
        time_col="close_time" if "close_time" in df2.columns else "open_time",
    )

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
            random_state=args.seed,
            class_weight="balanced",
        )

        clf.fit(X.iloc[tr_idx], y_map.iloc[tr_idx])

        proba = clf.predict_proba(X.iloc[va_idx])
        pred = np.argmax(proba, axis=1)

        print(f"\n=== fold {fold} ===")
        print(classification_report(y_map.iloc[va_idx], pred, zero_division=0))

        out = df2.iloc[va_idx][[wcfg.time_col, "open", "high", "low", "close", "volume"]].copy()
        out["fold"] = fold
        out["y"] = y.iloc[va_idx].astype(int).to_numpy()
        out["p_short"] = proba[:, 0]
        out["p_flat"] = proba[:, 1]
        out["p_long"] = proba[:, 2]
        preds.append(out)

    if not preds:
        raise SystemExit("no splits produced (check data span / split config)")

    out_df = pd.concat(preds, ignore_index=True).sort_values(wcfg.time_col)
    out_df.attrs["label_cfg"] = asdict(lcfg)
    out_df.attrs["split_cfg"] = asdict(wcfg)

    out_df.to_parquet(args.out, index=False)
    print(f"\nwrote {args.out} ({len(out_df)} rows)")


if __name__ == "__main__":
    main()
