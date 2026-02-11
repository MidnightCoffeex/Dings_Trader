"""Baseline model: supervised classifier on engineered features.

Outputs: long/flat/short probabilities.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

from features import make_features, label_future


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="../data/btcusdt_5m.parquet")
    ap.add_argument("--horizon", type=int, default=12)  # ~1h if 5m
    ap.add_argument("--tp", type=float, default=0.004)
    ap.add_argument("--sl", type=float, default=0.003)
    args = ap.parse_args()

    df = pd.read_parquet(args.data)
    df = make_features(df)
    df["y"] = label_future(df, horizon=args.horizon, tp=args.tp, sl=args.sl)

    feature_cols = [c for c in df.columns if c not in {"open_time","close_time","ignore","y"}]
    X = df[feature_cols]
    y = df["y"]

    # drop warmup NaNs
    mask = X.notna().all(axis=1)
    X = X[mask]
    y = y[mask]

    # map labels to classes 0..2 for sklearn
    y_map = y.map({-1: 0, 0: 1, 1: 2}).astype(int)

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced")),
    ])

    tss = TimeSeriesSplit(n_splits=5)
    reports = []

    for fold, (tr, te) in enumerate(tss.split(X), 1):
        pipe.fit(X.iloc[tr], y_map.iloc[tr])
        pred = pipe.predict(X.iloc[te])
        rep = classification_report(y_map.iloc[te], pred, output_dict=False, zero_division=0)
        reports.append(rep)
        print(f"\n=== fold {fold} ===")
        print(rep)

    print("\ndone.")


if __name__ == "__main__":
    main()
