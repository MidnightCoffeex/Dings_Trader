"""Train LightGBM on historical data with strict temporal splits."""

import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier
from labels import TripleBarrierConfig, triple_barrier_labels

def main():
    data_path = "../data/btcusdt_1h_historical_features.parquet"
    df = pd.read_parquet(data_path)
    
    # Labels: Strict 5% profit target within 48h horizon
    lcfg = TripleBarrierConfig(horizon=48, atr_k=0.05, fee_bps=6.0, slippage_bps=2.0, use_atr=False)
    df['y'] = triple_barrier_labels(df, lcfg)
    
    # Split
    train_df = df[df.open_time < "2024-01-01"].copy()
    val_df = df[(df.open_time >= "2024-01-01") & (df.open_time < "2025-01-01")].copy()
    test_df = df[df.open_time >= "2025-01-01"].copy()
    
    feature_cols = [c for c in df.columns if c not in {"open_time", "close_time", "ignore", "y"}]
    
    def prep(d):
        d = d.dropna(subset=feature_cols + ['y'])
        return d[feature_cols], d['y'].map({-1: 0, 0: 1, 1: 2})

    X_train, y_train = prep(train_df)
    X_val, y_val = prep(val_df)
    
    print(f"Training on {len(X_train)} rows...")
    clf = LGBMClassifier(n_estimators=1000, learning_rate=0.02, class_weight='balanced', random_state=42)
    clf.fit(X_train, y_train, eval_set=[(X_val, y_val)], eval_metric='multi_logloss')
    
    # Save model
    import joblib
    joblib.dump(clf, "model_v2.1.joblib")
    print("Model saved as model_v2.1.joblib")
    
    # Save predictions for backtest
    for name, d, X in [("val", val_df, X_val), ("test", test_df, prep(test_df)[0])]:
        proba = clf.predict_proba(X)
        d_out = d.copy()
        # Align with X indices
        d_out = d_out.loc[X.index]
        d_out['p_short'] = proba[:, 0]
        d_out['p_flat'] = proba[:, 1]
        d_out['p_long'] = proba[:, 2]
        d_out.to_parquet(f"../data/preds_strict_{name}.parquet")
        print(f"Saved {name} predictions.")

if __name__ == "__main__":
    main()
