"""Live inference and trading engine.

Supports multiple models and automated trading loops.
"""
import argparse
import time
import os
import sqlite3
import joblib
import pandas as pd
import requests
from datetime import datetime, timezone

from features import make_features
from db import init_db, ensure_model, check_kill_switch, get_open_position, open_position, close_position

DB_PATH = "/home/maxim/.openclaw/workspace/projects/dings-trader/data/trader.sqlite"
INITIAL_CAPITAL = 1100.0
KILL_SWITCH_EQUITY = 200.0

def fetch_latest_candles(symbol="BTCUSDT", interval="1h", limit=200):
    url = "https://api.binance.com/api/v3/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    r = requests.get(url, params=params, timeout=10)
    r.raise_for_status()
    
    cols = ["open_time", "open", "high", "low", "close", "volume", "close_time", "quote_asset_volume", "num_trades", "taker_buy_base", "taker_buy_quote", "ignore"]
    df = pd.DataFrame(r.json(), columns=cols)
    for c in ["open", "high", "low", "close", "volume", "quote_asset_volume", "taker_buy_base", "taker_buy_quote"]:
        df[c] = df[c].astype(float)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
    return df

def run_once(model_id: str, model_path: str):
    if not os.path.exists(DB_PATH):
        init_db()
        
    ensure_model(model_id, version="v2.1", initial_capital=INITIAL_CAPITAL)
    if check_kill_switch(model_id, threshold=KILL_SWITCH_EQUITY):
        print(f"[{model_id}] Kill-switch active. Stopping.")
        return

    # 1. Fetch & Features
    try:
        df = fetch_latest_candles()
        df_feat = make_features(df)
    except Exception as e:
        print(f"[{model_id}] Data fetch error: {e}")
        return

    # 2. Predict
    try:
        clf = joblib.load(model_path)
        feature_cols = [c for c in df_feat.columns if c not in {"open_time", "close_time", "ignore", "y"}]
        X_latest = df_feat[feature_cols].tail(1)
        
        if X_latest.isna().any().any():
            print(f"[{model_id}] Warmup - NaNs in features.")
            return

        proba = clf.predict_proba(X_latest)[0]
        p_short, p_flat, p_long = proba[0], proba[1], proba[2]
    except Exception as e:
        print(f"[{model_id}] Prediction error: {e}")
        return

    # 3. Execution Logic
    current_price = df['close'].iloc[-1]
    pos = get_open_position(model_id)
    
    print(f"[{model_id}] Price: {current_price:.2f} | P(L): {p_long:.2f} P(S): {p_short:.2f} P(F): {p_flat:.2f}")

    if pos:
        # Check Exit
        entry_price = pos["entry_price"]
        side = pos["side"]
        
        # Calc Unr. PnL
        if side == "Long":
            pnl_pct = (current_price - entry_price) / entry_price * 100
        else:
            pnl_pct = (entry_price - current_price) / entry_price * 100
            
        # Exit Rules (Simple)
        close_it = False
        reason = ""
        
        # 1. Hard SL/TP
        if pnl_pct < -5.0: 
            close_it = True; reason = "SL"
        elif pnl_pct > 5.0:
            close_it = True; reason = "TP"
            
        # 2. Model Signal (Flip)
        if side == "Long" and p_short > 0.4:
            close_it = True; reason = "Signal Flip"
        elif side == "Short" and p_long > 0.4:
            close_it = True; reason = "Signal Flip"
            
        if close_it:
            print(f"[{model_id}] Closing {side} ({reason}). PnL: {pnl_pct:.2f}%")
            close_position(pos["id"], current_price, pnl_pct)
    else:
        # Check Entry
        entry_threshold = 0.55
        size = 110.0 # 10% of 1100
        
        if p_long > entry_threshold:
            print(f"[{model_id}] Opening Long")
            open_position(model_id, "BTCUSDT", "Long", current_price, size)
        elif p_short > entry_threshold:
            print(f"[{model_id}] Opening Short")
            open_position(model_id, "BTCUSDT", "Short", current_price, size)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", default="v2")
    ap.add_argument("--model-file", default="model_v2.1.joblib")
    ap.add_argument("--loop", action="store_true")
    ap.add_argument("--sleep", type=int, default=300)
    args = ap.parse_args()
    
    # Resolve absolute path for model if needed
    if not os.path.isabs(args.model_file):
        base = os.path.dirname(os.path.abspath(__file__))
        args.model_file = os.path.join(base, args.model_file)

    if args.loop:
        print(f"Starting inference loop for {args.model_id}...")
        while True:
            try:
                run_once(args.model_id, args.model_file)
            except Exception as e:
                print(f"Loop error: {e}")
            time.sleep(args.sleep)
    else:
        run_once(args.model_id, args.model_file)

if __name__ == "__main__":
    main()
