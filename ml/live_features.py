"""Live feature builder for v3 (1h) candles.

Fetches fresh candles from Binance, computes features, and writes them to parquet.
Optionally runs in a loop for automation.
"""

from __future__ import annotations

import argparse
import time
from datetime import datetime, timezone

import pandas as pd
import requests

from features import make_features

BINANCE_URL = "https://api.binance.com/api/v3/klines"


def fetch_latest_candles(symbol: str = "BTCUSDT", interval: str = "1h", limit: int = 200) -> pd.DataFrame:
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    r = requests.get(BINANCE_URL, params=params, timeout=30)
    r.raise_for_status()

    cols = [
        "open_time",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "close_time",
        "quote_asset_volume",
        "num_trades",
        "taker_buy_base",
        "taker_buy_quote",
        "ignore",
    ]

    df = pd.DataFrame(r.json(), columns=cols)
    for c in ["open", "high", "low", "close", "volume", "quote_asset_volume", "taker_buy_base", "taker_buy_quote"]:
        df[c] = df[c].astype(float)

    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)

    return df


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    return make_features(df)


def save_features(df_feat: pd.DataFrame, out_path: str, latest_out: str | None = None) -> None:
    df_feat.to_parquet(out_path, index=False)
    if latest_out:
        df_feat.tail(1).to_parquet(latest_out, index=False)


def run_once(symbol: str, interval: str, limit: int, out_path: str, latest_out: str | None) -> None:
    df = fetch_latest_candles(symbol=symbol, interval=interval, limit=limit)
    df_feat = compute_features(df)
    save_features(df_feat, out_path=out_path, latest_out=latest_out)
    print(f"saved {len(df_feat):,} rows -> {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--interval", default="1h")
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--out", default="../data/live_features.parquet")
    ap.add_argument("--latest-out", default="../data/live_features_latest.parquet")
    ap.add_argument("--loop", action="store_true")
    ap.add_argument("--sleep", type=int, default=300)
    args = ap.parse_args()

    if args.loop:
        print("live_features: loop mode")
        while True:
            try:
                run_once(args.symbol, args.interval, args.limit, args.out, args.latest_out)
            except Exception as e:
                print(f"error: {e}")
            time.sleep(args.sleep)
    else:
        run_once(args.symbol, args.interval, args.limit, args.out, args.latest_out)


if __name__ == "__main__":
    main()
