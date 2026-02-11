"""Fetch public BTCUSDT candles and save as parquet.

Default source: Binance public klines endpoint (no key).
Timeframe: 1m/5m/15m/1h/4h/1d.
"""

from __future__ import annotations

import argparse
import time
from datetime import datetime, timezone

import pandas as pd
import requests
from tqdm import tqdm

BINANCE_URL = "https://api.binance.com/api/v3/klines"


def fetch_klines(symbol: str, interval: str, start_ms: int | None, end_ms: int | None, limit: int = 1000) -> list[list]:
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    if start_ms is not None:
        params["startTime"] = int(start_ms)
    if end_ms is not None:
        params["endTime"] = int(end_ms)
    r = requests.get(BINANCE_URL, params=params, timeout=30)
    r.raise_for_status()
    return r.json()


def ms(dt: datetime) -> int:
    return int(dt.timestamp() * 1000)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--interval", default="5m")
    ap.add_argument("--days", type=int, default=365)
    ap.add_argument("--out", default="../data/btcusdt_5m.parquet")
    args = ap.parse_args()

    end = datetime.now(timezone.utc)
    start = end - pd.Timedelta(days=args.days)

    start_ms = ms(start)
    end_ms = ms(end)

    rows = []
    cur = start_ms

    pbar = tqdm(total=args.days, desc="fetching")
    last_day = start

    while True:
        batch = fetch_klines(args.symbol, args.interval, cur, end_ms)
        if not batch:
            break
        rows.extend(batch)

        last_open = batch[-1][0]
        # next request starts after last open time
        cur = last_open + 1

        # progress by days
        cur_dt = datetime.fromtimestamp(last_open / 1000, tz=timezone.utc)
        delta_days = (cur_dt - last_day).total_seconds() / 86400
        if delta_days > 0:
            pbar.update(min(args.days - pbar.n, delta_days))
            last_day = cur_dt

        # stop if we're at the end
        if len(batch) < 1000:
            break
        time.sleep(0.2)

    pbar.close()

    # columns per Binance docs
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

    df = pd.DataFrame(rows, columns=cols)
    for c in ["open", "high", "low", "close", "volume", "quote_asset_volume", "taker_buy_base", "taker_buy_quote"]:
        df[c] = df[c].astype(float)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)

    df = df.drop_duplicates(subset=["open_time"]).sort_values("open_time").reset_index(drop=True)

    out_path = args.out
    df.to_parquet(out_path, index=False)
    print(f"saved {len(df):,} rows -> {out_path}")


if __name__ == "__main__":
    main()
