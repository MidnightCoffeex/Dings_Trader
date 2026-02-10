#!/usr/bin/env python3
"""
Binance Data Downloader for BTCUSDT
- Lädt historische OHLCV-Daten von Binance Futures API
- Speichert als Parquet in data_raw/
- Zeitfenster: 15m und 3m (default)
"""

import argparse
import os
import time
from datetime import datetime

import numpy as np
import pandas as pd
import requests

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA_DIR = os.path.join(BASE_DIR, "data_raw")
DEFAULT_SYMBOL = "BTCUSDT"
DEFAULT_INTERVALS = ["15m", "3m"]
DEFAULT_START_DATE = "2019-01-01"  # Binance Futures Start ~2019


def parse_args():
    parser = argparse.ArgumentParser(description="Download Binance futures OHLCV + funding data")
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR, help="Output directory for parquet files")
    parser.add_argument("--start-date", default=DEFAULT_START_DATE, help="Start date YYYY-MM-DD")
    parser.add_argument("--end-date", default=None, help="End date YYYY-MM-DD (default: today)")
    parser.add_argument("--symbol", default=DEFAULT_SYMBOL, help="Futures symbol, e.g. BTCUSDT")
    parser.add_argument("--intervals", default=",".join(DEFAULT_INTERVALS), help="Comma-separated intervals, e.g. 15m,3m")
    return parser.parse_args()


def ensure_dir(data_dir: str):
    """Erstelle data_raw/ falls nicht vorhanden."""
    os.makedirs(data_dir, exist_ok=True)
    print(f"📁 Datenverzeichnis: {data_dir}")


def _to_ms(date_str: str) -> int:
    return int(datetime.strptime(date_str, "%Y-%m-%d").timestamp() * 1000)


def fetch_klines(symbol, interval, start_str, end_str=None, limit=1000):
    """
    Lade Klines von Binance Futures API.
    https://binance-docs.github.io/apidocs/futures/en/#kline-candlestick-data
    """
    url = "https://fapi.binance.com/fapi/v1/klines"

    start_ms = _to_ms(start_str)
    end_ms = _to_ms(end_str) if end_str else None

    all_data = []
    current_start = start_ms

    print(f"⬇️ Lade {symbol} {interval} ab {start_str}...")

    while True:
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": current_start,
            "limit": limit,
        }
        if end_ms:
            params["endTime"] = end_ms

        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()

            if not data:
                break

            if isinstance(data, dict):
                raise RuntimeError(f"Unexpected Binance response: {data}")

            all_data.extend(data)

            current_start = data[-1][0] + 1  # letzter Timestamp + 1ms
            last_candle = datetime.fromtimestamp(data[-1][0] / 1000)
            print(f"  → {len(all_data)} candles, letzte: {last_candle}")

            time.sleep(0.5)  # rate limiting

            if end_ms and current_start >= end_ms:
                break
            if len(data) < limit:
                break

        except requests.exceptions.RequestException as e:
            print(f"⚠️ Netzwerkfehler: {e}")
            time.sleep(2)
            continue
        except Exception as e:
            print(f"⚠️ API/Parsing-Fehler: {e}")
            time.sleep(2)
            continue

    return all_data


def process_to_dataframe(klines):
    """Konvertiere Binance Klines zu DataFrame."""
    columns = [
        "open_time_ms",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "close_time_ms",
        "quote_volume",
        "trades",
        "taker_buy_base",
        "taker_buy_quote",
        "ignore",
    ]

    df = pd.DataFrame(klines, columns=columns)
    df = df[["open_time_ms", "open", "high", "low", "close", "volume"]]

    df["open_time_ms"] = df["open_time_ms"].astype(np.int64)
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)

    df["timestamp"] = pd.to_datetime(df["open_time_ms"], unit="ms")
    df.set_index("timestamp", inplace=True)

    return df


def download_funding_rates(symbol, start_str, end_str=None):
    """Lade Funding Rates von Binance Futures API."""
    url = "https://fapi.binance.com/fapi/v1/fundingRate"

    start_ms = _to_ms(start_str)
    end_ms = _to_ms(end_str) if end_str else None

    all_data = []
    current_start = start_ms

    print(f"⬇️ Lade Funding Rates für {symbol}...")

    while True:
        params = {
            "symbol": symbol,
            "startTime": current_start,
            "limit": 1000,
        }
        if end_ms:
            params["endTime"] = end_ms

        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()

            if not data:
                break

            if isinstance(data, dict):
                raise RuntimeError(f"Unexpected Binance response: {data}")

            all_data.extend(data)
            current_start = int(data[-1]["fundingTime"]) + 1

            print(f"  → {len(all_data)} funding entries")
            time.sleep(0.5)

            if end_ms and current_start >= end_ms:
                break
            if len(data) < 1000:
                break

        except requests.exceptions.RequestException as e:
            print(f"⚠️ Netzwerkfehler: {e}")
            time.sleep(2)
            continue
        except Exception as e:
            print(f"⚠️ API/Parsing-Fehler: {e}")
            time.sleep(2)
            continue

    return all_data


def main():
    args = parse_args()

    symbol = args.symbol.upper().strip()
    start_date = args.start_date
    end_date = args.end_date or datetime.now().strftime("%Y-%m-%d")
    data_dir = args.data_dir
    intervals = [i.strip() for i in args.intervals.split(",") if i.strip()]

    print("=" * 60)
    print("📊 Binance Data Downloader")
    print("=" * 60)
    print(f"Symbol: {symbol} | Intervals: {intervals}")
    print(f"Window: {start_date} → {end_date}")

    ensure_dir(data_dir)

    for interval in intervals:
        print(f"\n{'=' * 60}")
        print(f"🔄 Verarbeite Interval: {interval}")
        print(f"{'=' * 60}")

        klines = fetch_klines(symbol, interval, start_date, end_date)
        if not klines:
            print(f"❌ Keine Daten für {interval}")
            continue

        df = process_to_dataframe(klines)
        output_file = os.path.join(data_dir, f"{symbol.lower()}_{interval}.parquet")
        df.to_parquet(output_file, engine="pyarrow", compression="zstd")

        print(f"\n✅ Gespeichert: {output_file}")
        print(f"   Zeilen: {len(df)}")
        print(f"   Von: {df.index[0]}")
        print(f"   Bis: {df.index[-1]}")
        print(f"   Spalten: {list(df.columns)}")

    print(f"\n{'=' * 60}")
    print("🔄 Verarbeite Funding Rates...")
    print(f"{'=' * 60}")

    funding_data = download_funding_rates(symbol, start_date, end_date)
    if funding_data:
        df_funding = pd.DataFrame(funding_data)
        df_funding["fundingTime"] = pd.to_datetime(df_funding["fundingTime"], unit="ms")
        df_funding.set_index("fundingTime", inplace=True)
        df_funding["fundingRate"] = df_funding["fundingRate"].astype(float)

        output_file = os.path.join(data_dir, "funding_rates.parquet")
        df_funding.to_parquet(output_file, engine="pyarrow", compression="zstd")

        print(f"\n✅ Gespeichert: {output_file}")
        print(f"   Zeilen: {len(df_funding)}")

    print(f"\n{'=' * 60}")
    print("🎉 Download abgeschlossen!")
    print(f"{'=' * 60}")
    print(f"📁 Alle Dateien in: {data_dir}")


if __name__ == "__main__":
    main()
