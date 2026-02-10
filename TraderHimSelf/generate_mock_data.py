import pandas as pd
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(BASE_DIR, "data_raw")
os.makedirs(RAW_DIR, exist_ok=True)

def generate_ohlcv(start_str, end_str, freq, missing_prob=0.0):
    dates = pd.date_range(start=start_str, end=end_str, freq=freq, inclusive='left')
    n = len(dates)
    
    # Random walk for price
    price = 10000 + np.cumsum(np.random.randn(n))
    
    # Ensure nanoseconds -> milliseconds explicitly
    # Convert to datetime64[ns] first to be safe across pandas versions
    timestamps = dates.astype("datetime64[ns]").astype(np.int64) // 10**6
    
    df = pd.DataFrame({
        'open_time_ms': timestamps,
        'open': price,
        'high': price + 5,
        'low': price - 5,
        'close': price + np.random.randn(n),
        'volume': np.abs(np.random.randn(n) * 100)
    })
    
    # Introduce missing data
    if missing_prob > 0:
        mask = np.random.rand(n) > missing_prob
        df = df[mask]
        
    return df

def generate_funding(start_str, end_str):
    dates = pd.date_range(start=start_str, end=end_str, freq='8h', inclusive='left')
    n = len(dates)
    
    timestamps = dates.astype("datetime64[ns]").astype(np.int64) // 10**6
    
    df = pd.DataFrame({
        'time_ms': timestamps,
        'funding_rate': np.random.randn(n) * 0.0001
    })
    return df

def main():
    print("Generating mock data...")
    start = "2024-01-01"
    end = "2024-01-05"
    
    # 15m data (clean)
    df_15m = generate_ohlcv(start, end, "15min", missing_prob=0.0)
    df_15m.to_parquet(os.path.join(RAW_DIR, "btcusdt_15m.parquet"))
    print(f"Generated 15m: {len(df_15m)} rows")

    # 3m data (with holes to test alignment)
    df_3m = generate_ohlcv(start, end, "3min", missing_prob=0.05) # 5% missing
    df_3m.to_parquet(os.path.join(RAW_DIR, "btcusdt_3m.parquet"))
    print(f"Generated 3m: {len(df_3m)} rows")

    # Funding
    df_funding = generate_funding(start, end)
    df_funding.to_parquet(os.path.join(RAW_DIR, "btcusdt_funding.parquet"))
    print(f"Generated funding: {len(df_funding)} rows")

if __name__ == "__main__":
    main()
