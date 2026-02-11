import pandas as pd
import numpy as np
import os

def calculate_base_features(df):
    """Calculates basic features for BTC candles."""
    df = df.copy()
    
    # Ensure datetime index
    if 'open_time' in df.columns:
        df['timestamp'] = pd.to_datetime(df['open_time'], unit='ms')
        df = df.set_index('timestamp')
    
    # Basic price features
    df['log_return'] = np.log(df['close'] / df['close'].shift(1))
    
    # Rolling volatility (24h window for 5m data = 12 * 24 = 288 periods)
    df['volatility_24h'] = df['log_return'].rolling(window=288).std()
    
    # Simple Moving Averages
    df['sma_50'] = df['close'].rolling(window=50).mean()
    df['sma_200'] = df['close'].rolling(window=200).mean()
    
    # Price relative to SMAs
    df['price_vs_sma50'] = df['close'] / df['sma_50'] - 1
    df['price_vs_sma200'] = df['close'] / df['sma_200'] - 1
    
    # --- New Features: Trend Signals ---
    
    # Exponential Moving Averages
    df['ema_12'] = df['close'].ewm(span=12, adjust=False).mean()
    df['ema_26'] = df['close'].ewm(span=26, adjust=False).mean()
    
    # EMA Cross (MACD-ish Signal)
    df['ema_spread'] = df['ema_12'] / df['ema_26'] - 1
    
    # RSI (14 periods)
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi_14'] = 100 - (100 / (1 + rs))
    
    # MACD (12, 26, 9)
    df['macd'] = df['ema_12'] - df['ema_26']
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']

    # Price Slope (Linear regression over last 12 periods ~ 1 hour)
    def get_slope(array):
        y = array
        x = np.arange(len(y))
        slope, intercept = np.polyfit(x, y, 1)
        return slope / y[0] # Normalized slope

    df['slope_1h'] = df['close'].rolling(window=12).apply(get_slope, raw=True)
    
    return df

if __name__ == "__main__":
    data_path = "projects/dings-trader/data/btcusdt_5m.parquet"
    out_path = "projects/dings-trader/data/btcusdt_5m_features.parquet"
    
    if os.path.exists(data_path):
        print(f"Loading {data_path}...")
        df = pd.read_parquet(data_path)
        
        print("Calculating features (v2)...")
        df_features = calculate_base_features(df)
        
        print(f"Saving to {out_path}...")
        df_features.to_parquet(out_path)
        print("Done.")
    else:
        print(f"Error: {data_path} not found.")
