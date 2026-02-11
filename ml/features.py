"""Feature engineering for candle data.

Keeps dependencies minimal (no TA-Lib).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
    rs = avg_gain / (avg_loss.replace(0, np.nan))
    out = 100 - (100 / (1 + rs))
    return out.fillna(50)


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df["high"]
    low = df["low"]
    close = df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, adjust=False).mean()


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # returns
    df["ret_1"] = df["close"].pct_change()
    df["logret_1"] = np.log(df["close"]).diff()

    # moving averages
    for n in [10, 20, 50, 100]:
        ma = df["close"].rolling(n).mean()
        df[f"ma_{n}"] = ma
        df[f"ma_dist_{n}"] = (df["close"] - ma) / ma

    # volatility
    for n in [20, 50, 100]:
        df[f"vol_{n}"] = df["logret_1"].rolling(n).std() * np.sqrt(n)

    # RSI / ATR
    df["rsi_14"] = rsi(df["close"], 14)
    df["atr_14"] = atr(df, 14)
    df["atr_pct"] = df["atr_14"] / df["close"]

    # MACD (12, 26, 9)
    ema_12 = df['close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = ema_12 - ema_26
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']

    # Price Slope (Linear regression over last 12 periods ~ 1 hour)
    def get_slope(array):
        y = array
        x = np.arange(len(y))
        slope, intercept = np.polyfit(x, y, 1)
        return slope / y[0] # Normalized slope

    df['slope_1h'] = df['close'].rolling(window=12).apply(get_slope, raw=True)

    # candle shape
    df["body"] = (df["close"] - df["open"]) / df["open"]
    df["range"] = (df["high"] - df["low"]) / df["open"]
    df["upper_wick"] = (df[["high", "open", "close"]].max(axis=1) - df[["open", "close"]].max(axis=1)) / df["open"]
    df["lower_wick"] = (df[["open", "close"]].min(axis=1) - df[["low", "open", "close"]].min(axis=1)) / df["open"]

    # volume features
    df["vol_z_50"] = (df["volume"] - df["volume"].rolling(50).mean()) / df["volume"].rolling(50).std()

    # lagged returns (momentum)
    for n in [1, 3, 6, 12]:
        df[f"lag_ret_{n}"] = df["logret_1"].shift(n)

    # time features
    if "open_time" in df.columns:
        ts = pd.to_datetime(df["open_time"])
        df["hour"] = ts.dt.hour
        df["dayofweek"] = ts.dt.dayofweek

    # clean
    df = df.replace([np.inf, -np.inf], np.nan)
    return df


def label_future(df: pd.DataFrame, horizon: int = 12, tp: float = 0.004, sl: float = 0.003) -> pd.Series:
    """Simple 3-class label: 1=long, -1=short, 0=flat.

    Looks forward up to `horizon` bars to see whether TP or SL would hit first.
    This is a simplified proxy target for supervised learning.
    """
    close = df["close"].values
    high = df["high"].values
    low = df["low"].values

    y = np.zeros(len(df), dtype=np.int8)

    for i in range(len(df) - horizon - 1):
        entry = close[i]
        long_tp = entry * (1 + tp)
        long_sl = entry * (1 - sl)
        short_tp = entry * (1 - tp)
        short_sl = entry * (1 + sl)

        hit = 0
        for j in range(1, horizon + 1):
            if high[i + j] >= long_tp:
                hit = 1
                break
            if low[i + j] <= long_sl:
                hit = 0
                break
            if low[i + j] <= short_tp:
                hit = -1
                break
            if high[i + j] >= short_sl:
                hit = 0
                break
        y[i] = hit

    return pd.Series(y, index=df.index, name="y")
