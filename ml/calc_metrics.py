import pandas as pd
import numpy as np

def calculate_metrics(file_path):
    df = pd.read_parquet(file_path)
    if 'equity' not in df.columns:
        print(f"No equity column in {file_path}")
        return
    
    equity = df['equity'].values
    returns = df['equity'].pct_change().dropna()
    
    # Sharpe Ratio (annualized, assuming 1h bars)
    # (mean_ret / std_ret) * sqrt(number_of_bars_in_year)
    bars_per_year = 365 * 24
    sharpe = (returns.mean() / returns.std()) * np.sqrt(bars_per_year) if returns.std() != 0 else 0
    
    # Max Drawdown
    peak = np.maximum.accumulate(equity)
    drawdown = (equity - peak) / peak
    max_dd = drawdown.min()
    
    # Win Rate
    # In backtest_v2, we don't have separate trade logs yet, 
    # but we can look at changes in equity. 
    # Actually, the backtest_v2 script prints the final result.
    
    print(f"File: {file_path}")
    print(f"Final Equity: {equity[-1]:.2f}")
    print(f"Sharpe Ratio: {sharpe:.2f}")
    print(f"Max Drawdown: {max_dd:.2%}")
    print("-" * 30)

if __name__ == "__main__":
    files = [
        "../data/equity_val_2024_strict_48h_v2.parquet",
        "../data/equity_test_2025.parquet"
    ]
    for f in files:
        import os
        if os.path.exists(f):
            calculate_metrics(f)
        else:
            print(f"File not found: {f}")
