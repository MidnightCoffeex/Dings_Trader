"""Portfolio Backtester v2 - Maxim's Symbiomorphose Engine

Rules:
- Supports concurrent positions (max_concurrent).
- Exposure limit: Total invested value <= 10% of (Cash + Invested).
- Multi-split logic (Train/Val/Test).
"""

from __future__ import annotations
import argparse
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from features import make_features

@dataclass
class Position:
    entry_price: float
    side: int  # 1 long, -1 short
    entry_time: pd.Timestamp
    tp: float
    sl: float
    invested_amount: float
    bars_held: int = 0

def run_portfolio_backtest(df: pd.DataFrame, 
                           preds: pd.DataFrame | None, 
                           initial_capital: float = 1000.0,
                           max_concurrent: int = 5,
                           exposure_limit: float = 0.10, # 10%
                           leverage: float = 5.0,
                           threshold: float = 0.35,
                           fee_bps: float = 6.0,
                           slip_bps: float = 2.0,
                           horizon: int = 48,
                           min_profit_target: float = 0.05,
                           kill_switch_equity: float = 200.0): # 5%
    
    df = df.copy()
    time_col = "open_time"
    
    if preds is not None:
        p = preds[[time_col, "p_short", "p_flat", "p_long"]].copy()
        df = df.merge(p, on=time_col, how="left")
    
    df = df.dropna(subset=["p_long", "open", "high", "low", "close"]).reset_index(drop=True)
    
    # Ensure ATR is present
    if "atr_pct" not in df.columns:
        from features import make_features
        df = make_features(df)

    cash = initial_capital
    positions: list[Position] = []
    equity_history = []
    
    fee = fee_bps / 10_000.0
    slip = slip_bps / 10_000.0
    
    for i in range(len(df) - 1):
        row = df.iloc[i]
        next_row = df.iloc[i+1]
        
        # Trigger logic: allow new entry?
        # Regime shift: vol spike over median
        w = 96
        is_trigger = True
        if i > w:
            vol = df.loc[i-w:i, 'atr_pct'].median()
            if row['atr_pct'] < vol * 1.2: # Only trade in high-vol regimes
                is_trigger = False

        # 1. Update existing positions
        current_invested_value = 0
        for pos in positions[:]:
            pos.bars_held += 1
            # Check for exit (TP/SL/Horizon)
            exit_reason = None
            exit_price = None
            
            hi, lo = next_row['high'], next_row['low']
            
            if pos.side == 1: # Long
                if lo <= pos.sl:
                    exit_reason, exit_price = "SL", pos.sl * (1.0 - slip)
                elif hi >= pos.tp:
                    exit_reason, exit_price = "TP", pos.tp * (1.0 - slip)
                elif pos.bars_held >= horizon:
                    exit_reason, exit_price = "Timeout", next_row['open'] * (1.0 - slip)
            else: # Short
                if hi >= pos.sl:
                    exit_reason, exit_price = "SL", pos.sl * (1.0 + slip)
                elif lo <= pos.tp:
                    exit_reason, exit_price = "TP", pos.tp * (1.0 + slip)
                elif pos.bars_held >= horizon:
                    exit_reason, exit_price = "Timeout", next_row['open'] * (1.0 + slip)
            
            if exit_reason:
                # Calculate profit
                if pos.side == 1:
                    ret = (exit_price / pos.entry_price) - 1.0
                else:
                    ret = (pos.entry_price / exit_price) - 1.0
                
                realized = pos.invested_amount * (1.0 + leverage * ret)
                realized *= (1.0 - fee * leverage) # Exit fee
                cash += realized
                positions.remove(pos)
            else:
                current_invested_value += pos.invested_amount

        # 2. Check for new entries
        total_equity = cash + current_invested_value
        if total_equity < kill_switch_equity:
            equity_history.append(total_equity)
            print(f"KILL-SWITCH TRIGGERED (equity {total_equity:.2f} < {kill_switch_equity}).")
            break

        p_long, p_short = row['p_long'], row['p_short']
        edge = p_long - p_short
        
        if len(positions) < max_concurrent and is_trigger:
            # How much can we still invest?
            allowed_exposure = total_equity * exposure_limit
            remaining_exposure = allowed_exposure - current_invested_value
            
            if remaining_exposure > 1.0: # Minimum 1 unit
                side = 0
                if edge > threshold: side = 1
                elif edge < -threshold: side = -1
                
                if side != 0:
                    entry_price = next_row['open']
                    # Barriers: 5% price move
                    b = 0.05 
                    
                    # Entry cost
                    invest_size = min(remaining_exposure, total_equity / max_concurrent)
                    if cash >= invest_size:
                        cash -= invest_size
                        # Pay entry fee
                        invest_size *= (1.0 - fee * leverage)
                        
                        positions.append(Position(
                            entry_price = entry_price * (1.0 + slip if side == 1 else 1.0 - slip),
                            side = side,
                            entry_time = row[time_col],
                            tp = entry_price * (1.0 + b if side == 1 else 1.0 - b),
                            sl = entry_price * (1.0 - b/2.0 if side == 1 else 1.0 + b/2.0), # Asymmetric SL
                            invested_amount = invest_size
                        ))
        
        equity_history.append(total_equity)
        
    return equity_history

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True)
    ap.add_argument("--out", default="../data/portfolio_equity.parquet")
    ap.add_argument("--capital", type=float, default=1000.0)
    ap.add_argument("--exposure", type=float, default=0.10)
    ap.add_argument("--threshold", type=float, default=0.35)
    ap.add_argument("--target", type=float, default=0.05)
    args = ap.parse_args()

    df = pd.read_parquet(args.preds)
    # We pass None for preds because they are already merged in the df
    history = run_portfolio_backtest(df, None, initial_capital=args.capital, exposure_limit=args.exposure, threshold=args.threshold, min_profit_target=args.target)
    
    print(f"Final Equity: {history[-1]:.2f}")
    
    df_out = pd.DataFrame({"equity": history})
    df_out.to_parquet(args.out)

if __name__ == "__main__":
    main()
