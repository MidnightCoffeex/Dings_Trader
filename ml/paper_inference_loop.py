"""Paper Trading Auto-Inference Loop.

Läuft kontinuierlich und:
1. Holt ML-Signal von Binance Live-Daten
2. Aktualisiert Paper-Trading Positionen
3. Öffnet neue Positionen bei starken Signalen
4. Trackt Performance
"""
import argparse
import time
import os
import sys
import requests
from datetime import datetime

# Add ml directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from paper_trading import get_paper_engine

API_BASE = "http://127.0.0.1:8000"


def run_inference_cycle(model_id: str, symbol: str = "BTCUSDT"):
    """Führt einen einzelnen Inference-Zyklus aus."""
    try:
        # 1. Fetch current price
        price_res = requests.get(f"{API_BASE}/live-price?symbol={symbol}", timeout=10)
        if not price_res.ok:
            print(f"[{datetime.now()}] Error fetching price: {price_res.status_code}")
            return False
        
        price_data = price_res.json()
        current_price = float(price_data.get("price", 0))
        
        if current_price == 0:
            print(f"[{datetime.now()}] Invalid price received")
            return False
        
        # 2. Get ML Signal
        signal_res = requests.get(f"{API_BASE}/paper/ml-signal/{model_id}?symbol={symbol}", timeout=30)
        if not signal_res.ok:
            print(f"[{datetime.now()}] Error fetching signal: {signal_res.status_code}")
            return False
        
        signal_data = signal_res.json()
        signal = signal_data.get("signal", "FLAT")
        confidence = signal_data.get("confidence", 0)
        sentiment = signal_data.get("sentiment", "neutral")
        
        print(f"[{datetime.now()}] Price: ${current_price:,.2f} | Signal: {signal} ({confidence}% confidence, {sentiment})")
        
        # 3. Process Signal via Paper Trading API
        process_res = requests.post(
            f"{API_BASE}/paper/signal",
            json={
                "model_id": model_id,
                "symbol": symbol,
                "signal": signal,
                "confidence": confidence,
                "current_price": current_price
            },
            timeout=10
        )
        
        if not process_res.ok:
            print(f"[{datetime.now()}] Error processing signal: {process_res.status_code}")
            return False
        
        result = process_res.json()
        
        # Log actions
        for action in result.get("actions", []):
            print(f"  → {action}")
        
        # Log opened position
        opened = result.get("opened_position")
        if opened:
            print(f"  ✓ Opened {opened['side']} position: {opened['size_usdt']:.2f} USDT @ ${opened['entry_price']:,.2f}")
        
        # Log closed positions
        for closed in result.get("closed_positions", []):
            pnl = closed.get("pnl_pct", 0)
            pnl_emoji = "📈" if pnl > 0 else "📉"
            print(f"  ✗ Closed position ({closed.get('reason')}): {pnl:+.2f}% {pnl_emoji}")
        
        # Log stats
        stats = result.get("current_stats", {})
        print(f"  💰 Equity: ${stats.get('total_equity', 0):,.2f} | Return: {stats.get('total_return_pct', 0):+.2f}% | Win Rate: {stats.get('win_rate', 0):.1f}%")
        
        return True
        
    except Exception as e:
        print(f"[{datetime.now()}] Error in inference cycle: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Paper Trading Auto-Inference")
    parser.add_argument("--model-id", default="paper_v1", help="Model ID for paper trading")
    parser.add_argument("--symbol", default="BTCUSDT", help="Trading symbol")
    parser.add_argument("--interval", type=int, default=60, help="Interval in seconds between checks")
    parser.add_argument("--once", action="store_true", help="Run once and exit")
    parser.add_argument("--create-account", action="store_true", help="Create account if not exists")
    args = parser.parse_args()
    
    # Ensure account exists
    if args.create_account:
        engine = get_paper_engine()
        account = engine.get_or_create_account(
            args.model_id,
            initial_balance=10000.0,
            max_positions=5,
            default_leverage=7.0,
            profit_target_pct=5.0,
            time_limit_hours=48.0
        )
        print(f"[{datetime.now()}] Paper Trading Account: {account.model_id}")
        print(f"  Initial Balance: ${account.initial_balance:,.2f} USDT")
        print(f"  Max Positions: {account.max_positions}")
        print(f"  Default Leverage: {account.default_leverage}x")
        print(f"  Profit Target: {account.profit_target_pct}%")
        print(f"  Time Limit: {account.time_limit_hours}h")
    
    if args.once:
        print(f"[{datetime.now()}] Running single inference cycle...")
        run_inference_cycle(args.model_id, args.symbol)
    else:
        print(f"[{datetime.now()}] Starting Paper Trading Loop...")
        print(f"  Model ID: {args.model_id}")
        print(f"  Symbol: {args.symbol}")
        print(f"  Interval: {args.interval}s")
        print(f"  Press Ctrl+C to stop")
        print()
        
        while True:
            run_inference_cycle(args.model_id, args.symbol)
            print()
            time.sleep(args.interval)


if __name__ == "__main__":
    main()
