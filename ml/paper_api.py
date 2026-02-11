"""Paper Trading API Endpoints.

FastAPI Router für Paper Trading Operationen.
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime
import os
import sys
import requests

from paper_trading import get_paper_engine, PaperTradingEngine

# Model package registry
from db import get_model_package

# Try import new inference module
try:
    from ppo_forecast_inference import get_inference
except ImportError:
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from ppo_forecast_inference import get_inference

router = APIRouter(prefix="/paper", tags=["paper-trading"])


def _resolve_model_package_id(model_id: str) -> str:
    """Map paper trading account model_id to model package id.

    UI convention: modelId = <package_id>, paperModelId = paper_<package_id>
    """
    if model_id.startswith("paper_"):
        return model_id[len("paper_") :]
    return model_id


class CreateAccountRequest(BaseModel):
    model_id: str
    initial_balance: float = 10000.0
    max_positions: int = 5
    default_leverage: float = 7.0
    profit_target_pct: float = 5.0
    time_limit_hours: float = 48.0


class OpenPositionRequest(BaseModel):
    model_id: str
    symbol: str
    side: str  # "Long" or "Short"
    entry_price: float
    size_usdt: float
    leverage: Optional[float] = None
    take_profit_pct: Optional[float] = None
    stop_loss_pct: Optional[float] = None


class ClosePositionRequest(BaseModel):
    exit_price: float
    reason: str = "MANUAL"


class PriceUpdateRequest(BaseModel):
    model_id: str
    symbol: str
    current_price: float


class SignalRequest(BaseModel):
    model_id: str
    symbol: str
    signal: str  # "LONG", "SHORT", "FLAT"
    confidence: float  # 0-100
    current_price: float


# Account Endpoints
@router.post("/account/create")
async def create_account(req: CreateAccountRequest):
    """Erstellt ein neues Paper-Trading Konto."""
    engine = get_paper_engine()
    account = engine.create_account(
        model_id=req.model_id,
        initial_balance=req.initial_balance,
        max_positions=req.max_positions,
        default_leverage=req.default_leverage,
        profit_target_pct=req.profit_target_pct,
        time_limit_hours=req.time_limit_hours
    )
    return {
        "success": True,
        "account": {
            "model_id": account.model_id,
            "balance_usdt": account.balance_usdt,
            "total_equity": account.total_equity,
            "initial_balance": account.initial_balance,
            "max_positions": account.max_positions,
            "default_leverage": account.default_leverage
        }
    }


@router.get("/account/{model_id}")
async def get_account(model_id: str):
    """Holt Paper-Trading Konto Details."""
    engine = get_paper_engine()
    account = engine.get_account(model_id)
    
    if not account:
        # Auto-create account with defaults (so UI timers can work immediately)
        account = engine.create_account(
            model_id=model_id,
            initial_balance=10000.0,
            max_positions=5,
            default_leverage=7.0,
            profit_target_pct=5.0,
            time_limit_hours=48.0,
        )

    package_id = _resolve_model_package_id(model_id)
    pkg = get_model_package(package_id) if package_id else None

    return {
        "model_id": account.model_id,
        "model_package_id": package_id,
        "warmup_required": bool(pkg and int(pkg.get("warmup_required", 0)) == 1),
        "warmup_status": (pkg or {}).get("warmup_status"),
        "balance_usdt": account.balance_usdt,
        "total_equity": account.total_equity,
        "initial_balance": account.initial_balance,
        "total_trades": account.total_trades,
        "winning_trades": account.winning_trades,
        "losing_trades": account.losing_trades,
        "max_positions": account.max_positions,
        "default_leverage": account.default_leverage,
        "profit_target_pct": account.profit_target_pct,
        "time_limit_hours": account.time_limit_hours,
        "created_at": account.created_at.isoformat() + 'Z',
        "updated_at": account.updated_at.isoformat() + 'Z',
        "reset_at": account.reset_at.isoformat() + 'Z'
    }


@router.get("/account/{model_id}/performance")
async def get_performance(model_id: str):
    """Holt Performance-Statistiken."""
    engine = get_paper_engine()
    stats = engine.get_performance_stats(model_id)
    
    if not stats:
        raise HTTPException(status_code=404, detail="Account not found")
    
    return stats


@router.post("/account/{model_id}/reset")
async def reset_account(model_id: str, keep_history: bool = False):
    """Resetet ein Paper-Trading Konto."""
    engine = get_paper_engine()
    account = engine.reset_account(model_id, keep_history)
    
    return {
        "success": True,
        "account": {
            "model_id": account.model_id,
            "balance_usdt": account.balance_usdt,
            "total_equity": account.total_equity
        }
    }


# Position Endpoints
@router.get("/positions/{model_id}")
async def get_positions(model_id: str, status: str = "open"):
    """Holt Positionen eines Modells."""
    engine = get_paper_engine()
    
    if status == "open":
        positions = engine.get_open_positions(model_id)
    else:
        positions = engine.get_closed_positions(model_id)
    
    return {
        "positions": [
            {
                "id": p.id,
                "symbol": p.symbol,
                "side": p.side,
                "entry_price": p.entry_price,
                "exit_price": p.exit_price,
                "size_usdt": p.size_usdt,
                "leverage": p.leverage,
                "take_profit_pct": p.take_profit_pct,
                "stop_loss_pct": p.stop_loss_pct,
                "open_time": p.open_time.isoformat() + 'Z',
                "close_time": p.close_time.isoformat() + 'Z' if p.close_time else None,
                "realized_pnl_pct": p.realized_pnl_pct,
                "realized_pnl_usdt": p.realized_pnl_usdt,
                "status": p.status,
                "close_reason": p.close_reason
            }
            for p in positions
        ]
    }


@router.post("/position/open")
async def open_position(req: OpenPositionRequest):
    """Öffnet eine neue Paper-Position."""
    engine = get_paper_engine()
    
    # Validate side
    if req.side not in ["Long", "Short"]:
        raise HTTPException(status_code=400, detail="Side must be 'Long' or 'Short'")
    
    position = engine.open_position(
        model_id=req.model_id,
        symbol=req.symbol,
        side=req.side,
        entry_price=req.entry_price,
        size_usdt=req.size_usdt,
        leverage=req.leverage,
        take_profit_pct=req.take_profit_pct,
        stop_loss_pct=req.stop_loss_pct
    )
    
    if not position:
        raise HTTPException(status_code=400, detail="Could not open position")
    
    return {
        "success": True,
        "position": {
            "id": position.id,
            "symbol": position.symbol,
            "side": position.side,
            "entry_price": position.entry_price,
            "size_usdt": position.size_usdt,
            "leverage": position.leverage,
            "status": position.status,
            "open_time": position.open_time.isoformat() + 'Z'
        }
    }


@router.post("/position/{position_id}/close")
async def close_position(position_id: int, req: ClosePositionRequest):
    """Schließt eine Paper-Position."""
    engine = get_paper_engine()
    
    position = engine.close_position(position_id, req.exit_price, req.reason)
    
    if not position:
        raise HTTPException(status_code=404, detail="Position not found or already closed")
    
    return {
        "success": True,
        "position": {
            "id": position.id,
            "symbol": position.symbol,
            "side": position.side,
            "entry_price": position.entry_price,
            "exit_price": position.exit_price,
            "realized_pnl_pct": position.realized_pnl_pct,
            "realized_pnl_usdt": position.realized_pnl_usdt,
            "close_reason": position.close_reason,
            "status": position.status
        }
    }


@router.post("/price-update")
async def price_update(req: PriceUpdateRequest):
    """Aktualisiert Positionen mit aktuellem Preis."""
    engine = get_paper_engine()
    
    closed_positions = engine.update_positions_with_price(
        req.model_id, req.symbol, req.current_price
    )
    
    # Get current stats
    stats = engine.get_performance_stats(req.model_id)
    
    return {
        "success": True,
        "closed_positions": closed_positions,
        "current_stats": stats
    }


# Signal & Auto-Trading
@router.post("/signal")
async def process_signal(req: SignalRequest):
    """Verarbeitet ein ML-Signal und führt Auto-Trading aus."""
    engine = get_paper_engine()
    
    # Ensure account exists
    account = engine.get_or_create_account(req.model_id)
    
    result = {
        "signal": req.signal,
        "confidence": req.confidence,
        "actions": [],
        "opened_position": None,
        "closed_positions": []
    }
    
    # 1. Update existing positions with current price
    closed = engine.update_positions_with_price(req.model_id, req.symbol, req.current_price)
    result["closed_positions"] = closed
    
    # 2. Check for signal flip - close opposite positions
    open_positions = engine.get_open_positions(req.model_id)
    for pos in open_positions:
        if (req.signal == "LONG" and pos.side == "Short") or \
           (req.signal == "SHORT" and pos.side == "Long"):
            # Signal flip - close position
            closed_pos = engine.close_position(pos.id, req.current_price, "SIGNAL_FLIP")
            if closed_pos:
                result["closed_positions"].append({
                    "id": closed_pos.id,
                    "reason": "SIGNAL_FLIP",
                    "pnl_pct": closed_pos.realized_pnl_pct
                })
                result["actions"].append(f"Closed {pos.side} position due to signal flip")
    
    # 3. Block auto-trading until model package warmup is completed
    package_id = _resolve_model_package_id(req.model_id)
    pkg = get_model_package(package_id) if package_id else None
    if pkg and int(pkg.get("warmup_required", 0)) == 1 and pkg.get("warmup_status") != "DONE":
        result["actions"].append(
            f"Warmup pending for model package '{package_id}' (status={pkg.get('warmup_status')}). Trading is paused until first successful ML inference."
        )
        result["current_stats"] = engine.get_performance_stats(req.model_id)
        return result

    # 4. Open new position if signal is strong
    if req.confidence >= 60 and req.signal in ["LONG", "SHORT"]:
        # Allow multiple positions up to max_positions if total exposure < 10%
        open_positions = engine.get_open_positions(req.model_id)
        current_exposure = sum(p.size_usdt for p in open_positions)
        max_exposure = account.total_equity * 0.10
        
        if engine.can_open_position(req.model_id) and current_exposure < (max_exposure - 5): # Small buffer
            # Calculate position size (2% of equity per slot -> 10% total for 5 slots)
            position_size = account.total_equity * 0.02
            
            # Ensure we don't exceed max exposure
            if current_exposure + position_size > max_exposure:
                position_size = max_exposure - current_exposure
            
            # Minimum position size 10 USDT
            if position_size >= 10 and account.balance_usdt >= position_size / account.default_leverage:
                side = "Long" if req.signal == "LONG" else "Short"
                position = engine.open_position(
                    model_id=req.model_id,
                    symbol=req.symbol,
                    side=side,
                    entry_price=req.current_price,
                    size_usdt=position_size,
                    leverage=account.default_leverage
                )
                
                if position:
                    result["opened_position"] = {
                        "id": position.id,
                        "symbol": position.symbol,
                        "side": position.side,
                        "entry_price": position.entry_price,
                        "size_usdt": position.size_usdt,
                        "leverage": position.leverage
                    }
                    result["actions"].append(f"Opened {side} position with {req.confidence}% confidence (Slot {len(open_positions)+1})")
            else:
                result["actions"].append("Insufficient balance or exposure limit reached")
        else:
            result["actions"].append("Max positions or 10% exposure limit reached")
    
    # 4. Get updated stats
    result["current_stats"] = engine.get_performance_stats(req.model_id)
    
    return result


# ML Signal Endpoints
@router.get("/ml-signal/{model_id}")
async def get_ml_signal(model_id: str, symbol: str = "BTCUSDT"):
    """Holt aktuelles ML-Signal mit Confidence (PPO+Forecast)."""
    try:
        # Use PPO Forecast Inference (per model package)
        package_id = _resolve_model_package_id(model_id)
        inf = get_inference(model_package_id=package_id)
        result = inf.predict(symbol=symbol)
        
        if "error" in result:
             raise HTTPException(status_code=500, detail=result["error"])
        
        # Backward compatibility format
        signal = result.get("signal", "FLAT")
        confidence = result.get("confidence", 0)
        sentiment = result.get("sentiment", "neutral")
        current_price = result.get("current_price", 0.0)
        action_raw = result.get("action_raw", [])
        forecast_values = result.get("forecast_values", [])
        
        # Probabilities are fake/synthetic for UI compat
        probs = {
            "short": 90.0 if signal == "SHORT" else 10.0,
            "flat": 10.0,
            "long": 90.0 if signal == "LONG" else 10.0
        }
        
        return {
            "model_id": model_id,
            "symbol": symbol,
            "signal": signal,
            "confidence": confidence,
            "sentiment": sentiment,
            "current_price": current_price,
            "probabilities": probs,
            "action_raw": action_raw,
            "forecast_values": forecast_values,
            "timestamp": datetime.utcnow().isoformat() + 'Z',
            "diagnostics": result.get("diagnostics", {})
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")


@router.get("/ml-signal-lite/{model_id}")
async def get_ml_signal_lite(model_id: str, symbol: str = "BTCUSDT"):
    """Lite Signal: ohne Forecast-Payload (für schnelle UI-Updates)."""
    try:
        package_id = _resolve_model_package_id(model_id)
        inf = get_inference(model_package_id=package_id)
        result = inf.predict(symbol=symbol)
        
        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])
        
        return {
            "model_id": model_id,
            "symbol": symbol,
            "signal": result.get("signal", "FLAT"),
            "confidence": result.get("confidence", 0),
            "sentiment": result.get("sentiment", "neutral"),
            "current_price": result.get("current_price", 0.0),
            "action_raw": result.get("action_raw", []),
            "timestamp": datetime.utcnow().isoformat() + 'Z'
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")


# Dashboard Data
@router.get("/dashboard/{model_id}")
async def get_dashboard_data(model_id: str):
    """Holt alle Daten für das Dashboard."""
    engine = get_paper_engine()
    
    account = engine.get_account(model_id)
    if not account:
        # Auto-create account with defaults
        account = engine.create_account(
            model_id=model_id,
            initial_balance=10000.0,
            max_positions=5,
            default_leverage=7.0,
            profit_target_pct=5.0,
            time_limit_hours=48.0
        )
    
    # Get ML signal (PPO+Forecast)
    try:
        package_id = _resolve_model_package_id(model_id)
        inf = get_inference(model_package_id=package_id)
        res = inf.predict(symbol="BTCUSDT")
        
        if "error" in res:
            ml_signal = {"signal": "ERROR", "confidence": 0, "sentiment": "unknown", "error": res["error"]}
        else:
            ml_signal = {
                "signal": res.get("signal", "FLAT"), 
                "confidence": res.get("confidence", 0), 
                "sentiment": res.get("sentiment", "neutral"),
                "current_price": res.get("current_price", 0.0),
                "action_raw": res.get("action_raw", []),
                "forecast_values": res.get("forecast_values", []),
                "diagnostics": res.get("diagnostics", {})
            }
    except Exception as e:
        ml_signal = {"signal": "ERROR", "confidence": 0, "sentiment": "unknown", "error": str(e)}
    
    # Get positions with unrealized PnL
    open_positions = engine.get_open_positions(model_id)
    positions_with_pnl = []
    
    current_price = ml_signal.get("current_price", 0.0)
    if current_price and current_price > 0:
        for pos in open_positions:
            pnl_pct, pnl_usdt = engine.calculate_unrealized_pnl(pos, current_price)
            positions_with_pnl.append({
                "id": pos.id,
                "symbol": pos.symbol,
                "side": pos.side,
                "entry_price": pos.entry_price,
                "size_usdt": pos.size_usdt,
                "leverage": pos.leverage,
                "unrealized_pnl_pct": round(pnl_pct, 2),
                "unrealized_pnl_usdt": round(pnl_usdt, 2),
                "open_time": pos.open_time.isoformat() + 'Z'
            })
    
    # Get stats
    stats = engine.get_performance_stats(model_id)
    
    # Get closed trades
    closed_positions = engine.get_closed_positions(model_id, limit=10)
    recent_trades = [
        {
            "id": p.id,
            "symbol": p.symbol,
            "side": p.side,
            "entry_price": p.entry_price,
            "exit_price": p.exit_price,
            "pnl_pct": round(p.realized_pnl_pct, 2) if p.realized_pnl_pct else 0,
            "pnl_usdt": round(p.realized_pnl_usdt, 2) if p.realized_pnl_usdt else 0,
            "close_reason": p.close_reason,
            "close_time": p.close_time.isoformat() + 'Z' if p.close_time else None
        }
        for p in closed_positions
    ]
    
    package_id = _resolve_model_package_id(model_id)
    pkg = get_model_package(package_id) if package_id else None

    return {
        "account": {
            "model_id": account.model_id,
            "model_package_id": package_id,
            "warmup_required": bool(pkg and int(pkg.get("warmup_required", 0)) == 1),
            "warmup_status": (pkg or {}).get("warmup_status"),
            "initial_balance": account.initial_balance,
            "balance_usdt": account.balance_usdt,
            "total_equity": account.total_equity,
            "total_trades": account.total_trades,
            "winning_trades": account.winning_trades,
            "losing_trades": account.losing_trades,
            "win_rate": stats.get("win_rate", 0),
            "total_return_pct": stats.get("total_return_pct", 0),
            "total_return_usdt": stats.get("total_return_usdt", 0),
            "reset_at": account.reset_at.isoformat() + 'Z'
        },
        "ml_signal": ml_signal,
        "open_positions": positions_with_pnl,
        "open_positions_count": len(open_positions),
        "available_slots": account.max_positions - len(open_positions),
        "recent_trades": recent_trades,
        "performance": {
            "total_return_pct": stats.get("total_return_pct", 0),
            "win_rate": stats.get("win_rate", 0),
            "avg_pnl_pct": stats.get("avg_pnl_pct", 0),
            "avg_win_pct": stats.get("avg_win_pct", 0),
            "avg_loss_pct": stats.get("avg_loss_pct", 0),
            "open_exposure_pct": stats.get("open_exposure_pct", 0)
        }
    }


@router.get("/market-data/candles")
async def get_candles(symbol: str = "BTCUSDT", interval: str = "1h", limit: int = 100):
    """Holt Candles von Binance für Charts."""
    base_url = "https://api.binance.com/api/v3/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    
    try:
        r = requests.get(base_url, params=params, timeout=5)
        r.raise_for_status()
        data = r.json()
        
        # Format: [time, open, high, low, close, ...]
        candles = []
        for c in data:
            candles.append({
                "time": c[0], # ms timestamp
                "open": float(c[1]),
                "high": float(c[2]),
                "low": float(c[3]),
                "close": float(c[4]),
                "volume": float(c[5])
            })
            
        return {"symbol": symbol, "interval": interval, "candles": candles}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Binance API error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    from fastapi import FastAPI
    
    app = FastAPI(title="Paper Trading API")
    app.include_router(router)
    
    uvicorn.run(app, host="0.0.0.0", port=8001)
