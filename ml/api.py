from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import os
from datetime import datetime

import requests

from db import (
    init_db,
    ensure_model,
    update_equity,
    check_kill_switch,
    get_model,
    get_trades as get_db_trades,
    get_all_models,
    ensure_default_model_package,
    create_model_package,
    list_model_packages,
)


# Import Paper Trading
from paper_trading import get_paper_engine
from paper_api import router as paper_router

# Server start time - persistiert über Reloads
SERVER_START_TIME = datetime.utcnow().isoformat()

app = FastAPI(title="Symbiomorphose Trader API")

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include Paper Trading Router
app.include_router(paper_router)

# --- DB init + default model package seed ---
init_db()

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_FORECAST_REL = os.path.join("TraderHimSelf", "models", "forecast_model.pt")
DEFAULT_PPO_REL = os.path.join("TraderHimSelf", "models", "ppo_policy_final.zip")
ensure_default_model_package(
    package_id="ppo_v1",
    name="PPO v1 (ML)",
    forecast_rel_path=DEFAULT_FORECAST_REL,
    ppo_rel_path=DEFAULT_PPO_REL,
    status="READY",
    warmup_required=True,
)

MODEL_PACKAGES_DIR = os.environ.get(
    "MODEL_PACKAGES_DIR",
    os.path.join(PROJECT_ROOT, "TraderHimSelf", "models", "packages"),
)

DATA_DIR = "/home/maxim/.openclaw/workspace/projects/dings-trader/data"
INITIAL_CAPITAL = 1100.0
KILL_SWITCH_EQUITY = 200.0

@app.get("/")
async def root():
    return {"status": "Symbiomorphose Active", "version": "2.0", "paper_trading": True}

@app.get("/backend-status")
async def get_backend_status():
    """Returns server start time and uptime info"""
    return {
        "status": "running",
        "server_start_time": SERVER_START_TIME,
        "version": "2.0",
        "paper_trading": True
    }

@app.get("/live-price")
async def get_live_price(symbol: str = "BTCUSDT"):
    try:
        r = requests.get(f"https://api.binance.com/api/v3/ticker/price?symbol={symbol}")
        return r.json()
    except Exception:
        return {"error": "Failed to fetch price"}

@app.get("/equity")
async def get_equity(run: str = "test_2025"):
    # Handle paper trading models
    if run.startswith("paper_") or run == "paper_test_hf":
        engine = get_paper_engine()
        model_id = run if run == "paper_test_hf" else run
        
        # Ensure account exists
        account = engine.get_account(model_id)
        if not account:
            # Create account if it doesn't exist
            account = engine.create_account(model_id)
        
        # Get equity history
        history = engine.get_equity_history(model_id, limit=500)
        
        # If no history yet, create initial data point
        if not history:
            history = [{
                "timestamp": datetime.utcnow().isoformat(),
                "equity": account.total_equity,
                "balance": account.balance_usdt,
                "open_positions_count": 0,
                "unrealized_pnl": 0
            }]
        
        # Format for chart: use 'equity' key and convert timestamp
        formatted = []
        for h in history:
            formatted.append({
                "timestamp": h["timestamp"],
                "equity": float(h["equity"]),
                "balance": float(h["balance"]),
                "open_positions": h["open_positions_count"],
                "unrealized_pnl": float(h["unrealized_pnl"])
            })
        
        return formatted
    
    # Legacy backtest runs
    file_map = {
        "val_2024": "equity_val_2024_strict_48h_v2.parquet",
        "test_2025": "equity_test_2025.parquet"
    }
    file_name = file_map.get(run)
    if not file_name:
        return {"error": "Run not found"}
    
    path = os.path.join(DATA_DIR, file_name)
    if not os.path.exists(path):
        return {"error": "File not found"}
    
    df = pd.read_parquet(path)

    last_equity = None
    try:
        if "equity" in df.columns:
            last_equity = float(df["equity"].iloc[-1])
        else:
            numeric_cols = df.select_dtypes(include="number").columns
            if len(numeric_cols) > 0:
                last_equity = float(df[numeric_cols[-1]].iloc[-1])
    except Exception:
        last_equity = None

    if last_equity is not None:
        ensure_model(run, version=run, initial_capital=INITIAL_CAPITAL)
        update_equity(run, last_equity)
        check_kill_switch(run, threshold=KILL_SWITCH_EQUITY)

    # Return last 100 points for the chart
    return df.tail(100).to_dict(orient="records")

@app.get("/status")
async def get_status():
    roadmap_path = "/home/maxim/.openclaw/workspace/projects/dings-trader/docs/ROADMAP.md"
    with open(roadmap_path, "r") as f:
        content = f.read()
    return {"roadmap": content}

@app.get("/trades")
async def get_trades(run: str = "test_2025"):
    return get_db_trades(run)

@app.get("/metrics")
async def get_metrics(run: str = "test_2025"):
    # Mocked based on calc_metrics.py results for 2025
    if run == "test_2025":
        return {
            "sharpe_ratio": 0.22,
            "max_drawdown": "-22.50%",
            "final_equity": "1.027,40 €"
        }
    return {"error": "Metrics not found"}

@app.get("/model-status")
async def get_model_status(model_id: str = "test_2025"):
    ensure_model(model_id, version=model_id, initial_capital=INITIAL_CAPITAL)
    check_kill_switch(model_id, threshold=KILL_SWITCH_EQUITY)
    model = get_model(model_id)
    if not model:
        return {"error": "Model not found"}
    return model

@app.get("/models")
async def get_models():
    return get_all_models()


# --- Model package registry (for multi-model selection + upload) ---

@app.get("/model-packages")
async def get_model_packages():
    """List available model packages (forecast + PPO)."""
    return list_model_packages()


def _slugify(value: str) -> str:
    import re

    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("_")
    return value or "model"


@app.post("/model-packages/upload")
async def upload_model_package(
    name: str = Form(...),
    forecast_model: UploadFile = File(...),
    ppo_model: UploadFile = File(...),
):
    """Upload a new model package (multipart/form-data).

    Required parts:
    - name: string
    - forecast_model: .pt
    - ppo_model: .zip
    """
    # Validate extensions
    if not forecast_model.filename or not forecast_model.filename.endswith(".pt"):
        raise HTTPException(status_code=400, detail="forecast_model must be a .pt file")
    if not ppo_model.filename or not ppo_model.filename.endswith(".zip"):
        raise HTTPException(status_code=400, detail="ppo_model must be a .zip file")

    # Create deterministic ID from name, ensure uniqueness
    base_id = _slugify(name)
    existing = {p["id"] for p in list_model_packages()}
    package_id = base_id
    suffix = 2
    while package_id in existing:
        package_id = f"{base_id}_{suffix}"
        suffix += 1

    # Prepare paths
    os.makedirs(MODEL_PACKAGES_DIR, exist_ok=True)
    package_dir = os.path.join(MODEL_PACKAGES_DIR, package_id)
    os.makedirs(package_dir, exist_ok=True)

    forecast_filename = "forecast_model.pt"
    ppo_filename = "ppo_policy.zip"

    forecast_abs = os.path.join(package_dir, forecast_filename)
    ppo_abs = os.path.join(package_dir, ppo_filename)

    # Save files
    forecast_bytes = await forecast_model.read()
    ppo_bytes = await ppo_model.read()
    if len(forecast_bytes) == 0 or len(ppo_bytes) == 0:
        raise HTTPException(status_code=400, detail="Uploaded files must not be empty")

    with open(forecast_abs, "wb") as f:
        f.write(forecast_bytes)
    with open(ppo_abs, "wb") as f:
        f.write(ppo_bytes)

    # Store as rel paths (portable)
    forecast_rel = os.path.relpath(forecast_abs, PROJECT_ROOT)
    ppo_rel = os.path.relpath(ppo_abs, PROJECT_ROOT)

    create_model_package(
        package_id=package_id,
        name=name.strip(),
        forecast_rel_path=forecast_rel,
        ppo_rel_path=ppo_rel,
        status="UPLOADED",
        warmup_required=True,
    )

    return {
        "id": package_id,
        "name": name.strip(),
        "status": "UPLOADED",
        "warmup_required": True,
        "warmup_status": "PENDING",
    }


@app.get("/signals")
async def get_signals(run: str = "test_2025"):
    # Mock signals - would come from live_inference.py result
    # Only return signals for the requested run/model
    if run == "v2" or run == "test_2025":
        return [
            {"symbol": "BTC-USDT", "action": "Long", "confidence": "High", "variant": "success"}
        ]
    return []

@app.get("/positions")
async def get_positions(run: str = "test_2025"):
    # Mock open positions - would come from DB or exchange
    if run == "v2" or run == "test_2025":
        return [
            {
                "symbol": "BTC-USDT",
                "side": "Long",
                "size": "110,00 €",
                "entry": "$96.110",
                "unrealized_pnl": "+3.2%"
            }
        ]
    return []

# Paper Trading Endpoints (also available at root level for convenience)
@app.get("/paper-account/{model_id}")
async def paper_account_root(model_id: str):
    """Alias für /paper/account/{model_id}"""
    from fastapi import HTTPException
    
    engine = get_paper_engine()
    account = engine.get_account(model_id)
    
    if not account:
        return {"error": "Account not found"}
    
    return {
        "model_id": account.model_id,
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
        "reset_at": account.reset_at.isoformat()
    }

@app.get("/paper-dashboard/{model_id}")
async def paper_dashboard_root(model_id: str):
    """Alias für /paper/dashboard/{model_id}"""
    from paper_api import get_dashboard_data
    return await get_dashboard_data(model_id)

@app.post("/paper-signal")
async def paper_signal(signal_data: dict):
    """Shortcut für /paper/signal"""
    from paper_api import process_signal
    from pydantic import BaseModel
    
    class SignalRequest(BaseModel):
        model_id: str
        symbol: str
        signal: str
        confidence: float
        current_price: float
    
    req = SignalRequest(**signal_data)
    return await process_signal(req)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
