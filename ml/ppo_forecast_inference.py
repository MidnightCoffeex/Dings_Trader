"""
PPO Forecast Inference Module.
Handles live inference using TraderHimSelf models (PatchTST Forecast + PPO Policy).
"""
import os
import sys
import logging
import json
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import requests
import time
from typing import Dict, Any, Optional, List, Tuple
from stable_baselines3 import PPO

# Ensure we can import TraderHimSelf and ml modules
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ML_DIR = os.path.dirname(os.path.abspath(__file__))
for d in [PROJECT_ROOT, ML_DIR]:
    if d not in sys.path:
        sys.path.append(d)

# DB-backed model package registry (forecast + PPO artifacts)
try:
    from db import init_db, get_model_package, ensure_default_model_package, set_model_package_warmup_status

    init_db()
    ensure_default_model_package(
        package_id="ppo_v1",
        name="PPO v1 (ML)",
        forecast_rel_path=os.path.join("TraderHimSelf", "models", "forecast_model.pt"),
        ppo_rel_path=os.path.join("TraderHimSelf", "models", "ppo_policy_final.zip"),
        status="READY",
        warmup_required=True,
    )
except Exception as _e:
    # Inference can still run in standalone mode; registry features will be disabled.
    get_model_package = None  # type: ignore
    set_model_package_warmup_status = None  # type: ignore

from feature_cache import get_feature_cache

logger = logging.getLogger("PPOInference")
# Configure logger if not already configured
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

# Attempt imports from TraderHimSelf
try:
    from TraderHimSelf.feature_engine import compute_core_features, FEATURE_COLUMNS
except ImportError as e:
    logger.error(f"Failed to import TraderHimSelf modules: {e}")
    # We might fail hard here or later.
    pass

# --- PatchTST Model Definition (Must match training in TraderHimSelf/forecast/train_patchtst.py) ---

class PatchEmbedding(nn.Module):
    def __init__(self, patch_len, stride, d_model):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.project = nn.Linear(patch_len, d_model)

    def forward(self, x):
        B, C, L = x.shape
        # Patching: (B, C, L) -> (B, C, N_patches, patch_len)
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        x = self.project(x) 
        return x

class PatchTST(nn.Module):
    def __init__(self, input_dim=28, lookback=512, forecast_len=192, 
                 patch_len=16, stride=8, d_model=128, n_heads=4, n_layers=3, dropout=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.lookback = lookback
        self.forecast_len = forecast_len
        
        self.n_patches = (lookback - patch_len) // stride + 1
        
        self.patch_embed = PatchEmbedding(patch_len, stride, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, 1, self.n_patches, d_model))
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, 
                                                   dim_feedforward=d_model*4, dropout=dropout,
                                                   batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        self.flatten_dim = input_dim * self.n_patches * d_model
        
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.flatten_dim, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, forecast_len * 3) # 3 quantiles
        )

    def forward(self, x):
        # x: (B, L, C) -> transpose to (B, C, L)
        x = x.transpose(1, 2)
        
        x = self.patch_embed(x)
        x = x + self.pos_embed
        
        B, C, N, D = x.shape
        # Reshape for Transformer: (B*C, N, D) - Channel Independence
        x = x.reshape(B*C, N, D)
        
        x = self.encoder(x)
        
        # Reshape back
        x = x.reshape(B, C, N, D)
        
        out = self.head(x)
        
        out = out.reshape(B, self.forecast_len, 3)
        return out

def compute_forecast_features(forecast_seq):
    """
    Computes 35 forecast features from (192, 3) quantile predictions.
    Matches logic in TraderHimSelf/forecast/train_patchtst.py.
    """
    q10 = forecast_seq[:, 0]
    q50 = forecast_seq[:, 1]
    q90 = forecast_seq[:, 2]
    
    # 1. Horizon Block (15 features)
    indices = [3, 15, 47, 95, 191]
    horizon_feats = []
    for idx in indices:
        horizon_feats.extend([q10[idx], q50[idx], q90[idx]])
        
    # 2. Path Block (12 features)
    path_indices = np.arange(15, 192, 16)
    path_feats = q50[path_indices].tolist()
    
    # 3. Curve Stats (8 features) on q50
    curve = np.concatenate(([0], q50))
    
    min_ret = np.min(curve)
    max_ret = np.max(curve)
    
    time_to_min = np.argmin(curve) / 192.0
    time_to_max = np.argmax(curve) / 192.0
    
    running_max = np.maximum.accumulate(curve)
    dd = running_max - curve
    max_drawdown = np.max(dd)
    
    running_min = np.minimum.accumulate(curve)
    ru = curve - running_min
    max_runup = np.max(ru)
    
    slope_1 = curve[48] / 48.0
    slope_2 = (curve[192] - curve[48]) / (192.0 - 48.0)
    
    stats_feats = [min_ret, max_ret, time_to_min, time_to_max, max_drawdown, max_runup, slope_1, slope_2]
    
    all_feats = np.array(horizon_feats + path_feats + stats_feats, dtype=np.float32)
    return all_feats

def fetch_candles_shared(symbol="BTCUSDT", lookback_total=1500) -> pd.DataFrame:
    """
    Fetches enough 15m candles from Binance to support feature calc + lookback.
    Default lookback_total: 1500.
    """
    cache = get_feature_cache()
    cache_key = f"candles_{symbol}_{lookback_total}"
    cached = cache.get(cache_key)
    if cached is not None:
        return cached

    base_url = "https://api.binance.com/api/v3/klines"
    interval = "15m"
    
    # If we need more than 1000, we need multiple calls
    all_data = []
    end_time = None
    
    needed = lookback_total
    while needed > 0:
        limit = min(needed, 1000)
        params = {"symbol": symbol, "interval": interval, "limit": limit}
        if end_time:
            params["endTime"] = end_time - 1
            
        try:
            r = requests.get(base_url, params=params, timeout=10)
            r.raise_for_status()
            data = r.json()
            if not data:
                break
            all_data = data + all_data
            end_time = data[0][0]
            needed -= len(data)
        except Exception as e:
            logger.error(f"Binance API error: {e}")
            if not all_data:
                raise
            break

    cols = ["open_time", "open", "high", "low", "close", "volume", 
            "close_time", "quote_asset_volume", "num_trades", 
            "taker_buy_base", "taker_buy_quote", "ignore"]
    
    df = pd.DataFrame(all_data, columns=cols)
    
    # Convert numeric
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = df[c].astype(float)
        
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df = df.set_index("open_time").sort_index()
    
    # De-dupe index
    df = df[~df.index.duplicated(keep='last')]
    
    cache.set(cache_key, df, ttl=60) # Cache for 60s
    return df

def get_shared_features(symbol="BTCUSDT", lookback_total=1500) -> Tuple[pd.DataFrame, float]:
    cache = get_feature_cache()
    cache_key = f"features_data_{symbol}_{lookback_total}"
    cached = cache.get(cache_key)
    if cached is not None:
        return cached
    
    df = fetch_candles_shared(symbol, lookback_total)
    feats_df = compute_core_features(df)
    last_price = float(df['close'].iloc[-1])
    
    result = (feats_df, last_price)
    cache.set(cache_key, result, ttl=60)
    return result

class PPOForecastInference:
    def __init__(
        self,
        model_package_id: str = "ppo_v1",
        forecast_rel_path: str | None = None,
        ppo_rel_path: str | None = None,
        feature_mask: Optional[List[int]] = None,
    ):
        self.model_package_id = model_package_id
        self.feature_mask = feature_mask
        self.scaler = None
        self.forecast_model = None
        self.ppo_model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._cache = {}
        self._cache_ttl = 15  # seconds

        # Resolve artifact paths via registry if available
        if forecast_rel_path is None or ppo_rel_path is None:
            pkg = get_model_package(model_package_id) if callable(get_model_package) else None
            
            if pkg:
                forecast_rel_path = pkg.get("forecast_rel_path")
                ppo_rel_path = pkg.get("ppo_rel_path")
                if self.feature_mask is None and pkg.get("feature_mask"):
                    try:
                        self.feature_mask = json.loads(pkg.get("feature_mask"))
                    except Exception:
                        logger.warning(f"Failed to parse feature_mask for {model_package_id}")
            elif model_package_id == "ppo_v1":
                # Hardcoded fallback only for default model
                forecast_rel_path = os.path.join("TraderHimSelf", "models", "forecast_model.pt")
                ppo_rel_path = os.path.join("TraderHimSelf", "models", "ppo_policy_final.zip")
            else:
                raise ValueError(f"Model package '{model_package_id}' not found in registry and no default paths provided.")

        if forecast_rel_path is None or ppo_rel_path is None:
             raise ValueError(f"Could not resolve paths for model package '{model_package_id}'")

        self.forecast_path = os.path.join(PROJECT_ROOT, forecast_rel_path)
        self.ppo_path = os.path.join(PROJECT_ROOT, ppo_rel_path)

        self._load_artifacts()

    def _load_artifacts(self):
        base_dir = os.path.join(PROJECT_ROOT, "TraderHimSelf")
        scaler_path = os.path.join(base_dir, "data_processed", "scaler.pkl")

        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"Scaler not found at {scaler_path}")
        if not os.path.exists(self.forecast_path):
            raise FileNotFoundError(f"Forecast model not found at {self.forecast_path}")
        if not os.path.exists(self.ppo_path):
            raise FileNotFoundError(f"PPO model not found at {self.ppo_path}")

        logger.info(f"Loading Scaler from {scaler_path}...")
        self.scaler = joblib.load(scaler_path)

        logger.info(f"Loading Forecast Model from {self.forecast_path}...")
        self.forecast_model = PatchTST()
        # Handle loading: map_location to device
        state_dict = torch.load(self.forecast_path, map_location=self.device)
        
        # Normalize state dict keys (remove _orig_mod prefix)
        new_state_dict = {}
        for k, v in state_dict.items():
            key = k
            if key.startswith("_orig_mod."):
                key = key[10:]
            new_state_dict[key] = v
            
        self.forecast_model.load_state_dict(new_state_dict)
        self.forecast_model.to(self.device)
        self.forecast_model.eval()

        logger.info(f"Loading PPO Model from {self.ppo_path}...")
        # SB3 PPO load
        self.ppo_model = PPO.load(self.ppo_path, device=self.device)
        
        logger.info("All inference artifacts loaded.")

    def _get_cached(self, symbol: str):
        entry = self._cache.get(symbol)
        if not entry:
            return None
        if entry["expires_at"] < time.time():
            self._cache.pop(symbol, None)
            return None
        return entry["value"]

    def _set_cache(self, symbol: str, value: Dict[str, Any]):
        self._cache[symbol] = {
            "expires_at": time.time() + self._cache_ttl,
            "value": value,
        }

    def predict(self, symbol="BTCUSDT", account_state: Optional[List[float]] = None, lookback_total: int = 1500) -> Dict[str, Any]:
        """
        Main inference method.
        Returns dict with signal, confidence, etc.
        """
        symbol = symbol.upper()
        if account_state is None:
            cached = self._get_cached(symbol)
            if cached:
                return cached
        try:
            # Registry warmup status update (best-effort)
            pkg = None
            if callable(get_model_package) and callable(set_model_package_warmup_status):
                try:
                    pkg = get_model_package(self.model_package_id)
                    if pkg and int(pkg.get("warmup_required", 0)) == 1 and pkg.get("warmup_status") == "PENDING":
                        set_model_package_warmup_status(self.model_package_id, "RUNNING", completed=False)
                except Exception:
                    pkg = None

            # 1. Get Shared Features
            feats_df, current_price = get_shared_features(symbol, lookback_total)
            
            # 2. Prepare for Forecast Model
            # Remove initial NaNs caused by rolling windows
            valid_feats = feats_df.dropna()
            
            # Check length
            if len(valid_feats) < 512:
                # Not enough data
                return {
                    "signal": "FLAT",
                    "confidence": 0,
                    "error": f"Insufficient warmup data. Valid rows: {len(valid_feats)} < 512."
                }
                
            # Take last 512 rows
            input_df = valid_feats.iloc[-512:]
            
            # Scale
            # transform returns numpy array
            raw_features = input_df[FEATURE_COLUMNS].values
            
            # Apply feature mask if present (pro Modell feature selection)
            if self.feature_mask is not None:
                mask = np.array(self.feature_mask)
                if len(mask) == len(FEATURE_COLUMNS):
                    raw_features = raw_features * mask

            scaled_input = self.scaler.transform(raw_features)
            
            # To Tensor: (1, 512, 28)
            input_tensor = torch.tensor(scaled_input, dtype=torch.float32).unsqueeze(0).to(self.device)
            
            # 4. Forecast Inference
            with torch.no_grad():
                # Output: (1, 192, 3)
                forecast_out = self.forecast_model(input_tensor)
                
            forecast_out_np = forecast_out.cpu().numpy()[0]
            
            # 5. Compute Forecast Features (35 dim)
            forecast_feats = compute_forecast_features(forecast_out_np)
            
            # Convert forecast_out_np to list for JSON response
            forecast_values = forecast_out_np.tolist()
            
            # 6. Build PPO Observation
            # [Core28 (latest) + Forecast35 + Account9] = 72
            
            core_28 = scaled_input[-1] # Shape (28,)
            
            # Account State
            if account_state is None:
                # Default "Empty" state:
                # [pos_count=0, major_side=0, exp_open=0, not_open=0, upnl=0, 
                #  time_trade=0, time_left=192, liq_buff=1.0, avail_exp=0.1]
                # Assuming max_exposure_pct=0.1
                account_vec = np.array([0, 0, 0, 0, 0, 0, 192, 1.0, 0.1], dtype=np.float32)
            else:
                account_vec = np.array(account_state, dtype=np.float32)
                
            obs = np.concatenate([core_28, forecast_feats, account_vec]) # (72,)
            
            # 7. PPO Inference
            # predict returns (action, state)
            action, _ = self.ppo_model.predict(obs, deterministic=True)
            
            # 8. Decode Action
            # action shape (5,) -> [dir, size, lev, sl, tp] all in [-1, 1]
            act_dir_raw = float(action[0])
            act_size_raw = float(action[1])
            act_lev_raw = float(action[2])
            
            direction = "FLAT"
            confidence = 0.0
            sentiment = "neutral"
            
            if act_dir_raw > 0.33:
                direction = "LONG"
                confidence = min((act_dir_raw - 0.33) / 0.67 * 100 + 50, 99)
                sentiment = "bullish"
            elif act_dir_raw < -0.33:
                direction = "SHORT"
                confidence = min((abs(act_dir_raw) - 0.33) / 0.67 * 100 + 50, 99)
                sentiment = "bearish"
                
            result = {
                "signal": direction,
                "confidence": int(confidence),
                "sentiment": sentiment,
                "current_price": current_price,
                "action_raw": action.tolist(),
                "forecast_values": forecast_values,
                "diagnostics": {
                    "lookback_rows_used": len(valid_feats),
                    "warmup_ready": True,
                    "forecast_min": float(forecast_feats[27]), # min_ret from stats
                    "forecast_max": float(forecast_feats[28]), # max_ret
                }
            }

            if pkg and int(pkg.get("warmup_required", 0)) == 1 and callable(set_model_package_warmup_status):
                try:
                    # Mark warmup as completed on first successful inference
                    if pkg.get("warmup_status") != "DONE":
                        set_model_package_warmup_status(self.model_package_id, "DONE", completed=True)
                except Exception:
                    pass

            if account_state is None and "error" not in result:
                self._set_cache(symbol, result)

            return result

        except Exception as e:
            logger.error(f"Inference failed: {e}", exc_info=True)
            if pkg and int(pkg.get("warmup_required", 0)) == 1 and callable(set_model_package_warmup_status):
                try:
                    set_model_package_warmup_status(self.model_package_id, "FAILED", completed=False, error_msg=str(e))
                except Exception:
                    pass
            return {
                "signal": "ERROR",
                "confidence": 0,
                "error": str(e),
                "current_price": 0.0,
            }


# Multi-package singleton accessor
_inference_instances: Dict[str, PPOForecastInference] = {}


def get_inference(model_package_id: str = "ppo_v1") -> PPOForecastInference:
    instance = _inference_instances.get(model_package_id)
    if instance is None:
        instance = PPOForecastInference(model_package_id=model_package_id)
        _inference_instances[model_package_id] = instance
    return instance

if __name__ == "__main__":
    # Smoke test
    print("Running smoke test...")
    inf = get_inference()
    res = inf.predict()
    print("Result:", json.dumps(res, indent=2))
