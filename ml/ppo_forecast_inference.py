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
from typing import Dict, Any, Optional, List
from stable_baselines3 import PPO

# Ensure we can import TraderHimSelf modules
# Assuming this file is in projects/dings-trader/ml, we want to access projects/dings-trader/
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

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

class PPOForecastInference:
    def __init__(self):
        self.scaler = None
        self.forecast_model = None
        self.ppo_model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._load_artifacts()

    def _load_artifacts(self):
        base_dir = os.path.join(PROJECT_ROOT, "TraderHimSelf")
        scaler_path = os.path.join(base_dir, "data_processed", "scaler.pkl")
        forecast_path = os.path.join(base_dir, "models", "forecast_model.pt")
        ppo_path = os.path.join(base_dir, "models", "ppo_policy_final.zip")

        if not os.path.exists(scaler_path):
             raise FileNotFoundError(f"Scaler not found at {scaler_path}")
        if not os.path.exists(forecast_path):
             raise FileNotFoundError(f"Forecast model not found at {forecast_path}")
        if not os.path.exists(ppo_path):
             raise FileNotFoundError(f"PPO model not found at {ppo_path}")

        logger.info(f"Loading Scaler from {scaler_path}...")
        self.scaler = joblib.load(scaler_path)
        
        logger.info(f"Loading Forecast Model from {forecast_path}...")
        self.forecast_model = PatchTST()
        # Handle loading: map_location to device
        state_dict = torch.load(forecast_path, map_location=self.device)
        
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

        logger.info(f"Loading PPO Model from {ppo_path}...")
        # SB3 PPO load
        self.ppo_model = PPO.load(ppo_path, device=self.device)
        
        logger.info("All inference artifacts loaded.")

    def fetch_candles(self, symbol="BTCUSDT") -> pd.DataFrame:
        """
        Fetches enough 15m candles from Binance to support feature calc + lookback.
        We need: 672 (feature warm up) + 512 (model lookback) = 1184 candles.
        Safe buffer: 1500.
        """
        base_url = "https://api.binance.com/api/v3/klines"
        interval = "15m"
        limit_per_call = 1000
        
        # 1. Fetch Latest
        params = {"symbol": symbol, "interval": interval, "limit": limit_per_call}
        try:
            r = requests.get(base_url, params=params, timeout=10)
            r.raise_for_status()
            data1 = r.json()
        except Exception as e:
            logger.error(f"Binance API error: {e}")
            raise

        if not data1:
            raise ValueError("No data returned from Binance.")

        # 2. Fetch Previous
        start_time = data1[0][0]
        params["endTime"] = start_time - 1
        # We need ~500 more. limit=500 is enough.
        params["limit"] = 500
        
        try:
            r2 = requests.get(base_url, params=params, timeout=10)
            r2.raise_for_status()
            data2 = r2.json()
        except Exception as e:
            logger.warning(f"Binance API secondary fetch error: {e}. Proceeding with partial data.")
            data2 = []

        full_data = data2 + data1
        
        cols = ["open_time", "open", "high", "low", "close", "volume", 
                "close_time", "quote_asset_volume", "num_trades", 
                "taker_buy_base", "taker_buy_quote", "ignore"]
        
        df = pd.DataFrame(full_data, columns=cols)
        
        # Convert numeric
        for c in ["open", "high", "low", "close", "volume"]:
            df[c] = df[c].astype(float)
            
        df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
        df = df.set_index("open_time").sort_index()
        
        # De-dupe index
        df = df[~df.index.duplicated(keep='last')]
        
        return df

    def predict(self, symbol="BTCUSDT", account_state: Optional[List[float]] = None) -> Dict[str, Any]:
        """
        Main inference method.
        Returns dict with signal, confidence, etc.
        """
        try:
            # 1. Get Data
            df = self.fetch_candles(symbol)
            
            # 2. Compute Core Features
            # feature_engine.compute_core_features expects open, high, low, close, volume
            # and DatetimeIndex.
            feats_df = compute_core_features(df)
            
            # 3. Prepare for Forecast Model
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
            scaled_input = self.scaler.transform(input_df[FEATURE_COLUMNS].values)
            
            # To Tensor: (1, 512, 28)
            input_tensor = torch.tensor(scaled_input, dtype=torch.float32).unsqueeze(0).to(self.device)
            
            # 4. Forecast Inference
            with torch.no_grad():
                # Output: (1, 192, 3)
                forecast_out = self.forecast_model(input_tensor)
                
            forecast_out_np = forecast_out.cpu().numpy()[0]
            
            # 5. Compute Forecast Features (35 dim)
            forecast_feats = compute_forecast_features(forecast_out_np)
            
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
                
            current_price = float(df['close'].iloc[-1])
            
            return {
                "signal": direction,
                "confidence": int(confidence),
                "sentiment": sentiment,
                "current_price": current_price,
                "action_raw": action.tolist(),
                "diagnostics": {
                    "lookback_rows_used": len(valid_feats),
                    "warmup_ready": True,
                    "forecast_min": float(forecast_feats[27]), # min_ret from stats
                    "forecast_max": float(forecast_feats[28]), # max_ret
                }
            }

        except Exception as e:
            logger.error(f"Inference failed: {e}", exc_info=True)
            return {
                "signal": "ERROR", 
                "confidence": 0, 
                "error": str(e),
                "current_price": 0.0
            }

# Singleton accessor
_inference_instance = None

def get_inference():
    global _inference_instance
    if _inference_instance is None:
        _inference_instance = PPOForecastInference()
    return _inference_instance

if __name__ == "__main__":
    # Smoke test
    print("Running smoke test...")
    inf = get_inference()
    res = inf.predict()
    print("Result:", json.dumps(res, indent=2))
