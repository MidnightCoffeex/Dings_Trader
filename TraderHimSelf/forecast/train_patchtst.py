import os
import glob
import logging
import argparse
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
import math
from typing import List, Optional
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from feature_engine import FEATURE_COLUMNS

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("Forecast-PatchTST")

# Constants
LOOKBACK = 512
FORECAST_HORIZON = 192  # Max horizon (48h * 4 steps/h)
INPUT_CHANNELS = 28
QUANTILES = [0.1, 0.5, 0.9]
HORIZON_STEPS = [4, 16, 48, 96, 192]  # 1h, 4h, 12h, 24h, 48h
HORIZON_WEIGHTS = {4: 1.0, 16: 1.0, 48: 0.8, 96: 0.6, 192: 0.4}
BATCH_SIZE = 32 # Can be adjusted
EPOCHS = 20
LR = 1e-4
NUM_WORKERS = 0

# Paths
DATA_DIR = os.path.join(BASE_DIR, "data_processed")
MODEL_DIR = os.path.join(BASE_DIR, "models")
SCALER_PATH = os.path.join(DATA_DIR, "scaler.pkl")
DATA_FILE_FEATURES = os.path.join(DATA_DIR, "features.parquet")      # scaled 28 core features
DATA_FILE_15M_RAW = os.path.join(DATA_DIR, "aligned_15m.parquet")    # for 'close' prices
OUTPUT_FEATURES_PATH = os.path.join(DATA_DIR, "forecast_features.parquet")

os.makedirs(MODEL_DIR, exist_ok=True)

# --- 1. Dataset ---

class ForecastDataset(Dataset):
    def __init__(self, df, lookback=LOOKBACK, forecast_horizon=FORECAST_HORIZON, mode='train', feature_cols=None):
        """
        df: DataFrame indexed by UTC timestamps. Must contain:
            - 28 core feature columns (scaled)
            - 'close' column (raw close prices)
        mode: 'train', 'val', 'test', 'inference'
        feature_cols: explicit list of the 28 feature columns (recommended)
        """
        self.lookback = lookback
        self.forecast_horizon = forecast_horizon
        self.mode = mode
        
        # Ensure data is sorted
        # Assuming df has index or time column, but here we assume passed df is correct order
        self.data = df
        
        # Identify feature columns (assuming first 28 columns match core features or passed explicitly)
        # For simplicity, we assume the first 28 columns are the features, or we define them.
        # Based on roadmap: "28 Core Features (normalisiert)". 
        # We assume the dataframe passed here already has the scaler applied or we handle normalization outside.
        # Ideally, we select the specific columns.
        # Feature column selection
        if feature_cols is not None:
            self.feature_cols = list(feature_cols)
        else:
            # Fallback: assume the first 28 columns are features (and 'close' is separate/last)
            self.feature_cols = list(df.columns[:INPUT_CHANNELS])
        
        # We need 'close' for target calculation if not in features (it usually isn't raw close)
        if 'close' in df.columns:
            self.close_prices = df['close'].values
        else:
            # Fallback for inference if close not provided separately but needed? 
            # Actually targets are derived from close. For inference we don't need targets.
            self.close_prices = np.zeros(len(df))

        # Convert features to float32 numpy
        self.features = df[self.feature_cols].values.astype(np.float32)
        
        # Calculate valid indices (t = index of the *last* row included in the lookback window)
        # We want x to include rows [t-lookback+1 .. t] (length=lookback).
        # For training targets we need future closes up to t+forecast_horizon.
        t_min = self.lookback - 1
        if self.mode in ['train', 'val', 'test']:
            t_max = len(df) - self.forecast_horizon - 1
            self.valid_indices = range(t_min, max(t_min, t_max) + 1)
        else:
            # Inference: we only need history
            self.valid_indices = range(t_min, len(df))

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        t = self.valid_indices[idx]
        
        # Input window includes row t (last known information at time t)
        x = self.features[t - self.lookback + 1 : t + 1]

        if self.mode == 'inference':
            return torch.tensor(x), torch.zeros(self.forecast_horizon)  # Dummy target

        # Target: log returns for the next forecast_horizon steps relative to close[t]
        # Roadmap: y_{t,s} = log(close_{t+s} / close_t), with s in 1..H
        current_close = self.close_prices[t]
        future_closes = self.close_prices[t + 1 : t + 1 + self.forecast_horizon]

        eps = 1e-8
        y = np.log((future_closes + eps) / (current_close + eps))

        return torch.tensor(x), torch.tensor(y, dtype=torch.float32)

# --- 2. Model: PatchTST (Simplified & Adapted) ---

class PatchEmbedding(nn.Module):
    def __init__(self, patch_len, stride, d_model):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.project = nn.Linear(patch_len, d_model)

    def forward(self, x):
        # x: (B, C, L)
        B, C, L = x.shape
        # Patching
        # Output: (B, C, N_patches, patch_len)
        # Using unfold
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        # x: (B, C, N_patches, patch_len)
        
        # Project to d_model
        x = self.project(x) 
        # x: (B, C, N_patches, d_model)
        return x

class PatchTST(nn.Module):
    def __init__(self, input_dim=28, lookback=512, forecast_len=192, 
                 patch_len=16, stride=8, d_model=128, n_heads=4, n_layers=3, dropout=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.lookback = lookback
        self.forecast_len = forecast_len
        
        # Calculate number of patches
        self.n_patches = (lookback - patch_len) // stride + 1
        
        self.patch_embed = PatchEmbedding(patch_len, stride, d_model)
        
        # Positional Encoding (Learnable)
        self.pos_embed = nn.Parameter(torch.randn(1, 1, self.n_patches, d_model))
        
        # Transformer Encoder (Channel Independent -> Batch dimension includes channels)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, 
                                                   dim_feedforward=d_model*4, dropout=dropout,
                                                   batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # Head: Multivariate to Univariate (3 quantiles per step)
        # We flatten all channels and patches: 28 * N_patches * d_model -> very large.
        # Strategy: Global pooling or Flatten? 
        # Roadmap implies "Architectur: PatchTST". Original PatchTST uses Linear Head per channel.
        # But we need to predict Close Return based on ALL 28 features.
        # So we must mix channels at the end.
        
        self.flatten_dim = input_dim * self.n_patches * d_model
        
        # Simple MLP Head
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
        
        # Embedding: (B, C, N, D)
        x = self.patch_embed(x)
        
        # Add pos embed
        x = x + self.pos_embed
        
        B, C, N, D = x.shape
        # Reshape for Transformer: (B*C, N, D) - Channel Independence
        x = x.reshape(B*C, N, D)
        
        # Encoder
        x = self.encoder(x)
        
        # Reshape back: (B, C, N, D)
        x = x.reshape(B, C, N, D)
        
        # Head
        out = self.head(x) # (B, H*3)
        
        # Reshape to (B, H, 3)
        out = out.reshape(B, self.forecast_len, 3)
        return out

# --- 3. Loss: Quantile / Pinball ---

def quantile_loss(preds, target, quantiles=QUANTILES, weights=None, horizon_indices=None):
    """
    preds: (B, H, 3) - predicted quantiles
    target: (B, H) - actual returns
    weights: list of weights for specific horizons (optional)
    horizon_indices: list of indices to apply loss on (optional)
    """
    loss = 0
    
    # If we focus on specific horizons
    if horizon_indices is not None:
        # Select steps
        preds = preds[:, horizon_indices, :]
        target = target[:, horizon_indices]
        # weights should match horizon_indices length
        if weights is not None:
            w = torch.tensor(weights, device=preds.device).view(1, -1)
        else:
            w = 1.0
    else:
        w = 1.0
        
    for i, q in enumerate(quantiles):
        pred_q = preds[:, :, i]
        errors = target - pred_q
        loss_q = torch.max((q - 1) * errors, q * errors)
        loss += (loss_q * w).mean()
        
    return loss

# --- 4. Feature Block Generation ---

def compute_forecast_features(forecast_seq):
    """
    forecast_seq: (192, 3) numpy array [q10, q50, q90]
    Returns: 35-dim vector
    """
    q10 = forecast_seq[:, 0]
    q50 = forecast_seq[:, 1]
    q90 = forecast_seq[:, 2]
    
    # 1. Horizon Block (15 features)
    # Indices: 4, 16, 48, 96, 192 (0-indexed: 3, 15, 47, 95, 191)
    # Ensure we don't go out of bounds if forecast len < 192 (should not happen)
    indices = [3, 15, 47, 95, 191]
    horizon_feats = []
    for idx in indices:
        horizon_feats.extend([q10[idx], q50[idx], q90[idx]])
        
    # 2. Path Block (12 features)
    # 12 points over 48h (every 4h) from q50
    # 4h = 16 steps. Indices: 15, 31, 47, ..., 191
    path_indices = np.arange(15, 192, 16) # 16, 32... 192 (length 12)
    path_feats = q50[path_indices].tolist()
    
    # 3. Curve Stats (8 features) on q50
    # Prepend 0 to represent the start point (t=0, return=0) for accurate stats
    curve = np.concatenate(([0], q50))
    
    # min_ret, max_ret
    min_ret = np.min(curve)
    max_ret = np.max(curve)
    
    # time_to_min, time_to_max (normalized 0..1)
    # indices shifted by 1 due to prepended 0, but we divide by 192 total steps
    time_to_min = np.argmin(curve) / 192.0
    time_to_max = np.argmax(curve) / 192.0
    
    # max_drawdown, max_runup
    # Drawdown: max drop from a peak
    # Runup: max rise from a valley
    running_max = np.maximum.accumulate(curve)
    dd = running_max - curve
    max_drawdown = np.max(dd)
    
    running_min = np.minimum.accumulate(curve)
    ru = curve - running_min
    max_runup = np.max(ru)
    
    # Slopes
    # slope_0_12h (0 to step 48) -> Index 48 in curve (which is step 48)
    # slope_12_48h (step 48 to 192)
    
    # Index 48 is step 48 (12h)
    # Index 192 is step 192 (48h)
    slope_1 = curve[48] / 48.0
    slope_2 = (curve[192] - curve[48]) / (192.0 - 48.0)
    
    stats_feats = [min_ret, max_ret, time_to_min, time_to_max, max_drawdown, max_runup, slope_1, slope_2]
    
    # Concatenate
    all_feats = np.array(horizon_feats + path_feats + stats_feats, dtype=np.float32)
    return all_feats

# --- 5. Main Execution ---

def _ensure_datetime_index(df: pd.DataFrame, *, time_col_candidates: List[str]) -> pd.DataFrame:
    if isinstance(df.index, pd.DatetimeIndex):
        return df.sort_index()
    for c in time_col_candidates:
        if c in df.columns:
            out = df.copy()
            out.index = pd.to_datetime(out[c], unit="ms", utc=True)
            return out.sort_index()
    raise ValueError("DataFrame has no DatetimeIndex and no known time column.")


def load_data() -> pd.DataFrame:
    if not os.path.exists(DATA_FILE_FEATURES):
        raise FileNotFoundError(f"{DATA_FILE_FEATURES} not found. Run feature_engine.py build first.")
    if not os.path.exists(DATA_FILE_15M_RAW):
        raise FileNotFoundError(f"{DATA_FILE_15M_RAW} not found. Run build_dataset.py first.")

    feats = pd.read_parquet(DATA_FILE_FEATURES)
    raw_15m = pd.read_parquet(DATA_FILE_15M_RAW)

    feats = _ensure_datetime_index(feats, time_col_candidates=["open_time_ms"])
    raw_15m = _ensure_datetime_index(raw_15m, time_col_candidates=["open_time_ms"])

    missing = [c for c in FEATURE_COLUMNS if c not in feats.columns]
    if missing:
        raise ValueError(f"features.parquet missing columns: {missing}")
    if "close" not in raw_15m.columns:
        raise ValueError("aligned_15m.parquet missing required column: close")

    # Align close to feature index (scaled core features are the base)
    close = raw_15m["close"].reindex(feats.index)
    if close.isna().any():
        raise ValueError("Missing close prices after aligning aligned_15m to features index.")

    df = feats[FEATURE_COLUMNS].copy()
    df["close"] = close.values
    return df

def train_model(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    df = load_data()
    
    feature_cols = FEATURE_COLUMNS
    
    # Split
    train_size = int(len(df) * 0.8)
    train_df = df.iloc[:train_size]
    val_df = df.iloc[train_size:]
    
    train_ds = ForecastDataset(train_df, mode='train', feature_cols=feature_cols)
    val_ds = ForecastDataset(val_df, mode='val', feature_cols=feature_cols)
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    
    model = PatchTST().to(device)
    optimizer = AdamW(model.parameters(), lr=LR)
    scheduler = OneCycleLR(optimizer, max_lr=LR, steps_per_epoch=len(train_loader), epochs=EPOCHS)
    
    # Loss config
    horizon_indices = [3, 15, 47, 95, 191] # corresponding to 4, 16, 48, 96, 192 steps (0-indexed)
    weights = [HORIZON_WEIGHTS[h] for h in HORIZON_STEPS]
    
    best_val_loss = float('inf')
    
    logger.info("Starting training...")
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(x) # (B, 192, 3)
            loss = quantile_loss(pred, y, weights=weights, horizon_indices=horizon_indices)
            loss.backward()
            optimizer.step()
            scheduler.step()
            train_loss += loss.item()
            
        train_loss /= len(train_loader)
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x)
                loss = quantile_loss(pred, y, weights=weights, horizon_indices=horizon_indices)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        
        logger.info(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(MODEL_DIR, "forecast_model.pt"))
            logger.info("Saved best model.")

def precompute_features(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Starting Precompute...")
    
    df = load_data()
    ds = ForecastDataset(df, mode='inference', feature_cols=FEATURE_COLUMNS)
    loader = DataLoader(ds, batch_size=BATCH_SIZE*2, shuffle=False, num_workers=NUM_WORKERS)
    
    model = PatchTST().to(device)
    model_path = os.path.join(MODEL_DIR, "forecast_model.pt")
    if not os.path.exists(model_path):
        logger.warning(f"Model not found at {model_path}. Using randomly initialized weights.")
    else:
        model.load_state_dict(torch.load(model_path, map_location=device))
        
    model.eval()
    
    all_features = []
    
    # Pre-allocate array for efficiency? 
    # Length will be len(ds). 
    # We will use a list and concat later, safer.
    
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)
            preds = model(x) # (B, 192, 3)
            preds_np = preds.cpu().numpy()
            
            # Compute features for batch
            batch_feats = [compute_forecast_features(p) for p in preds_np]
            all_features.extend(batch_feats)
            
    # Pad the beginning (lookback period) with NaNs or zeros to match original df length
    # ForecastDataset only iterates over valid_indices (lookback to end).
    # So we have results for indices [lookback, len(df)].
    # The first 'lookback' rows have no forecast.
    
    if len(all_features) == 0:
        features_array = np.empty((0, 35), dtype=np.float32)
    else:
        features_array = np.array(all_features, dtype=np.float32)

    # Full-length output aligned to input index with NaN padding for initial lookback rows.
    full = np.full((len(df), 35), np.nan, dtype=np.float32)
    valid_positions = np.array(list(ds.valid_indices), dtype=int)
    if len(valid_positions) != len(features_array):
        raise RuntimeError("Forecast feature count mismatch with dataset indices.")
    full[valid_positions] = features_array

    feat_df = pd.DataFrame(full, index=df.index)
    feat_df.columns = [f"forecast_{i}" for i in range(35)]
    
    # Save
    feat_df.to_parquet(OUTPUT_FEATURES_PATH)
    logger.info(f"Saved forecast features to {OUTPUT_FEATURES_PATH}. Shape: {feat_df.shape}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['train', 'precompute'], help='Mode: train or precompute')
    args = parser.parse_args()
    
    if args.mode == 'train':
        train_model(args)
    elif args.mode == 'precompute':
        precompute_features(args)
