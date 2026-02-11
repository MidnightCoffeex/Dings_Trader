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
from contextlib import nullcontext
from typing import List, Optional
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from feature_engine_train30 import FEATURE_COLUMNS_TRAIN30 as FEATURE_COLUMNS

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("Forecast-PatchTST")

# Constants (Model Architecture & Data)
LOOKBACK = 512
FORECAST_HORIZON = 192  # Max horizon (48h * 4 steps/h)
INPUT_CHANNELS = len(FEATURE_COLUMNS)
QUANTILES = [0.1, 0.5, 0.9]
HORIZON_STEPS = [4, 16, 48, 96, 192]  # 1h, 4h, 12h, 24h, 48h
HORIZON_WEIGHTS = {4: 1.0, 16: 1.0, 48: 0.8, 96: 0.6, 192: 0.4}

# Training Defaults (can be overridden by args)
DEFAULT_EPOCHS = 20
DEFAULT_LR = 1e-4

# Paths
BASE_DATA_DIR = os.path.join(BASE_DIR, "data_processed")
TRAIN30_DIR = os.path.join(BASE_DATA_DIR, "train30")  # separate training artifacts; live stays untouched
MODEL_DIR = os.path.join(BASE_DIR, "models")

SCALER_PATH = os.path.join(TRAIN30_DIR, "scaler.pkl")
DATA_FILE_FEATURES = os.path.join(TRAIN30_DIR, "features.parquet")      # scaled core features (30D; FEATURE_COLUMNS)
DATA_FILE_15M_RAW = os.path.join(BASE_DATA_DIR, "aligned_15m.parquet")  # for raw 'close' prices
OUTPUT_FEATURES_PATH = os.path.join(TRAIN30_DIR, "forecast_features.parquet")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(TRAIN30_DIR, exist_ok=True)

# --- 1. Dataset ---

class ForecastDataset(Dataset):
    def __init__(self, df, lookback=LOOKBACK, forecast_horizon=FORECAST_HORIZON, mode='train', feature_cols=None):
        """
        df: DataFrame indexed by UTC timestamps. Must contain:
            - core feature columns (scaled; see feature_engine_train30.FEATURE_COLUMNS_TRAIN30)
            - 'close' column (raw close prices)
        mode: 'train', 'val', 'test', 'inference'
        feature_cols: explicit list of the feature columns (recommended)
        """
        self.lookback = lookback
        self.forecast_horizon = forecast_horizon
        self.mode = mode
        
        # Ensure data is sorted
        self.data = df
        
        # Feature column selection
        if feature_cols is not None:
            self.feature_cols = list(feature_cols)
        else:
            # Fallback: assume the first INPUT_CHANNELS columns are features
            self.feature_cols = list(df.columns[:INPUT_CHANNELS])
        
        # We need 'close' for target calculation if not in features (it usually isn't raw close)
        if 'close' in df.columns:
            self.close_prices = df['close'].values
        else:
            self.close_prices = np.zeros(len(df))

        # Convert features to float32 numpy
        self.features = df[self.feature_cols].values.astype(np.float32)
        
        # Calculate valid indices (t = index of the *last* row included in the lookback window)
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
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        # x: (B, C, N_patches, patch_len)
        x = self.project(x) 
        # x: (B, C, N_patches, d_model)
        return x

class PatchTST(nn.Module):
    def __init__(self, input_dim=INPUT_CHANNELS, lookback=512, forecast_len=192, 
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
        
        # Embedding
        x = self.patch_embed(x)
        x = x + self.pos_embed
        
        B, C, N, D = x.shape
        # Reshape for Transformer: (B*C, N, D) - Channel Independence
        x = x.reshape(B*C, N, D)
        
        # Encoder
        x = self.encoder(x)
        
        # Reshape back
        x = x.reshape(B, C, N, D)
        
        # Head
        out = self.head(x)
        
        # Reshape
        out = out.reshape(B, self.forecast_len, 3)
        return out

# --- 3. Loss: Quantile / Pinball ---

def quantile_loss(preds, target, quantiles=QUANTILES, weights=None, horizon_indices=None):
    loss = 0
    
    # If we focus on specific horizons
    if horizon_indices is not None:
        preds = preds[:, horizon_indices, :]
        target = target[:, horizon_indices]
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
        raise FileNotFoundError(f"{DATA_FILE_FEATURES} not found. Run feature_engine_train30.py build first (train30 artifacts).")
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

    close = raw_15m["close"].reindex(feats.index)
    if close.isna().any():
        raise ValueError("Missing close prices after aligning aligned_15m to features index.")

    df = feats[FEATURE_COLUMNS].copy()
    df["close"] = close.values

    # Strict numeric sanity checks to avoid silent NaN training
    numeric_cols = FEATURE_COLUMNS + ["close"]
    invalid_mask = ~np.isfinite(df[numeric_cols].to_numpy(dtype=np.float64)).all(axis=1)
    dropped = int(invalid_mask.sum())
    if dropped > 0:
        logger.warning(f"Dropping {dropped} rows with NaN/Inf in features or close before training.")
        df = df.loc[~invalid_mask].copy()

    min_required = LOOKBACK + FORECAST_HORIZON + 1
    if len(df) < min_required:
        raise ValueError(
            f"Not enough clean rows after filtering ({len(df)} < {min_required}). "
            "Rebuild features or extend dataset window."
        )

    return df

def get_config(args):
    """Resolve configuration based on profile and explicit args."""
    config = {
        'batch_size': 32,
        'num_workers': 0,
        'pin_memory': False,
        'prefetch_factor': None,
        'persistent_workers': False,
        'amp': False,
        'compile': False
    }
    
    if args.profile == 'high-util':
        config.update({
            'batch_size': 256,
            'num_workers': 4,
            'pin_memory': True,
            'prefetch_factor': 2,
            'persistent_workers': True,
            'amp': True,
            'compile': True
        })
    
    # Override with explicit args if provided
    if args.batch_size is not None: config['batch_size'] = args.batch_size
    if args.num_workers is not None: config['num_workers'] = args.num_workers
    if args.pin_memory is not None: config['pin_memory'] = args.pin_memory
    if args.prefetch_factor is not None: config['prefetch_factor'] = args.prefetch_factor
    if args.persistent_workers is not None: config['persistent_workers'] = args.persistent_workers
    if args.amp is not None: config['amp'] = args.amp
    if args.compile is not None: config['compile'] = args.compile

    # Safety checks
    if config['num_workers'] == 0:
        config['prefetch_factor'] = None
        config['persistent_workers'] = False
        
    return config

def _make_autocast(device: torch.device, enabled: bool):
    if not enabled or device.type != 'cuda':
        return nullcontext()
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast(device_type='cuda', enabled=True)
    return torch.cuda.amp.autocast(enabled=True)


def _make_grad_scaler(device: torch.device, enabled: bool):
    if device.type != 'cuda' or not enabled:
        if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
            return torch.amp.GradScaler("cuda", enabled=False)
        return torch.cuda.amp.GradScaler(enabled=False)

    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda", enabled=True)
    return torch.cuda.amp.GradScaler(enabled=True)


def _state_dict_for_saving(model: nn.Module):
    base_model = getattr(model, "_orig_mod", model)
    return base_model.state_dict()


def _normalize_loaded_state_dict(state_dict: dict):
    if not isinstance(state_dict, dict):
        raise TypeError(f"Expected state_dict dict, got {type(state_dict)}")

    has_orig_mod_prefix = any(str(k).startswith("_orig_mod.") for k in state_dict.keys())
    if not has_orig_mod_prefix:
        return state_dict

    normalized = {}
    for k, v in state_dict.items():
        k = str(k)
        if k.startswith("_orig_mod."):
            k = k[len("_orig_mod."):]
        normalized[k] = v
    return normalized


def train_model(args):
    cfg = get_config(args)
    logger.info(f"Configuration: {cfg}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    if device.type == 'cuda' and cfg['amp']:
        logger.info("AMP enabled.")
        
    if cfg['compile'] and not hasattr(torch, 'compile'):
        logger.warning("torch.compile not supported in this torch version. Disabling compile.")
        cfg['compile'] = False

    df = load_data()
    feature_cols = FEATURE_COLUMNS
    
    train_size = int(len(df) * 0.8)
    train_df = df.iloc[:train_size]
    val_df = df.iloc[train_size:]
    
    train_ds = ForecastDataset(train_df, mode='train', feature_cols=feature_cols)
    val_ds = ForecastDataset(val_df, mode='val', feature_cols=feature_cols)
    
    loader_kwargs = {
        'batch_size': cfg['batch_size'],
        'num_workers': cfg['num_workers'],
        'pin_memory': cfg['pin_memory'],
    }
    if cfg['num_workers'] > 0:
        loader_kwargs['prefetch_factor'] = cfg['prefetch_factor']
        loader_kwargs['persistent_workers'] = cfg['persistent_workers']
    
    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)
    
    model = PatchTST().to(device)
    
    if cfg['compile']:
        logger.info("Compiling model...")
        try:
            model = torch.compile(model)
        except Exception as e:
            logger.error(f"Compilation failed: {e}. Continuing without compilation.")

    optimizer = AdamW(model.parameters(), lr=DEFAULT_LR)
    scheduler = OneCycleLR(optimizer, max_lr=DEFAULT_LR, steps_per_epoch=len(train_loader), epochs=DEFAULT_EPOCHS)
    scaler = _make_grad_scaler(device, cfg['amp'])

    horizon_indices = [3, 15, 47, 95, 191]
    weights = [HORIZON_WEIGHTS[h] for h in HORIZON_STEPS]

    best_val_loss = float('inf')

    batch_log_every = int(getattr(args, 'log_every_batches', 0) or 0)
    total_train_batches = len(train_loader)

    logger.info("Starting training...")
    for epoch in range(DEFAULT_EPOCHS):
        model.train()
        train_loss = 0.0
        train_batches = 0

        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            if not torch.isfinite(x).all() or not torch.isfinite(y).all():
                raise ValueError(f"Non-finite values detected in training batch {batch_idx}. Check features.parquet.")

            optimizer.zero_grad(set_to_none=True)

            with _make_autocast(device, cfg['amp']):
                pred = model(x)
                loss = quantile_loss(pred, y, weights=weights, horizon_indices=horizon_indices)

            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Non-finite training loss at epoch {epoch + 1}, batch {batch_idx}. "
                    "Try --amp false and/or smaller --batch-size."
                )

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            loss_value = float(loss.detach().item())
            train_loss += loss_value
            train_batches += 1

            if batch_log_every > 0:
                should_log = (
                    (batch_idx + 1) == 1
                    or ((batch_idx + 1) % batch_log_every == 0)
                    or ((batch_idx + 1) == total_train_batches)
                )
                if should_log:
                    current_lr = float(scheduler.get_last_lr()[0])
                    logger.info(
                        f"Batch {batch_idx + 1}/{total_train_batches} | "
                        f"Epoch {epoch + 1}/{DEFAULT_EPOCHS} | "
                        f"Loss: {loss_value:.6f} | LR: {current_lr:.2e}"
                    )

        if train_batches == 0:
            raise RuntimeError("No training batches produced. Dataset too short or batch size too large.")
        train_loss /= train_batches

        model.eval()
        val_loss = 0.0
        val_batches = 0
        with torch.no_grad():
            for batch_idx, (x, y) in enumerate(val_loader):
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                if not torch.isfinite(x).all() or not torch.isfinite(y).all():
                    raise ValueError(f"Non-finite values detected in validation batch {batch_idx}.")

                with _make_autocast(device, cfg['amp']):
                    pred = model(x)
                    loss = quantile_loss(pred, y, weights=weights, horizon_indices=horizon_indices)

                if not torch.isfinite(loss):
                    raise FloatingPointError(
                        f"Non-finite validation loss at epoch {epoch + 1}, batch {batch_idx}."
                    )

                val_loss += float(loss.detach().item())
                val_batches += 1

        if val_batches == 0:
            raise RuntimeError("No validation batches produced. Dataset split invalid.")
        val_loss /= val_batches

        logger.info(f"Epoch {epoch+1}/{DEFAULT_EPOCHS} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(_state_dict_for_saving(model), os.path.join(MODEL_DIR, "forecast_model.pt"))
            logger.info("Saved best model.")

def precompute_features(args):
    cfg = get_config(args)
    # For inference, we can often use larger batches if no explicit override is provided.
    if args.batch_size is None:
        cfg['batch_size'] = cfg['batch_size'] * 2
    logger.info(f"Precompute Configuration: {cfg}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Starting Precompute...")
    
    df = load_data()
    ds = ForecastDataset(df, mode='inference', feature_cols=FEATURE_COLUMNS)
    
    loader_kwargs = {
        'batch_size': cfg['batch_size'],
        'num_workers': cfg['num_workers'],
        'pin_memory': cfg['pin_memory'],
    }
    if cfg['num_workers'] > 0:
        loader_kwargs['prefetch_factor'] = cfg['prefetch_factor']
        loader_kwargs['persistent_workers'] = cfg['persistent_workers']

    loader = DataLoader(ds, shuffle=False, **loader_kwargs)
    
    model = PatchTST().to(device)
    model_path = os.path.join(MODEL_DIR, "forecast_model.pt")
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found at {model_path}. Run train mode successfully before precompute."
        )

    # Load weights before compile
    raw_state = torch.load(model_path, map_location=device)
    state_dict = _normalize_loaded_state_dict(raw_state)
    model.load_state_dict(state_dict)

    if cfg['compile']:
        model = torch.compile(model)

    model.eval()

    batch_log_every = int(getattr(args, 'log_every_batches', 0) or 0)
    total_precompute_batches = len(loader)
    if total_precompute_batches == 0:
        raise RuntimeError("No precompute batches produced. Dataset too short or batch size too large.")

    all_features = []

    with torch.no_grad():
        for batch_idx, (x, _) in enumerate(loader):
            x = x.to(device, non_blocking=True)
            if not torch.isfinite(x).all():
                raise ValueError(f"Non-finite inputs during precompute at batch {batch_idx}.")

            with _make_autocast(device, cfg['amp']):
                preds = model(x)
            if not torch.isfinite(preds).all():
                raise FloatingPointError(f"Non-finite predictions during precompute at batch {batch_idx}.")

            preds_np = preds.float().cpu().numpy()  # Ensure float32 for CPU

            # Compute features for batch (CPU bound)
            batch_feats = [compute_forecast_features(p) for p in preds_np]
            all_features.extend(batch_feats)

            if batch_log_every > 0:
                should_log = (
                    (batch_idx + 1) == 1
                    or ((batch_idx + 1) % batch_log_every == 0)
                    or ((batch_idx + 1) == total_precompute_batches)
                )
                if should_log:
                    logger.info(
                        f"Precompute Batch {batch_idx + 1}/{total_precompute_batches} | "
                        f"Generated feature rows: {len(all_features)}"
                    )
            
    if len(all_features) == 0:
        features_array = np.empty((0, 35), dtype=np.float32)
    else:
        features_array = np.array(all_features, dtype=np.float32)

    full = np.full((len(df), 35), np.nan, dtype=np.float32)
    valid_positions = np.array(list(ds.valid_indices), dtype=int)
    if len(valid_positions) != len(features_array):
        raise RuntimeError("Forecast feature count mismatch with dataset indices.")
    full[valid_positions] = features_array

    feat_df = pd.DataFrame(full, index=df.index)
    feat_df.columns = [f"forecast_{i}" for i in range(35)]
    
    feat_df.to_parquet(OUTPUT_FEATURES_PATH)
    logger.info(f"Saved forecast features to {OUTPUT_FEATURES_PATH}. Shape: {feat_df.shape}")

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['train', 'precompute'], help='Mode: train or precompute')
    
    # Profile arg
    parser.add_argument('--profile', choices=['default', 'high-util'], default='default', help='Hardware profile')
    
    # Overrides
    parser.add_argument('--batch-size', type=int, help='Batch size')
    parser.add_argument('--num-workers', type=int, help='Number of dataloader workers')
    parser.add_argument('--pin-memory', type=str2bool, help='Pin memory for DataLoader')
    parser.add_argument('--persistent-workers', type=str2bool, help='Persistent workers')
    parser.add_argument('--prefetch-factor', type=int, help='Prefetch factor')
    parser.add_argument('--amp', type=str2bool, help='Use Automatic Mixed Precision')
    parser.add_argument('--compile', type=str2bool, help='Use torch.compile')
    parser.add_argument('--log-every-batches', type=int, default=0, help='Log training batch stats every N batches (0=off)')

    args = parser.parse_args()
    
    if args.mode == 'train':
        train_model(args)
    elif args.mode == 'precompute':
        precompute_features(args)
