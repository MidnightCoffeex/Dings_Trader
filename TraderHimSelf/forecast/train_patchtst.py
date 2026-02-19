import os
import glob
import logging
import argparse
import pickle
import json
import re
import shutil
from datetime import datetime
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
DEFAULT_LOOKBACK = 512
DEFAULT_FORECAST_HORIZON = 192  # Max horizon (48h * 4 steps/h)
INPUT_CHANNELS = len(FEATURE_COLUMNS)
DEFAULT_QUANTILES = [0.2, 0.5, 0.8]  # v4 default: slightly tighter central interval than 0.1/0.9
DEFAULT_HORIZON_STEPS = [4, 16, 48, 96, 192]  # 1h, 4h, 12h, 24h, 48h
DEFAULT_HORIZON_WEIGHTS = [1.0, 1.0, 0.8, 0.6, 0.4]

# Runtime-mutable globals (set in apply_runtime_config)
LOOKBACK = DEFAULT_LOOKBACK
FORECAST_HORIZON = DEFAULT_FORECAST_HORIZON
QUANTILES = DEFAULT_QUANTILES.copy()
HORIZON_STEPS = DEFAULT_HORIZON_STEPS.copy()
HORIZON_WEIGHTS = {s: w for s, w in zip(DEFAULT_HORIZON_STEPS, DEFAULT_HORIZON_WEIGHTS)}

# Training Defaults (can be overridden by args)
DEFAULT_EPOCHS = 20
DEFAULT_LR = 1e-4

# Paths
BASE_DATA_DIR = os.path.join(BASE_DIR, "data_processed")
TRAIN30_DIR = os.path.join(BASE_DATA_DIR, "train30")  # separate training artifacts; live stays untouched
MODEL_DIR = os.path.join(BASE_DIR, "models")
PACKAGES_DIR = os.path.join(MODEL_DIR, "packages")

SCALER_PATH = os.path.join(TRAIN30_DIR, "scaler.pkl")
DATA_FILE_FEATURES = os.path.join(TRAIN30_DIR, "features.parquet")      # scaled core features (30D; FEATURE_COLUMNS)
DATA_FILE_15M_RAW = os.path.join(BASE_DATA_DIR, "aligned_15m.parquet")  # for raw 'close' prices
DATA_FILE_3M_RAW = os.path.join(BASE_DATA_DIR, "aligned_3m.parquet")    # for intrabar sim (PPO stage)
OUTPUT_FEATURES_PATH = os.path.join(TRAIN30_DIR, "forecast_features.parquet")
PIPELINE_JSON_DEFAULT = os.path.join(MODEL_DIR, "pipeline_args.json")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(TRAIN30_DIR, exist_ok=True)
os.makedirs(PACKAGES_DIR, exist_ok=True)


def _parse_csv_ints(raw: Optional[str]) -> Optional[List[int]]:
    if raw is None:
        return None
    vals = [x.strip() for x in str(raw).split(",") if x.strip()]
    if not vals:
        return None
    out = [int(v) for v in vals]
    if any(v <= 0 for v in out):
        raise ValueError("CSV ints must be > 0")
    return out


def _parse_csv_floats(raw: Optional[str]) -> Optional[List[float]]:
    if raw is None:
        return None
    vals = [x.strip() for x in str(raw).split(",") if x.strip()]
    if not vals:
        return None
    out = [float(v) for v in vals]
    if any(v <= 0 for v in out):
        raise ValueError("CSV floats must be > 0")
    return out


def _default_horizon_steps_for(horizon: int) -> List[int]:
    # Keep legacy anchor semantics (~2%, 8%, 25%, 50%, 100%).
    anchors = [0.0208, 0.0833, 0.25, 0.5, 1.0]
    steps = sorted({max(1, int(round(horizon * a))) for a in anchors})
    if steps[-1] != horizon:
        steps.append(horizon)
    return steps


def _slugify(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", s).strip("_")


def _normalize_tf(tf: str) -> str:
    return str(tf).strip().lower()


def _default_decision_file_for_tf(tf: str) -> str:
    return os.path.join(BASE_DATA_DIR, f"aligned_{_normalize_tf(tf)}.parquet")


def _default_features_file_for(feature_set: str, decision_tf: str) -> str:
    base = os.path.join(BASE_DATA_DIR, str(feature_set))
    tf_norm = _normalize_tf(decision_tf)
    if tf_norm == "15m":
        return os.path.join(base, "features.parquet")
    return os.path.join(base, f"features_{tf_norm}.parquet")


def _default_scaler_file_for(feature_set: str, decision_tf: str) -> str:
    base = os.path.join(BASE_DATA_DIR, str(feature_set))
    tf_norm = _normalize_tf(decision_tf)
    if tf_norm == "15m":
        return os.path.join(base, "scaler.pkl")
    return os.path.join(base, f"scaler_{tf_norm}.pkl")


def _load_pipeline_json(path: Optional[str]) -> dict:
    if not path:
        return {}
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"Failed to read pipeline json {path}: {e}")
        return {}


def _write_json(path: str, payload: dict):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def _resolve_runtime_config(args) -> dict:
    # For precompute, load prior config by default.
    existing = _load_pipeline_json(args.pipeline_json)
    existing_keys = set(existing.keys())

    cfg = {
        "symbol": "BTCUSDT",
        "decision_tf": "15m",
        "intrabar_tf": "3m",
        "feature_set": "train30",
        "lookback_steps": DEFAULT_LOOKBACK,
        "forecast_horizon_steps": DEFAULT_FORECAST_HORIZON,
        "forecast_quantiles": DEFAULT_QUANTILES.copy(),
        "horizon_steps": DEFAULT_HORIZON_STEPS.copy(),
        "horizon_weights": DEFAULT_HORIZON_WEIGHTS.copy(),
        "patch_epochs": DEFAULT_EPOCHS,
        "patch_learning_rate": DEFAULT_LR,
        "patch_batch_size": None,
        "patch_num_workers": None,
        "patch_pin_memory": None,
        "patch_prefetch_factor": None,
        "patch_persistent_workers": None,
        "patch_amp": None,
        "patch_compile": None,
        # default PPO knobs also live in same JSON handover
        "ppo_total_timesteps": 1_000_000,
        "ppo_learning_rate": 3e-4,
        "ppo_n_steps": 2048,
        "ppo_batch_size": 64,
        "ppo_n_epochs": 10,
        "ppo_gamma": 0.99,
        "ppo_gae_lambda": 0.95,
        "ppo_clip_range": 0.2,
        "ppo_ent_coef": 0.01,
        "model_tag": "default",
        "package_root": PACKAGES_DIR,
        "pipeline_json": args.pipeline_json or PIPELINE_JSON_DEFAULT,
        "data_file_features": DATA_FILE_FEATURES,
        "data_file_raw_decision": DATA_FILE_15M_RAW,
        "data_file_raw_intrabar": DATA_FILE_3M_RAW,
        "scaler_path": SCALER_PATH,
        "forecast_features_output": None,
        "forecast_features_compat_path": OUTPUT_FEATURES_PATH,
        "forecast_model_alias_path": os.path.join(MODEL_DIR, "forecast_model.pt"),
        "ppo_model_alias_path": os.path.join(MODEL_DIR, "ppo_policy_final.zip"),
    }

    # Load from existing JSON first.
    cfg.update(existing)

    # CLI overrides.
    if args.symbol is not None:
        cfg["symbol"] = args.symbol

    # convenience alias --candles for decision timeframe
    decision_tf_cli = args.decision_tf if args.decision_tf is not None else getattr(args, "candles", None)
    if decision_tf_cli is not None:
        cfg["decision_tf"] = _normalize_tf(decision_tf_cli)

    if args.intrabar_tf is not None:
        cfg["intrabar_tf"] = _normalize_tf(args.intrabar_tf)
    if args.lookback_steps is not None:
        cfg["lookback_steps"] = int(args.lookback_steps)
    if args.forecast_horizon_steps is not None:
        cfg["forecast_horizon_steps"] = int(args.forecast_horizon_steps)
    if args.epochs is not None:
        cfg["patch_epochs"] = int(args.epochs)
    if args.learning_rate is not None:
        cfg["patch_learning_rate"] = float(args.learning_rate)

    if args.batch_size is not None:
        cfg["patch_batch_size"] = int(args.batch_size)
    if args.num_workers is not None:
        cfg["patch_num_workers"] = int(args.num_workers)
    if args.pin_memory is not None:
        cfg["patch_pin_memory"] = bool(args.pin_memory)
    if args.prefetch_factor is not None:
        cfg["patch_prefetch_factor"] = int(args.prefetch_factor)
    if args.persistent_workers is not None:
        cfg["patch_persistent_workers"] = bool(args.persistent_workers)
    if args.amp is not None:
        cfg["patch_amp"] = bool(args.amp)
    if args.compile is not None:
        cfg["patch_compile"] = bool(args.compile)

    if args.model_tag is not None:
        cfg["model_tag"] = args.model_tag
    if args.package_root is not None:
        cfg["package_root"] = args.package_root
    if args.features_file is not None:
        cfg["data_file_features"] = args.features_file
    elif "data_file_features" not in existing_keys:
        cfg["data_file_features"] = _default_features_file_for(
            cfg.get("feature_set", "train30"), cfg["decision_tf"]
        )

    if args.raw_price_file is not None:
        cfg["data_file_raw_decision"] = args.raw_price_file
    elif "data_file_raw_decision" not in existing_keys:
        # derive default candle file binding from decision_tf (e.g. aligned_15m.parquet, aligned_1h.parquet)
        cfg["data_file_raw_decision"] = _default_decision_file_for_tf(cfg["decision_tf"])

    if getattr(args, "intrabar_candles_file", None) is not None:
        cfg["data_file_raw_intrabar"] = args.intrabar_candles_file

    if args.scaler_file is not None:
        cfg["scaler_path"] = args.scaler_file
    elif "scaler_path" not in existing_keys:
        cfg["scaler_path"] = _default_scaler_file_for(
            cfg.get("feature_set", "train30"), cfg["decision_tf"]
        )
    if args.forecast_features_output is not None:
        cfg["forecast_features_output"] = args.forecast_features_output
    if args.forecast_model_path is not None:
        cfg["forecast_model_path"] = args.forecast_model_path
    if args.package_id is not None:
        cfg["package_id"] = args.package_id

    # Horizon schedule overrides.
    custom_steps = _parse_csv_ints(args.horizon_steps)
    custom_weights = _parse_csv_floats(args.horizon_weights)
    custom_quantiles = _parse_csv_floats(getattr(args, "quantiles", None))

    if custom_steps is not None:
        cfg["horizon_steps"] = custom_steps
    elif "horizon_steps" not in existing:
        cfg["horizon_steps"] = _default_horizon_steps_for(int(cfg["forecast_horizon_steps"]))

    if custom_weights is not None:
        cfg["horizon_weights"] = custom_weights
    elif "horizon_weights" not in existing:
        n = len(cfg["horizon_steps"])
        if n == len(DEFAULT_HORIZON_WEIGHTS):
            cfg["horizon_weights"] = DEFAULT_HORIZON_WEIGHTS.copy()
        else:
            # Smooth fallback: earlier horizons weigh higher.
            cfg["horizon_weights"] = np.linspace(1.0, 0.5, n).round(4).tolist()

    if custom_quantiles is not None:
        cfg["forecast_quantiles"] = custom_quantiles
    elif "forecast_quantiles" not in existing:
        cfg["forecast_quantiles"] = DEFAULT_QUANTILES.copy()

    # Normalize / validate steps + weights + quantiles
    horizon = int(cfg["forecast_horizon_steps"])
    steps = [int(s) for s in cfg["horizon_steps"] if int(s) > 0]
    steps = sorted(set(min(horizon, s) for s in steps))
    if not steps:
        steps = _default_horizon_steps_for(horizon)
    weights = [float(w) for w in cfg["horizon_weights"]]
    if len(weights) != len(steps):
        raise ValueError(
            f"horizon_weights length ({len(weights)}) must match horizon_steps length ({len(steps)})."
        )

    quantiles = [float(q) for q in cfg.get("forecast_quantiles", DEFAULT_QUANTILES)]
    if len(quantiles) != 3:
        raise ValueError(f"forecast_quantiles must contain exactly 3 values, got {quantiles}")
    if any((q <= 0.0 or q >= 1.0) for q in quantiles):
        raise ValueError(f"forecast_quantiles must be in (0,1), got {quantiles}")
    if any(quantiles[i] >= quantiles[i + 1] for i in range(len(quantiles) - 1)):
        raise ValueError(f"forecast_quantiles must be strictly increasing, got {quantiles}")

    cfg["horizon_steps"] = steps
    cfg["horizon_weights"] = weights
    cfg["forecast_quantiles"] = quantiles

    cfg["decision_tf"] = _normalize_tf(cfg.get("decision_tf", "15m"))
    cfg["intrabar_tf"] = _normalize_tf(cfg.get("intrabar_tf", "3m"))

    # Derive package id/paths.
    package_id = cfg.get("package_id")
    if not package_id:
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S") if args.mode == "train" else "active"
        package_id = _slugify(
            f"{cfg['decision_tf']}_h{cfg['forecast_horizon_steps']}_{cfg['feature_set']}_{cfg['model_tag']}_{ts}"
        )
    cfg["package_id"] = package_id

    package_dir = cfg.get("package_dir") or os.path.join(cfg["package_root"], package_id)
    cfg["package_dir"] = package_dir
    cfg["manifest_path"] = os.path.join(package_dir, "manifest.json")

    if "forecast_model_path" not in cfg or not cfg["forecast_model_path"]:
        cfg["forecast_model_path"] = os.path.join(package_dir, "forecast_model.pt")

    if (
        ("forecast_features_output" not in existing_keys)
        and (args.forecast_features_output is None)
        and (not cfg.get("forecast_features_output"))
    ):
        cfg["forecast_features_output"] = os.path.join(package_dir, "forecast_features.parquet")

    if not cfg.get("forecast_features_output"):
        cfg["forecast_features_output"] = os.path.join(package_dir, "forecast_features.parquet")

    # Backward compatibility artifact (so existing PPO/inference paths still work for 15m/train30).
    if str(cfg.get("feature_set", "train30")) == "train30" and str(cfg.get("decision_tf")) == "15m":
        cfg["forecast_features_compat_path"] = OUTPUT_FEATURES_PATH
    else:
        cfg["forecast_features_compat_path"] = cfg["forecast_features_output"]

    # Manifest-level meta
    cfg["pipeline_version"] = "arg_interface_v1"
    cfg["updated_at"] = datetime.utcnow().isoformat() + "Z"

    return cfg


def _persist_pipeline_config(cfg: dict):
    # Ensure package dir contains key artifacts (e.g. scaler) for future deployment.
    try:
        package_dir = cfg.get("package_dir")
        scaler_src = cfg.get("scaler_path")
        if package_dir and scaler_src and os.path.exists(scaler_src):
            os.makedirs(package_dir, exist_ok=True)
            scaler_dst = os.path.join(package_dir, "scaler.pkl")
            if os.path.abspath(scaler_src) != os.path.abspath(scaler_dst):
                shutil.copy2(scaler_src, scaler_dst)
            cfg["scaler_packaged_path"] = scaler_dst
    except Exception as e:
        logger.warning(f"Failed to package scaler into model package: {e}")

    # Main handover JSON (for notebook / next steps)
    pipeline_json = cfg.get("pipeline_json") or PIPELINE_JSON_DEFAULT
    _write_json(pipeline_json, cfg)
    # Package-local manifest
    _write_json(cfg["manifest_path"], cfg)


def _apply_runtime_config(cfg: dict):
    global LOOKBACK, FORECAST_HORIZON, QUANTILES, HORIZON_STEPS, HORIZON_WEIGHTS
    LOOKBACK = int(cfg.get("lookback_steps", DEFAULT_LOOKBACK))
    FORECAST_HORIZON = int(cfg.get("forecast_horizon_steps", DEFAULT_FORECAST_HORIZON))
    QUANTILES = [float(q) for q in cfg.get("forecast_quantiles", DEFAULT_QUANTILES)]
    HORIZON_STEPS = [int(s) for s in cfg.get("horizon_steps", DEFAULT_HORIZON_STEPS)]
    HORIZON_WEIGHTS = {
        int(s): float(w)
        for s, w in zip(HORIZON_STEPS, cfg.get("horizon_weights", DEFAULT_HORIZON_WEIGHTS))
    }


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

def quantile_loss(preds, target, quantiles=None, weights=None, horizon_indices=None):
    loss = 0
    quantiles = QUANTILES if quantiles is None else quantiles

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

def compute_forecast_features(forecast_seq, horizon: Optional[int] = None):
    """
    Build a stable 35-dim feature vector from quantile forecast output.

    Supports variable forecast horizon and keeps output shape fixed:
      - Horizon block: 5 anchors * 3 quantiles = 15
      - Path block: 12 anchors on q50 = 12
      - Curve stats on q50 = 8
      -> total = 35
    """
    arr = np.asarray(forecast_seq, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] < 3:
        raise ValueError(f"forecast_seq must have shape (H,3), got {arr.shape}")

    h = int(horizon) if horizon is not None else int(arr.shape[0])
    if h <= 1:
        raise ValueError(f"forecast horizon must be > 1, got {h}")

    # Use exactly h rows (truncate if caller passed larger array by accident).
    # Columns correspond to low / median / high quantile heads configured by QUANTILES.
    arr = arr[:h, :3]
    q_low = arr[:, 0]
    q_mid = arr[:, 1]
    q_high = arr[:, 2]

    # 1) Horizon Block (15): ~2%, 8%, 25%, 50%, 100%
    horizon_fracs = [0.0208, 0.0833, 0.25, 0.5, 1.0]
    horizon_indices = [min(h - 1, max(0, int(round((h - 1) * f)))) for f in horizon_fracs]
    horizon_feats = []
    for idx in horizon_indices:
        horizon_feats.extend([q_low[idx], q_mid[idx], q_high[idx]])

    # 2) Path Block (12): median path sampled evenly from ~8%..100%
    start_idx = min(h - 1, max(0, int(round((h - 1) * 0.0833))))
    path_indices = np.linspace(start_idx, h - 1, 12).round().astype(int)
    path_feats = q_mid[path_indices].tolist()

    # 3) Curve Stats (8) on median path
    curve = np.concatenate(([0.0], q_mid.astype(np.float32)))  # include "now" baseline
    min_ret = float(np.min(curve))
    max_ret = float(np.max(curve))

    # Normalize timing by horizon length
    time_to_min = float(np.argmin(curve) / max(1, h))
    time_to_max = float(np.argmax(curve) / max(1, h))

    running_max = np.maximum.accumulate(curve)
    dd = running_max - curve
    max_drawdown = float(np.max(dd))

    running_min = np.minimum.accumulate(curve)
    ru = curve - running_min
    max_runup = float(np.max(ru))

    # Slope anchors: 25% and end
    idx_q1 = min(h, max(1, int(round(h * 0.25))))
    slope_1 = float(curve[idx_q1] / max(1, idx_q1))
    slope_2 = float((curve[h] - curve[idx_q1]) / max(1, h - idx_q1))

    stats_feats = [
        min_ret,
        max_ret,
        time_to_min,
        time_to_max,
        max_drawdown,
        max_runup,
        slope_1,
        slope_2,
    ]

    all_feats = np.array(horizon_feats + path_feats + stats_feats, dtype=np.float32)
    if all_feats.shape[0] != 35:
        raise RuntimeError(f"Expected 35 forecast features, got {all_feats.shape[0]}")
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


def load_data(runtime_cfg: Optional[dict] = None) -> pd.DataFrame:
    cfg = runtime_cfg or {}
    features_path = cfg.get("data_file_features", DATA_FILE_FEATURES)
    raw_decision_path = cfg.get("data_file_raw_decision", DATA_FILE_15M_RAW)

    if not os.path.exists(features_path):
        raise FileNotFoundError(
            f"{features_path} not found. Build matching feature artifacts first (e.g. feature_engine_train30.py build)."
        )
    if not os.path.exists(raw_decision_path):
        raise FileNotFoundError(
            f"{raw_decision_path} not found. Build/download matching candle dataset first."
        )

    feats = pd.read_parquet(features_path)
    raw_15m = pd.read_parquet(raw_decision_path)

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

    min_required = int(cfg.get("lookback_steps", LOOKBACK)) + int(cfg.get("forecast_horizon_steps", FORECAST_HORIZON)) + 1
    if len(df) < min_required:
        raise ValueError(
            f"Not enough clean rows after filtering ({len(df)} < {min_required}). "
            "Rebuild features or extend dataset window."
        )

    return df

def get_config(args, runtime_cfg: Optional[dict] = None):
    """Resolve DataLoader/runtime configuration from profile, JSON config, and explicit CLI args."""
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

    # JSON overrides (if present)
    rc = runtime_cfg or {}
    if rc.get('patch_batch_size') is not None:
        config['batch_size'] = int(rc['patch_batch_size'])
    if rc.get('patch_num_workers') is not None:
        config['num_workers'] = int(rc['patch_num_workers'])
    if rc.get('patch_pin_memory') is not None:
        config['pin_memory'] = bool(rc['patch_pin_memory'])
    if rc.get('patch_prefetch_factor') is not None:
        config['prefetch_factor'] = int(rc['patch_prefetch_factor'])
    if rc.get('patch_persistent_workers') is not None:
        config['persistent_workers'] = bool(rc['patch_persistent_workers'])
    if rc.get('patch_amp') is not None:
        config['amp'] = bool(rc['patch_amp'])
    if rc.get('patch_compile') is not None:
        config['compile'] = bool(rc['patch_compile'])

    # CLI overrides win
    if args.batch_size is not None:
        config['batch_size'] = args.batch_size
    if args.num_workers is not None:
        config['num_workers'] = args.num_workers
    if args.pin_memory is not None:
        config['pin_memory'] = args.pin_memory
    if args.prefetch_factor is not None:
        config['prefetch_factor'] = args.prefetch_factor
    if args.persistent_workers is not None:
        config['persistent_workers'] = args.persistent_workers
    if args.amp is not None:
        config['amp'] = args.amp
    if args.compile is not None:
        config['compile'] = args.compile

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


def train_model(args, runtime_cfg: dict):
    _apply_runtime_config(runtime_cfg)

    loader_cfg = get_config(args, runtime_cfg)
    logger.info(f"DataLoader/Runtime profile: {loader_cfg}")
    logger.info(
        "PatchTST config => tf=%s horizon=%s lookback=%s epochs=%s lr=%s quantiles=%s",
        runtime_cfg.get("decision_tf"),
        runtime_cfg.get("forecast_horizon_steps"),
        runtime_cfg.get("lookback_steps"),
        runtime_cfg.get("patch_epochs"),
        runtime_cfg.get("patch_learning_rate"),
        runtime_cfg.get("forecast_quantiles", DEFAULT_QUANTILES),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    if device.type == 'cuda' and loader_cfg['amp']:
        logger.info("AMP enabled.")

    if loader_cfg['compile'] and not hasattr(torch, 'compile'):
        logger.warning("torch.compile not supported in this torch version. Disabling compile.")
        loader_cfg['compile'] = False

    df = load_data(runtime_cfg)
    feature_cols = FEATURE_COLUMNS
    
    train_size = int(len(df) * 0.8)
    train_df = df.iloc[:train_size]
    val_df = df.iloc[train_size:]
    
    train_ds = ForecastDataset(
        train_df,
        lookback=LOOKBACK,
        forecast_horizon=FORECAST_HORIZON,
        mode='train',
        feature_cols=feature_cols,
    )
    val_ds = ForecastDataset(
        val_df,
        lookback=LOOKBACK,
        forecast_horizon=FORECAST_HORIZON,
        mode='val',
        feature_cols=feature_cols,
    )

    loader_kwargs = {
        'batch_size': loader_cfg['batch_size'],
        'num_workers': loader_cfg['num_workers'],
        'pin_memory': loader_cfg['pin_memory'],
    }
    if loader_cfg['num_workers'] > 0:
        loader_kwargs['prefetch_factor'] = loader_cfg['prefetch_factor']
        loader_kwargs['persistent_workers'] = loader_cfg['persistent_workers']

    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)

    model = PatchTST(
        input_dim=len(feature_cols),
        lookback=LOOKBACK,
        forecast_len=FORECAST_HORIZON,
    ).to(device)

    if loader_cfg['compile']:
        logger.info("Compiling model...")
        try:
            model = torch.compile(model)
        except Exception as e:
            logger.error(f"Compilation failed: {e}. Continuing without compilation.")

    lr = float(runtime_cfg.get("patch_learning_rate", DEFAULT_LR))
    epochs = int(runtime_cfg.get("patch_epochs", DEFAULT_EPOCHS))

    optimizer = AdamW(model.parameters(), lr=lr)
    scheduler = OneCycleLR(optimizer, max_lr=lr, steps_per_epoch=len(train_loader), epochs=epochs)
    scaler = _make_grad_scaler(device, loader_cfg['amp'])

    horizon_indices = [max(0, min(FORECAST_HORIZON - 1, int(h) - 1)) for h in HORIZON_STEPS]
    weights = [float(HORIZON_WEIGHTS[h]) for h in HORIZON_STEPS]

    best_val_loss = float('inf')

    batch_log_every = int(getattr(args, 'log_every_batches', 0) or 0)
    total_train_batches = len(train_loader)

    logger.info("Starting training...")
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_batches = 0

        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            if not torch.isfinite(x).all() or not torch.isfinite(y).all():
                raise ValueError(f"Non-finite values detected in training batch {batch_idx}. Check features.parquet.")

            optimizer.zero_grad(set_to_none=True)

            with _make_autocast(device, loader_cfg['amp']):
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
                        f"Epoch {epoch + 1}/{epochs} | "
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

                with _make_autocast(device, loader_cfg['amp']):
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

        logger.info(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model_path = runtime_cfg["forecast_model_path"]
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            torch.save(_state_dict_for_saving(model), model_path)
            logger.info(f"Saved best model: {model_path}")

    # Write/update handover JSON + package manifest and maintain alias path.
    runtime_cfg["forecast_model_path"] = runtime_cfg["forecast_model_path"]
    runtime_cfg["forecast_train_completed_at"] = datetime.utcnow().isoformat() + "Z"
    runtime_cfg["forecast_best_val_loss"] = float(best_val_loss)

    alias_path = runtime_cfg.get("forecast_model_alias_path")
    if alias_path and os.path.abspath(alias_path) != os.path.abspath(runtime_cfg["forecast_model_path"]):
        os.makedirs(os.path.dirname(alias_path), exist_ok=True)
        shutil.copy2(runtime_cfg["forecast_model_path"], alias_path)
        logger.info(f"Updated alias model path: {alias_path}")

    _persist_pipeline_config(runtime_cfg)

def precompute_features(args, runtime_cfg: dict):
    _apply_runtime_config(runtime_cfg)

    loader_cfg = get_config(args, runtime_cfg)
    # For inference, we can often use larger batches if no explicit override is provided.
    if args.batch_size is None:
        loader_cfg['batch_size'] = loader_cfg['batch_size'] * 2
    logger.info(f"Precompute DataLoader config: {loader_cfg}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Starting Precompute...")

    df = load_data(runtime_cfg)
    ds = ForecastDataset(
        df,
        lookback=LOOKBACK,
        forecast_horizon=FORECAST_HORIZON,
        mode='inference',
        feature_cols=FEATURE_COLUMNS,
    )

    loader_kwargs = {
        'batch_size': loader_cfg['batch_size'],
        'num_workers': loader_cfg['num_workers'],
        'pin_memory': loader_cfg['pin_memory'],
    }
    if loader_cfg['num_workers'] > 0:
        loader_kwargs['prefetch_factor'] = loader_cfg['prefetch_factor']
        loader_kwargs['persistent_workers'] = loader_cfg['persistent_workers']

    loader = DataLoader(ds, shuffle=False, **loader_kwargs)

    model = PatchTST(
        input_dim=len(FEATURE_COLUMNS),
        lookback=LOOKBACK,
        forecast_len=FORECAST_HORIZON,
    ).to(device)

    model_path = runtime_cfg.get("forecast_model_path", os.path.join(MODEL_DIR, "forecast_model.pt"))
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found at {model_path}. Run train mode successfully before precompute."
        )

    # Load weights before compile
    raw_state = torch.load(model_path, map_location=device)
    state_dict = _normalize_loaded_state_dict(raw_state)
    model.load_state_dict(state_dict)

    if loader_cfg['compile']:
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

            with _make_autocast(device, loader_cfg['amp']):
                preds = model(x)
            if not torch.isfinite(preds).all():
                raise FloatingPointError(f"Non-finite predictions during precompute at batch {batch_idx}.")

            preds_np = preds.float().cpu().numpy()  # Ensure float32 for CPU

            # Compute features for batch (CPU bound)
            batch_feats = [compute_forecast_features(p, horizon=FORECAST_HORIZON) for p in preds_np]
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

    out_path = runtime_cfg.get("forecast_features_output", OUTPUT_FEATURES_PATH)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    feat_df.to_parquet(out_path)
    logger.info(f"Saved forecast features to {out_path}. Shape: {feat_df.shape}")

    compat_path = runtime_cfg.get("forecast_features_compat_path")
    if compat_path and os.path.abspath(compat_path) != os.path.abspath(out_path):
        os.makedirs(os.path.dirname(compat_path), exist_ok=True)
        feat_df.to_parquet(compat_path)
        logger.info(f"Updated compatibility forecast features path: {compat_path}")

    runtime_cfg["forecast_features_output"] = out_path
    runtime_cfg["forecast_precompute_completed_at"] = datetime.utcnow().isoformat() + "Z"
    _persist_pipeline_config(runtime_cfg)

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

    # DataLoader/runtime overrides
    parser.add_argument('--batch-size', type=int, help='Batch size')
    parser.add_argument('--num-workers', type=int, help='Number of dataloader workers')
    parser.add_argument('--pin-memory', type=str2bool, help='Pin memory for DataLoader')
    parser.add_argument('--persistent-workers', type=str2bool, help='Persistent workers')
    parser.add_argument('--prefetch-factor', type=int, help='Prefetch factor')
    parser.add_argument('--amp', type=str2bool, help='Use Automatic Mixed Precision')
    parser.add_argument('--compile', type=str2bool, help='Use torch.compile')
    parser.add_argument('--log-every-batches', type=int, default=0, help='Log training/precompute batch stats every N batches (0=off)')

    # Arg-first pipeline interface
    parser.add_argument('--pipeline-json', type=str, default=PIPELINE_JSON_DEFAULT, help='Shared pipeline config/manifest JSON path')
    parser.add_argument('--symbol', type=str, default=None, help='Trading symbol (metadata + downstream handover)')
    parser.add_argument('--decision-tf', type=str, default=None, help='Decision candle timeframe (e.g. 15m)')
    parser.add_argument('--candles', type=str, default=None, help='Alias for --decision-tf (e.g. 15m, 1h)')
    parser.add_argument('--intrabar-tf', type=str, default=None, help='Intrabar candle timeframe for execution logic (e.g. 3m)')

    parser.add_argument('--lookback-steps', type=int, default=None, help='PatchTST input lookback window')
    parser.add_argument('--forecast-horizon-steps', type=int, default=None, help='PatchTST forecast horizon steps')
    parser.add_argument('--horizon-steps', type=str, default=None, help='Comma-separated weighted horizon anchors (step units, e.g. "4,16,48,96,192")')
    parser.add_argument('--horizon-weights', type=str, default=None, help='Comma-separated horizon weights (same length as --horizon-steps)')
    parser.add_argument('--quantiles', type=str, default=None, help='Comma-separated forecast quantiles (exactly 3, e.g. "0.2,0.5,0.8")')

    parser.add_argument('--epochs', type=int, default=None, help='PatchTST epochs')
    parser.add_argument('--learning-rate', type=float, default=None, help='PatchTST learning rate')

    parser.add_argument('--features-file', type=str, default=None, help='Path to scaled core feature parquet')
    parser.add_argument('--raw-price-file', '--decision-candles-file', dest='raw_price_file', type=str, default=None, help='Path to raw decision candle parquet (close/open_time_ms source)')
    parser.add_argument('--intrabar-candles-file', type=str, default=None, help='Path to intrabar candle parquet (aligned_3m.parquet)')
    parser.add_argument('--scaler-file', type=str, default=None, help='Path to scaler.pkl (metadata handover)')
    parser.add_argument('--forecast-features-output', type=str, default=None, help='Output path for precomputed forecast feature parquet')

    parser.add_argument('--model-tag', type=str, default=None, help='Freeform run tag used in package naming')
    parser.add_argument('--package-root', type=str, default=None, help='Root folder for package outputs')
    parser.add_argument('--package-id', type=str, default=None, help='Explicit package id (otherwise auto-generated)')
    parser.add_argument('--forecast-model-path', type=str, default=None, help='Path to save/load forecast model')

    args = parser.parse_args()

    runtime_cfg = _resolve_runtime_config(args)
    _apply_runtime_config(runtime_cfg)

    logger.info(f"Loaded runtime config from {runtime_cfg.get('pipeline_json')}")

    if args.mode == 'train':
        train_model(args, runtime_cfg)
    elif args.mode == 'precompute':
        precompute_features(args, runtime_cfg)
