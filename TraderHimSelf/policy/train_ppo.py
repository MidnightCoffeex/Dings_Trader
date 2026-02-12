"""
TraderHimSelf/policy/train_ppo.py

Implementiert Roadmap Schritt 9 — PPO Policy Training.
Trainiert einen PPO Agenten auf dem PerpEnv.

Funktionen:
1. Lädt Daten (15m Candles, 3m Candles, Core Features, Forecast Features).
2. Bereitet das Environment vor (Integration von Features in Observation).
3. Trainiert Stable-Baselines3 PPO.
4. Speichert das Modell.

Anforderungen:
- Observation Space (dim = CORE_DIM + 35 + 9): Core (FEATURE_COLUMNS) + Forecast (35) + Account (9).
- Action Space: Direction, Size, Leverage, SL, TP.
- Reward: Delta Equity - penalties.

Nutzung:
    python train_ppo.py --profile high-util
"""

import os
import sys
import argparse
import json
import re
import shutil
from datetime import datetime
import numpy as np
import pandas as pd
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from typing import List, Optional

# Add project root to path to allow imports
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from env.perp_env import PerpEnv
from gymnasium import spaces
from feature_engine_train30 import FEATURE_COLUMNS_TRAIN30 as CORE_FEATURE_COLUMNS

# --- CONFIG ---
BASE_DATA_DIR = os.path.join(project_root, "data_processed")
TRAIN30_DIR = os.path.join(BASE_DATA_DIR, "train30")  # separate training artifacts; live stays untouched
MODELS_DIR = os.path.join(project_root, "models")
PACKAGES_DIR = os.path.join(MODELS_DIR, "packages")
PIPELINE_JSON_DEFAULT = os.path.join(MODELS_DIR, "pipeline_args.json")
LOG_DIR = os.path.join(project_root, "runs/ppo_logs")
CHECKPOINT_DIR = os.path.join(project_root, "checkpoints/ppo")

FORECAST_DIM = 35
ACCOUNT_DIM = 9
CORE_DIM = len(CORE_FEATURE_COLUMNS)
OS_OBS_DIM = CORE_DIM + FORECAST_DIM + ACCOUNT_DIM

# --- FORECAST FEATURE COLUMNS ---
FORECAST_COLUMNS = [f"forecast_{i}" for i in range(FORECAST_DIM)]


def _slugify(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", str(s)).strip("_")


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


def _load_pipeline_json(path: Optional[str]) -> dict:
    if not path:
        return {}
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"WARN: failed to read pipeline json {path}: {e}")
        return {}


def _write_json(path: str, payload: dict):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def _resolve_runtime_config(args) -> dict:
    existing = _load_pipeline_json(args.pipeline_json)
    existing_keys = set(existing.keys())

    cfg = {
        "pipeline_version": "arg_interface_v1",
        "pipeline_json": args.pipeline_json or PIPELINE_JSON_DEFAULT,
        "symbol": "BTCUSDT",
        "decision_tf": "15m",
        "intrabar_tf": "3m",
        "feature_set": "train30",
        "data_file_raw_decision": os.path.join(BASE_DATA_DIR, "aligned_15m.parquet"),
        "data_file_raw_intrabar": os.path.join(BASE_DATA_DIR, "aligned_3m.parquet"),
        "data_file_features": os.path.join(TRAIN30_DIR, "features.parquet"),
        "forecast_features_output": os.path.join(TRAIN30_DIR, "forecast_features.parquet"),
        "package_root": PACKAGES_DIR,
        "package_id": None,
        "package_dir": None,
        "manifest_path": None,
        "ppo_model_alias_path": os.path.join(MODELS_DIR, "ppo_policy_final.zip"),
        "ppo_model_base": None,
        "ppo_model_path": None,
        # PPO hyperparams
        "ppo_total_timesteps": 1_000_000,
        "ppo_learning_rate": 3e-4,
        "ppo_n_steps": 2048,
        "ppo_batch_size": 64,
        "ppo_n_epochs": 10,
        "ppo_gamma": 0.99,
        "ppo_gae_lambda": 0.95,
        "ppo_clip_range": 0.2,
        "ppo_ent_coef": 0.01,
        "ppo_n_envs": 1,
        "ppo_vec_env": "auto",
        "ppo_device": "auto",
    }

    # Keep any forecast-stage metadata present.
    cfg.update(existing)

    # --- CLI overrides (metadata + paths) ---
    if getattr(args, "symbol", None) is not None:
        cfg["symbol"] = args.symbol

    decision_tf_cli = args.decision_tf if getattr(args, "decision_tf", None) is not None else getattr(args, "candles", None)
    if decision_tf_cli is not None:
        cfg["decision_tf"] = _normalize_tf(decision_tf_cli)

    if getattr(args, "intrabar_tf", None) is not None:
        cfg["intrabar_tf"] = _normalize_tf(args.intrabar_tf)

    if getattr(args, "decision_candles_file", None) is not None:
        cfg["data_file_raw_decision"] = args.decision_candles_file
    elif ("data_file_raw_decision" not in existing_keys):
        # TF-based default file binding (e.g. aligned_15m.parquet, aligned_1h.parquet)
        cfg["data_file_raw_decision"] = _default_decision_file_for_tf(cfg["decision_tf"])

    if getattr(args, "intrabar_candles_file", None) is not None:
        cfg["data_file_raw_intrabar"] = args.intrabar_candles_file

    if getattr(args, "features_file", None) is not None:
        cfg["data_file_features"] = args.features_file
    elif ("data_file_features" not in existing_keys):
        cfg["data_file_features"] = _default_features_file_for(
            cfg.get("feature_set", "train30"), cfg["decision_tf"]
        )

    if getattr(args, "forecast_features_file", None) is not None:
        cfg["forecast_features_output"] = args.forecast_features_file

    if getattr(args, "package_root", None) is not None:
        cfg["package_root"] = args.package_root
    if getattr(args, "package_id", None) is not None:
        cfg["package_id"] = args.package_id

    # Apply profile defaults only if not explicitly set via JSON and not overridden via CLI.
    if args.profile == "high-util":
        if ("ppo_n_steps" not in existing_keys) and (args.n_steps is None):
            cfg["ppo_n_steps"] = 8192
        if ("ppo_batch_size" not in existing_keys) and (args.batch_size is None):
            cfg["ppo_batch_size"] = 512

    # --- CLI overrides (PPO hyperparams) ---
    if getattr(args, "total_timesteps", None) is not None:
        cfg["ppo_total_timesteps"] = int(args.total_timesteps)
    if getattr(args, "learning_rate", None) is not None:
        cfg["ppo_learning_rate"] = float(args.learning_rate)
    if getattr(args, "n_steps", None) is not None:
        cfg["ppo_n_steps"] = int(args.n_steps)
    if getattr(args, "batch_size", None) is not None:
        cfg["ppo_batch_size"] = int(args.batch_size)
    if getattr(args, "n_epochs", None) is not None:
        cfg["ppo_n_epochs"] = int(args.n_epochs)
    if getattr(args, "gamma", None) is not None:
        cfg["ppo_gamma"] = float(args.gamma)
    if getattr(args, "gae_lambda", None) is not None:
        cfg["ppo_gae_lambda"] = float(args.gae_lambda)
    if getattr(args, "clip_range", None) is not None:
        cfg["ppo_clip_range"] = float(args.clip_range)
    if getattr(args, "ent_coef", None) is not None:
        cfg["ppo_ent_coef"] = float(args.ent_coef)

    if getattr(args, "n_envs", None) is not None:
        cfg["ppo_n_envs"] = int(args.n_envs)
    if getattr(args, "vec_env", None) is not None:
        cfg["ppo_vec_env"] = str(args.vec_env)
    if getattr(args, "device", None) is not None:
        cfg["ppo_device"] = str(args.device)

    cfg["decision_tf"] = _normalize_tf(cfg.get("decision_tf", "15m"))
    cfg["intrabar_tf"] = _normalize_tf(cfg.get("intrabar_tf", "3m"))

    # Package inference: prefer package_id from forecast stage, else create.
    if not cfg.get("package_id"):
        horizon = cfg.get("forecast_horizon_steps") or cfg.get("forecast_horizon") or "?"
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        cfg["package_id"] = _slugify(
            f"{cfg.get('decision_tf','15m')}_h{horizon}_{cfg.get('feature_set','train30')}_ppo_{ts}"
        )

    if not cfg.get("package_dir"):
        cfg["package_dir"] = os.path.join(cfg.get("package_root", PACKAGES_DIR), cfg["package_id"])

    if not cfg.get("manifest_path"):
        cfg["manifest_path"] = os.path.join(cfg["package_dir"], "manifest.json")

    if not cfg.get("ppo_model_base"):
        cfg["ppo_model_base"] = os.path.join(cfg["package_dir"], "ppo_policy_final")

    if not cfg.get("ppo_model_path"):
        cfg["ppo_model_path"] = cfg["ppo_model_base"] + ".zip"

    cfg["updated_at"] = datetime.utcnow().isoformat() + "Z"
    return cfg


def _persist_pipeline_config(cfg: dict):
    pipeline_json = cfg.get("pipeline_json") or PIPELINE_JSON_DEFAULT
    _write_json(pipeline_json, cfg)
    manifest_path = cfg.get("manifest_path")
    if manifest_path:
        _write_json(manifest_path, cfg)


class TrainingPerpEnv(PerpEnv):
    """
    Wrapper / Subclass von PerpEnv für das Training.
    Implementiert _get_obs korrekt, indem es die echten Features aus dem DataFrame liest.
    """
    def __init__(self, df_15m: pd.DataFrame, df_3m: pd.DataFrame, feature_cols: List[str]):
        super().__init__(df_15m, df_3m)
        self.feature_cols = feature_cols
        
        # Validierung
        assert len(self.feature_cols) == (CORE_DIM + FORECAST_DIM), \
            f"Expected {CORE_DIM + FORECAST_DIM} feature columns, got {len(self.feature_cols)}"
            
        # Override observation space to match our training feature-set (may differ from live env constants).
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(OS_OBS_DIM,),
            dtype=np.float32,
        )

    def _get_obs(self):
        """
        Holt den Observation Vector für den aktuellen Step.
        Aufbau: [Core Features (CORE_DIM) | Forecast Features (35) | Account State (9)]
        """
        # 1. ML Features aus DataFrame holen
        if not hasattr(self, 'features_np'):
             self.features_np = self.df_15m[self.feature_cols].values.astype(np.float32)
        
        idx = self.current_step
        if idx >= len(self.features_np):
            ml_features = np.zeros(CORE_DIM + FORECAST_DIM, dtype=np.float32)
        else:
            ml_features = self.features_np[idx]
            
        # 2. Account State (Logic copied from PerpEnv to ensure self-contained consistency)
        pos_count = len(self.open_positions)
        major_side = 0
        if pos_count > 0:
            major_side = 1 if self.open_positions[0].side == 'long' else 2
            
        total_margin = sum(p.margin_used for p in self.open_positions)
        total_notional = sum(p.notional for p in self.open_positions)
        total_upnl = sum(p.current_pnl for p in self.open_positions)
        
        equity = self.equity if self.equity > 0 else 1.0
        
        exp_open_pct = total_margin / equity
        not_open_pct = total_notional / equity
        upnl_open_pct = total_upnl / equity
        
        time_in_trade_max = 0
        if pos_count > 0:
            time_in_trade_max = max(p.time_in_trade_steps_15m for p in self.open_positions)
            
        time_left_min = 192 - time_in_trade_max
        liq_buffer_min = 1.0 
        avail_exp_pct = (self.equity * self.max_exposure_pct - total_margin) / equity
        
        account_vec = np.array([
            pos_count, major_side, exp_open_pct, not_open_pct, upnl_open_pct,
            time_in_trade_max, time_left_min, liq_buffer_min, avail_exp_pct
        ], dtype=np.float32)

        obs = np.concatenate([ml_features, account_vec]).astype(np.float32)
        if not np.isfinite(obs).all():
            raise ValueError(f"Non-finite observation at step={idx}. Check feature/forecast preprocessing.")
        return obs


def _rename_fc_feat_columns(df: pd.DataFrame) -> pd.DataFrame:
    fc_cols = [c for c in df.columns if c.startswith("fc_feat_")]
    if not fc_cols:
        return df
    renamed = {c: c.replace("fc_feat_", "forecast_") for c in fc_cols}
    return df.rename(columns=renamed)


def _slot15m_to_ms(series: pd.Series) -> pd.Series:
    """Normalize slot_15m values to epoch milliseconds.

    Handles datetime columns that may be stored at different resolutions (ns/us/ms)
    after Parquet round-trips.
    """
    s = series.copy()

    if pd.api.types.is_datetime64_any_dtype(s):
        arr_ns = pd.to_datetime(s, utc=True, errors="coerce").to_numpy(dtype="datetime64[ns]")
        ms = arr_ns.view("int64") // 10**6
        # NaT becomes int64 min; convert to NaN so downstream filters can drop it.
        ms = ms.astype("float64")
        ms[ms <= -9e18] = np.nan
        return pd.Series(ms, index=s.index, name=s.name)

    if pd.api.types.is_numeric_dtype(s):
        s_num = pd.to_numeric(s, errors="coerce")
        med = float(s_num.dropna().median()) if s_num.dropna().size else 0.0
        # Heuristics:
        # - ns epoch ~1e18
        # - us epoch ~1e15
        # - ms epoch ~1e12
        # - s  epoch ~1e9
        if med > 10**17:
            s_num = s_num // 10**6
        elif med > 10**14:
            s_num = s_num // 10**3
        elif med > 10**11:
            s_num = s_num
        elif med > 10**9:
            s_num = s_num * 1000
        return s_num

    # Fallback: parse strings/objects as datetime
    s_dt = pd.to_datetime(s, utc=True, errors="coerce")
    if s_dt.isna().any():
        bad = int(s_dt.isna().sum())
        raise ValueError(f"slot_15m contains {bad} unparsable values")
    arr_ns = s_dt.to_numpy(dtype="datetime64[ns]")
    ms = arr_ns.view("int64") // 10**6
    return pd.Series(ms.astype("int64"), index=s.index, name=s.name)


def load_data(*, allow_dummy_forecast: bool = False, runtime_cfg: Optional[dict] = None):
    """Lädt und mergt alle benötigten Daten."""
    cfg = runtime_cfg or {}
    print("Lade Daten...")

    # 1. Candles (decision tf, currently expected 15m semantics)
    p_15m = cfg.get("data_file_raw_decision") or os.path.join(BASE_DATA_DIR, "aligned_15m.parquet")
    if not os.path.exists(p_15m):
        raise FileNotFoundError(f"{p_15m} nicht gefunden. Bitte erst build_dataset.py ausführen.")
    df_15m = pd.read_parquet(p_15m)

    # Ensure open_time_ms is truly epoch-ms (robust against Parquet timestamp resolution).
    if isinstance(df_15m.index, pd.DatetimeIndex):
        idx_ns = df_15m.index.to_numpy(dtype="datetime64[ns]")
        df_15m["open_time_ms"] = (idx_ns.view("int64") // 10**6).astype("int64")

    
    # 2. Candles (intrabar tf, currently expected 3m semantics + slot_15m column)
    p_3m = cfg.get("data_file_raw_intrabar") or os.path.join(BASE_DATA_DIR, "aligned_3m.parquet")
    if not os.path.exists(p_3m):
        raise FileNotFoundError(f"{p_3m} nicht gefunden.")
    df_3m = pd.read_parquet(p_3m)

    # 3. Core Features
    p_feat = cfg.get("data_file_features") or os.path.join(TRAIN30_DIR, "features.parquet")
    if not os.path.exists(p_feat):
        raise FileNotFoundError(
            f"{p_feat} nicht gefunden. Bitte feature_engine_train30.py build ausführen (train30 artifacts)."
        )
    df_feat = pd.read_parquet(p_feat)
    missing_core = [c for c in CORE_FEATURE_COLUMNS if c not in df_feat.columns]
    if missing_core:
        raise ValueError(f"features.parquet missing core columns: {missing_core}")
    
    # 4. Forecast Features (default: uses output path from pipeline JSON)
    p_forecast = cfg.get("forecast_features_output") or os.path.join(TRAIN30_DIR, "forecast_features.parquet")
    if not os.path.exists(p_forecast):
        if allow_dummy_forecast:
            print(f"WARNUNG: {p_forecast} nicht gefunden. Dummy-Forecast-Features werden erstellt.")
            df_forecast = pd.DataFrame(0.0, index=df_feat.index, columns=FORECAST_COLUMNS)
        else:
            raise FileNotFoundError(
                f"{p_forecast} nicht gefunden. Bitte train_patchtst.py precompute ausführen "
                "oder --allow-dummy-forecast verwenden."
            )
    else:
        df_forecast = pd.read_parquet(p_forecast)
        df_forecast = _rename_fc_feat_columns(df_forecast)
        missing_fc = [c for c in FORECAST_COLUMNS if c not in df_forecast.columns]
        if missing_fc:
            if allow_dummy_forecast:
                print(f"WARNUNG: Forecast-Spalten fehlen: {missing_fc[:5]}... Dummy ergänzt.")
                for c in missing_fc:
                    df_forecast[c] = 0.0
            else:
                raise ValueError(f"Missing forecast columns: {missing_fc}")
        
    print("Merge DataFrames...")
    common_idx = df_15m.index.intersection(df_feat.index).intersection(df_forecast.index)
    if len(common_idx) < 1000:
        print(f"Warnung: Sehr wenig gemeinsame Datenpunkte ({len(common_idx)}). Check Alignment.")
        
    df_15m = df_15m.loc[common_idx].copy()
    df_feat = df_feat.loc[common_idx].copy()
    df_forecast = df_forecast.loc[common_idx].copy()
    
    for col in CORE_FEATURE_COLUMNS:
        df_15m[col] = df_feat[col]
        
    for col in FORECAST_COLUMNS:
        df_15m[col] = df_forecast[col]
            
    if 'atr_14' in df_15m.columns:
        df_15m['atr'] = df_15m['atr_14']
    else:
        df_15m['atr'] = df_15m['close'] * 0.01

    # Strict row cleanup for PPO: no NaN/Inf in observation-driving columns.
    required_cols = [
        'open_time_ms', 'open', 'high', 'low', 'close', 'atr'
    ] + CORE_FEATURE_COLUMNS + FORECAST_COLUMNS

    missing_req = [c for c in required_cols if c not in df_15m.columns]
    if missing_req:
        raise ValueError(f"df_15m missing required columns for PPO: {missing_req[:10]}")

    req_num = df_15m[required_cols].apply(pd.to_numeric, errors='coerce')
    valid_mask = np.isfinite(req_num.to_numpy(dtype=np.float64)).all(axis=1)
    dropped_rows = int((~valid_mask).sum())
    if dropped_rows > 0:
        print(f"WARN: dropping {dropped_rows} 15m rows with NaN/Inf before PPO training")
        df_15m = df_15m.loc[valid_mask].copy()

    if len(df_15m) < 5000:
        raise ValueError(
            f"Too few clean 15m rows for PPO ({len(df_15m)}). "
            "Rebuild dataset/features/forecast and ensure full history is present."
        )

    if "slot_15m" not in df_3m.columns:
        raise ValueError("aligned_3m.parquet missing required column: slot_15m")
    if "open_time_ms" not in df_15m.columns:
        raise ValueError("aligned_15m.parquet missing required column: open_time_ms")

    # Normalize slot_15m to ms so it matches 15m open_time_ms domain.
    df_3m = df_3m.copy()
    df_3m["slot_15m"] = _slot15m_to_ms(df_3m["slot_15m"])

    # Remove rows with invalid slot values before matching.
    df_3m = df_3m[np.isfinite(pd.to_numeric(df_3m["slot_15m"], errors='coerce'))].copy()

    valid_slots = set(pd.to_numeric(df_15m["open_time_ms"], errors="coerce").astype("int64").values)
    df_3m = df_3m[df_3m["slot_15m"].isin(valid_slots)].copy()

    if len(df_3m) == 0:
        raise ValueError(
            "No aligned 3m rows after slot matching. "
            "Likely slot_15m/open_time_ms mismatch; rerun build_dataset.py and verify slot units."
        )

    print(f"Daten geladen: {len(df_15m)} 15m Steps, {len(df_3m)} 3m Steps.")
    return df_15m, df_3m


def make_env(df_15m, df_3m):
    feature_cols = CORE_FEATURE_COLUMNS + FORECAST_COLUMNS
    missing = [c for c in feature_cols if c not in df_15m.columns]
    if missing:
        raise ValueError(f"Missing columns in df_15m: {missing[:5]}...")
    return TrainingPerpEnv(df_15m, df_3m, feature_cols)


def make_env_fn(df_15m, df_3m):
    """Factory for VecEnv workers."""
    return lambda: make_env(df_15m, df_3m)


def build_vec_env(df_15m, df_3m, *, n_envs: int, vec_env: str):
    if n_envs <= 1:
        return DummyVecEnv([make_env_fn(df_15m, df_3m)]), "dummy"

    mode = vec_env
    if mode == "auto":
        mode = "subproc"

    if mode == "subproc":
        try:
            env = SubprocVecEnv([make_env_fn(df_15m, df_3m) for _ in range(n_envs)], start_method="fork")
        except TypeError:
            env = SubprocVecEnv([make_env_fn(df_15m, df_3m) for _ in range(n_envs)])
        return env, "subproc"

    if mode == "dummy":
        env = DummyVecEnv([make_env_fn(df_15m, df_3m) for _ in range(n_envs)])
        return env, "dummy"

    raise ValueError(f"Unknown vec env mode: {vec_env}")

# legacy get_config removed; runtime config is resolved via _resolve_runtime_config().

def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(PACKAGES_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(TRAIN30_DIR, exist_ok=True)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pipeline-json",
        type=str,
        default=PIPELINE_JSON_DEFAULT,
        help="Shared pipeline config/manifest JSON (from PatchTST stage).",
    )
    parser.add_argument(
        "--allow-dummy-forecast", action="store_true", help="Use zero forecast features if missing."
    )
    parser.add_argument("--profile", choices=['default', 'high-util'], default='default')

    # Metadata + data paths (prefer JSON; CLI overrides win)
    parser.add_argument("--symbol", type=str, default=None)
    parser.add_argument("--decision-tf", type=str, default=None)
    parser.add_argument("--candles", type=str, default=None, help="Alias for --decision-tf")
    parser.add_argument("--intrabar-tf", type=str, default=None)

    parser.add_argument(
        "--decision-candles-file",
        type=str,
        default=None,
        help="Decision candle parquet (expected aligned_15m.parquet schema)",
    )
    parser.add_argument(
        "--intrabar-candles-file",
        type=str,
        default=None,
        help="Intrabar candle parquet (expected aligned_3m.parquet schema incl. slot_15m)",
    )
    parser.add_argument(
        "--features-file",
        type=str,
        default=None,
        help="Core features parquet (train30/features.parquet)",
    )
    parser.add_argument(
        "--forecast-features-file",
        type=str,
        default=None,
        help="Forecast features parquet (train30/forecast_features.parquet or package-local)",
    )

    parser.add_argument("--package-root", type=str, default=None, help="Root folder for packaged outputs")
    parser.add_argument(
        "--package-id",
        type=str,
        default=None,
        help="Explicit package id (else reuse from pipeline JSON or auto-generate)",
    )

    # PPO hyperparams (prefer JSON; CLI overrides win)
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=None,
        help="Total timesteps (e.g. 2000000 or 8000000)",
    )
    parser.add_argument("--learning-rate", type=float, default=None, help="Learning rate")
    parser.add_argument("--n-steps", type=int, default=None, help="PPO n_steps")
    parser.add_argument("--batch-size", type=int, default=None, help="PPO batch_size")
    parser.add_argument("--n-epochs", type=int, default=None, help="PPO n_epochs")
    parser.add_argument("--gamma", type=float, default=None, help="Discount gamma")
    parser.add_argument("--gae-lambda", type=float, default=None, help="GAE lambda")
    parser.add_argument("--clip-range", type=float, default=None, help="Clip range")
    parser.add_argument("--ent-coef", type=float, default=None, help="Entropy coefficient")

    # Runtime
    parser.add_argument("--n-envs", type=int, default=None, help="Number of parallel environments")
    parser.add_argument(
        "--vec-env",
        choices=["auto", "dummy", "subproc"],
        default=None,
        help="Vectorized env backend",
    )
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default=None, help="Training device")
    args = parser.parse_args()

    runtime_cfg = _resolve_runtime_config(args)

    ppo_cfg = {
        'total_timesteps': int(runtime_cfg['ppo_total_timesteps']),
        'learning_rate': float(runtime_cfg['ppo_learning_rate']),
        'n_steps': int(runtime_cfg['ppo_n_steps']),
        'batch_size': int(runtime_cfg['ppo_batch_size']),
        'n_epochs': int(runtime_cfg['ppo_n_epochs']),
        'gamma': float(runtime_cfg['ppo_gamma']),
        'gae_lambda': float(runtime_cfg['ppo_gae_lambda']),
        'clip_range': float(runtime_cfg['ppo_clip_range']),
        'ent_coef': float(runtime_cfg['ppo_ent_coef']),
    }

    print(f"Pipeline JSON: {runtime_cfg.get('pipeline_json')}")
    print(
        f"Data: decision={runtime_cfg.get('data_file_raw_decision')} intrabar={runtime_cfg.get('data_file_raw_intrabar')}"
    )
    print(
        f"Features: core={runtime_cfg.get('data_file_features')} forecast={runtime_cfg.get('forecast_features_output')}"
    )
    print(f"PPO Configuration: {ppo_cfg}")

    if runtime_cfg.get('decision_tf') != '15m' or runtime_cfg.get('intrabar_tf') != '3m':
        print(
            "WARN: PerpEnv is currently hard-wired for 15m decision + 3m intrabar (slot_15m). "
            "Other TFs are metadata-only for now."
        )
    
    try:
        df_15m, df_3m = load_data(allow_dummy_forecast=args.allow_dummy_forecast, runtime_cfg=runtime_cfg)
    except (FileNotFoundError, ValueError) as e:
        print(f"Fehler beim Laden der Daten: {e}")
        return

    n_envs = max(1, int(runtime_cfg.get("ppo_n_envs", 1)))
    vec_env = str(runtime_cfg.get("ppo_vec_env", "auto"))
    device_pref = str(runtime_cfg.get("ppo_device", "auto"))

    env, env_mode = build_vec_env(df_15m, df_3m, n_envs=n_envs, vec_env=vec_env)
    print(f"VecEnv mode: {env_mode} | n_envs={n_envs}")

    if device_pref == "cpu":
        model_device = "cpu"
    elif device_pref == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested but CUDA is not available")
        model_device = "cuda"
    else:
        model_device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Initialisiere PPO Agent auf device={model_device}...")
    policy_kwargs = dict(
        net_arch=dict(pi=[256, 256], vf=[256, 256]),
        activation_fn=torch.nn.Tanh,
    )
    
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=ppo_cfg['learning_rate'],
        n_steps=ppo_cfg['n_steps'],
        batch_size=ppo_cfg['batch_size'],
        n_epochs=ppo_cfg['n_epochs'],
        gamma=ppo_cfg['gamma'],
        gae_lambda=ppo_cfg['gae_lambda'],
        clip_range=ppo_cfg['clip_range'],
        ent_coef=ppo_cfg['ent_coef'],
        verbose=1,
        tensorboard_log=LOG_DIR,
        policy_kwargs=policy_kwargs,
        device=model_device
    )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=50000,
        save_path=CHECKPOINT_DIR,
        name_prefix="ppo_model"
    )
    
    runtime_cfg["ppo_train_started_at"] = datetime.utcnow().isoformat() + "Z"
    runtime_cfg["ppo_device"] = model_device
    runtime_cfg["ppo_vec_env"] = env_mode
    runtime_cfg["ppo_n_envs"] = n_envs
    _persist_pipeline_config(runtime_cfg)

    print(f"Starte Training für {ppo_cfg['total_timesteps']} Timesteps...")
    try:
        model.learn(total_timesteps=ppo_cfg['total_timesteps'], callback=checkpoint_callback, progress_bar=True)
        print("Training abgeschlossen.")
    except KeyboardInterrupt:
        print("Training unterbrochen. Speichere Zwischenstand...")
    
    save_base = runtime_cfg.get("ppo_model_base") or os.path.join(MODELS_DIR, "ppo_policy_final")
    os.makedirs(os.path.dirname(save_base), exist_ok=True)
    model.save(save_base)
    saved_zip = save_base + ".zip"

    runtime_cfg["ppo_model_base"] = save_base
    runtime_cfg["ppo_model_path"] = saved_zip
    runtime_cfg["ppo_train_completed_at"] = datetime.utcnow().isoformat() + "Z"

    alias_path = runtime_cfg.get("ppo_model_alias_path")
    if alias_path and os.path.abspath(alias_path) != os.path.abspath(saved_zip):
        os.makedirs(os.path.dirname(alias_path), exist_ok=True)
        shutil.copy2(saved_zip, alias_path)
        print(f"Alias aktualisiert: {alias_path}")

    _persist_pipeline_config(runtime_cfg)

    print(f"Modell gespeichert unter: {saved_zip}")


if __name__ == "__main__":
    main()
