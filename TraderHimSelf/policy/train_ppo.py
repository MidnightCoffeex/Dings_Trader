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
import numpy as np
import pandas as pd
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from typing import List, Optional, Dict, Any

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
LOG_DIR = os.path.join(project_root, "runs/ppo_logs")
CHECKPOINT_DIR = os.path.join(project_root, "checkpoints/ppo")

FORECAST_DIM = 35
ACCOUNT_DIM = 9
CORE_DIM = len(CORE_FEATURE_COLUMNS)
OS_OBS_DIM = CORE_DIM + FORECAST_DIM + ACCOUNT_DIM

# --- FORECAST FEATURE COLUMNS ---
FORECAST_COLUMNS = [f"forecast_{i}" for i in range(FORECAST_DIM)]

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
    """Normalize slot_15m values to int64 millisecond timestamps."""
    s = series.copy()

    if pd.api.types.is_datetime64_any_dtype(s):
        return (s.astype("int64") // 10**6).astype("int64")

    if pd.api.types.is_numeric_dtype(s):
        s_num = pd.to_numeric(s, errors="coerce")
        # Heuristic: nanosecond epochs are much larger than ms epochs.
        if s_num.dropna().median() > 10**14:
            s_num = s_num // 10**6
        return s_num.astype("int64")

    # Fallback: parse strings/objects as datetime
    s_dt = pd.to_datetime(s, utc=True, errors="coerce")
    if s_dt.isna().any():
        bad = int(s_dt.isna().sum())
        raise ValueError(f"slot_15m contains {bad} unparsable values")
    return (s_dt.astype("int64") // 10**6).astype("int64")


def load_data(*, allow_dummy_forecast: bool = False):
    """Lädt und mergt alle benötigten Daten."""
    print("Lade Daten...")
    
    # 1. Candles 15m
    p_15m = os.path.join(BASE_DATA_DIR, "aligned_15m.parquet")
    if not os.path.exists(p_15m):
        raise FileNotFoundError(f"{p_15m} nicht gefunden. Bitte erst build_dataset.py ausführen.")
    df_15m = pd.read_parquet(p_15m)
    
    # 2. Candles 3m
    p_3m = os.path.join(BASE_DATA_DIR, "aligned_3m.parquet")
    if not os.path.exists(p_3m):
        raise FileNotFoundError(f"{p_3m} nicht gefunden.")
    df_3m = pd.read_parquet(p_3m)
    
    # 3. Core Features
    p_feat = os.path.join(TRAIN30_DIR, "features.parquet")
    if not os.path.exists(p_feat):
        raise FileNotFoundError(f"{p_feat} nicht gefunden. Bitte feature_engine_train30.py build ausführen (train30 artifacts).")
    df_feat = pd.read_parquet(p_feat)
    missing_core = [c for c in CORE_FEATURE_COLUMNS if c not in df_feat.columns]
    if missing_core:
        raise ValueError(f"features.parquet missing core columns: {missing_core}")
    
    # 4. Forecast Features
    p_forecast = os.path.join(TRAIN30_DIR, "forecast_features.parquet")
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

def get_config(args) -> Dict[str, Any]:
    config = {
        'total_timesteps': 1_000_000,
        'learning_rate': 3e-4,
        'n_steps': 2048,
        'batch_size': 64,
        'n_epochs': 10,
        'gamma': 0.99,
        'gae_lambda': 0.95,
        'clip_range': 0.2,
        'ent_coef': 0.01
    }
    
    if args.profile == 'high-util':
        config.update({
            'n_steps': 8192,
            'batch_size': 512,
        })
    
    # Explicit overrides
    if args.n_steps is not None: config['n_steps'] = args.n_steps
    if args.batch_size is not None: config['batch_size'] = args.batch_size
    
    return config

def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(TRAIN30_DIR, exist_ok=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-dummy-forecast", action="store_true", help="Use zero forecast features if missing.")
    parser.add_argument("--profile", choices=['default', 'high-util'], default='default')
    parser.add_argument("--n-steps", type=int, help="PPO n_steps")
    parser.add_argument("--batch-size", type=int, help="PPO batch_size")
    parser.add_argument("--n-envs", type=int, default=1, help="Number of parallel environments")
    parser.add_argument("--vec-env", choices=["auto", "dummy", "subproc"], default="auto", help="Vectorized env backend")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto", help="Training device")
    args = parser.parse_args()
    
    cfg = get_config(args)
    print(f"Configuration: {cfg}")
    
    try:
        df_15m, df_3m = load_data(allow_dummy_forecast=args.allow_dummy_forecast)
    except (FileNotFoundError, ValueError) as e:
        print(f"Fehler beim Laden der Daten: {e}")
        return

    env, env_mode = build_vec_env(df_15m, df_3m, n_envs=max(1, int(args.n_envs)), vec_env=args.vec_env)
    print(f"VecEnv mode: {env_mode} | n_envs={max(1, int(args.n_envs))}")

    if args.device == "cpu":
        model_device = "cpu"
    elif args.device == "cuda":
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
        learning_rate=cfg['learning_rate'],
        n_steps=cfg['n_steps'],
        batch_size=cfg['batch_size'],
        n_epochs=cfg['n_epochs'],
        gamma=cfg['gamma'],
        gae_lambda=cfg['gae_lambda'],
        clip_range=cfg['clip_range'],
        ent_coef=cfg['ent_coef'],
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
    
    print(f"Starte Training für {cfg['total_timesteps']} Timesteps...")
    try:
        model.learn(total_timesteps=cfg['total_timesteps'], callback=checkpoint_callback, progress_bar=True)
        print("Training abgeschlossen.")
    except KeyboardInterrupt:
        print("Training unterbrochen. Speichere Zwischenstand...")
    
    save_path = os.path.join(MODELS_DIR, "ppo_policy_final")
    model.save(save_path)
    print(f"Modell gespeichert unter: {save_path}.zip")


if __name__ == "__main__":
    main()
