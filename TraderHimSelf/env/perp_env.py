"""
perp_env.py

Gymnasium Environment für den BTCUSDT Perpetual Bot.
Implementiert die Logik aus Roadmap Schritt 6:
- 15m Decision Steps + 3m Intrabar Simulation
- Portfolio-Management (Multi-Position, max 10 Lots)
- Gebühren, Funding, Slippage, Liquidation
- SL/TP Logik (SL-first)

Author: Sub-Agent
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import math

# Versuche data_contract zu importieren, fallback falls Pfad nicht stimmt
try:
    from ..data_contract import TradingConfig, CandleRecord
except ImportError:
    # Fallback für Standalone-Tests oder abweichende Pfade
    class TradingConfig:
        SYMBOL = "BTCUSDT"
        DECISION_TIMEFRAME = "15m"
        INTRABAR_TIMEFRAME = "3m"
        MAX_HOLD_STEPS = 192
        MAX_EXPOSURE_PCT = 0.10
        MAX_POSITIONS = 10
        TAKER_FEE = 0.0006
        MAKER_FEE = 0.0002
        NO_HEDGE = True
        BUFFER_STEPS = 0

@dataclass
class EnvLot:
    """
    Repräsentiert ein einzelnes Position-Lot im Environment.
    Entspricht Roadmap Schritt 6.1.
    """
    side: str            # 'long' oder 'short'
    margin_used: float   # Margin in USDT
    leverage: float      # Hebel
    notional: float      # margin_used * leverage
    entry_price: float   # Eintrittspreis
    qty: float           # notional / entry_price (in BTC)
    sl_price: float      # Stop Loss Preis
    tp_price: float      # Take Profit Preis
    open_time_ms: int    # Timestamp des Openings
    time_in_trade_steps_15m: int = 0
    
    # Tracking für uPnL und Liquidation Check
    current_pnl: float = 0.0
    
    def update_pnl(self, current_price: float):
        """Berechnet uPnL (linear USDT-M)."""
        if self.side == 'long':
            self.current_pnl = self.qty * (current_price - self.entry_price)
        else:
            self.current_pnl = self.qty * (self.entry_price - current_price)


class PerpEnv(gym.Env):
    """
    Trading Environment für BTCUSDT Perp.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, df_15m: pd.DataFrame, df_3m: pd.DataFrame, funding_df: Optional[pd.DataFrame] = None):
        """
        Initialisiert das Environment.
        
        Args:
            df_15m: DataFrame mit 15m Candles (muss 'open', 'high', 'low', 'close', 'atr', 'open_time_ms' enthalten).
                    Erwartet zudem vorbrechnete Features für die Observation.
            df_3m: DataFrame mit 3m Candles (für Intrabar Sim). Muss aligned sein.
            funding_df: DataFrame mit Funding Rates (optional).
        """
        super(PerpEnv, self).__init__()

        self.df_15m = df_15m.reset_index(drop=True)
        self.df_3m = df_3m.reset_index(drop=True)
        self.funding_df = funding_df

        if "open_time_ms" not in self.df_15m.columns:
            raise ValueError("df_15m missing required column: open_time_ms")
        if "slot_15m" not in self.df_3m.columns:
            raise ValueError("df_3m missing required column: slot_15m")

        # Precompute 15m->3m mapping via slot_15m
        # Maps 15m open_time_ms -> array of 3m indices (expected 5 subbars)
        self._slot_to_3m_idx = self.df_3m.groupby("slot_15m").indices

        # Funding rates aligned to 15m steps
        if "funding_rate" in self.df_15m.columns:
            self.funding_rates = self.df_15m["funding_rate"].astype(float).values
        elif self.funding_df is not None:
            f = self.funding_df.copy()
            if not isinstance(f.index, pd.DatetimeIndex) and "time_ms" in f.columns:
                f.index = pd.to_datetime(f["time_ms"], unit="ms", utc=True)
            f = f.sort_index()
            # Map by asof on open_time_ms
            temp = pd.DataFrame({"open_time_ms": self.df_15m["open_time_ms"].values})
            temp["_ts"] = pd.to_datetime(temp["open_time_ms"], unit="ms", utc=True)
            f["_ts"] = f.index
            merged = pd.merge_asof(temp.sort_values("_ts"), f[["_ts", "funding_rate"]], on="_ts", direction="backward")
            merged = merged.sort_index()
            self.funding_rates = merged["funding_rate"].astype(float).values
        else:
            self.funding_rates = np.zeros(len(self.df_15m), dtype=float)

        # Mapping von 15m Index zu 3m Indizes (simple Annahme: 5x 3m pro 15m, aligned by timestamp)
        # Für Performance bauen wir hier einen Lookup oder gehen davon aus, dass df_3m 5x so lang ist
        # und synchron startet. Der Einfachheit halber nutzen wir hier Time-Matching.
        # In einer optimierten Version würde 'build_dataset.py' dies vorbreiten (z.B. Array-Struktur).
        
        # Action Space (Step 9.2 & 6.8):
        # Direction: 0=Flat, 1=Long, 2=Short
        # Size: 0..1 (float) -> via Box
        # Leverage: 1..10 (int) -> via Box (rounded) oder Discrete? 
        # Roadmap sagt "mapped zu integer", wir nutzen Box für continuous output des PPO Agents
        # SL_mult: 0.5..3.0
        # TP_mult: 0.5..6.0
        # Struktur: [direction_logit_long, direction_logit_short, direction_logit_flat, size, leverage, sl_mult, tp_mult]
        # Oder einfacher (Standard SB3 PPO mag Box spaces):
        # Wir definieren hier einen Box Space und mappen im step()
        
        # Wir nutzen einen kombinierten Box Space für Action Params. 
        # Die Entscheidung Flat/Long/Short machen wir meist über Argmax der ersten 3 Werte oder Discrete.
        # Roadmap sagt "Action Space (fix): direction, size, leverage, sl_mult, tp_mult".
        # Um PPO "native" zu unterstützen, ist ein rein kontinuierlicher Space oft einfacher, 
        # wobei direction diskretisiert wird (z.B. 0.0-0.33=Flat, etc. oder via MultiDiscrete).
        # Hier implementieren wir es als:
        # [action_type (float), size (float), leverage (float), sl_mult (float), tp_mult (float)]
        # action_type: < -0.33 -> Short, > 0.33 -> Long, dazwischen Flat (Beispiel)
        # Oder besser: 3 Output Neuronen für Action Type (Softmax/Argmax).
        # Da Gym "Space" nur einen Typ haben darf (Tuple geht, aber komplexer für PPO):
        # Wir nehmen Box(low=-1, high=1, shape=(5,))
        # 0: Direction (-1: Short, 0: Flat, 1: Long) -> threshold logic
        # 1: Size (0..1)
        # 2: Leverage (mapped to 1..10)
        # 3: SL Mult (mapped to 0.5..3.0)
        # 4: TP Mult (mapped to 0.5..6.0)
        
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(5,), dtype=np.float32)

        # Observation Space (Step 9.1): 72 Dimensionen
        # Wir nehmen an, die Input DFs haben die Features schon oder wir füllen Nullen.
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(72,), dtype=np.float32)

        # State Variables
        self.current_step = 0
        self.equity = 10000.0  # Startkapital (Beispiel)
        self.open_positions: List[EnvLot] = []
        self.balance = self.equity # Balance = Equity wenn keine Pos offen. 
        # Equity = Balance + uPnL
        
        self.done = False
        
        # Constants
        self.maker_fee = TradingConfig.MAKER_FEE
        self.taker_fee = TradingConfig.TAKER_FEE
        self.max_pos = TradingConfig.MAX_POSITIONS
        self.max_exposure_pct = TradingConfig.MAX_EXPOSURE_PCT

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.current_step = 0 # Sollte idealerweise random start im Training sein (Bootstrapping beachten)
        if options and 'start_step' in options:
            self.current_step = options['start_step']
        else:
            # Skip Bootstrapping buffer (e.g. 800 steps)
            self.current_step = TradingConfig.BUFFER_STEPS

        self.equity = 10000.0
        self.balance = 10000.0
        self.open_positions = []
        self.done = False
        
        return self._get_obs(), {}

    def step(self, action):
        """
        Führt einen 15m Step aus.
        1. Parse Action
        2. Execute Order (Open Position) at Next Open (T+1)
        3. Intrabar Simulation (5x 3m of T+1) -> Check SL/TP/Liq
        4. Calculate Reward
        5. Return State
        """
        if self.done:
            return self._get_obs(), 0.0, True, False, {}
        
        # Check boundary for T+1
        if self.current_step + 1 >= len(self.df_15m):
            self.done = True
            return self._get_obs(), 0.0, True, False, {}

        # 0. Data for current step (Decision Time T)
        current_candle = self.df_15m.iloc[self.current_step]
        current_close = current_candle['close']
        current_atr = current_candle.get('atr', current_close * 0.01) # Fallback ATR
        
        # Next Candle (Execution Time T+1)
        next_step_idx = self.current_step + 1
        next_candle = self.df_15m.iloc[next_step_idx]
        next_open = next_candle['open'] # EXECUTION PRICE
        next_close = next_candle['close']

        # 1. Parse Action
        # Mapping Box(-1..1) to logic
        act_dir_raw, act_size_raw, act_lev_raw, act_sl_raw, act_tp_raw = action
        
        # Direction Logic (Thresholds)
        direction = 'flat'
        if act_dir_raw > 0.33:
            direction = 'long'
        elif act_dir_raw < -0.33:
            direction = 'short'
        
        # Mapping Parameters
        size_pct = (act_size_raw + 1) / 2.0 # Map -1..1 to 0..1
        leverage = 1.0 + ((act_lev_raw + 1) / 2.0) * 9.0 # Map -1..1 to 1..10
        leverage = round(leverage) # Integer Leverage
        
        sl_mult = 0.5 + ((act_sl_raw + 1) / 2.0) * 2.5 # Map -1..1 to 0.5..3.0
        tp_mult = 0.5 + ((act_tp_raw + 1) / 2.0) * 5.5 # Map -1..1 to 0.5..6.0

        # --- INTRABAR SUBBARS (5x 3m of T+1) ---
        slot_key = next_candle["open_time_ms"]
        idxs = self._slot_to_3m_idx.get(slot_key)
        sub_candles = self.df_3m.iloc[idxs] if idxs is not None else self.df_3m.iloc[0:0]

        missing_intrabar = False
        # Missing if no subbars, is_missing flag, or NaNs in OHLC
        if len(sub_candles) == 0:
            missing_intrabar = True
        elif "is_missing" in sub_candles.columns and sub_candles["is_missing"].astype(bool).any():
            missing_intrabar = True
        elif sub_candles[["open", "high", "low", "close"]].isna().any().any():
            missing_intrabar = True

        # Also treat missing 15m candle conservatively
        if "is_missing" in next_candle and bool(next_candle["is_missing"]):
            missing_intrabar = True
        if pd.isna(next_open) or pd.isna(next_close):
            missing_intrabar = True
            next_open = current_close
            next_close = current_close

        # --- ACTION EXECUTION (At T+1 Open) ---
        entry_penalty = 0.0
        trade_opened = False
        
        # Check Rules
        # 1. No Hedge
        current_side_major = self._get_major_side()
        can_open = True
        if direction == 'flat':
            can_open = False
        elif len(self.open_positions) > 0:
            if TradingConfig.NO_HEDGE and direction != current_side_major:
                can_open = False # Block hedge
        
        # 2. Max Positions
        if len(self.open_positions) >= self.max_pos:
            can_open = False
            
        # 3. Exposure Cap
        current_exposure = sum(p.margin_used for p in self.open_positions)
        available_exposure = (self.equity * self.max_exposure_pct) - current_exposure
        
        if missing_intrabar:
            can_open = False

        if can_open and available_exposure > 0:
            # Calculate Margin Used
            desired_margin = available_exposure * size_pct
            
            # Min trade size check (z.B. 5 USDT) - hier ignorieren oder simple check
            if desired_margin > 1.0: # Min 1$ margin
                # Calculate SL/TP Prices based on T Open (Execution Price)
                # Note: ATR is from T (known at decision time)
                sl_dist = sl_mult * current_atr
                tp_dist = tp_mult * current_atr
                
                # Slippage (Entry) applied to Next Open
                slippage_pct = 0.0002 + 0.1 * (current_atr / next_open)
                
                if direction == 'long':
                    exec_price = next_open * (1 + slippage_pct)
                    sl_price = exec_price - sl_dist
                    tp_price = exec_price + tp_dist
                else: # short
                    exec_price = next_open * (1 - slippage_pct)
                    sl_price = exec_price + sl_dist
                    tp_price = exec_price - tp_dist
                    
                # Create Lot
                notional = desired_margin * leverage
                qty = notional / exec_price
                
                new_lot = EnvLot(
                    side=direction,
                    margin_used=desired_margin,
                    leverage=leverage,
                    notional=notional,
                    entry_price=exec_price,
                    qty=qty,
                    sl_price=sl_price,
                    tp_price=tp_price,
                    open_time_ms=next_candle['open_time_ms'],
                    time_in_trade_steps_15m=0
                )
                
                self.open_positions.append(new_lot)
                
                # Pay Entry Fee immediately from Balance
                fee = notional * self.taker_fee
                self.balance -= fee
                entry_penalty = 0.0002 * self.equity # Reward shaping
                trade_opened = True

        # --- INTRABAR SIMULATION ---
        # If missing intrabar data, fall back to single 15m candle checks (conservative).
        if missing_intrabar:
            if not pd.isna(next_candle["high"]):
                sub_candles_iter = [(None, next_candle)]
                step_increment = 1.0
            else:
                sub_candles_iter = []
                step_increment = 0.0
        else:
            sub_candles_iter = list(sub_candles.iterrows())
            step_increment = 1.0 / max(len(sub_candles_iter), 1)
        
        total_pnl_delta = 0.0
        liquidation_penalty = 0.0
        
        # Iteration über Subbars
        for _, sub_bar in sub_candles_iter:
            sb_open = sub_bar["open"]
            sb_high = sub_bar["high"]
            sb_low = sub_bar["low"]
            sb_close = sub_bar["close"]
            
            # Check for each position
            # Use list copy to allow removal
            for lot in self.open_positions[:]:
                lot.time_in_trade_steps_15m += step_increment
                
                # 1. Update uPnL (for Liq check)
                # Wir checken Liq gegen Low/High (worst case intrabar)
                # Long Liq danger at Low, Short at High
                
                # Liquidation Logic Step 6.6
                # Maintenance Margin Requirement (MMR)
                mmr = 0.005
                
                # Check Liquidation Trigger
                is_liquidated = False
                liq_price = 0.0
                
                if lot.side == 'long':
                    # PnL at Low
                    pnl_at_low = lot.qty * (sb_low - lot.entry_price)
                    # Check: margin + pnl - costs <= maintenance
                    # Isolated Margin Proxy:
                    if lot.margin_used + pnl_at_low <= (mmr * lot.notional):
                        is_liquidated = True
                        liq_price = sb_low # Approximation
                else: # short
                    pnl_at_high = lot.qty * (lot.entry_price - sb_high)
                    if lot.margin_used + pnl_at_high <= (mmr * lot.notional):
                        is_liquidated = True
                        liq_price = sb_high

                if is_liquidated:
                    # Close Position
                    self.open_positions.remove(lot)
                    
                    # Liquidation Fix: Prevent double counting
                    # Realized PnL is total loss of margin.
                    # Balance only needs to be reduced by the Margin Used (since it was part of Equity).
                    # Actually, if Balance was 10000, and we opened a pos with 1000 margin.
                    # Balance is 9999.4 (fee). uPnL starts 0. Equity 9999.4.
                    # If liquidated: Margin is gone. Equity should be 8999.4.
                    # So Balance should become 8999.4.
                    # We subtract margin_used from Balance.
                    
                    self.balance -= lot.margin_used 
                    
                    # No additional PnL subtraction!
                    
                    liquidation_penalty += 50.0 # Reward penalty
                    continue # Lot is gone

                # 2. Check SL/TP (Step 6.8 & 6.7)
                # SL-first Regel
                closed_reason = None
                exec_exit_price = 0.0
                
                if lot.side == 'long':
                    sl_hit = sb_low <= lot.sl_price
                    tp_hit = sb_high >= lot.tp_price
                else:
                    sl_hit = sb_high >= lot.sl_price
                    tp_hit = sb_low <= lot.tp_price
                    
                if sl_hit:
                    closed_reason = 'SL'
                    exec_exit_price = lot.sl_price # Slippage logic could apply here too
                elif tp_hit:
                    closed_reason = 'TP'
                    exec_exit_price = lot.tp_price
                    
                if closed_reason:
                    self.open_positions.remove(lot)
                    
                    # Apply Slippage on Exit
                    # simplified: accept trigger price as exec price for now
                    exit_val = lot.qty * exec_exit_price
                    fee = exit_val * self.taker_fee
                    
                    if lot.side == 'long':
                        pnl = lot.qty * (exec_exit_price - lot.entry_price)
                    else:
                        pnl = lot.qty * (lot.entry_price - exec_exit_price)
                        
                    self.balance += (pnl - fee)
                    continue

        # If we had no subbars at all, close positions conservatively at next_open.
        if missing_intrabar and len(sub_candles_iter) == 0 and self.open_positions:
            for lot in self.open_positions[:]:
                self.open_positions.remove(lot)
                exit_price = next_open
                slip_pct = 0.0002 + 0.1 * (current_atr / next_open) if next_open > 0 else 0.0
                if lot.side == 'long':
                    exit_price *= (1 - slip_pct)
                    pnl = lot.qty * (exit_price - lot.entry_price)
                else:
                    exit_price *= (1 + slip_pct)
                    pnl = lot.qty * (lot.entry_price - exit_price)
                exit_val = lot.qty * exit_price
                fee = exit_val * self.taker_fee
                self.balance += (pnl - fee)

        # If we had no subbars, still advance time in trade conservatively.
        if missing_intrabar and len(sub_candles_iter) == 0:
            for lot in self.open_positions:
                lot.time_in_trade_steps_15m += 1.0

        # --- TIMEOUT & FUNDING CHECKS (End of Step) ---
        # Update Time in Trade (Round up for safety)
        for lot in self.open_positions[:]:
            lot.time_in_trade_steps_15m = math.ceil(lot.time_in_trade_steps_15m)
            
            # Timeout Check (192 steps)
            if lot.time_in_trade_steps_15m >= TradingConfig.MAX_HOLD_STEPS:
                self.open_positions.remove(lot)
                
                # Market Close at T+1 Close
                exit_price = next_close
                
                # Slippage
                slip_pct = 0.0002 + 0.1 * (current_atr / next_close)
                if lot.side == 'long': exit_price *= (1 - slip_pct)
                else: exit_price *= (1 + slip_pct)
                
                exit_val = lot.qty * exit_price
                fee = exit_val * self.taker_fee
                
                if lot.side == 'long':
                    pnl = lot.qty * (exit_price - lot.entry_price)
                else:
                    pnl = lot.qty * (lot.entry_price - exit_price)
                    
                self.balance += (pnl - fee)

            # Funding (applied per 15m step based on aligned funding_rate)
            funding_rate = 0.0
            if 0 <= next_step_idx < len(self.funding_rates):
                fr = self.funding_rates[next_step_idx]
                if np.isfinite(fr):
                    funding_rate = float(fr)
            if funding_rate != 0.0:
                sign = -1.0 if lot.side == 'long' else 1.0
                self.balance += sign * lot.notional * funding_rate

        # --- REWARD CALCULATION ---
        # Calculate current Equity based on T+1 Close
        upnl_total = 0.0
        for lot in self.open_positions:
            lot.update_pnl(next_close)
            upnl_total += lot.current_pnl
            
        new_equity = self.balance + upnl_total
        
        # Reward = Delta Equity - penalties
        prev_equity = getattr(self, 'prev_equity', 10000.0) # Init logic handle
        if self.current_step == TradingConfig.BUFFER_STEPS: # First step
            prev_equity = 10000.0
            
        delta_equity = new_equity - prev_equity
        self.prev_equity = new_equity
        
        # Risk Penalty (Step 9.3)
        total_notional = sum(p.notional for p in self.open_positions)
        notional_open_pct = total_notional / new_equity if new_equity > 0 else 0
        risk_penalty = 0.1 * abs(notional_open_pct) * (current_atr / next_close)
        
        reward = delta_equity - risk_penalty - entry_penalty - liquidation_penalty
        
        # Update State to T+1
        self.equity = new_equity
        self.current_step += 1
        
        # Termination
        if self.equity <= 0: # Bust
            self.done = True
            reward -= 1000 # Bust penalty
        
        if self.current_step >= len(self.df_15m) - 1:
            self.done = True
            
        return self._get_obs(), reward, self.done, False, {}

    def _get_obs(self):
        """
        Erstellt den Observation Vector.
        Placeholder Implementierung, die Nullen zurückgibt für fehlende Features,
        aber den Portfolio State korrekt füllt.
        """
        # 1. Features & Forecast (aus DataFrame)
        # Wir nehmen an, der DF hat Spalten 'feat_0'...'feat_62' oder wir füllen 0
        # Hier returnen wir Nullen für den ML-Part, da feature_engine nicht integriert ist
        ml_features = np.zeros(28 + 35, dtype=np.float32)
        
        # 2. Portfolio State (9 Dim)
        # pos_count, pos_side_major, exposure_open_pct, notional_open_pct, uPnL_open_pct,
        # time_in_trade_max, time_left_min, liq_buffer_min_pct, available_exposure_pct
        
        pos_count = len(self.open_positions)
        
        major_side = 0
        if pos_count > 0:
            major_side = 1 if self.open_positions[0].side == 'long' else 2
            
        total_margin = sum(p.margin_used for p in self.open_positions)
        total_notional = sum(p.notional for p in self.open_positions)
        total_upnl = sum(p.current_pnl for p in self.open_positions)
        
        equity = self.equity if self.equity > 0 else 1.0 # Div/0 protect
        
        exp_open_pct = total_margin / equity
        not_open_pct = total_notional / equity
        upnl_open_pct = total_upnl / equity
        
        time_in_trade_max = 0
        if pos_count > 0:
            time_in_trade_max = max(p.time_in_trade_steps_15m for p in self.open_positions)
            
        time_left_min = TradingConfig.MAX_HOLD_STEPS - time_in_trade_max
        
        liq_buffer_min = 1.0 # TODO: Calculate real distance to liquidation
        
        avail_exp_pct = (self.equity * self.max_exposure_pct - total_margin) / equity
        
        state_vec = np.array([
            pos_count, major_side, exp_open_pct, not_open_pct, upnl_open_pct,
            time_in_trade_max, time_left_min, liq_buffer_min, avail_exp_pct
        ], dtype=np.float32)
        
        return np.concatenate([ml_features, state_vec])

    def _get_major_side(self):
        if not self.open_positions:
            return 'flat'
        # Simple Logic: First position defines side (since hedge is blocked)
        return self.open_positions[0].side

    def render(self, mode='human'):
        print(f"Step: {self.current_step} | Eq: {self.equity:.2f} | Pos: {len(self.open_positions)}")
