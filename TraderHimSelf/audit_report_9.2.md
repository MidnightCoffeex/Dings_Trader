# Audit Report: Step 9.2 Bug-Hunting
**Date:** 2026-02-09
**Scope:** `projects/dings-trader/TraderHimSelf/` (Focus: `env/perp_env.py`, `feature_engine.py`, `build_dataset.py`)

## 🚨 Critical Issues (High Priority)

### 1. Time-Travel / Causality Violation in `perp_env.py`
**Location:** `PerpEnv.step()`
**Description:**
The environment executes trades using `current_close` (from `self.df_15m.iloc[self.current_step]`), which corresponds to the price at the *end* of the 15-minute bar. Immediately after, it runs the **Intrabar Simulation** using 3m candles starting at `current_time` (the *start* of that same 15m bar).
**Impact:**
- **Impossible Logic:** You are entering a trade at 10:15 (Close) but checking if you were stopped out between 10:00 and 10:15.
- **Backtest Reliability:** 0%. The agent effectively "sees" the future (the bar is already closed) and checks risk on the past.

**Correction Proposal:**
Shift the execution or simulation logic.
- **Option A (Trade at Open):** Use `obs` from `step-1`, execute at `df_15m.iloc[step]['open']`, simulate on `df_3m` for `step`.
- **Option B (Trade at Close):** Decide at `step` (Close), execute at `df_15m.iloc[step+1]['open']`, simulate on `df_3m` for `step+1`.

### 2. Liquidation PnL Calculation Error
**Location:** `PerpEnv.step()` (Liquidation block)
**Description:**
```python
self.balance += realized_pnl # realized_pnl is -margin_used
# ...
final_pnl = ... # Calculated based on Liq Price vs Entry (large negative value)
self.balance += final_pnl # Adds negative value AGAIN
```
**Impact:**
- **Double Counting:** The account balance is penalized twice for a liquidation. First by removing the margin, then by subtracting the calculated PnL (which essentially equals the lost margin).
- **Result:** Agent goes bankrupt much faster than it should.

**Correction Proposal:**
If using Isolated Margin, the maximum loss is the `margin_used`. Remove the second `self.balance += final_pnl`.

## ⚠️ Performance Bottlenecks

### 3. Inefficient Intrabar Lookup
**Location:** `PerpEnv.step()`
**Description:**
```python
sub_candles = self.df_3m[(self.df_3m['open_time_ms'] >= start_ts) & ...]
```
This performs a boolean mask scan over the entire `df_3m` dataframe (millions of rows) *every single step*.
**Impact:** Training will be extremely slow (seconds per step instead of microseconds).

**Correction Proposal:**
Leverage the strict alignment from `build_dataset.py`. Since every 15m bar has exactly 5 corresponding 3m slots (even if empty/NaN):
```python
# O(1) Lookup
idx_start = self.current_step * 5
sub_candles = self.df_3m.iloc[idx_start : idx_start + 5]
```

## 🔍 Other Findings

### 4. Code Duplication (Risk Logic)
- **Observation:** `env/perp_env.py` implements its own checks for `NO_HEDGE`, `MAX_POSITIONS`, and `MAX_EXPOSURE`.
- **Issue:** `env/risk_manager.py` exists but is not used in `perp_env.py`. This leads to "drift" (e.g., `perp_env` allows checking major side, `risk_manager` returns `'flat'`).
- **Fix:** Integrate `RiskManager` into `PerpEnv` to centralize validation.

### 5. Missing Imports
- `perp_env.py` relies on `from ..data_contract import ...` which works if run as a module from root, but might fail in standalone notebook tests. (Addressed by try/except block in code, but fragile).

## Summary
The codebase needs an immediate refactor of `PerpEnv.step()` to fix the time-travel bug and performance issue before any RL training (`train_ppo.py`) can yield valid results.
