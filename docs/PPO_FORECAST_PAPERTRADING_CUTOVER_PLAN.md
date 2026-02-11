# PPO Forecast Paper Trading Cutover Plan

## 1. Executive Summary
This document outlines the migration strategy from the legacy `model_v2.1.joblib` pipeline to the new **TraderHimSelf Dual-Model Stack** (PatchTST Forecast + PPO Policy). The new architecture processes live 15m candles through a complex feature engineering and forecasting pipeline before feeding the policy agent.

**Goal:** Enable immediate paper trading with the new stack without waiting for 512 live candles (approx. 5 days) by implementing a robust historical warmup mechanism.

## 2. Current State Audit

### Legacy Components to Replace
-   **Endpoints:** `GET /paper/ml-signal/{model_id}` (Hardcoded to `paper_test_hf` or legacy `model_v2.1`).
-   **Files:**
    -   `projects/dings-trader/ml/model_v2.1.joblib` (Old Classifier).
    -   `projects/dings-trader/ml/features.py` (Old Feature Engineering).
-   **Logic:** Simple probability-based logic (`p_long > p_short`).

### Constraints
-   The current API explicitly blocks non-`paper_test_hf` models.
-   The new models (`forecast_model.pt`, `ppo_policy_final.zip`) are located in `TraderHimSelf/models/` but not yet integrated into the `ml/` runtime.

## 3. Target Architecture

### 3.1 Data Flow Pipeline
The new inference pipeline is significantly more complex:

1.  **Ingestion:** Raw 15m Candles (Binance).
2.  **Core Features (28 dims):**
    -   Requires **672 candles** of history (due to `vol_672` rolling window).
    -   Transforms: Log Returns, Volatility, EMA Distances, RSI, MACD, Volume Z-Score, Funding.
    -   **Scaling:** Must apply `StandardScaler` (pre-fitted on 2019-2023 data).
3.  **Forecast (PatchTST):**
    -   **Input:** Sequence of 512 steps of *Scaled Core Features* (512 x 28).
    -   **Model:** `forecast_model.pt` (PyTorch).
    -   **Output:** Quantile Forecasts (192 steps x 3 quantiles).
4.  **Forecast Features (35 dims):**
    -   Post-processing of PatchTST output (Horizon stats, Path stats, Curve stats).
5.  **Observation Construction (72 dims):**
    -   Concatenation: `[Core Features (28) | Forecast Features (35) | Account State (9)]`.
6.  **Policy (PPO):**
    -   **Input:** 72-dim Observation.
    -   **Model:** `ppo_policy_final.zip` (Stable Baselines 3).
    -   **Output:** Action (Side, Size, Leverage, SL, TP).

### 3.2 Warmup Requirements
To produce the *first* valid action immediately at startup:

*   **Feature Engine:** Needs 672 raw candles to produce the first valid feature row.
*   **PatchTST:** Needs 512 *valid* feature rows for one inference.
*   **Total Minimum History:** 672 + 512 = **1184 candles**.
*   **Recommended Buffer:** **1500 candles** (~16 days of 15m history).

### 3.3 3m Candle Requirement
*   **Policy Inference:** Does NOT use 3m candles for decision making.
*   **Execution:** 3m candles are used for stop-loss/take-profit monitoring during the 15m interval.
*   **Warmup:** **0** (Start listening to live stream immediately).

## 4. Implementation Plan

### Phase 1: Backend Preparation
1.  **Model Deployment:**
    -   Copy `forecast_model.pt` and `ppo_policy_final.zip` to `projects/dings-trader/ml/models/`.
    -   **CRITICAL:** Ensure `scaler.pkl` exists in `projects/dings-trader/ml/models/`. If missing, generate it using `feature_engine.py` on 2023-2025 data.
2.  **Inference Engine (`ml/inference_v3.py`):**
    -   Implement `DualModelEngine` class.
    -   Implement `BinanceHistoryLoader` to fetch 1500 candles on init.
    -   Implement `FeaturePipeline` (feature_engine logic + scaler + PatchTST post-processing).

### Phase 2: API Integration
1.  **New Endpoint:** Create `POST /paper/inference/v3`.
    -   Accepts: `symbol`, `current_price`, `account_state`.
    -   Returns: `action` (formatted for execution).
2.  **Update `paper_api.py`:**
    -   Allow `paper_v3_ppo` model ID.
    -   Route requests to the new engine.

### Phase 3: Rollout & UI Cutover
1.  **Update Loops:**
    -   Modify `paper_inference_loop.py` to instantiate the V3 engine.
    -   Configure `ecosystem.config.js` to run a `dt-loop-v3` with the new stack.
2.  **UI Updates:**
    -   Ensure UI displays the new "Confidence" (derived from PPO deterministic vs stochastic difference, or placeholder).

### Phase 4: Cleanup
1.  Retire `model_v2.1.joblib`.
2.  Remove legacy endpoint logic in `paper_api.py`.

## 5. API Contract (Backend -> UI/Loop)

**Request:**
```json
{
  "model_id": "paper_v3_ppo",
  "symbol": "BTCUSDT",
  "current_price": 96500.50,
  "account_state": {
    "equity": 10500.00,
    "positions": [...]
  }
}
```

**Response:**
```json
{
  "action": {
    "type": "OPEN_LONG",  # or CLOSE, HOLD, FLIP
    "size_usdt": 2000.00,
    "leverage": 5,
    "stop_loss": 94000.00,
    "take_profit": 102000.00
  },
  "debug": {
    "forecast_trend": "bullish",
    "feature_validity": true
  }
}
```

## 6. Rollback Strategy
1.  Keep `paper_test_hf` running in parallel.
2.  If V3 fails (e.g., OOM, crash), the ecosystem auto-restarts.
3.  If trading logic is broken, stop `dt-loop-v3` via PM2 (`pm2 stop dt-loop-v3`).

## 7. Checklist
- [ ] **Scaler Availability:** Verify `scaler.pkl` is present/generated.
- [ ] **Warmup Logic:** Verify fetching 1500 candles takes < 30s.
- [ ] **Model Loading:** Verify `forecast_model.pt` loads in PyTorch without path errors.
- [ ] **PPO Observation:** Ensure account state in live loop matches `train_ppo.py` logic perfectly.
