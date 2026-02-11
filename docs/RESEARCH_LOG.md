# Research Log - dings-trader

## 2026-02-04: The 5m-Chart Massacre
- **Status:** Model v2 (LightGBM, 180d window) reached 46% accuracy.
- **Problem:** Backtest equity went to zero in all configurations.
- **Root Cause:** Transaction costs exceed average trade move on 5m.

## 2026-02-04: Portfolio Engine v2 (Maxim's Split)
- **Status:** Implemented strict Train/Val/Test pipeline and Portfolio Logic.
- **Rules:** Max 5 concurrent positions, max 10% total equity exposure.
- **Timeframe:** 1h (2 years historical data).
- **Results:**
    - **Training:** Data pre-2024.
    - **Validation (2024):** Final Equity **1130.64** (Profit +13%) with 0.50 confidence threshold.
    - **Test (2025+):** Final Equity **1027.40** (Profit +2.7%) with 0.50 confidence threshold.
- **Insight:** The portfolio logic and higher timeframe actually produce profit! The "10% rule" prevents total ruin and allows the model's signal to compound.
