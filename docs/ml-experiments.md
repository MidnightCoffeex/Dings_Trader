# ML Experiments & Benchmarks

## 2026-02-04: Baseline LightGBM Test
- **Goal:** Verify walk-forward pipeline and basic feature set.
- **Config:** 
  - Horizon: 12 bars (1h)
  - Train: 30 days, Val: 7 days, Step: 7 days.
- **Results:**
  - Fold 10 Accuracy: 42% (Weighted F1: 0.42)
  - Classes: short=0, flat=1, long=2
  - Observations: Model shows some predictive signal (accuracy > random 33%), but high class imbalance needs tuning.
- **Output:** `projects/dings-trader/data/test_preds.parquet`
