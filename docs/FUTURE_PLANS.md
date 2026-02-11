# Dings-Trader Future Plans

## Dynamic Scaling & Position Management

### Overview
This document outlines the planned enhancements for dynamic position scaling and flexible risk management in the Dings-Trader system.

### Configurable Max Positions

**Concept:** Allow dynamic configuration of the maximum number of concurrent positions (currently fixed at 5).

- **Range:** Configurable from 1 to 10 positions
- **Configuration:** Via environment variable or dashboard setting
- **Impact:** More flexible trading strategies based on market conditions and model confidence

### Dynamic Risk-Meter Slots

**Concept:** The Risk-Meter visualization should adapt dynamically based on the configured max positions setting.

- **Current:** Fixed 5-slot display
- **Future:** Variable slot count (1-10) matching the max positions configuration
- **Visual:** Slots scale proportionally within the Risk-Meter card
- **Color Coding:** Maintain current green/yellow/red scheme based on utilization

### Dynamic Position Sizing by ML Model

**Concept:** Replace fixed 2% per slot sizing with intelligent, model-driven position sizing.

#### Key Constraints

1. **Total Equity Cap:** Sum of all position sizes must not exceed 10% of total equity
2. **ML-Driven Allocation:** The ML model decides individual position sizes
3. **Flexible Distribution:** Examples of valid allocations:
   - One 5% trade + several smaller 1-2% trades
   - Ten 1% trades
   - Two 4% trades + two 1% trades
   - Any combination summing to ≤ 10%

#### Model Decision Factors

The ML model should consider:
- Signal strength and confidence score
- Market volatility
- Current portfolio exposure
- Historical performance of similar setups
- Correlation between concurrent positions

#### Implementation Notes

- Position sizing becomes part of the model's output
- Risk management enforces the 10% total cap as a hard limit
- Dashboard updates to show actual vs. maximum allocated per position
- Position cards display allocated percentage prominently

### Benefits

1. **Capital Efficiency:** Better allocation based on opportunity quality
2. **Risk Optimization:** Dynamic sizing reduces overexposure to correlated trades
3. **Strategy Flexibility:** Supports both concentrated and diversified approaches
4. **ML Empowerment:** Model has more control over risk/reward balance

### Timeline

These features are planned for a future v2.0 release following the stabilization of the current paper trading system.
