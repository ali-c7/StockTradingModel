# Train/Test Split Control Feature

## Overview

Added user control over the train/test split ratio to allow more flexibility in model evaluation.

## Location

**Sidebar → Advanced Settings → Train/Test Split slider**

## Parameters

- **Range**: 60% to 90% (training data percentage)
- **Default**: 80% (classic 80/20 split)
- **Step**: 5%
- **Display**: Shows both training and testing percentages

## How It Works

### Split Ratios

| Split | Training | Testing | Use Case |
|-------|----------|---------|----------|
| 60% | 60% | 40% | Maximum test data for validation |
| 70% | 70% | 30% | Balanced approach |
| **80%** | **80%** | **20%** | **Standard (default)** |
| 85% | 85% | 15% | More training data |
| 90% | 90% | 10% | Maximum training data |

### Example: 2Y Timeframe with Different Splits

Assuming 2Y = 504 trading days:

| Split | Train Days | Train Period | Test Days | Test Period |
|-------|------------|--------------|-----------|-------------|
| 60% | 302 days | Sept 2023 - Aug 2024 | 202 days | Aug 2024 - Oct 2025 |
| 70% | 353 days | Sept 2023 - Oct 2024 | 151 days | Oct 2024 - Oct 2025 |
| **80%** | **403 days** | **Sept 2023 - Jul 2025** | **101 days** | **Jul 2025 - Oct 2025** |
| 90% | 454 days | Sept 2023 - Aug 2025 | 50 days | Aug 2025 - Oct 2025 |

## Why Adjust the Split?

### Use 60-70% Training (More Test Data) When:
✅ You want **more test trades** to evaluate performance  
✅ You want **higher confidence** in test metrics  
✅ You have **plenty of data** (5Y timeframe)  
✅ You want to see **how the model performs over longer test periods**

### Use 80-90% Training (Less Test Data) When:
✅ You have **limited data** (1Y timeframe)  
✅ Your model needs **more examples** to learn patterns  
✅ You want **longer training history** for better learning  
✅ You're okay with **fewer test samples** for validation

## Real-World Scenario

### Problem: "I only have 1 test trade!"

**Before (80% split with 2Y data):**
- Training: Sept 2023 → Jul 2025 (403 days)
- Testing: Aug 2025 → Oct 2025 (101 days, but only 8 have occurred)
- **Result**: 1 test trade (most of test period is in the future!)

**After (70% split with 2Y data):**
- Training: Sept 2023 → Oct 2024 (353 days)
- Testing: Oct 2024 → Oct 2025 (151 days, all have occurred!)
- **Result**: ~8-15 test trades (entire test period has happened!)

## Best Practices

### For Historical Analysis (Past Data)
- Use **60-70% split** to maximize test data
- The entire test period will have occurred
- More test trades = better evaluation

### For Live Trading (Current Data)
- Use **80-90% split** to maximize training data
- Accept that some test period may not have occurred yet
- Focus on model accuracy metrics rather than test trade count

### For Long Timeframes (5Y)
- Use **70% split** (3.5Y train, 1.5Y test)
- Plenty of data for both training and testing
- More robust evaluation

### For Short Timeframes (1Y)
- Use **80-85% split** (better learning with limited data)
- Accept smaller test set
- Consider using longer timeframe instead

## Technical Details

### Code Location
- **UI Control**: `app_new.py` lines 400-411
- **Parameter Passing**: Lines 450, 479
- **TradingSystem**: `core/trading_system.py` line 38

### Default Behavior
- If no split specified: defaults to 0.8 (80/20)
- UI slider: defaults to 0.80
- Time-series chronological split (no shuffling)

## Visual Feedback

The UI now shows:
1. **Split percentage** in the slider
2. **Caption** showing "📊 Split: 80% training, 20% testing"
3. **Test Days Elapsed** metric showing "8/60 days" (how much of test period has occurred)
4. **Period column** in Trade History showing TRAIN vs 🎯 TEST for each trade

## Impact on Metrics

Changing the split affects:
- **Test Accuracy**: More test samples = more reliable accuracy
- **Test Trades**: More test days = more executed trades
- **Training Quality**: More training data = potentially better learning
- **Validation Confidence**: Larger test set = higher confidence in results

## Recommendations

| Your Goal | Recommended Split | Reasoning |
|-----------|------------------|-----------|
| Evaluate historical performance | 60-70% | Maximize test trades |
| Train for live trading | 80-90% | Maximize training data |
| General research | 80% | Industry standard |
| Limited data (1Y) | 80-85% | Need more training examples |
| Abundant data (5Y) | 70-75% | Can afford more test data |

## Example Usage

1. **Default**: Leave at 80% for standard evaluation
2. **More test trades**: Move slider to 70% (30% test)
3. **More training**: Move slider to 85% (15% test)
4. **Maximum validation**: Move slider to 60% (40% test)

## Related Files

- `app_new.py` - UI implementation
- `core/trading_system.py` - Split logic
- `docs/MODEL_PREDICTIONS_VS_EXECUTED_TRADES.md` - Understanding train/test concepts

---

**Updated**: 2025-11-01  
**Feature**: User-controllable train/test split ratio

