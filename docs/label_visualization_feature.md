# Label Visualization Feature

## Overview
Visual debugging tool to understand how the ML model creates training labels and identify potential issues.

## Purpose
Stock prediction models often fail due to:
1. **Class imbalance** - Too many HOLD, not enough BUY/SELL
2. **Poor label quality** - Labels don't match actual price patterns
3. **Insufficient data** - Not enough samples for each class
4. **Threshold issues** - Threshold too high/low for stock volatility

This visualization helps identify these issues **before** you waste time training.

## What It Shows

### 1. Price Chart with Markers (Top Panel)
- **Green triangles ▲**: BUY signals (where model expects price to rise)
- **Red triangles ▼**: SELL signals (where model expects price to fall)
- **Orange dots ●**: HOLD signals (10% sample to avoid clutter)

**What to look for:**
- ✅ **Good**: Buys at local minima, sells at local maxima
- ❌ **Bad**: Random distribution, buys at peaks, sells at troughs

### 2. Label Distribution Over Time (Middle Panel)
Stacked bar chart showing how labels are distributed by month.

**What to look for:**
- ✅ **Good**: Relatively balanced (around 30-40% each)
- ⚠️ **Warning**: One class dominates (e.g., 80% HOLD)
- ❌ **Bad**: 95% HOLD, 2.5% BUY, 2.5% SELL

### 3. Future Returns (Bottom Panel)
Scatter plot of actual future returns that created the labels.

**What to look for:**
- Green dots above +1% line = BUY labels
- Red dots below -1% line = SELL labels
- Orange dots in middle = HOLD labels
- Threshold lines show the cutoffs (default ±1%)

**Insights:**
- If most dots are clustered near zero → Stock has low volatility → Increase threshold
- If dots are spread out → Stock is volatile → Current threshold is good
- If you see many dots crossing threshold lines → Labels are capturing real movements ✅

## How to Use

### In the UI:
1. Check **"Show label visualization"** in the sidebar (under Model Validation)
2. Run an analysis
3. Scroll to the bottom to see the visualization

### What the Metrics Tell You:

**Total Labels**: Should be at least 400+ for good training

**Label Distribution**:
- **Ideal**: ~33% BUY, ~33% HOLD, ~33% SELL
- **Acceptable**: 20-45% each
- **Problem**: <15% or >60% for any class

**Imbalance Score**:
- **< 60**: ✅ Good balance
- **60-100**: ⚠️ Moderate imbalance (acceptable)
- **> 100**: ❌ Severe imbalance (will hurt performance)

## Common Problems and Solutions

### Problem 1: 90% HOLD, 5% BUY, 5% SELL
**Cause**: Threshold too high for this stock's volatility  
**Solution**: Lower threshold from 1% to 0.5% or 0.7%

### Problem 2: BUY signals at price peaks
**Cause**: Look-ahead bias or inverted logic  
**Solution**: Check that forward_days is positive (not negative)

### Problem 3: Very few total labels (< 200)
**Cause**: Too much data lost to rolling windows  
**Solution**: Use longer timeframe (5Y instead of 6M)

### Problem 4: Clustered patterns (all buys in one month)
**Cause**: Stock had a major event or trend change  
**Solution**: This is actually normal! Stock patterns change over time

## Technical Details

**Label Generation Logic:**
```python
forward_return = (price_in_3_days - price_today) / price_today

if forward_return >= +1%:  # threshold
    label = BUY (1)
elif forward_return <= -1%:  # -threshold
    label = SELL (-1)
else:
    label = HOLD (0)
```

**Parameters (coming in Phase 5):**
- `forward_days`: How far ahead to look (default: 3 days)
- `threshold`: Percentage for buy/sell (default: 1% = 0.01)

**Data Loss:**
- Last `forward_days` rows have no label (can't see future)
- Rows with NaN features are removed
- Typical loss: 3-5% of original data

## Example Interpretation

### Good Visualization (AAPL 2Y)
```
Total Labels: 450
BUY:  142 (31.6%)
HOLD: 166 (36.9%)
SELL: 142 (31.6%)
Imbalance Score: 5.4

✅ Well balanced
✅ Enough samples
✅ BUYs at local lows, SELLs at local highs
→ Model should perform well (50-65% accuracy)
```

### Bad Visualization (Low-volume penny stock)
```
Total Labels: 89
BUY:  4 (4.5%)
HOLD: 81 (91.0%)
SELL: 4 (4.5%)
Imbalance Score: 165

❌ Severe imbalance
❌ Insufficient samples
❌ Model will just predict HOLD every time
→ Use longer timeframe or different stock
```

## Future Enhancements (Phase 5)

- [ ] Adjustable threshold slider
- [ ] Adjustable forward_days
- [ ] Show feature importance overlay
- [ ] Compare labels vs. actual outcomes
- [ ] Export label data as CSV
- [ ] Interactive hover to see why each label was assigned

## Related Files

- **Implementation**: `core/utils/label_visualizer.py`
- **UI Integration**: `app.py` (lines 156-163, 397-403, 812-850)
- **Label Logic**: `core/models/label_generator.py`

## Performance Impact

- Minimal (< 1 second to generate)
- Only runs if checkbox is enabled
- Uses cached prediction data
- No impact on model training

