# Model Predictions vs Executed Trades

## The Problem

**Users often get confused why they see very few trades in the test period, even with a low signal threshold.**

### Example Scenario:
- User sets **Signal Threshold = 0.01%** (very low)
- User selects **2Y timeframe** (plenty of data)
- User expects: **Many signals in the 61-day test period**
- User sees: **Only 1 SELL signal in the chart** 😕

## The Explanation

There's a critical distinction between:

### 1. **Model Predictions** (What the AI thinks)
The model makes a prediction for **every single day**:
- **BUY (1)**: Expects price to rise ≥ threshold
- **HOLD (0)**: Expects price movement < threshold  
- **SELL (-1)**: Expects price to drop ≥ threshold

With a 0.01% threshold and 61 test days, the model might predict:
- 25 BUY signals
- 20 SELL signals
- 16 HOLD signals

### 2. **Executed Trades** (What actually happens in backtest)
The backtest can **only execute trades when conditions allow**:

#### BUY Signal Requirements:
```python
if signal == 1 and self.position_type is None:
    # Execute BUY
```
✅ **Can only BUY if you have NO open position**

#### SELL Signal Requirements:
```python
elif signal == -1 and self.position_type == 'LONG':
    # Execute SELL
```
✅ **Can only SELL if you have an OPEN position**

---

## Why You Only See 1 Trade

Even if the model predicts **20 SELL signals** in the test period:

```
Day 1: Model predicts SELL (-1)
       → No open position → Cannot execute ❌

Day 5: Model predicts BUY (1)
       → No position → Execute BUY ✅

Day 6-12: Model predicts SELL (-1)
          → Have open position → Execute SELL ✅ (only once!)

Day 13-60: Model predicts SELL (-1)
           → No position (already sold) → Cannot execute ❌
```

**Result:** Only 1 SELL trade appears in the chart, even though the model predicted SELL 14 times!

---

## How to See All Predictions

The new **🔮 Model Predictions** tab shows:
- **ALL raw model outputs** (not just executed trades)
- Every single BUY/SELL prediction the model makes
- Helps you understand model behavior vs backtest constraints

### Comparison:

| Tab | What It Shows | Purpose |
|-----|---------------|---------|
| **🎯 Executed Trades** | Only trades that were actually executed | See backtest performance |
| **🔮 Model Predictions** | Every prediction the model makes | See model behavior and signal frequency |

---

## Why This Design Makes Sense

### Real-World Trading Constraints
The backtest mimics **real trading**:
- You can't BUY if you already own the stock (no double-buying)
- You can't SELL if you don't own the stock (no short selling in this system)
- You can only execute ONE trade at a time per position

### Strategy Implications
If you want **more frequent trading**:
1. **Lower threshold** → More BUY/SELL predictions (but not necessarily more trades)
2. **Adjust position sizing** → Faster entries/exits
3. **Multiple positions** → Trade different stocks simultaneously (future feature)

---

## Key Takeaways

1. ✅ **Low threshold = More predictions**, not necessarily more executed trades
2. ✅ **Executed trades are limited by position state** (have stock or not)
3. ✅ **Use Model Predictions tab** to see full model behavior
4. ✅ **Executed Trades tab shows realistic backtest** results
5. ✅ **Both views are important** for different reasons

---

## Technical Details

### Label Generation
```python
def generate_labels(df, forward_days=3, threshold=0.01):
    future_close = df['Close'].shift(-forward_days)
    forward_return = (future_close - df['Close']) / df['Close']
    
    # Labels based on threshold
    labels[forward_return >= threshold] = 1   # BUY
    labels[forward_return <= -threshold] = -1  # SELL
    labels[between -threshold and +threshold] = 0  # HOLD
```

### Trade Execution
```python
def execute_trade(signal):
    # Can only BUY if no position
    if signal == 1 and position_type is None:
        buy_stock()
    
    # Can only SELL if have position
    elif signal == -1 and position_type == 'LONG':
        sell_stock()
    
    # Otherwise, do nothing (HOLD)
```

---

## Related Files

- **UI Implementation**: `app_new.py` (lines 76-174)
  - `create_predictions_chart()` - Shows all predictions
  - `create_trade_markers_chart()` - Shows executed trades
- **Backtest Logic**: `core/backtest/portfolio_simulator.py` (lines 169-290)
  - `execute_trade()` - Handles position constraints
- **Label Generation**: `core/labels/label_generator.py`
  - `generate_labels()` - Creates training labels

---

## Future Improvements

Potential enhancements to increase trade frequency:
1. **Multi-stock portfolio**: Trade 5-10 stocks simultaneously
2. **Short selling**: Allow SELL without owning (with margin requirements)
3. **Partial positions**: Buy/sell in increments (25%, 50%, 75%, 100%)
4. **Options trading**: Use options for leverage without full position

---

## FAQ

**Q: Why is my threshold so low but I still have few trades?**  
A: Low threshold affects **predictions**, not **execution**. Trades are limited by position state.

**Q: Should I increase my threshold to get more trades?**  
A: No! Higher threshold = FEWER predictions. Check the Model Predictions tab first.

**Q: How do I get more frequent trading?**  
A: Trade multiple stocks, use shorter forward_days, or implement partial position sizing.

**Q: Is this a bug?**  
A: No, this is realistic trading simulation. You can't buy what you already own or sell what you don't have!

---

## Updated: 2025-11-01
**Feature:** Added Model Predictions tab to visualize all predictions vs executed trades

