# Initial Capital & Position Sizing Fix

## Problem Identified

**User Question:** "Is the initial capital value hardcoded to 10000?"

**Investigation Results:**
- ✅ UI has configurable input in Advanced Settings
  - Range: $1,000 - $100,000
  - Default: $10,000
  - Step: $1,000
- ❌ **BUG:** The value was NOT being passed to the trading system!
- ❌ System always used hardcoded default: $10,000

**Same issue with Position Sizing:**
- ✅ UI has dropdown: `kelly`, `fixed`, `equal`
- ❌ Not being passed to trading system

---

## Root Cause

### The UI Collected the Values:
```python
# app_new.py - Line 299
initial_capital = st.number_input(
    "Initial Capital ($)",
    min_value=1000,
    max_value=100000,
    value=10000,
    step=1000
)

position_sizing = st.selectbox(
    "Position Sizing",
    options=['kelly', 'fixed', 'equal'],
    index=0
)
```

### But TradingSystem Didn't Accept Them:
```python
# OLD: core/trading_system.py - __init__
def __init__(
    self,
    ticker: str,
    timeframe: str,
    model_type: str = 'xgboost',
    forward_days: int = 5,
    threshold: float = 0.02,
    train_split: float = 0.8,
    # ❌ Missing: initial_capital, position_sizing
):
```

### And run_backtest Used Defaults:
```python
# OLD: core/trading_system.py - run_complete_pipeline
backtest_results = self.run_backtest(signals, confidences, verbose=verbose)
# ❌ Not passing initial_capital or position_sizing
```

---

## Solution Implemented

### 1. Updated TradingSystem Constructor

**File:** `core/trading_system.py`

```python
def __init__(
    self,
    ticker: str,
    timeframe: str,
    model_type: str = 'xgboost',
    forward_days: int = 5,
    threshold: float = 0.02,
    train_split: float = 0.8,
    initial_capital: float = 10000.0,      # ✅ NEW
    position_sizing: str = 'kelly'          # ✅ NEW
):
    # Store parameters
    self.initial_capital = initial_capital
    self.position_sizing = position_sizing
```

### 2. Updated run_complete_pipeline

**File:** `core/trading_system.py`

```python
# Step 7: Backtest
backtest_results = self.run_backtest(
    signals, 
    confidences, 
    initial_capital=self.initial_capital,    # ✅ Pass stored value
    position_sizing=self.position_sizing,     # ✅ Pass stored value
    verbose=verbose
)
```

### 3. Updated UI to Pass Parameters

**File:** `app_new.py`

```python
# For all 3 models comparison
system = TradingSystem(
    ticker=ticker,
    timeframe=timeframe,
    model_type=model_name,
    forward_days=forward_days,
    threshold=threshold,
    train_split=0.8,
    initial_capital=initial_capital,    # ✅ Now passed
    position_sizing=position_sizing      # ✅ Now passed
)

# For single model
system = TradingSystem(
    ticker=ticker,
    timeframe=timeframe,
    model_type=model_type,
    forward_days=forward_days,
    threshold=threshold,
    train_split=0.8,
    initial_capital=initial_capital,    # ✅ Now passed
    position_sizing=position_sizing      # ✅ Now passed
)
```

---

## Testing Instructions

### Before Fix:
1. Open Advanced Settings → Change Initial Capital to $50,000
2. Run Analysis
3. Check Final Value → Still based on $10,000 ❌

### After Fix:
1. **Restart Streamlit** (to load new code)
2. Open Advanced Settings → Change Initial Capital to $50,000
3. Change Position Sizing to `fixed` (instead of `kelly`)
4. Click "Run Analysis"
5. Check results:
   - **Final Value should be ~$140,000+** (5x higher than before) ✅
   - Position sizing should follow `fixed` strategy ✅

---

## Impact

### Initial Capital Examples:

| Setting | $10K (default) | $50K | $100K |
|---------|----------------|------|-------|
| **Returns:** +182% | $28,206 | $141,000 | $282,000 |
| **Drawdown:** -11% | -$1,100 | -$5,500 | -$11,000 |

**Note:** Returns percentage stays same, but absolute $ values scale linearly.

### Position Sizing Options:

| Method | Description | When to Use |
|--------|-------------|-------------|
| **kelly** (default) | Size positions based on edge & confidence | Best for optimizing growth |
| **fixed** | Same dollar amount per trade | Conservative, predictable |
| **equal** | Same % of portfolio per trade | Balanced approach |

---

## Files Changed

1. **`core/trading_system.py`**:
   - Added `initial_capital` and `position_sizing` to `__init__`
   - Updated `run_complete_pipeline` to pass these to `run_backtest`

2. **`app_new.py`**:
   - Updated both `TradingSystem` instantiation points (all models + single model)
   - Now passes `initial_capital` and `position_sizing` from UI inputs

---

## Backward Compatibility

✅ **Fully backward compatible!**

Both parameters have defaults:
- `initial_capital=10000.0`
- `position_sizing='kelly'`

Old code that doesn't pass these will still work with defaults.

---

## Related Features

### Initial Capital Range:
- **Min:** $1,000 (for small accounts)
- **Max:** $100,000 (UI limit, code can handle more)
- **Step:** $1,000
- **Recommended:** 
  - Beginners: $10,000-$25,000
  - Experienced: $50,000-$100,000

### Position Sizing Methods:
1. **Kelly Criterion** (`kelly`):
   - Optimal bet sizing based on win probability
   - Can be aggressive (use `kelly_fraction=0.5` for half-Kelly)
   - Best for: Maximizing long-term growth

2. **Fixed** (`fixed`):
   - Same dollar amount per trade
   - Simpler, more conservative
   - Best for: Risk-averse traders

3. **Equal** (`equal`):
   - Same % of portfolio per trade
   - Balanced approach
   - Best for: Diversified portfolios

---

## Future Enhancements

1. **Dynamic Capital Scaling**: Allow capital to grow/shrink with profits/losses
2. **Custom Position Limits**: Set max $ or % per trade
3. **Capital Allocation**: Split capital across multiple tickers
4. **Risk Per Trade**: Set max % of capital at risk per trade

---

**Status:** ✅ **FIXED** - Initial capital and position sizing now fully configurable!

