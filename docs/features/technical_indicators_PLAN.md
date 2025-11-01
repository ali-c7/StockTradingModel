# Technical Indicators Computation - Technical Plan

## Overview
Calculate real technical indicators (RSI, MACD, Bollinger Bands) from historical stock data using the `ta` library. Replace mock indicator values with actual calculations based on price data.

## Files to Create/Modify

### 1. `data/indicators/indicators_data.py` (CREATE)
Module for calculating technical indicators from price data.

**Functions to implement:**

#### `calculate_rsi(df: pd.DataFrame, period: int = 14) -> float`
Calculate Relative Strength Index for the most recent period.

**Parameters:**
- `df`: DataFrame with OHLCV data
- `period`: RSI period (default: 14 days)

**Returns:**
- Current RSI value (0-100)

**Interpretation:**
- RSI > 70: Overbought (potential sell signal)
- RSI < 30: Oversold (potential buy signal)
- RSI 30-70: Neutral

**Implementation:**
```python
from ta.momentum import RSIIndicator

def calculate_rsi(df, period=14):
    rsi_indicator = RSIIndicator(close=df['Close'], window=period)
    rsi_series = rsi_indicator.rsi()
    return rsi_series.iloc[-1] if not rsi_series.empty else 50.0
```

#### `calculate_macd(df: pd.DataFrame) -> dict`
Calculate MACD (Moving Average Convergence Divergence) indicators.

**Parameters:**
- `df`: DataFrame with OHLCV data

**Returns:**
```python
{
    "macd": float,           # MACD line value
    "macd_signal": float,    # Signal line value
    "macd_diff": float,      # Histogram (MACD - Signal)
    "trend": str             # "bullish" or "bearish"
}
```

**Interpretation:**
- MACD > Signal: Bullish (potential buy)
- MACD < Signal: Bearish (potential sell)
- Histogram growing: Strengthening trend
- Histogram shrinking: Weakening trend

**Implementation:**
```python
from ta.trend import MACD

def calculate_macd(df):
    macd = MACD(close=df['Close'])
    macd_line = macd.macd().iloc[-1]
    signal_line = macd.macd_signal().iloc[-1]
    macd_diff = macd.macd_diff().iloc[-1]
    
    return {
        "macd": macd_line,
        "macd_signal": signal_line,
        "macd_diff": macd_diff,
        "trend": "bullish" if macd_line > signal_line else "bearish"
    }
```

#### `calculate_bollinger_bands(df: pd.DataFrame, period: int = 20, std_dev: int = 2) -> dict`
Calculate Bollinger Bands indicators.

**Parameters:**
- `df`: DataFrame with OHLCV data
- `period`: Moving average period (default: 20)
- `std_dev`: Standard deviation multiplier (default: 2)

**Returns:**
```python
{
    "bb_upper": float,      # Upper band
    "bb_middle": float,     # Middle band (SMA)
    "bb_lower": float,      # Lower band
    "bb_width": float,      # Band width (volatility measure)
    "position": str         # "above", "below", or "within"
}
```

**Interpretation:**
- Price > Upper Band: Overbought
- Price < Lower Band: Oversold
- Price within bands: Normal range
- Width expanding: Increasing volatility
- Width contracting: Decreasing volatility

**Implementation:**
```python
from ta.volatility import BollingerBands

def calculate_bollinger_bands(df, period=20, std_dev=2):
    bb = BollingerBands(close=df['Close'], window=period, window_dev=std_dev)
    
    upper = bb.bollinger_hband().iloc[-1]
    middle = bb.bollinger_mavg().iloc[-1]
    lower = bb.bollinger_lband().iloc[-1]
    current_price = df['Close'].iloc[-1]
    
    # Determine position
    if current_price > upper:
        position = "Above Upper Band"
    elif current_price < lower:
        position = "Below Lower Band"
    else:
        position = "Within Bands"
    
    return {
        "bb_upper": upper,
        "bb_middle": middle,
        "bb_lower": lower,
        "bb_width": upper - lower,
        "position": position
    }
```

#### `calculate_all_indicators(df: pd.DataFrame) -> dict`
Calculate all indicators at once for efficiency.

**Returns:**
```python
{
    "rsi": float,
    "rsi_status": str,      # "Overbought", "Oversold", "Neutral"
    "macd": dict,
    "bollinger": dict
}
```

### 2. `app.py` (MODIFY)
Replace mock indicator values with real calculations.

**Changes:**

1. Import indicator functions
2. Calculate indicators after fetching data
3. Replace mock values in results dictionary

**Current (Mock):**
```python
mock_signal_data = generate_mock_results(ticker_input, timeframe_value)
# Contains: rsi (random), macd (random), bb_status (random)
```

**New (Real):**
```python
# Calculate real indicators
indicators = calculate_all_indicators(stock_df)

# Use real values
results = {
    "rsi": indicators["rsi"],
    "rsi_status": indicators["rsi_status"],
    "macd": indicators["macd"]["macd"],
    "macd_diff": indicators["macd"]["macd_diff"],
    "macd_trend": indicators["macd"]["trend"],
    "bb_status": indicators["bollinger"]["position"],
    "bb_upper": indicators["bollinger"]["bb_upper"],
    "bb_lower": indicators["bollinger"]["bb_lower"],
    # ... other real data
}
```

3. Update display to show real indicator values

## Technical Indicator Details

### RSI (Relative Strength Index)
- **Type**: Momentum oscillator
- **Range**: 0-100
- **Formula**: RS = Average Gain / Average Loss over period
- **Common period**: 14 days
- **Use**: Identify overbought/oversold conditions

### MACD (Moving Average Convergence Divergence)
- **Type**: Trend-following momentum indicator
- **Components**:
  - MACD Line: 12-day EMA - 26-day EMA
  - Signal Line: 9-day EMA of MACD
  - Histogram: MACD - Signal
- **Use**: Identify trend changes and momentum

### Bollinger Bands
- **Type**: Volatility indicator
- **Components**:
  - Upper Band: SMA + (2 × STD)
  - Middle Band: 20-day SMA
  - Lower Band: SMA - (2 × STD)
- **Use**: Identify overbought/oversold and volatility

## Error Handling

### Insufficient Data
```python
if len(df) < 30:  # Need minimum data for indicators
    return default_values
```

### Calculation Errors
```python
try:
    rsi = calculate_rsi(df)
except Exception as e:
    st.warning(f"Could not calculate RSI: {e}")
    rsi = 50.0  # Neutral default
```

## Dependencies

- **ta** (already in requirements.txt)
- Uses pandas DataFrames
- No new dependencies needed

## Testing

**Test cases:**
- [ ] RSI calculated correctly (compare with known values)
- [ ] MACD values match expected results
- [ ] Bollinger Bands positioned correctly
- [ ] Works with different timeframes
- [ ] Handles edge cases (insufficient data)
- [ ] Display shows real values (not mock)
- [ ] Status interpretations correct

## Display Updates

### Current Display (Mock):
```
RSI: 65.2 (random)
Status: ✅ Neutral range
```

### New Display (Real):
```
RSI: 68.4 (calculated from actual prices)
Status: ⚠️ Approaching overbought
```

## Benefits

- ✅ Real analysis based on actual market data
- ✅ Industry-standard calculations
- ✅ Properly interpreted signals
- ✅ Foundation for Phase 3 (real predictions)
- ✅ Users can trust the indicators

## Notes

- Keep signal generation (Buy/Sell/Hold) as mock for now (Phase 3)
- Indicators are the foundation for the prediction model
- `ta` library provides tested, reliable calculations
- Common parameters used (RSI=14, MACD=12/26/9, BB=20/2)
- Can be customized later if needed


