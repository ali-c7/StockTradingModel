# Indicator Validation Guide

## How to Validate Indicators with TradingView

Our app now displays all the information you need to validate indicators against TradingView or other charting platforms.

---

## What's Displayed

### 1. Calculation Context (Top of Indicators Section)
```
📅 Calculated from Daily data | Last update: 2025-10-31 | Compare with TradingView on 1D chart
```

This tells you:
- **Timeframe**: DAILY data (not hourly, not minute)
- **Last data point**: The exact date of the last data used
- **How to compare**: Use TradingView's 1D (Daily) chart

### 2. Indicator-Specific Settings

Each indicator now shows its calculation parameters:

**RSI**
```
RSI
68.4
⚙️ 14-period
✅ Neutral range (30-70)
```

**MACD**
```
MACD
-0.52 ↓-0.34
⚙️ 12/26/9 (Fast/Slow/Signal)
📉 Bearish signal
```

**Bollinger Bands**
```
Bollinger Bands
Within Bands
⚙️ 20-period, 2σ (StdDev)
✅ Normal range
Bands: $170.23 - $185.67
```

---

## Step-by-Step Validation

### On TradingView

1. **Search for the ticker** (e.g., AAPL, BTC-USD)

2. **Set chart to 1D (Daily)**
   - Click the timeframe dropdown
   - Select "1D" (NOT 1h, 4h, or 1W)

3. **Add RSI indicator**
   - Click "Indicators" → Search "RSI"
   - Settings: Length = **14**
   - Check the **rightmost value** on the chart

4. **Add MACD indicator**
   - Click "Indicators" → Search "MACD"
   - Settings: Fast = **12**, Slow = **26**, Signal = **9**
   - Check the **rightmost value** (MACD line and histogram)

5. **Add Bollinger Bands indicator**
   - Click "Indicators" → Search "Bollinger Bands"
   - Settings: Length = **20**, StdDev = **2**
   - Check the **current price position** relative to bands

### Compare Values

The values should match within **±0.1** due to:
- Rounding differences
- Data source timing (Yahoo Finance vs TradingView)
- Intraday updates (our app may show last close, TradingView may show real-time)

---

## Common Differences & Why

### Minor Price Differences (<$0.50)
**Cause**: Data refresh timing
- Our app: Uses Yahoo Finance API
- TradingView: Uses their own data feed
- **Solution**: Check the date/time - if it's the same day, values should be close

### RSI Differs by ~1-2 points
**Cause**: Rounding or calculation method variant
- Both use Wilder's smoothing, but implementation details vary slightly
- **Solution**: If within 2 points, it's normal variance

### MACD Differs Slightly
**Cause**: EMA calculation precision
- Exponential moving averages are sensitive to precision
- **Solution**: Trend direction (bullish/bearish) should match, even if exact values differ slightly

### Bollinger Bands Show Different Position
**Cause**: Price is near the band threshold
- Our app: Uses last close price
- TradingView: May use current real-time price
- **Solution**: If price is within 1-2% of a band, position status may differ

---

## Data Source: Yahoo Finance

Our app uses **yfinance** library which pulls data from Yahoo Finance:
- **Granularity**: Daily (1 bar = 1 trading day)
- **Update frequency**: Typically 15-20 minutes delayed during market hours
- **After-hours**: Shows last closing price
- **Historical**: Accurate back to listing date

### Ticker Format Differences

| Asset Type | Our Format | TradingView Format |
|------------|------------|-------------------|
| US Stocks | AAPL | AAPL |
| Crypto | BTC-USD | BTCUSD or BTC |
| Forex | EURUSD=X | EURUSD |
| Indices | ^GSPC | SPX or SPX500 |

---

## Quick Validation Checklist

- [ ] Set TradingView to **1D chart**
- [ ] Use **same ticker format** (add -USD for crypto in our app)
- [ ] Set RSI to **14**
- [ ] Set MACD to **12/26/9**
- [ ] Set Bollinger Bands to **20/2**
- [ ] Check **rightmost value** (latest day)
- [ ] Expect values within **±0.1 to ±2** range

---

## Still Not Matching?

If values are significantly different (>5% off):

1. **Check the date**: Make sure you're looking at the same day
2. **Check ticker symbol**: BTC ≠ BTC-USD (different assets!)
3. **Check market hours**: During trading, TradingView updates real-time, we show last close
4. **Check data source**: Some tickers have multiple data sources (different exchanges)
5. **Try another ticker**: If one ticker is off, test with AAPL or SPY (highly liquid, consistent data)

---

## Feedback

If you notice persistent discrepancies, please report them with:
- Ticker symbol
- Date of analysis
- Expected value (from TradingView)
- Actual value (from our app)
- Screenshot (if possible)

This helps us improve data accuracy!


