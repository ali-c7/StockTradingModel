# Data Retrieval Module - Technical Plan

## Overview
Implement Yahoo Finance data fetching to retrieve real historical stock data (OHLCV) for the selected ticker and timeframe. Replace mock data with actual market data.

## Files to Create/Modify

### 1. `data/stock/stock_data.py` (CREATE)
Module for fetching and managing stock data from Yahoo Finance.

**Functions to implement:**

#### `calculate_date_range(timeframe: str) -> tuple[str, str]`
Calculate start and end dates based on timeframe selection.

**Parameters:**
- `timeframe`: "1M", "3M", "6M", "1Y", "2Y", "5Y"

**Returns:**
- Tuple of (start_date, end_date) as strings "YYYY-MM-DD"

**Implementation:**
```python
from datetime import datetime, timedelta

def calculate_date_range(timeframe: str) -> tuple[str, str]:
    end_date = datetime.now()
    
    timeframe_map = {
        "1M": timedelta(days=30),
        "3M": timedelta(days=90),
        "6M": timedelta(days=180),
        "1Y": timedelta(days=365),
        "2Y": timedelta(days=730),
        "5Y": timedelta(days=1825)
    }
    
    delta = timeframe_map.get(timeframe, timedelta(days=180))
    start_date = end_date - delta
    
    return start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")
```

#### `fetch_stock_data(ticker: str, timeframe: str) -> pd.DataFrame | None`
Fetch historical OHLCV data from Yahoo Finance.

**Parameters:**
- `ticker`: Stock ticker symbol (e.g., "AAPL")
- `timeframe`: Analysis timeframe ("1M", "3M", etc.)

**Returns:**
- DataFrame with columns: Date (index), Open, High, Low, Close, Volume
- None if fetch fails

**Implementation:**
```python
import yfinance as yf

@st.cache_data(ttl=3600)  # Cache for 1 hour
def fetch_stock_data(ticker: str, timeframe: str) -> pd.DataFrame | None:
    try:
        start_date, end_date = calculate_date_range(timeframe)
        
        # Download data from Yahoo Finance
        stock = yf.Ticker(ticker)
        df = stock.history(start=start_date, end=end_date)
        
        if df.empty:
            return None
        
        # Ensure we have required columns
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in df.columns for col in required_cols):
            return None
        
        return df[required_cols]
    
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {str(e)}")
        return None
```

#### `get_current_price(ticker: str) -> dict | None`
Get current price and change for display.

**Returns:**
```python
{
    "price": 150.25,
    "change": 2.50,
    "change_pct": 1.69,
    "volume": "50M"
}
```

#### `validate_ticker_data(df: pd.DataFrame, ticker: str) -> bool`
Validate that fetched data is usable.

**Checks:**
- DataFrame not empty
- Has minimum number of data points (at least 5 days)
- No all-NaN columns
- Reasonable price values (> 0)

### 2. `app.py` (MODIFY)
Replace mock data generation with real data fetching.

**Changes:**
1. Import stock data functions
2. Replace `generate_mock_results()` call with real data fetch
3. Update results structure to use real data
4. Handle fetch failures gracefully

## Data Structure

### DataFrame from Yahoo Finance
```
Date (index)  | Open    | High    | Low     | Close   | Volume
2024-05-01    | 170.00  | 172.50  | 169.00  | 171.25  | 50000000
2024-05-02    | 171.50  | 173.00  | 170.50  | 172.75  | 48000000
...
```

### Results Dictionary (Updated)
```python
{
    "ticker": "AAPL",
    "current_price": 172.75,
    "price_change": 2.50,
    "price_change_pct": 1.47,
    "volume": "48M",
    "data": df,  # Full OHLCV DataFrame
    "signal": "BUY",  # Still mock for now
    "confidence": 85.0,  # Still mock for now
    "rsi": 65.0,  # Still mock for now
    "macd": 1.5,  # Still mock for now
    "bb_status": "Within Bands",  # Still mock for now
    "reasoning": "...",  # Still mock for now
    "last_updated": "2024-11-01 21:00:00"
}
```

## Implementation Strategy

### Phase 2.1a: Basic Data Fetching
1. Create stock data module
2. Implement date calculation
3. Implement basic fetch function
4. Test with single ticker

### Phase 2.1b: Integration
1. Update app.py to use real data
2. Replace mock prices with real prices
3. Handle errors gracefully
4. Keep mock signals for now (Phase 3)

### Phase 2.1c: Validation & Caching
1. Add data validation
2. Implement caching strategy
3. Add error handling for edge cases
4. Test with various tickers

## Caching Strategy

**Stock Data Cache:**
- TTL: 1 hour (3600 seconds)
- Reason: Market data changes during trading hours
- After hours: Data doesn't change, cache saves API calls

**Cache Key:** Combination of ticker + timeframe + date
- Different tickers have separate caches
- Different timeframes have separate caches
- Cache expires after 1 hour

## Error Handling

### Invalid Ticker
- yfinance returns empty DataFrame
- Show error: "Invalid ticker symbol. Please try again."
- Keep analysis_triggered = False
- Allow user to try again

### Network Error
- Catch connection exceptions
- Show error: "Network error. Please check your connection."
- Suggest retry

### No Data Available
- Ticker valid but no data for timeframe
- Show error: "No data available for this timeframe."
- Suggest different timeframe

### Delisted/Suspended Stock
- Empty DataFrame returned
- Show warning: "This ticker may be delisted or suspended."

## Dependencies

- **yfinance** (already in requirements.txt)
- Uses pandas (already included)
- Uses datetime (Python standard library)

## Testing

**Test cases:**
- [ ] Valid ticker (AAPL) → Data fetched successfully
- [ ] Invalid ticker (INVALID123) → Error handled
- [ ] Different timeframes → Correct date ranges
- [ ] Caching works → Second call instant
- [ ] Network error → Graceful handling
- [ ] Empty data → Proper error message
- [ ] Crypto ticker (BTC-USD) → Works
- [ ] International ticker (0700.HK) → Works

## Notes

- Keep signal generation (Buy/Sell/Hold) as mock for now (Phase 3)
- Technical indicators (RSI, MACD, BB) still mock (Phase 2.4)
- Focus on getting real price data and charts working
- Price data is foundation for all future analysis

