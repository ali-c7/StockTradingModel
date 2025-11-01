# Ticker List & Searchable Dropdown - Technical Plan

## Overview
Implement a searchable dropdown for ticker selection by fetching a curated list of common stock tickers at app startup. This improves UX by providing suggestions and validation, while still allowing manual entry.

## Files to Create/Modify

### 1. `data/tickers/ticker_list_data.py` (CREATE)
Module for fetching and managing ticker lists.

**Functions to implement:**

#### `load_ticker_list() -> list[dict]`
Returns list of ticker dictionaries with company names.

**Implementation approach:**
Fetch S&P 500 tickers from Wikipedia using pandas:
- Source: https://en.wikipedia.org/wiki/List_of_S%26P_500_companies
- Includes ticker symbol and company name
- ~500 major US companies
- Yahoo Finance compatible tickers
- Cache for 24 hours to avoid repeated requests

**Return format:**
```python
[
    {"ticker": "AAPL", "name": "Apple Inc."},
    {"ticker": "GOOGL", "name": "Alphabet Inc. Class A"},
    {"ticker": "MSFT", "name": "Microsoft Corporation"},
    ...
]
```

**Implementation:**
```python
import pandas as pd

@st.cache_data(ttl=86400)  # Cache for 24 hours
def load_ticker_list():
    try:
        # Fetch S&P 500 list from Wikipedia
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        tables = pd.read_html(url)
        sp500_table = tables[0]
        
        # Extract ticker and company name
        tickers = []
        for _, row in sp500_table.iterrows():
            tickers.append({
                "ticker": row['Symbol'],
                "name": row['Security']
            })
        
        return tickers
    except Exception as e:
        # Fallback: return empty list and show warning
        st.warning("Could not load ticker list. You can still enter tickers manually.")
        return []
```

#### `get_ticker_options() -> list[str]`
Returns formatted list for selectbox display.
- Format: "AAPL - Apple Inc."
- Searchable by ticker or company name

#### `extract_ticker(selected_option: str) -> str`
Extracts ticker symbol from formatted string.
- Input: "AAPL - Apple Inc."
- Output: "AAPL"

#### `@st.cache_data` decorator
Cache ticker list to avoid reloading on every rerun.

### 2. `app.py` (MODIFY)
Update ticker input section to use searchable dropdown.

**Changes needed:**

#### Replace text input with selectbox
Current:
```python
ticker_input = st.text_input(...)
```

New:
```python
ticker_options = get_ticker_options()
# Add "Type custom ticker..." as first option
ticker_options.insert(0, "Type custom ticker...")

selected = st.selectbox(
    "Stock Ticker Symbol",
    options=ticker_options,
    help="Select a ticker or choose 'Type custom ticker...' to enter manually"
)

# If custom option selected, show text input
if selected == "Type custom ticker...":
    ticker_input = st.text_input(
        "Enter custom ticker",
        max_chars=10,
        placeholder="e.g., AAPL"
    ).upper()
else:
    ticker_input = extract_ticker(selected)
```

#### Update validation
- Validation still applies to custom entries
- Dropdown selections are pre-validated

## Implementation Details

### Fetching Ticker List from Wikipedia
Use pandas `read_html()` to scrape S&P 500 companies:

```python
import pandas as pd
import streamlit as st

@st.cache_data(ttl=86400)  # Cache for 24 hours
def load_ticker_list():
    """
    Fetch S&P 500 ticker list from Wikipedia
    
    Returns:
        List of dictionaries with ticker and company name
    """
    try:
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        tables = pd.read_html(url)
        sp500_table = tables[0]  # First table contains the companies
        
        tickers = []
        for _, row in sp500_table.iterrows():
            tickers.append({
                "ticker": row['Symbol'],
                "name": row['Security']
            })
        
        return sorted(tickers, key=lambda x: x['ticker'])
    except Exception as e:
        st.warning(f"Could not load ticker list: {str(e)}. You can still enter tickers manually.")
        return []
```

**Benefits:**
- Always up-to-date with current S&P 500 composition
- Includes all ~500 major US companies
- Yahoo Finance compatible tickers
- Company names for better UX

### Searchable Functionality
Streamlit's `st.selectbox` is searchable by default:
- User types "APP" → "AAPL - Apple Inc." appears
- User types "Apple" → also finds "AAPL - Apple Inc."
- No additional code needed

### Caching Strategy
```python
@st.cache_data(ttl=None)  # Cache forever (static list)
def load_ticker_list():
    return POPULAR_TICKERS
```

## User Experience Flow

### Standard Flow
1. User opens app
2. Ticker list loads (cached after first load)
3. User clicks dropdown
4. User types "AAPL" or "Apple"
5. List filters to matching entries
6. User selects "AAPL - Apple Inc."
7. Click Analyze

### Custom Ticker Flow
1. User clicks dropdown
2. Selects "Type custom ticker..."
3. Text input field appears
4. User enters custom ticker (e.g., "BTC-USD")
5. Validation runs
6. Click Analyze

## Session State Updates

No new session state variables needed.
Ticker value stored in existing `st.session_state.ticker`.

## Error Handling

- Invalid custom ticker → Show validation error (existing logic)
- Empty selection → Validation catches it
- Delisted ticker in list → Yahoo Finance API will catch it

## Benefits

- **Better UX**: Easy to find tickers without memorizing symbols
- **Validation**: Pre-validated tickers reduce errors
- **Discovery**: Users can browse available stocks
- **Flexibility**: Still allows custom entries for crypto, forex, etc.
- **Fast**: Cached list loads instantly

## Alternative Data Sources

### Selected: Wikipedia S&P 500 List (Implemented)
- ✅ Comprehensive (~500 companies)
- ✅ Always up-to-date
- ✅ Yahoo Finance compatible
- ✅ Includes company names
- ⚠️ Requires network access on first load
- ⚠️ Cached for 24 hours to minimize requests

### Alternative: NASDAQ-listed Tickers
Could fetch from: `https://www.nasdaq.com/market-activity/stocks/screener`
- More comprehensive (includes all exchanges)
- More complex parsing
- Slower to load

### Alternative: Multiple Indices
Could combine S&P 500 + NASDAQ 100 + Dow Jones:
- Most comprehensive
- Longer load time
- More complex implementation

**Recommendation**: S&P 500 from Wikipedia provides excellent coverage for most use cases.

## Dependencies

- **pandas** (already included) - for `read_html()` to scrape Wikipedia
- **lxml** or **html5lib** (NEW) - required by pandas `read_html()`
  - Add to requirements.txt: `lxml>=4.9.0`
- **requests** (already included via streamlit) - for HTTP requests

**Update requirements.txt:**
```txt
lxml>=4.9.0  # Required for pandas read_html
```

## File Structure
```
data/
  tickers/
    __init__.py
    ticker_list_data.py
```

## Testing

- [ ] Dropdown displays correctly
- [ ] Search filters results
- [ ] Selected ticker extracts properly
- [ ] Custom ticker option works
- [ ] Validation works for custom entries
- [ ] Caching works (no reload on rerun)
- [ ] Works with existing analyze flow

## Notes

- S&P 500 covers most popular US stocks users will search for
- List automatically updates when S&P 500 composition changes
- Cache expires after 24 hours, ensuring reasonably fresh data
- Custom ticker option handles: crypto (BTC-USD), forex, ETFs, international stocks
- First load may take 1-2 seconds to fetch from Wikipedia
- Subsequent loads are instant (cached)
- Consider adding sector/category filters in future phases
- Could expand to multiple indices (NASDAQ 100, Russell 2000) in Phase 5

