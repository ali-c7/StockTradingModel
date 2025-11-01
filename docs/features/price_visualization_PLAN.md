# Basic Price Visualization - Technical Plan

## Overview
Create interactive price charts using Plotly to visualize the historical stock data we're now fetching from Yahoo Finance. Replace the chart placeholder with real, interactive visualizations.

## Files to Create/Modify

### 1. `plots/stock/stock_plot.py` (CREATE)
Module for creating stock price visualizations.

**Functions to implement:**

#### `create_price_chart(df: pd.DataFrame, ticker: str) -> plotly.graph_objects.Figure`
Create interactive candlestick or line chart with volume subplot.

**Parameters:**
- `df`: DataFrame with OHLCV data (index=Date, columns=Open/High/Low/Close/Volume)
- `ticker`: Stock ticker symbol for title

**Returns:**
- Plotly Figure object with 2 subplots (price + volume)

**Features:**
- Main chart: Candlestick chart (preferred) or line chart
- Volume subplot below main chart
- Interactive features: zoom, pan, hover tooltips
- Clean styling with proper axes labels
- Responsive design

**Implementation approach:**
```python
from plotly.subplots import make_subplots
import plotly.graph_objects as go

def create_price_chart(df, ticker):
    # Create figure with 2 subplots
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.7, 0.3],
        subplot_titles=(f'{ticker} Price', 'Volume')
    )
    
    # Add candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name='Price'
        ),
        row=1, col=1
    )
    
    # Add volume bars
    colors = ['red' if df['Close'].iloc[i] < df['Open'].iloc[i] 
              else 'green' for i in range(len(df))]
    
    fig.add_trace(
        go.Bar(
            x=df.index,
            y=df['Volume'],
            name='Volume',
            marker_color=colors
        ),
        row=2, col=1
    )
    
    # Update layout
    fig.update_layout(
        xaxis_rangeslider_visible=False,
        height=600,
        hovermode='x unified',
        showlegend=False
    )
    
    return fig
```

#### `create_simple_line_chart(df: pd.DataFrame, ticker: str) -> plotly.graph_objects.Figure`
Simplified line chart (fallback if candlestick has issues).

**Features:**
- Single line showing closing price
- Volume subplot
- Cleaner for very large datasets

### 2. `app.py` (MODIFY)
Replace chart placeholder with real chart display.

**Changes:**
1. Import plot functions
2. Check if results contain DataFrame
3. Generate chart using DataFrame
4. Display with `st.plotly_chart()`

**Location:**
Replace this section:
```python
# Price chart placeholder
st.subheader("📈 Stock Price Chart & Historical Signals")
st.info("📊 **Coming in Phase 2**: ...")
```

With:
```python
# Price chart
st.subheader("📈 Stock Price Chart & Historical Signals")
if 'data' in results and results['data'] is not None:
    chart = create_price_chart(results['data'], st.session_state.ticker)
    st.plotly_chart(chart, use_container_width=True)
else:
    st.info("No chart data available")
```

## Chart Features

### Candlestick Chart
- **Green candles**: Close > Open (price went up)
- **Red candles**: Close < Open (price went down)
- **Wicks**: Show high/low of the day
- Industry standard for stock charts

### Volume Bars
- **Green bars**: Up day (close > open)
- **Red bars**: Down day (close < open)
- Shows trading activity

### Interactive Features
- **Zoom**: Click and drag to zoom into specific time period
- **Pan**: Shift+drag to move left/right
- **Hover**: Shows OHLC values at any point
- **Reset**: Double-click to reset zoom
- **Download**: Built-in download as PNG

## Layout Structure

```
┌─────────────────────────────────────────────────────┐
│  📈 Stock Price Chart & Historical Signals          │
│                                                     │
│  ┌───────────────────────────────────────────────┐ │
│  │           Candlestick Chart                   │ │
│  │  ┃  ┃   ┃  ┃  ┃  ┃   ┃  ┃                   │ │
│  │  ┃  ┃   ┃  ┃  ┃  ┃   ┃  ┃  (70% height)     │ │
│  └───────────────────────────────────────────────┘ │
│                                                     │
│  ┌───────────────────────────────────────────────┐ │
│  │           Volume Bars                         │ │
│  │  ▓  ▓   ▓  ▓  ▓  ▓   ▓  ▓  (30% height)     │ │
│  └───────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────┘
```

## Styling

### Colors
- **Up/Green**: #26a69a (teal green)
- **Down/Red**: #ef5350 (red)
- **Background**: White
- **Grid**: Light gray

### Typography
- Chart title: Bold, 16px
- Axis labels: 12px
- Hover text: 11px

## Error Handling

### No Data
```python
if df is None or df.empty:
    st.warning("No data available to chart")
    return None
```

### Invalid Data
- Check for required columns
- Handle missing values
- Skip if < 2 data points

## Performance Considerations

- Plotly handles up to ~10k points well
- For very large datasets (5+ years daily), consider resampling
- Chart rendering is client-side (fast)
- Use `use_container_width=True` for responsive design

## Dependencies

- **plotly** (already in requirements.txt)
- Uses pandas DataFrames (already included)

## Testing

**Test cases:**
- [ ] Chart displays for valid data
- [ ] Candlesticks show correct colors
- [ ] Volume bars match price direction
- [ ] Zoom/pan works smoothly
- [ ] Hover shows correct values
- [ ] Works with different timeframes (1M - 5Y)
- [ ] Works with crypto (BTC-USD)
- [ ] Works with international stocks
- [ ] Responsive on different screen sizes

## Future Enhancements (Phase 4)

- Add technical indicators overlay (Moving averages, Bollinger Bands)
- Add buy/sell signal markers on chart
- Add drawing tools
- Add comparison with other tickers
- Add different chart types (Heikin-Ashi, Renko)
- Add custom date range selector

## Notes

- Candlestick is industry standard for stock charts
- Plotly provides excellent interactivity out of the box
- Chart should load quickly (<1 second)
- Mobile-friendly and responsive
- No additional configuration needed from user

