# Signal Visualization Approach

## Overview
This document describes how buy/sell/hold signals will be visualized in the application.

## Visualization Strategy

### Primary: Integrated Stock Chart with Signal Markers

Historical buy/sell/hold signals will be displayed as **markers overlaid on the main stock price chart**. This is the industry-standard approach used by professional trading platforms.

**Benefits:**
- Shows signals in context with price action
- Easy to see if signals were accurate
- Reduces cognitive load (one chart to read)
- Shows relationship between price and recommendations

### Chart Components

#### 1. Candlestick/Line Chart (Base Layer)
- OHLCV data (Open, High, Low, Close, Volume)
- Timeframe selected by user
- Interactive zoom/pan with Plotly

#### 2. Signal Markers (Overlay Layer)
Markers will be placed at the closing price on the date the signal was generated:

- **🟢 BUY Signal**: Green upward triangle marker (▲)
  - Positioned slightly below the price candle
  - Green color (#00CC00)
  - Tooltip shows: "BUY - Date, Price, Confidence"

- **🔴 SELL Signal**: Red downward triangle marker (▼)
  - Positioned slightly above the price candle
  - Red color (#FF3333)
  - Tooltip shows: "SELL - Date, Price, Confidence"

- **🟡 HOLD Signal**: Yellow circle marker (●)
  - Positioned at the closing price
  - Orange/Yellow color (#FFA500)
  - Tooltip shows: "HOLD - Date, Price, Confidence"

#### 3. Volume Subplot (Below Main Chart)
- Bar chart showing trading volume
- Color-coded: green bars (up days), red bars (down days)

#### 4. Technical Indicator Overlays (Optional Toggles)
- Bollinger Bands (shaded area)
- Moving averages (SMA 20, 50, 200)
- User can toggle on/off via sidebar/checkboxes

## Implementation Plan

### Phase 2: Data Pipeline
- Fetch historical OHLCV data
- Store in DataFrame with date index
- Prepare data structure for Plotly

### Phase 3: Signal Generation
- Generate historical signals for the entire timeframe
- Store signals in DataFrame with columns:
  - `date`: Signal date
  - `signal`: "BUY"/"SELL"/"HOLD"
  - `price`: Price at signal generation
  - `confidence`: Confidence percentage
  - `reasoning`: Explanation text

### Phase 4: Visualization
Create `plots/stock/stock_plot.py`:
```python
def create_stock_chart_with_signals(
    df: pd.DataFrame,           # OHLCV data
    signals: pd.DataFrame,      # Historical signals
    ticker: str,
    show_volume: bool = True,
    show_indicators: bool = False
) -> plotly.graph_objects.Figure
```

**Chart Features:**
- Main candlestick chart
- Signal markers with custom shapes and colors
- Hover tooltips with detailed information
- Volume subplot
- Optional indicator overlays
- Responsive design
- Export functionality (PNG/HTML)

## Alternative: Separate Signal Timeline (Optional)

If the main chart becomes too cluttered, we can add a **secondary visualization** below the main chart:

### Signal Timeline Chart
- Horizontal timeline showing signal history
- Color-coded segments (green/red/yellow)
- Shows signal frequency and distribution
- Compact view for quick overview

**Implementation**: Only add if user feedback suggests main chart is cluttered.

## UI Layout in Application

```
┌─────────────────────────────────────────────────────────────┐
│  Current Signal: 🟢 BUY (87% confidence)                    │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│  📈 Stock Price Chart & Historical Signals                  │
│                                                             │
│  [Candlestick Chart]                                        │
│  ▲ ▼ ● ← Signal markers overlaid on price                  │
│                                                             │
│  [Volume Chart Below]                                       │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│  🔧 Technical Indicators (RSI, MACD, BB)                    │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│  📊 Performance: Returns over time chart                    │
└─────────────────────────────────────────────────────────────┘
```

## Example Code Structure

```python
# Phase 4 Implementation
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def create_stock_chart_with_signals(df, signals, ticker):
    # Create figure with subplots
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.7, 0.3]
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
    
    # Add BUY signals
    buy_signals = signals[signals['signal'] == 'BUY']
    fig.add_trace(
        go.Scatter(
            x=buy_signals['date'],
            y=buy_signals['price'],
            mode='markers',
            marker=dict(
                symbol='triangle-up',
                size=12,
                color='#00CC00'
            ),
            name='BUY',
            text=buy_signals['reasoning'],
            hovertemplate='<b>BUY</b><br>Price: $%{y:.2f}<br>%{text}'
        ),
        row=1, col=1
    )
    
    # Add SELL signals (similar)
    # Add HOLD signals (similar)
    
    # Add volume
    fig.add_trace(
        go.Bar(x=df.index, y=df['Volume'], name='Volume'),
        row=2, col=1
    )
    
    return fig
```

## User Experience

1. **Initial View**: User sees current signal prominently displayed at top
2. **Chart View**: Scrolls down to see historical context with all past signals
3. **Interactivity**: Hovers over markers to see signal details
4. **Analysis**: Can zoom/pan to examine specific time periods
5. **Understanding**: Visual correlation between signals and subsequent price movement

## Success Metrics

- Users can quickly identify signal patterns
- Clear visual distinction between signal types
- Tooltips provide necessary context without cluttering
- Chart remains readable with multiple signals
- Performance is smooth even with many data points

## Next Steps

1. **Phase 2**: Implement data fetching with OHLCV storage
2. **Phase 3**: Generate historical signals DataFrame
3. **Phase 4**: Implement Plotly chart with signal markers
4. **User Testing**: Gather feedback on clarity and usability
5. **Iterate**: Adjust marker styles, colors, or add alternative views if needed

## Notes

- This approach follows industry best practices (TradingView, MetaTrader, etc.)
- Keeps all relevant information in one place
- Easy to implement with Plotly's annotation/marker system
- Scalable to additional features (support/resistance lines, pattern recognition, etc.)

