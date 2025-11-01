# Results Display Framework - Technical Plan

## Overview
Create a comprehensive results display framework that shows mock prediction signals, stock information, and placeholder visualizations. This framework will be populated with real data in Phase 2+.

## Files to Modify

### 1. `app.py` (MODIFY)
Replace placeholder sections with structured results display components.

**Requirements:**

#### Signal Display Section (Top Priority)
- Large, prominent display of Buy/Sell/Hold recommendation
- Use `st.metric()` or custom styled container
- Color-coded signal:
  - **BUY**: Green background/text (🟢)
  - **SELL**: Red background/text (🔴)
  - **HOLD**: Yellow/Orange background/text (🟡)
- Confidence score percentage (mock for now)
- Signal reasoning/explanation text
- Show current price and change (mock data)
- Display only when `st.session_state.analysis_triggered == True`

#### Stock Information Card
- Display key stock information:
  - Company name (mock or use ticker for now)
  - Current price (mock)
  - Day change amount and percentage
  - Volume (mock)
  - Market cap (optional, mock)
- Use `st.columns()` for multi-metric display
- Professional card-style layout

#### Price Chart Placeholder
- Area for candlestick/line chart
- Show mockup message: "📈 Stock price chart will display here"
- Use `st.empty()` or `st.container()` for future chart integration
- Add dimensions/aspect ratio suitable for Plotly charts

#### Technical Indicators Section
- Create 3-column layout for main indicators:
  - RSI (Relative Strength Index)
  - MACD (Moving Average Convergence Divergence)
  - Bollinger Bands
- Each indicator shows:
  - Indicator name and current value (mock)
  - Small visual placeholder (e.g., "📊 Chart")
  - Brief status text (e.g., "Neutral", "Oversold", "Bullish")
- Use `st.metric()` for clean display

#### Performance Metrics Section
- Show model performance information:
  - Accuracy percentage (mock)
  - Last updated timestamp
  - Number of signals analyzed (mock)
- Use simple layout with metrics

#### Layout Structure

**When analysis NOT triggered:**
- Show placeholder messages
- Display help text

**When analysis IS triggered:**
```
+----------------------------------+------------------+
|                                  |  Signal Display  |
|  Stock Info Card                 |  (BUY/SELL/HOLD) |
|  (Price, Change, Volume)         |  Confidence: XX% |
|                                  |  Reasoning       |
+----------------------------------+------------------+

+-------------------------------------------------------+
|  Stock Price Chart Placeholder                        |
|  (Full width)                                         |
+-------------------------------------------------------+

+------------------+------------------+------------------+
|  RSI             |  MACD            |  Bollinger Bands |
|  Value: XX       |  Value: XX       |  Upper/Lower     |
|  Status: XXX     |  Status: XXX     |  Status: XXX     |
+------------------+------------------+------------------+

+-------------------------------------------------------+
|  Model Performance Metrics                            |
|  Accuracy: XX% | Last Updated: DATE | Signals: XX     |
+-------------------------------------------------------+
```

## Mock Data Function

Create `generate_mock_results(ticker: str, timeframe: str) -> dict`:
- Returns dictionary with mock data:
  - `signal`: "BUY", "SELL", or "HOLD"
  - `confidence`: percentage (60-95%)
  - `current_price`: mock price based on ticker
  - `price_change`: mock change amount
  - `price_change_pct`: mock percentage
  - `volume`: mock volume
  - `rsi`: value (0-100)
  - `macd`: value
  - `bb_status`: Bollinger Bands status
  - `reasoning`: explanation text for signal
  - `accuracy`: model accuracy percentage
  - `last_updated`: current timestamp

## Display Logic

- Only show results when `st.session_state.analysis_triggered == True`
- Store mock results in `st.session_state.prediction_result`
- Persist results across reruns
- Clear results when new analysis is triggered

## Styling Requirements

- Use Streamlit's built-in styling (metrics, columns, containers)
- Color-coded signals:
  - Buy: Green (#00CC00 or similar)
  - Sell: Red (#FF0000 or similar)
  - Hold: Orange/Yellow (#FFA500 or similar)
- Use emojis for visual appeal:
  - 🟢 BUY
  - 🔴 SELL
  - 🟡 HOLD
  - 📈 Charts
  - 💰 Price
  - 📊 Indicators

## Session State Updates

Add to session state:
- Store mock results when analysis runs
- Clear old results when new analysis starts

## Dependencies
- No new dependencies (uses built-in Streamlit components)

## Notes
- All data is mock/placeholder for Phase 1
- Framework should be easy to replace with real data in Phase 2
- Focus on clean, professional layout
- Ensure responsive design (works on different screen sizes)
- Make signal display visually prominent
- Add explanatory text to help users understand results

