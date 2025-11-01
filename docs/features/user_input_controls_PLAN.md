# User Input Controls - Technical Plan

## Overview
Implement interactive input controls in the Streamlit UI allowing users to enter a stock ticker symbol, select a timeframe, and trigger the analysis. Includes input validation and error handling.

## Files to Modify

### 1. `app.py` (MODIFY)
Update the input section placeholder with functional controls.

**Requirements:**

#### Ticker Input Field
- Use `st.text_input()` for ticker symbol entry
- Convert input to uppercase automatically
- Placeholder text: "e.g., AAPL, GOOGL, TSLA"
- Label: "Stock Ticker Symbol"
- Max length validation (reasonable ticker length: 5-10 characters)
- Store in `st.session_state.ticker`

#### Timeframe Selector
- Use `st.selectbox()` for dropdown menu
- Label: "Analysis Timeframe"
- Options:
  - "1 Month" → "1M"
  - "3 Months" → "3M"
  - "6 Months" → "6M" (default)
  - "1 Year" → "1Y"
  - "2 Years" → "2Y"
  - "5 Years" → "5Y"
- Store selected value in `st.session_state.timeframe`

#### Analyze Button
- Use `st.button()` to trigger analysis
- Label: "🔍 Analyze Stock"
- Primary button style (type="primary")
- Disabled state if ticker is empty or invalid
- On click: validate inputs and set `st.session_state.analysis_triggered = True`

#### Input Validation
- **Ticker validation**:
  - Not empty
  - Contains only alphanumeric characters (A-Z, 0-9)
  - Length between 1-10 characters
  - Display error with `st.error()` if invalid
- **Error messages**:
  - "Please enter a stock ticker symbol" (if empty)
  - "Invalid ticker format. Use only letters and numbers" (if invalid characters)
  - "Ticker symbol too long. Maximum 10 characters" (if too long)

#### User Feedback
- Use `st.info()` for helpful hints
- Use `st.success()` when valid inputs are ready
- Use `st.error()` for validation errors
- Use `st.warning()` for non-critical issues (e.g., timeframe considerations)

## Layout Structure

Replace the placeholder in `input_container` with:
```
Column layout (col1, col2, col3):
- col1 (50%): Ticker input
- col2 (30%): Timeframe selector
- col3 (20%): Analyze button (vertically aligned)
```

## Session State Variables

Add to `initialize_session_state()`:
- `analysis_triggered`: Boolean flag when analyze is clicked
- `last_ticker`: Store last analyzed ticker to detect changes
- `last_timeframe`: Store last timeframe to detect changes

## Validation Logic

Create helper function `validate_ticker(ticker: str) -> tuple[bool, str]`:
- Returns (is_valid, error_message)
- Check emptiness, format, length
- Reusable for future phases

## Mock Behavior (Phase 1.2 Only)

When "Analyze" button is clicked with valid inputs:
- Display `st.success()` message with ticker and timeframe
- Show mock loading spinner with `st.spinner()`
- Display message: "Data fetching will be implemented in Phase 2"
- Update session state to reflect button click

## Dependencies
- No new dependencies (uses built-in Streamlit components)

## Notes
- Keep validation simple but effective
- Focus on user experience with clear feedback
- Button should be visually prominent
- Prepare session state structure for Phase 2 data integration
- Use columns for responsive layout

