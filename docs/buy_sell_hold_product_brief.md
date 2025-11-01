# Buy, Sell, and Hold Predictive Model - Product Brief

## 1. Project Overview / Description

An interactive Streamlit web application that generates **Buy, Sell, or Hold** trading signals for stocks based on historical price data and technical indicators. Users input a ticker symbol and timeframe, and the app retrieves data from Yahoo Finance, computes key technical indicators (RSI, MACD, Bollinger Bands, Volume MA), and applies a predictive model to recommend trading actions. The system supports both rule-based and machine learning approaches, with walk-forward validation for robust evaluation.

## 2. Target Audience

- **Retail investors** seeking data-driven trading insights
- **Day traders and swing traders** looking for technical analysis automation
- **Finance students and enthusiasts** exploring algorithmic trading concepts
- **Quant developers** prototyping trading strategies with minimal setup

## 3. Primary Benefits / Features

- **Simple inputs**: Enter any stock ticker and select a timeframe (e.g., 1 month, 6 months, 1 year)
- **Automated technical analysis**: Calculates RSI, MACD, Bollinger Bands, and Volume Moving Average
- **Actionable signals**: Clear Buy/Sell/Hold recommendations based on model predictions
- **Backtesting visualization**: View historical performance metrics and validation results
- **Modular design**: Easily extend with additional indicators or ML models
- **Walk-forward validation**: Ensures robust, time-aware model evaluation

## 4. High-Level Tech / Architecture

**Framework**: Streamlit (Python)  
**Data Source**: Yahoo Finance API (`yfinance`)  
**Data Processing**: pandas, numpy  
**Technical Indicators**: `ta` (technical analysis library)  
**Visualization**: Plotly for interactive charts  
**Modeling**: Starts with rule-based logic, with architecture to support scikit-learn, LightGBM, or XGBoost for ML-based predictions  
**Validation**: Walk-forward backtesting with performance metrics (accuracy, precision, recall)

**Project Structure**:
- `ui/` — Streamlit interface modules
- `data/` — Data retrieval, preprocessing, and indicator computation
- `core/` — Model training, signal generation, and backtesting logic
- `plots/` — Plotly visualizations for results and metrics
- `tests/` — Unit and integration tests
- `docs/` — Architecture and workflow documentation

**Development Flow**:
1. **Phase 1 (UI)**: Build user input controls and display framework
2. **Phase 2 (Data)**: Implement data fetching and technical indicator pipeline
3. **Phase 3 (Training)**: Create predictive model with rule-based and ML options
4. **Phase 4 (Testing)**: Add validation, backtesting, and performance visualization

