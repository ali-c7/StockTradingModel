# Buy, Sell, Hold Predictive Model - Features Plan

## Development Roadmap

This document outlines all features and their implementation order for the Buy, Sell, Hold Predictive Model application.

---

## Phase 1: UI Foundation (Streamlit Interface)

### 1.1 Basic App Structure ✅
- [x] Create main Streamlit app entry point (`app.py`)
- [x] Set up page configuration (title, icon, layout)
- [x] Add application header and description
- [x] Implement basic navigation/layout structure

### 1.2 User Input Controls ✅
- [x] Ticker symbol input field with validation
- [x] Timeframe selector (dropdown: 1M, 3M, 6M, 1Y, 2Y, 5Y)
- [x] "Analyze" button to trigger prediction
- [x] Input error handling and user feedback messages

### 1.3 Results Display Framework ✅
- [x] Create placeholder sections for signal display
- [x] Design layout for prediction results (Buy/Sell/Hold)
- [x] Add section for stock price chart
- [x] Add section for technical indicators visualization
- [x] Add section for model performance metrics

### 1.4 UI Polish ✅
- [x] Add loading spinners for long operations
- [x] Implement session state management
- [x] Add sidebar for configuration options
- [x] Style components with custom CSS (optional)
- [x] Add app footer with disclaimers

**Deliverable**: Polished Streamlit UI with sidebar configuration and display preferences

---

## Phase 2: Data Pipeline

### 2.0 Ticker List & Searchable Dropdown ✅
- [x] Create `data/tickers/ticker_list_data.py` module
- [x] Fetch S&P 500 ticker list from Wikipedia at app startup
- [x] Store ticker list with company names in cache (24-hour TTL)
- [x] Replace text input with searchable `st.selectbox` for ticker selection
- [x] Implement filtering/search functionality (built-in with selectbox)
- [x] Add option to manually enter ticker if not in list
- [x] Cache ticker list to avoid repeated fetching
- [x] Add lxml dependency for pandas read_html()

### 2.1 Data Retrieval Module ✅
- [x] Create `data/stock/stock_data.py` module
- [x] Implement Yahoo Finance data fetching function
- [x] Add date range calculation based on timeframe
- [x] Implement data validation (check for empty/invalid data)
- [x] Add caching mechanism (`@st.cache_data`) for performance
- [x] Handle API errors and edge cases (delisted stocks, invalid tickers)
- [x] Store OHLCV data (Open, High, Low, Close, Volume) for chart visualization
- [x] Replace mock prices with real Yahoo Finance data
- [x] Get current price, change, and volume from real data

### 2.2 Basic Price Visualization ✅
- [x] Create `plots/stock/stock_plot.py` module
- [x] Implement candlestick chart for price visualization
- [x] Use Plotly for interactive chart (zoom, pan, hover)
- [x] Display price over selected timeframe
- [x] Add volume subplot below price chart
- [x] Integrate into UI to replace chart placeholder
- [x] Color-coded volume bars (green/red)
- [x] Interactive tooltips and controls

### 2.3 Data Preprocessing ⏭️
- [SKIPPED] Clean Yahoo Finance data (not needed - data is already clean)
- [SKIPPED] Handle missing values (rare with Yahoo Finance)
- [SKIPPED] Outlier detection (will handle in Phase 3 if needed)

### 2.4 Technical Indicators Computation ✅
- [x] Create `data/indicators/indicators_data.py` module
- [x] Implement RSI (Relative Strength Index) calculation
- [x] Implement MACD (Moving Average Convergence Divergence)
- [x] Implement Bollinger Bands calculation
- [x] Create indicator interpretation functions (overbought/oversold etc.)
- [x] Replace mock indicator values with real calculations
- [x] Display real RSI, MACD, and Bollinger Bands data
- [x] Add status indicators (Overbought, Oversold, Bullish, Bearish)
- [ ] Implement Volume Moving Average (future enhancement)
- [ ] Add SMA (Simple Moving Average) - 20, 50, 200 day (future enhancement)
- [ ] Add EMA (Exponential Moving Average) (future enhancement)
- [ ] Unit tests for indicator calculations (Phase 5)

### 2.5 Feature Engineering
- [ ] Create `data/features/features_data.py` module
- [ ] Generate price momentum features
- [ ] Create volatility features
- [ ] Add volume-based features
- [ ] Implement lagged features (previous N days)
- [ ] Create feature selection utilities
- [ ] Add feature importance analysis

**Deliverable**: Complete data pipeline that fetches, cleans, and enriches stock data with technical indicators

---

## Phase 3: Prediction Model ✅

### 3.1 Label Generation ✅
- [x] Create `core/models/label_generator.py` module
- [x] Create label generation logic (future returns → Buy/Sell/Hold)
- [x] Implement configurable forward_days and threshold
- [x] Add label distribution analysis
- [x] Add threshold optimization function

### 3.2 Feature Engineering ✅
- [x] Create `core/models/feature_engineer.py` module
- [x] Add moving averages (SMA 20, 50)
- [x] Add momentum features (5-day, 10-day)
- [x] Add volume features (ratio, moving average)
- [x] Add volatility features (ATR, volatility ratio)
- [x] Add Bollinger Band derived features (width, position)
- [x] Add lagged features (RSI, MACD, Close)
- [x] Add rate of change features
- [x] Create feature validation function
- [x] Total: 25+ engineered features

### 3.3 ML Model Training ✅
- [x] Create `core/models/model_trainer.py` module
- [x] Implement Random Forest classifier
- [x] Add time-series aware train/test split (80/20)
- [x] Implement class weighting for imbalance
- [x] Add model evaluation metrics (accuracy, per-class)
- [x] Implement model serialization (save/load with joblib)
- [x] Add feature importance extraction
- [x] Add prediction probability methods
- [x] Retrain on full data for production

### 3.4 Prediction Interface ✅
- [x] Create `core/signals/signal_predictor.py` module
- [x] Implement SignalPredictor class
- [x] Add complete training pipeline
- [x] Add predict_latest() for current prediction
- [x] Add predict_historical() for backtesting
- [x] Generate reasoning from indicator values
- [x] Return confidence scores and probabilities
- [x] Add metrics summary function

### 3.5 App Integration ✅
- [x] Import SignalPredictor into app.py
- [x] Replace mock predictions with real ML
- [x] Add error handling for ML failures
- [x] Display real confidence scores
- [x] Display real reasoning
- [x] Show model accuracy in UI
- [x] Remove deprecated mock function

**Deliverable**: Functional prediction model (rule-based + ML) with validation framework

---

## Phase 4: Testing & Visualization

### 4.1 Performance Metrics
- [ ] Create `core/metrics/metrics_core.py` module
- [ ] Implement basic accuracy calculation
- [ ] Add simple profit/loss tracking
- [ ] Create user-friendly performance summary

### 4.2 Backtesting Engine (Portfolio Simulation)
- [ ] Create `core/backtest/backtest_core.py` module
- [ ] Implement portfolio simulator with $1000 initial capital
- [ ] Add realistic trade execution:
  - Transaction costs (commission fees)
  - Slippage modeling
  - Position sizing (Kelly Criterion or fixed %)
- [ ] Track portfolio metrics:
  - Total return (%)
  - Win rate (profitable trades %)
  - Max drawdown
  - Sharpe ratio
  - Best/worst trades
- [ ] Generate trade history log (date, action, price, shares, P/L)
- [ ] Calculate time-series portfolio value
- [ ] Add comparison vs Buy & Hold strategy
- [ ] Export trade log to CSV (optional)

### 4.3 Visualization Modules
- [ ] Update `plots/stock/stock_plot.py` for enhanced price charts
- [ ] Add historical buy/sell/hold signal markers on price chart
  - Green markers (▲) for historical BUY signals
  - Red markers (▼) for historical SELL signals
  - Yellow markers (●) for historical HOLD signals
- [ ] Add technical indicators overlay on price chart (optional toggles)
- [ ] Create `plots/performance/portfolio_plot.py`
  - Portfolio value over time (line chart)
  - Comparison with buy & hold strategy
  - Trade markers (buy/sell points)
  - Drawdown visualization
  - Cumulative returns chart
- [ ] Add `plots/indicators/indicators_plot.py` (optional, Phase 5)
  - RSI chart with threshold lines
  - MACD histogram and signal lines
  - Bollinger Bands visualization

### 4.4 Integration & UI Updates
- [ ] Connect data pipeline to UI
- [ ] Connect prediction model to UI
- [ ] Display real prediction results with clear Buy/Sell/Hold signal
- [ ] Show model accuracy in simple percentage format
- [ ] Integrate all visualizations into Streamlit UI
- [ ] Add interactive chart controls (zoom, pan, hover)
- [ ] Display signal reasoning (why Buy/Sell/Hold)

### 4.5 Unit & Integration Tests
- [ ] Create `tests/data/test_stock_data.py`
- [ ] Create `tests/data/test_indicators_data.py`
- [ ] Create `tests/core/test_signal_core.py`
- [ ] Create `tests/core/test_model_core.py`
- [ ] Create `tests/core/test_backtest_core.py`
- [ ] Add integration tests for full pipeline
- [ ] Implement smoke tests for UI components

**Deliverable**: Full-featured application with simple, user-friendly visualizations and clear signal output

---

## Phase 5: Enhancement & Polish (Post-MVP)

### 5.1 Advanced Features
- [ ] Add multi-ticker comparison mode
- [ ] Implement portfolio optimization
- [ ] Add real-time data streaming option
- [ ] Create custom indicator builder
- [ ] Add alerts and notifications system
- [ ] Implement strategy builder interface

### 5.2 Advanced Analytics (Optional)
- [ ] Add confusion matrix and classification metrics
- [ ] Implement ROC curve and AUC score
- [ ] Add F1 score and precision/recall metrics
- [ ] Implement Sharpe ratio and risk-adjusted returns
- [ ] Add drawdown analysis and visualization
- [ ] Create commission/slippage simulation
- [ ] Implement advanced performance comparison utilities

### 5.3 Performance & Optimization
- [ ] Optimize data loading with async operations
- [ ] Implement advanced caching strategies
- [ ] Add data compression for large datasets
- [ ] Profile and optimize bottlenecks
- [ ] Implement lazy loading for visualizations

### 5.4 Documentation & Deployment
- [ ] Create user guide (`docs/user_guide.md`)
- [ ] Write API documentation
- [ ] Add inline code documentation (docstrings)
- [ ] Create `requirements.txt` with pinned versions
- [ ] Write deployment guide (Streamlit Cloud, Docker)
- [ ] Create demo video/screenshots
- [ ] Add CONTRIBUTING.md for open source

### 5.5 Production Readiness
- [ ] Add comprehensive error handling
- [ ] Implement logging framework
- [ ] Add monitoring and analytics
- [ ] Create health check endpoints
- [ ] Implement rate limiting for API calls
- [ ] Add security best practices (input sanitization)

**Deliverable**: Production-ready application with documentation and deployment support

---

## Feature Implementation Guidelines

**Priority Levels**:
- **P0 (Critical)**: Must-have for MVP (Phases 1-4, core features)
- **P1 (High)**: Important but not blocking (Phase 4 polish)
- **P2 (Medium)**: Nice to have (Phase 5 enhancements)
- **P3 (Low)**: Future considerations

**Development Principles**:
1. Complete each phase before moving to the next
2. Write tests alongside feature implementation
3. Update documentation as features are added
4. Commit after each completed feature
5. Review and refactor before moving to next phase

**Success Criteria**:
- All P0 features implemented and tested
- Application runs without errors
- User can input ticker, get prediction, and view results
- Model achieves reasonable accuracy (>55% on validation set)
- Code passes lint checks and follows style guide
- Core functionality has test coverage >80%

