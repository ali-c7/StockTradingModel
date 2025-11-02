# Training Pipeline: From Yahoo Finance to Trained Model

## Complete 7-Step Pipeline

Here's exactly how we train models on Yahoo Finance data:

---

## **STEP 1: Fetch Data from Yahoo Finance**

**File**: `data/stock/stock_data.py`

### Process:
```python
def fetch_stock_data(ticker: str, timeframe: str) -> pd.DataFrame:
    # Calculate date range (e.g., 2Y = 730 days ago to today)
    start_date, end_date = calculate_date_range(timeframe)
    
    # Fetch from Yahoo Finance using yfinance library
    stock = yf.Ticker(ticker)
    df = stock.history(start=start_date, end=end_date)
    
    # Returns OHLCV data: Open, High, Low, Close, Volume
    return df[['Open', 'High', 'Low', 'Close', 'Volume']]
```

### Output:
- **Raw OHLCV data** with daily candles
- **Example**: 2Y for AAPL = ~504 trading days
- **Indexed by date** (timezone-aware timestamps)

### Caching:
- Data is **cached for 1 hour** using `@st.cache_data(ttl=3600)`
- Cache key includes current date for daily refresh
- Prevents repeated API calls during same session

---

## **STEP 2: Engineer Technical Features**

**File**: `core/features/technical_features.py`

### Process:
```python
def engineer_all_features(df: pd.DataFrame) -> pd.DataFrame:
    # 1. Price features (returns, ranges)
    df = add_price_features(df)
    
    # 2. Trend indicators (EMA, ADX, Ichimoku)
    df = add_trend_features(df)
    
    # 3. Momentum indicators (RSI, Stochastic, CCI)
    df = add_momentum_features(df)
    
    # 4. Volatility indicators (Bollinger, ATR, Keltner)
    df = add_volatility_features(df)
    
    # 5. Volume indicators (OBV, MFI, VWAP)
    df = add_volume_features(df)
    
    return df
```

### Feature Categories:

| Category | Features | Count |
|----------|----------|-------|
| **Price** | returns, log_returns, high_low_range, close_open_range | 4 |
| **Trend** | EMA (9,12,26,50,200), SMA (20,50,200), ADX, Ichimoku | 15+ |
| **Momentum** | RSI (14), Stochastic (K,D), CCI, ROC, Williams %R | 8+ |
| **Volatility** | Bollinger Bands, ATR, Keltner, Historical Vol | 10+ |
| **Volume** | OBV, MFI, VWAP, Volume MA, price_vs_vwap | 6+ |

### Total Features:
- **~50-60 technical indicators** per row
- All calculated using TA-Lib or pandas rolling functions

### Data Loss:
- **EMA 200/SMA 200** require 200 days of history
- First ~200 rows will have NaN values for long-period indicators
- For short timeframes (<400 days), long-period indicators are **automatically dropped**

---

## **STEP 3: Generate Training Labels**

**File**: `core/labels/label_generator.py`

### Process:
```python
def generate_labels(df, forward_days=5, threshold=0.02):
    # Look forward in time to see actual price movement
    future_close = df['Close'].shift(-forward_days)
    forward_return = (future_close - df['Close']) / df['Close']
    
    # Adaptive threshold based on volatility
    volatility = df['Close'].pct_change().std()
    adjusted_threshold = max(threshold, volatility * 1.5)
    
    # Generate 3-class labels
    labels = 0  # Default: HOLD
    labels[forward_return >= adjusted_threshold] = 1   # BUY
    labels[forward_return <= -adjusted_threshold] = -1  # SELL
    
    # Remove last N rows (no future data available)
    labels.iloc[-forward_days:] = NaN
    
    return labels
```

### Label Logic:
- **BUY (1)**: Price will rise ≥ threshold in next N days
- **HOLD (0)**: Price movement < threshold (not worth trading)
- **SELL (-1)**: Price will drop ≥ threshold in next N days

### Example (forward_days=5, threshold=2%):
- Day 100: Close = $100
- Day 105: Close = $103 → **3% gain → BUY label**
- Day 200: Close = $100
- Day 205: Close = $99 → **-1% change → HOLD label**
- Day 300: Close = $100
- Day 305: Close = $97 → **-3% loss → SELL label**

### Adaptive Threshold:
- **Base threshold**: User-set (e.g., 2%)
- **Volatility adjustment**: `volatility × 1.5`
- **Final threshold**: `max(base, volatility × 1.5)`
- **Why?** High-volatility stocks need higher thresholds to avoid noise

### Output:
- **Labels series** with values: -1 (SELL), 0 (HOLD), 1 (BUY)
- **Last N rows removed** (no future data to label them)
- **Distribution stats** (% of BUY/HOLD/SELL)

---

## **STEP 4: Prepare Train/Test Split**

**File**: `core/trading_system.py` (lines 133-164)

### Process:
```python
def prepare_train_test_split(self):
    # Get all features
    feature_list = get_feature_list('all')  # ~50 features
    X = prepare_feature_matrix(self.feature_data, features=feature_list)
    
    # Remove rows with NaN labels (last forward_days rows + indicator warmup)
    valid_idx = self.labels.dropna().index
    X = X.loc[valid_idx]
    y = self.labels.loc[valid_idx]
    
    # Time-series chronological split (NO SHUFFLING!)
    split_point = int(len(X) * self.train_split)  # e.g., 80%
    
    self.X_train = X.iloc[:split_point]        # First 80%
    self.X_test = X.iloc[split_point:]          # Last 20%
    self.y_train = y.iloc[:split_point]
    self.y_test = y.iloc[split_point:]
```

### Key Characteristics:

#### Time-Series Split (Chronological)
```
[Training Data]            [Test Data]
Sept 2023 → Jul 2025      Aug 2025 → Oct 2025
    (80%)                      (20%)
```

#### Why NO Shuffling?
- **Time-series data**: Future cannot predict past
- **Realistic evaluation**: Model never sees future during training
- **Prevents data leakage**: Training data always comes before test data

#### Data Reduction:
**Original data**: 504 days (2Y)  
**After EMA 200**: ~304 days (200 lost to warmup)  
**After forward labels**: ~299 days (5 lost to future labels)  
**Final usable**: ~299 samples  
**Train (80%)**: ~239 samples  
**Test (20%)**: ~60 samples

---

## **STEP 5: Train ML Model**

**File**: `core/models/baseline_models.py`

### Supported Models:

#### 1. **XGBoost** (Default - Usually Best)
```python
XGBClassifier(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.05,
    random_state=42,
    eval_metric='mlogloss'
)
```

#### 2. **Random Forest** (Most Stable)
```python
RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_split=20,
    min_samples_leaf=10,
    random_state=42
)
```

#### 3. **LightGBM** (Fastest)
```python
LGBMClassifier(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.05,
    num_leaves=31,
    random_state=42
)
```

### Training Process:
```python
def train(X_train, y_train, X_test, y_test):
    # 1. Encode labels: -1 → 0, 0 → 1, 1 → 2
    y_train_encoded = label_encoder.transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)
    
    # 2. Fit model on training data
    model.fit(X_train, y_train_encoded)
    
    # 3. Evaluate on test data (never seen during training!)
    y_pred_test = model.predict(X_test)
    y_prob_test = model.predict_proba(X_test)
    
    # 4. Calculate metrics
    test_accuracy = accuracy_score(y_test_encoded, y_pred_test)
    precision = precision_score(y_test_encoded, y_pred_test, average='weighted')
    recall = recall_score(y_test_encoded, y_pred_test, average='weighted')
    
    return metrics
```

### What the Model Learns:
- **Input**: 50+ technical indicators for each day
- **Output**: Probability distribution [P(SELL), P(HOLD), P(BUY)]
- **Learning**: Patterns that predict future price movements

### Example:
```
Day 100 features:
- RSI: 65 (overbought)
- MACD: Negative divergence
- Volume: 2x average
- Price: Above EMA 50

Model prediction: [0.10, 0.20, 0.70]
→ 70% probability of BUY
→ Makes BUY signal
```

---

## **STEP 6: Generate Trading Signals**

**File**: `core/trading_system.py` (lines 185-216)

### Process:
```python
def generate_signals(self):
    # Get full feature matrix (all dates)
    X = prepare_feature_matrix(self.feature_data, features=feature_list)
    
    # Generate predictions for ALL dates
    predictions = model.predict(X)  # -1, 0, or 1
    probabilities = model.predict_proba(X)  # [P(SELL), P(HOLD), P(BUY)]
    
    # Extract confidence (max probability)
    confidences = probabilities.max(axis=1)
    
    return signals, confidences
```

### Output:
- **Signals series**: -1 (SELL), 0 (HOLD), 1 (BUY) for every day
- **Confidences series**: 0.0-1.0 (model certainty)

### Important:
- Predictions made for **ALL dates** (training + testing)
- But we only **evaluate on test period**
- Training predictions used for backtesting visualization only

---

## **STEP 7: Backtest Strategy**

**File**: `core/backtest/portfolio_simulator.py`

### Process:
```python
def run_backtest(df, signals, confidences):
    for each_day in data:
        signal = signals[day]
        price = df['Close'][day]
        
        # Check stop-loss / take-profit on existing positions
        if has_position:
            if return >= take_profit_pct:
                sell_position()
            elif return <= -stop_loss_pct:
                stop_loss()
        
        # Execute new signals
        if signal == 1 and no_position:
            buy_stock(price, confidence)
        elif signal == -1 and has_position:
            sell_stock(price)
        
        # Track portfolio value
        track_portfolio_value()
```

### Risk Management:
- **Position Sizing**: Kelly Criterion (adjusts based on confidence)
- **Stop Loss**: 5% (automatic exit on -5% loss)
- **Take Profit**: 10% (automatic exit on +10% gain)
- **Transaction Cost**: 0.1% per trade

### Constraints:
- ✅ Can only BUY if no position open
- ✅ Can only SELL if position open
- ✅ No short selling (long-only strategy)
- ✅ No double-buying (one position at a time)

---

## Complete Pipeline Visualization

```
┌─────────────────────────────────────────────────────────────────┐
│  STEP 1: Fetch Data (Yahoo Finance)                            │
│  Input: ticker="AAPL", timeframe="2Y"                          │
│  Output: 504 rows × 5 columns (OHLCV)                          │
└──────────────────────┬──────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 2: Engineer Features                                      │
│  Output: 504 rows × 55 columns (OHLCV + 50 indicators)         │
│  Data loss: First 200 rows have NaN (EMA 200 warmup)           │
└──────────────────────┬──────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 3: Generate Labels                                        │
│  Output: 299 valid labels (504 - 200 warmup - 5 forward)       │
│  Distribution: 30% BUY, 40% HOLD, 30% SELL (example)           │
└──────────────────────┬──────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 4: Train/Test Split (80/20)                              │
│  X_train: 239 samples × 50 features  (Sept 2023 - Jul 2025)   │
│  X_test:   60 samples × 50 features  (Aug 2025 - Oct 2025)    │
└──────────────────────┬──────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 5: Train Model (XGBoost)                                 │
│  Input: X_train (239 × 50), y_train (239 labels)               │
│  Training: Learns patterns from 50 indicators → BUY/HOLD/SELL  │
│  Output: Trained model + metrics (test_accuracy: 38%)          │
└──────────────────────┬──────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 6: Generate Signals (ALL dates)                          │
│  Output: 299 predictions + confidences                         │
│  Example: [BUY, HOLD, BUY, SELL, HOLD, BUY...]                 │
└──────────────────────┬──────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 7: Backtest Strategy                                     │
│  Input: Signals + prices + risk rules                          │
│  Simulation: Execute trades, track portfolio, apply stop/profit │
│  Output: 26 executed trades, final value: $12,688 (+26.88%)    │
└─────────────────────────────────────────────────────────────────┘
```

---

## Key Design Choices

### 1. **Supervised Learning Approach**
- We use **labeled data** (future returns) to train
- Model learns: "Given these indicators, price goes up/down"
- **Not** reinforcement learning or unsupervised clustering

### 2. **3-Class Classification**
- **BUY/HOLD/SELL** instead of just BUY/SELL
- HOLD accounts for sideways markets
- More realistic than binary classification

### 3. **Time-Series Aware**
- Chronological split (no shuffling)
- Future never predicts past
- Realistic out-of-sample testing

### 4. **Adaptive Thresholds**
- High volatility stocks → Higher thresholds
- Prevents noise trading
- Auto-adjusts to market conditions

### 5. **Feature-Rich**
- 50+ indicators capture multiple market aspects
- Trend, momentum, volatility, volume all represented
- Model learns which matter most

---

## Common Questions

### Q: Why is test accuracy only 38%?
**A**: 3-class problem with imbalanced data. Even 33% (random) is baseline. 38% is better than random, and **profitability matters more than accuracy**.

### Q: Why adaptive threshold?
**A**: AAPL (low vol) shouldn't use same threshold as crypto (high vol). Adaptive prevents overtrading in noise.

### Q: Why no shuffling?
**A**: Time-series data! If you shuffle, the model can "see the future" during training → unrealistic performance.

### Q: Why only 299 samples from 504 days?
**A**: 200 lost to EMA/SMA warmup, 5 lost to forward labels. This is normal and necessary.

### Q: Can I use more training data?
**A**: Yes! Adjust Train/Test Split slider to 85% or 90% (more training, less testing).

---

## Files Summary

| Step | File | Purpose |
|------|------|---------|
| 1 | `data/stock/stock_data.py` | Fetch from Yahoo Finance |
| 2 | `core/features/technical_features.py` | Add 50+ indicators |
| 3 | `core/labels/label_generator.py` | Create BUY/HOLD/SELL labels |
| 4-7 | `core/trading_system.py` | Split, train, predict, backtest |
| Models | `core/models/baseline_models.py` | XGBoost/RF/LGBM implementation |
| Backtest | `core/backtest/portfolio_simulator.py` | Simulate trading with risk rules |

---

**Updated**: 2025-11-01  
**Complete training pipeline from Yahoo Finance to executed trades**

