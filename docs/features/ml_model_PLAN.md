# Phase 3: ML Prediction Model - Implementation Plan

## Overview

Build a machine learning model that predicts Buy/Sell/Hold signals based on technical indicators. We'll use Random Forest as the primary model with proper walk-forward validation.

---

## Architecture

```
core/
  ├── models/
  │   ├── __init__.py
  │   ├── model_core.py          # Main ML model class
  │   ├── label_generator.py     # Create Buy/Sell/Hold labels
  │   ├── feature_engineer.py    # Feature engineering
  │   └── model_trainer.py       # Training pipeline
  ├── validation/
  │   ├── __init__.py
  │   └── walk_forward.py        # Walk-forward validation
  └── signals/
      ├── __init__.py
      └── signal_predictor.py    # Generate predictions
```

---

## Phase 3.1: Label Generation

### Goal
Create training labels (Buy/Sell/Hold) based on future returns.

### Logic (from hackathon code, but improved)

```python
# For each day, look N days ahead
forward_days = 3  # Look 3 trading days ahead
threshold = 0.01  # 1% threshold

future_return = (price[t+3] - price[t]) / price[t]

if future_return >= +0.01:    # Price goes up 1%+
    label = BUY (1)
elif future_return <= -0.01:  # Price goes down 1%+
    label = SELL (-1)
else:                         # Between -1% and +1%
    label = HOLD (0)
```

### Key Considerations

**Avoid Look-Ahead Bias:**
- Labels use FUTURE data (correct for training)
- But predictions must NEVER see future data
- This is why we need walk-forward validation

**Label Distribution:**
- Market is often neutral → many HOLD labels
- Class imbalance: BUY (20%), HOLD (60%), SELL (20%)
- Solution: Use `class_weight='balanced'` in Random Forest

**Tunable Parameters:**
- `forward_days`: 1, 3, 5, 10 (shorter = day trading, longer = swing trading)
- `threshold`: 0.005 (0.5%), 0.01 (1%), 0.02 (2%)

### Files to Create
- `core/models/label_generator.py`

---

## Phase 3.2: Feature Engineering

### Goal
Transform raw data into ML-ready features.

### Current Features (from indicators)
We already have these from Phase 2:
- RSI (14-period)
- MACD (value, signal, histogram)
- Bollinger Bands (upper, middle, lower, position)

### Additional Features to Add

**1. Lagged Features** (past values)
```python
# Yesterday's values
rsi_lag_1 = RSI shifted by 1 day
macd_lag_1 = MACD shifted by 1 day

# 5 days ago
rsi_lag_5 = RSI shifted by 5 days
```

**2. Rate of Change**
```python
rsi_roc = (RSI_today - RSI_yesterday) / RSI_yesterday
macd_roc = (MACD_today - MACD_yesterday) / MACD_yesterday
```

**3. Moving Averages**
```python
sma_20 = 20-day simple moving average
sma_50 = 50-day simple moving average
price_vs_sma20 = (price - sma_20) / sma_20  # Distance from MA
```

**4. Volume Features**
```python
volume_ma_20 = 20-day volume average
volume_ratio = volume / volume_ma_20  # Above/below average?
```

**5. Price Momentum**
```python
momentum_5 = (price_today - price_5_days_ago) / price_5_days_ago
momentum_10 = (price_today - price_10_days_ago) / price_10_days_ago
```

**6. Volatility (from hackathon - ATR)**
```python
atr = Average True Range (14-period)
volatility_ratio = atr / price  # Normalized volatility
```

**7. Bollinger Band Derived Features**
```python
bb_width = (bb_upper - bb_lower) / bb_middle  # Band width %
bb_position = (price - bb_lower) / (bb_upper - bb_lower)  # 0-1 scale
```

### Final Feature List (~25 features)

```python
FEATURES = [
    # Technical Indicators
    'rsi', 'macd', 'macd_signal', 'macd_histogram',
    'bb_upper', 'bb_middle', 'bb_lower', 'bb_width', 'bb_position',
    
    # Moving Averages
    'sma_20', 'sma_50', 'price_vs_sma20', 'price_vs_sma50',
    
    # Momentum
    'momentum_5', 'momentum_10', 'roc_5',
    
    # Volume
    'volume_ma_20', 'volume_ratio',
    
    # Volatility
    'atr', 'volatility_ratio',
    
    # Lagged Features
    'rsi_lag_1', 'macd_lag_1', 'rsi_lag_5',
    
    # Rate of Change
    'rsi_roc', 'macd_roc'
]
```

### Files to Create
- `core/models/feature_engineer.py`

---

## Phase 3.3: Model Training (Walk-Forward)

### Goal
Train Random Forest with proper time-series validation.

### Walk-Forward Validation Strategy

**Problem with Standard Train/Test Split:**
```python
# WRONG - randomly splits data
train, test = train_test_split(data, test_size=0.2)
```

This creates look-ahead bias because model sees future data!

**Correct: Walk-Forward (Expanding Window)**
```python
# Train on past, test on future (never seen)

Train Size: 80% of data
Test Size: 20% of data

Window 1:
├─ Train: [Day 1    → Day 292]  (80% = 292 days)
└─ Test:  [Day 293  → Day 365]  (20% = 73 days)

Then retrain on all data for final model:
└─ Train: [Day 1    → Day 365]  (for production predictions)
```

**More Advanced (Multiple Windows):**
```python
# Test multiple time periods for robustness

Window 1:
├─ Train: [Day 1-200]
└─ Test:  [Day 201-250]

Window 2:
├─ Train: [Day 1-250]  (expanded window)
└─ Test:  [Day 251-300]

Window 3:
├─ Train: [Day 1-300]  (expanded window)
└─ Test:  [Day 301-350]

Average performance across all windows
```

### Random Forest Configuration

```python
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(
    n_estimators=100,        # Number of trees
    max_depth=10,            # Max tree depth (prevent overfitting)
    min_samples_split=20,    # Min samples to split node
    min_samples_leaf=10,     # Min samples in leaf
    class_weight='balanced', # Handle class imbalance
    random_state=42,         # Reproducibility
    n_jobs=-1               # Use all CPU cores
)
```

### Training Pipeline

```python
def train_model(df, features, target):
    """
    Train Random Forest with walk-forward validation
    """
    # 1. Split data (80/20, time-aware)
    split_idx = int(len(df) * 0.8)
    train_data = df.iloc[:split_idx]
    test_data = df.iloc[split_idx:]
    
    # 2. Prepare X, y
    X_train = train_data[features]
    y_train = train_data[target]
    X_test = test_data[features]
    y_test = test_data[target]
    
    # 3. Train model
    model = RandomForestClassifier(...)
    model.fit(X_train, y_train)
    
    # 4. Evaluate on test set (unseen data)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    # 5. Retrain on ALL data for production
    X_all = df[features]
    y_all = df[target]
    final_model = RandomForestClassifier(...)
    final_model.fit(X_all, y_all)
    
    return final_model, accuracy
```

### Files to Create
- `core/models/model_trainer.py`
- `core/validation/walk_forward.py`

---

## Phase 3.4: Prediction Interface

### Goal
Create clean API for making predictions.

### Usage Example

```python
from core.signals.signal_predictor import SignalPredictor

# Initialize predictor
predictor = SignalPredictor(ticker="AAPL", timeframe="1Y")

# Train model
predictor.train()

# Get prediction for today
signal = predictor.predict_latest()
# Returns: {"signal": "BUY", "confidence": 0.73, "probabilities": {...}}

# Get historical predictions (for backtesting)
historical_signals = predictor.predict_historical()
```

### Prediction Output Format

```python
{
    "signal": "BUY",  # BUY, SELL, or HOLD
    "confidence": 0.73,  # 0-1 scale
    "probabilities": {
        "buy": 0.73,
        "hold": 0.18,
        "sell": 0.09
    },
    "reasoning": "Strong buy signal: RSI oversold (32.4), MACD bullish crossover, price near lower Bollinger Band",
    "model_accuracy": 0.68,  # From validation
    "feature_importance": {
        "rsi": 0.25,
        "macd": 0.18,
        "bb_position": 0.15,
        # ... top features
    }
}
```

### Files to Create
- `core/signals/signal_predictor.py`

---

## Integration with App

### Current Flow (Mock Data)
```
User clicks "Analyze" 
→ Fetch stock data 
→ Calculate indicators 
→ Show MOCK prediction (random)
```

### New Flow (Real ML)
```
User clicks "Analyze"
→ Fetch stock data
→ Calculate indicators
→ Generate features
→ Train/load ML model
→ Make prediction
→ Show REAL prediction with confidence
```

### App Changes

**File: `app.py`**

```python
# Current (Phase 2)
from data.indicators.indicators_data import calculate_all_indicators

# Add (Phase 3)
from core.signals.signal_predictor import SignalPredictor

# In analysis section
if analyze_button:
    # ... existing data fetch ...
    
    # NEW: Train model and predict
    predictor = SignalPredictor(ticker, timeframe)
    predictor.train()
    prediction = predictor.predict_latest()
    
    # Display real prediction
    st.session_state.prediction_result = {
        "signal": prediction["signal"],        # Real signal!
        "confidence": prediction["confidence"],  # Real confidence!
        "reasoning": prediction["reasoning"],    # Real reasoning!
        "model_accuracy": prediction["model_accuracy"],
        # ... rest of data ...
    }
```

---

## Dependencies to Add

Update `requirements.txt`:

```txt
# Existing
streamlit
pandas
numpy
yfinance
ta
plotly
scikit-learn  # Already have this!

# May need to add
joblib>=1.3.0        # Model serialization
```

---

## Testing Strategy

### Unit Tests

```python
# tests/models/test_label_generator.py
def test_label_generation():
    """Test that labels are generated correctly"""
    prices = [100, 101, 102, 103, 98]  # Price sequence
    labels = generate_labels(prices, forward_days=2, threshold=0.01)
    assert labels[0] == 1   # BUY (goes to 102 = +2%)
    assert labels[-1] == 0  # HOLD (no future data)

# tests/models/test_walk_forward.py
def test_no_data_leakage():
    """Ensure test data never appears in training"""
    train, test = walk_forward_split(data, train_ratio=0.8)
    assert train.index.max() < test.index.min()  # No overlap
```

### Integration Tests

```python
# tests/integration/test_ml_pipeline.py
def test_full_ml_pipeline():
    """Test complete ML pipeline on sample data"""
    predictor = SignalPredictor("AAPL", "1Y")
    predictor.train()
    prediction = predictor.predict_latest()
    
    assert prediction["signal"] in ["BUY", "SELL", "HOLD"]
    assert 0 <= prediction["confidence"] <= 1
    assert prediction["model_accuracy"] > 0.3  # Better than random
```

---

## Deliverables Checklist

**Phase 3.1 - Label Generation**
- [ ] `core/models/label_generator.py` created
- [ ] Function: `generate_labels(df, forward_days, threshold)`
- [ ] Unit tests pass
- [ ] Handles edge cases (end of data)

**Phase 3.2 - Feature Engineering**
- [ ] `core/models/feature_engineer.py` created
- [ ] Function: `engineer_features(df, indicators)`
- [ ] All 25 features calculated
- [ ] No NaN values (handled properly)

**Phase 3.3 - Model Training**
- [ ] `core/models/model_trainer.py` created
- [ ] `core/validation/walk_forward.py` created
- [ ] Walk-forward validation implemented
- [ ] Random Forest trained successfully
- [ ] Validation accuracy > 50% (better than random)

**Phase 3.4 - Prediction Interface**
- [ ] `core/signals/signal_predictor.py` created
- [ ] Clean API for predictions
- [ ] Returns signal + confidence + reasoning
- [ ] Integrates with app.py

**Phase 3.5 - App Integration**
- [ ] Mock predictions replaced with real ML
- [ ] Confidence score shows real values
- [ ] Reasoning shows actual feature values
- [ ] Model accuracy displayed

---

## Success Criteria

✅ Model achieves >50% accuracy on test data (out-of-sample)  
✅ No data leakage (walk-forward validation working)  
✅ Predictions complete in <5 seconds  
✅ App displays real ML predictions (not mock)  
✅ Confidence scores make sense (high for strong signals)  
✅ All unit tests pass  

---

## Estimated Timeline

- **Phase 3.1** (Label Generation): 30 minutes
- **Phase 3.2** (Feature Engineering): 1 hour
- **Phase 3.3** (Model Training): 1.5 hours
- **Phase 3.4** (Prediction Interface): 1 hour
- **Phase 3.5** (App Integration): 30 minutes

**Total: ~4.5 hours**

---

## Let's Start! 🚀

First step: Create the label generator module.

This will generate Buy/Sell/Hold labels based on future returns.

