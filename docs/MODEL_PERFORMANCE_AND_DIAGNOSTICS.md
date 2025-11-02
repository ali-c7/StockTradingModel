# Model Performance & Diagnostics Guide

## Part 1: How to Increase Model Accuracy

### Current Performance Context

Your model shows:
- **Training Accuracy**: 100% 🚩 (Red flag - likely overfitting!)
- **Test Accuracy**: 38.3% 
- **Precision**: 66.6%

This gap indicates **overfitting** - the model memorizes training data but doesn't generalize.

---

## Strategies to Increase Accuracy

### 1. **More Training Data** ⭐ (Easiest & Most Effective)

**Why It Works**: More examples → better pattern recognition

**How**:
```python
# Current: 2Y = ~300 usable samples
# Better:  5Y = ~1,000 usable samples

# In UI: Select "5Y" timeframe
```

**Expected Improvement**: +5-15% test accuracy

**Trade-off**: Older data may not reflect current market conditions

---

### 2. **Feature Engineering** 🔧

#### A. Add More Indicators
```python
# Currently: ~50 indicators
# Add:
- Fibonacci retracements
- Pivot points  
- Market breadth indicators
- Sentiment indicators (if available)
- Intermarket data (bonds, VIX, sector ETFs)
```

#### B. Feature Interactions
```python
# Create interaction features
df['rsi_macd_interaction'] = df['rsi'] * df['macd']
df['volume_price_momentum'] = df['volume_change'] * df['returns']
df['trend_strength'] = df['adx'] * (df['ema_50'] > df['ema_200'])
```

#### C. Time-Based Features
```python
# Add temporal context
df['day_of_week'] = df.index.dayofweek
df['month'] = df.index.month
df['quarter'] = df.index.quarter
df['is_month_end'] = (df.index.day > 25)  # Month-end effects
```

---

### 3. **Feature Selection** 🎯

**Problem**: Too many features → model overfits to noise

**Solution**: Keep only the most predictive features

```python
from sklearn.feature_selection import SelectKBest, mutual_info_classif

# Select top 30 features
selector = SelectKBest(mutual_info_classif, k=30)
X_train_selected = selector.fit_transform(X_train, y_train)
X_test_selected = selector.transform(X_test)

# Get selected feature names
selected_features = X_train.columns[selector.get_support()]
```

**Expected Improvement**: +3-8% by removing noise

---

### 4. **Hyperparameter Tuning** ⚙️

#### Current XGBoost Params (Default):
```python
XGBClassifier(
    n_estimators=200,      # Good
    max_depth=5,           # Too shallow?
    learning_rate=0.05,    # Too high?
    random_state=42
)
```

#### Optimized Params (Suggested):
```python
XGBClassifier(
    n_estimators=500,           # More trees
    max_depth=3,                # Shallower (prevent overfit)
    learning_rate=0.01,         # Lower (more careful learning)
    min_child_weight=5,         # Prevent overfit
    subsample=0.8,              # Random sampling
    colsample_bytree=0.8,       # Feature sampling
    gamma=1,                    # Regularization
    reg_alpha=0.1,              # L1 regularization
    reg_lambda=1.0,             # L2 regularization
    random_state=42
)
```

**Expected Improvement**: +5-10% test accuracy

---

### 5. **Handle Class Imbalance** ⚖️

**Problem**: If 60% HOLD, 25% BUY, 15% SELL → model ignores minority classes

**Solutions**:

#### A. Class Weights
```python
from sklearn.utils.class_weight import compute_class_weight

# Automatically balance classes
class_weights = compute_class_weight(
    'balanced', 
    classes=np.unique(y_train), 
    y=y_train
)

model = XGBClassifier(
    scale_pos_weight=class_weights,  # Give more weight to rare classes
    ...
)
```

#### B. SMOTE (Synthetic Minority Oversampling)
```python
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
```

**Expected Improvement**: +2-5% especially for rare classes

---

### 6. **Better Labeling Strategy** 🏷️

#### Current Issues:
- **Fixed forward_days**: Always looks 5 days ahead
- **Binary threshold**: Either BUY/SELL or HOLD
- **No confidence levels**: All labels treated equally

#### Improvements:

#### A. Multi-Horizon Labels
```python
# Look at multiple timeframes
labels_3d = generate_labels(df, forward_days=3, threshold=0.015)
labels_5d = generate_labels(df, forward_days=5, threshold=0.020)
labels_10d = generate_labels(df, forward_days=10, threshold=0.030)

# Combine: Only BUY if ALL agree
labels_combined = (labels_3d + labels_5d + labels_10d) / 3
```

#### B. Tiered Thresholds
```python
# Instead of 2% fixed:
strong_buy_threshold = 0.04    # +4% → Strong BUY
buy_threshold = 0.02           # +2% → BUY
sell_threshold = -0.02         # -2% → SELL
strong_sell_threshold = -0.04  # -4% → Strong SELL

# 5-class problem: Strong BUY, BUY, HOLD, SELL, Strong SELL
```

#### C. Regime-Based Labels
```python
# Different strategies for different market conditions
if adx > 25:  # Trending market
    use_trend_following_labels()
else:  # Ranging market
    use_mean_reversion_labels()
```

**Expected Improvement**: +5-12% by giving clearer signals

---

### 7. **Ensemble Methods** 🎭

**Idea**: Combine multiple models for better predictions

```python
from sklearn.ensemble import VotingClassifier

# Train 3 different models
xgb_model = XGBClassifier(...)
rf_model = RandomForestClassifier(...)
lgbm_model = LGBMClassifier(...)

# Combine via voting
ensemble = VotingClassifier(
    estimators=[
        ('xgb', xgb_model),
        ('rf', rf_model),
        ('lgbm', lgbm_model)
    ],
    voting='soft'  # Use probability averaging
)

ensemble.fit(X_train, y_train)
```

**Expected Improvement**: +3-7% via wisdom of crowds

---

### 8. **Regularization** 🛡️

**Problem**: Model is too complex → overfits

**Solution**: Add penalties for complexity

```python
# XGBoost
model = XGBClassifier(
    reg_alpha=0.5,    # L1 regularization (feature selection)
    reg_lambda=1.0,   # L2 regularization (weight shrinkage)
    min_child_weight=5,  # Minimum samples per leaf
    max_depth=3,      # Limit tree depth
    subsample=0.8,    # Use 80% of data per tree
    colsample_bytree=0.8  # Use 80% of features per tree
)
```

**Expected Improvement**: Reduces overfitting gap by 5-20%

---

### 9. **External Data** 📊

Add context beyond technical indicators:

```python
# Market context
df['vix'] = fetch_vix_data()  # Volatility index
df['spy_return'] = fetch_spy_return()  # Market benchmark
df['sector_performance'] = fetch_sector_etf()

# Fundamental data (if available)
df['pe_ratio'] = stock.info['trailingPE']
df['earnings_date_proximity'] = days_to_earnings()

# Sentiment (advanced)
df['news_sentiment'] = analyze_news_sentiment()
df['social_sentiment'] = analyze_twitter_sentiment()
```

**Expected Improvement**: +5-15% by adding real-world context

---

### 10. **Different Model Architectures** 🏗️

#### A. LSTM (Recurrent Neural Network)
```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Good for sequential patterns
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(lookback, n_features)),
    LSTM(50),
    Dense(3, activation='softmax')
])
```

#### B. Transformer (Attention-Based)
```python
# State-of-the-art for time series
from transformers import TimeSeriesTransformer
```

**Expected Improvement**: +10-25% but requires more data (5Y+)

---

## Part 2: Tracking Predictive Performance

### Current Metrics (Limited)

```python
metrics = {
    'train_accuracy': 100%,  # Too high!
    'test_accuracy': 38.3%,  # Gap = overfitting
    'precision': 66.6%,
    'recall': ??,
    'f1': ??
}
```

### Comprehensive Metrics to Add

---

### 1. **Confusion Matrix** 📊

Shows exactly where model makes mistakes

```python
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

cm = confusion_matrix(y_test, y_pred)

#              Predicted
#              SELL  HOLD  BUY
# Actual SELL   15    3    2   ← 15 correct SELLs, 3 wrong HOLDs, 2 wrong BUYs
#       HOLD    5    20    5   ← 20 correct HOLDs
#        BUY    2    4    14   ← 14 correct BUYs
```

**Insights**:
- Diagonal = correct predictions
- Off-diagonal = mistakes
- Shows which classes are confused

---

### 2. **Per-Class Performance** 🎯

```python
from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred, target_names=['SELL', 'HOLD', 'BUY']))

#              precision    recall  f1-score   support
#
#        SELL       0.68      0.75      0.71        20
#        HOLD       0.74      0.67      0.70        30
#         BUY       0.67      0.70      0.68        20
#
#    accuracy                           0.70        70
```

**Interpretation**:
- **Precision**: Of all predicted BUYs, how many were correct?
- **Recall**: Of all actual BUYs, how many did we catch?
- **F1**: Harmonic mean of precision & recall

---

### 3. **Learning Curves** 📈

Track performance as training set grows

```python
from sklearn.model_selection import learning_curve

train_sizes, train_scores, test_scores = learning_curve(
    model, X, y, 
    train_sizes=np.linspace(0.1, 1.0, 10),
    cv=5
)

# Plot
plt.plot(train_sizes, train_scores.mean(axis=1), label='Training')
plt.plot(train_sizes, test_scores.mean(axis=1), label='Testing')
```

**Diagnosis**:
```
High train, low test → Overfitting (need regularization)
Both low → Underfitting (need more features/complexity)
Both high, converging → Good fit!
```

---

### 4. **Validation Curves** 📉

Track performance vs model complexity

```python
from sklearn.model_selection import validation_curve

param_range = [1, 2, 3, 5, 7, 10]
train_scores, test_scores = validation_curve(
    model, X, y,
    param_name='max_depth',
    param_range=param_range,
    cv=5
)

# Plot train vs test accuracy for different max_depth values
```

**Diagnosis**:
```
Test accuracy peaks at depth=3 → Optimal complexity
Test drops after depth=5 → Overfitting starts
```

---

### 5. **Cross-Validation (Time Series Aware)** ⏱️

```python
from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=5)

scores = []
for train_idx, val_idx in tscv.split(X):
    X_train_fold, X_val_fold = X[train_idx], X[val_idx]
    y_train_fold, y_val_fold = y[train_idx], y[val_idx]
    
    model.fit(X_train_fold, y_train_fold)
    score = model.score(X_val_fold, y_val_fold)
    scores.append(score)

print(f"CV Scores: {scores}")
print(f"Mean: {np.mean(scores):.2%} ± {np.std(scores):.2%}")
```

**Example Output**:
```
CV Scores: [0.35, 0.38, 0.42, 0.36, 0.39]
Mean: 38.0% ± 2.5%
→ Model is consistent (low variance)
```

---

### 6. **Probability Calibration** 🎲

Are the model's confidence scores reliable?

```python
from sklearn.calibration import calibration_curve

prob_true, prob_pred = calibration_curve(
    y_test, 
    model.predict_proba(X_test)[:, 1],  # BUY probability
    n_bins=10
)

# Perfect calibration: line y=x
# Under-confident: curve below diagonal
# Over-confident: curve above diagonal
```

**Interpretation**:
- Model says "70% BUY" → Actually BUYs 70% of the time ✅
- Model says "70% BUY" → Actually BUYs 40% of the time ❌ (over-confident)

---

### 7. **Feature Importance Stability** 🔍

```python
# Train multiple times with different random seeds
importances = []
for seed in range(10):
    model = XGBClassifier(random_state=seed)
    model.fit(X_train, y_train)
    importances.append(model.feature_importances_)

# Calculate mean and std
mean_importance = np.mean(importances, axis=0)
std_importance = np.std(importances, axis=0)

# High std = unstable features (likely noise)
```

---

### 8. **Out-of-Sample Walk-Forward Testing** 🚶

Most realistic evaluation for time series

```python
# Expanding window
results = []
for i in range(len(data) - test_period):
    train_data = data[:train_period + i]
    test_data = data[train_period + i:train_period + i + 1]
    
    model.fit(train_data.X, train_data.y)
    pred = model.predict(test_data.X)
    results.append(pred == test_data.y)

# Average accuracy over all walk-forward steps
```

---

## Part 3: Detecting Overfitting/Underfitting

### Overfitting Indicators 🚨

| Symptom | Diagnosis |
|---------|-----------|
| Train accuracy >> Test accuracy | Overfitting |
| 100% train accuracy | Severe overfitting |
| High variance in CV scores | Unstable model |
| Test accuracy decreases with more epochs | Overfitting during training |
| Complex model (deep trees) + small dataset | High risk of overfit |

**Your Case**: Train=100%, Test=38% → **Severe overfitting!**

### Underfitting Indicators 🐌

| Symptom | Diagnosis |
|---------|-----------|
| Train accuracy ≈ Test accuracy (both low) | Underfitting |
| Both < 40% for 3-class problem | Model too simple |
| Adding features doesn't help | Model capacity insufficient |
| Learning curves plateau early | Can't learn patterns |

---

### Diagnostic Checklist ✅

```python
def diagnose_model_fit(train_acc, test_acc, cv_std):
    """
    Diagnose overfitting/underfitting
    """
    gap = train_acc - test_acc
    
    if gap > 0.30:  # 30% gap
        return "🚨 SEVERE OVERFITTING"
        # Solution: Regularization, more data, simpler model
    
    elif gap > 0.15:  # 15% gap
        return "⚠️ MODERATE OVERFITTING"
        # Solution: Light regularization, feature selection
    
    elif train_acc < 0.45 and test_acc < 0.45:
        return "🐌 UNDERFITTING"
        # Solution: More features, complex model, more data
    
    elif cv_std > 0.10:  # 10% variance
        return "📊 HIGH VARIANCE (Unstable)"
        # Solution: More data, cross-validation tuning
    
    else:
        return "✅ GOOD FIT"

# Your case:
diagnose_model_fit(1.00, 0.383, 0.025)
# → "🚨 SEVERE OVERFITTING"
```

---

## Recommended Action Plan

### Immediate (Quick Wins):

1. ✅ **Use 5Y timeframe** instead of 2Y (+1000 samples)
2. ✅ **Add regularization** to XGBoost (reg_alpha=0.5, reg_lambda=1.0)
3. ✅ **Reduce max_depth** to 3 (prevent memorization)
4. ✅ **Feature selection** (keep top 30 features)

### Short-Term (This Week):

5. ✅ **Implement learning curves** (see training progress)
6. ✅ **Add confusion matrix** (understand mistakes)
7. ✅ **Cross-validation** (get reliable estimate)
8. ✅ **Hyperparameter tuning** (Grid search best params)

### Long-Term (Next Sprint):

9. ✅ **Add external data** (VIX, sector ETFs, market breadth)
10. ✅ **Ensemble models** (combine XGB + RF + LGBM)
11. ✅ **Better labeling** (multi-horizon, regime-based)
12. ✅ **LSTM/Transformer** (if 5Y+ data available)

---

## Expected Improvements

| Change | Expected Gain | Difficulty |
|--------|---------------|------------|
| 5Y data instead of 2Y | +10-15% | Easy |
| Regularization | +5-10% | Easy |
| Feature selection | +3-8% | Medium |
| Hyperparameter tuning | +5-10% | Medium |
| External data | +5-15% | Hard |
| Ensemble | +3-7% | Medium |
| Better labels | +5-12% | Hard |
| LSTM | +10-25% | Very Hard |

**Realistic Target**: 50-60% test accuracy (from current 38%)

---

## Monitoring Dashboard (To Build)

```
┌────────────────────────────────────────────────────────┐
│  MODEL HEALTH DASHBOARD                                │
├────────────────────────────────────────────────────────┤
│  Overfitting Check:                                    │
│  ├─ Train Acc: 65% ✅                                  │
│  ├─ Test Acc:  58% ✅                                  │
│  └─ Gap: 7% ✅ (< 15% threshold)                       │
│                                                        │
│  Cross-Validation:                                     │
│  ├─ Mean: 56% ± 3%                                     │
│  └─ Status: ✅ Stable                                  │
│                                                        │
│  Per-Class Performance:                                │
│  ├─ BUY:  Precision 0.65, Recall 0.70 ✅              │
│  ├─ HOLD: Precision 0.72, Recall 0.68 ✅              │
│  └─ SELL: Precision 0.63, Recall 0.67 ✅              │
│                                                        │
│  Feature Health:                                       │
│  ├─ Top 10 stable (low variance) ✅                    │
│  └─ 5 features flagged (high variance) ⚠️             │
└────────────────────────────────────────────────────────┘
```

---

## Next Steps

Want me to implement any of these? I can add:

1. **Learning curves visualization** to the UI
2. **Confusion matrix** in Advanced view
3. **Cross-validation scores** display
4. **Hyperparameter tuning** slider
5. **Model diagnostics dashboard**

Let me know which would be most helpful!

---

**Created**: 2025-11-01  
**Topic**: Model performance optimization and overfitting detection

