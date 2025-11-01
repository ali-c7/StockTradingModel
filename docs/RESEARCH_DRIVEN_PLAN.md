# Research-Driven Trading Algorithm - Development Plan

## Executive Summary
Complete rebuild of the prediction system using state-of-the-art ML techniques from academic research, focusing on:
- Advanced models (LSTM, ensemble methods)
- Ticker-specific training
- Profit-centric optimization (not just accuracy)
- Comprehensive risk management
- Low false-positive rate through multi-indicator confirmation

## Phase 1: Foundation & Data Infrastructure (Current)

### 1.1 Advanced Feature Engineering ✓ (Existing)
- [x] OHLCV data fetching
- [x] Basic technical indicators (RSI, MACD, BB)
- [ ] **NEW: Expanded indicator suite**
  - Trend: EMA, Ichimoku Cloud, ADX
  - Momentum: Stochastic, CCI, Williams %R
  - Volatility: ATR, Keltner Channels
  - Volume: OBV, Volume MA, VWAP
- [ ] **Feature normalization & scaling**
- [ ] **Temporal window creation** (sequences for LSTM)

### 1.2 Smart Label Generation
- [ ] **Triple-barrier method** (profit/stop/time boundaries)
- [ ] **Magnitude-weighted labeling** (prioritize large moves)
- [ ] **Adaptive thresholds** per ticker volatility
- [ ] **Forward return analysis** (optimize forward_days per ticker)

### 1.3 Data Pipeline
- [ ] **Time-series cross-validation** (walk-forward)
- [ ] **Sequence generators** for LSTM
- [ ] **Class balancing** (SMOTE, class weights)
- [ ] **Data augmentation** for financial time series

---

## Phase 2: Model Architecture

### 2.1 Baseline Models (Start Simple)
- [ ] **Random Forest** (current baseline)
- [ ] **XGBoost** (gradient boosting)
- [ ] **LightGBM** (faster, more efficient)
- [ ] Performance comparison on validation set

### 2.2 Deep Learning Models
- [ ] **LSTM Network**
  - Bidirectional LSTM layers
  - Attention mechanism
  - Dropout for regularization
- [ ] **1D CNN** for pattern recognition
- [ ] **Hybrid CNN-LSTM** (feature extraction + temporal)
- [ ] **Transformer** (if data is sufficient)

### 2.3 Ensemble Methods
- [ ] **Stacking**: Combine RF + XGBoost + LSTM
- [ ] **Voting**: Weighted average of predictions
- [ ] **Confidence scoring**: Only act when models agree

---

## Phase 3: Signal Generation & Filtering

### 3.1 Confidence Thresholding
- [ ] **Probability cutoffs** (only trade if P(signal) > threshold)
- [ ] **Genetic algorithm optimization** for thresholds
- [ ] **Per-class thresholds** (different for BUY/SELL)

### 3.2 Multi-Indicator Confirmation
- [ ] **Trend filter**: Only buy in uptrends (price > EMA200)
- [ ] **Volatility filter**: Avoid trades in extreme volatility
- [ ] **Volume confirmation**: Require volume surge for signals
- [ ] **ADX filter**: Only trade when trend strength is high

### 3.3 Signal Logic
- [ ] **State machine**: Track position (LONG/FLAT/SHORT)
- [ ] **Signal overlap handling**
- [ ] **Cooldown periods** (prevent overtrading)
- [ ] **Regime detection** (bull/bear/sideways)

---

## Phase 4: Profit Optimization

### 4.1 Custom Loss Functions
- [ ] **Return-weighted loss** (prioritize big moves)
- [ ] **Sharpe-maximizing loss**
- [ ] **Profit factor optimization**

### 4.2 Position Sizing
- [ ] **Kelly Criterion** implementation
- [ ] **Fractional Kelly** (half-Kelly for safety)
- [ ] **Volatility-adjusted sizing**
- [ ] **Fixed fractional** (fallback)

### 4.3 Risk Management
- [ ] **Stop-loss**: ATR-based dynamic stops
- [ ] **Take-profit**: Risk-reward ratio targets
- [ ] **Max drawdown limits**
- [ ] **Consecutive loss circuit breaker**

---

## Phase 5: Backtesting & Evaluation

### 5.1 Simulation Engine
- [ ] **Walk-forward backtesting**
- [ ] **Transaction costs** (slippage, commissions)
- [ ] **Position tracking**
- [ ] **Portfolio metrics calculation**

### 5.2 Performance Metrics
- [ ] **Returns**: Total, annualized, CAGR
- [ ] **Risk-adjusted**: Sharpe, Sortino, Calmar
- [ ] **Drawdown**: Max DD, recovery time
- [ ] **Trade stats**: Win rate, profit factor, avg win/loss
- [ ] **Benchmark comparison**: vs. Buy & Hold

### 5.3 Analysis Tools
- [ ] **Equity curve visualization**
- [ ] **Drawdown chart**
- [ ] **Trade distribution**
- [ ] **Confusion matrix**
- [ ] **Feature importance**

---

## Phase 6: Production Features

### 6.1 Ticker-Specific Training
- [ ] **Auto-training pipeline** for any ticker
- [ ] **Model persistence** (save/load per ticker)
- [ ] **Transfer learning** (pre-train on index, fine-tune on ticker)

### 6.2 Adaptive Learning
- [ ] **Periodic retraining** schedule
- [ ] **Performance monitoring**
- [ ] **Auto-adjustment** when accuracy drops
- [ ] **Regime change detection**

### 6.3 Advanced UI
- [ ] **Model comparison dashboard**
- [ ] **Live training progress**
- [ ] **Hyperparameter tuning interface**
- [ ] **Strategy configuration**

---

## Technical Stack

### ML Frameworks
- **TensorFlow/Keras**: LSTM, CNN, deep learning
- **scikit-learn**: RF, baseline models, preprocessing
- **XGBoost/LightGBM**: Gradient boosting
- **TA-Lib**: Advanced technical indicators (optional)

### Optimization
- **Optuna**: Hyperparameter tuning
- **DEAP**: Genetic algorithms for threshold optimization
- **Ray Tune**: Distributed training (if needed)

### Backtesting
- **Backtrader**: Professional backtesting framework
- **Zipline**: Quantitative trading library
- **Custom**: Build our own (more control)

---

## Success Criteria

### Minimum Acceptable Performance (per ticker)
- **Sharpe Ratio**: > 1.5 (good), > 2.0 (excellent)
- **Win Rate**: > 45% (quality over quantity)
- **Profit Factor**: > 1.5
- **Max Drawdown**: < 20%
- **vs. Buy & Hold**: Outperform in Sharpe, even if not in raw return

### False Positive Control
- **Precision**: > 60% (6 out of 10 signals are correct)
- **Conservative default**: "Hold" unless strong evidence
- **Trade frequency**: 1-10 trades per month (not daily churn)

---

## Key Research References

1. **LSTM for Stock Prediction**: MDPI studies showing 60-70% directional accuracy
2. **DenseNet + Technical Indicators**: State-of-the-art multi-horizon forecasting
3. **Genetic Algorithm Optimization**: Wu et al. - threshold optimization for profitability
4. **Kelly Criterion**: Mathematical position sizing for growth maximization
5. **Ensemble Methods**: Combining multiple models reduces false positives
6. **Triple-Barrier Method**: Structured labeling that balances profit/risk
7. **Regime-Specific Models**: Better than one-size-fits-all (Alpha Architect)

---

## Implementation Order

### Week 1-2: Foundation
1. Advanced feature engineering
2. Smart label generation
3. Data pipeline + validation split

### Week 3-4: Models
4. Baseline models (RF, XGBoost)
5. LSTM implementation
6. Ensemble framework

### Week 5-6: Optimization
7. Custom loss functions
8. Position sizing (Kelly)
9. Risk management

### Week 7-8: Backtesting
10. Simulation engine
11. Metrics calculation
12. Visualization

### Week 9-10: Production
13. Ticker-specific training
14. UI integration
15. Testing & refinement

---

## Next Steps

**Immediate Action**: Start with Phase 1.1 - Expand indicator suite and feature engineering.

**Philosophy**: Start simple (baseline models), validate, then add complexity (deep learning) only if it improves results.

**Mantra**: "Profit-first, not accuracy-first" - optimize for Sharpe ratio, not classification accuracy.

