# Why Test Accuracy ≠ Win Rate

## The Confusion

You see:
- **Training Accuracy**: 100% 🚩
- **Test Accuracy**: 41.3%
- **Win Rate**: 91.7% ✅

This seems contradictory! How can the model be "bad" (41% test accuracy) but trades be "good" (91.7% win rate)?

---

## The Answer: They Measure Different Things

### **Test Accuracy** (41.3%)
**Question**: "Of ALL predictions on test days, how many were correct?"

```
Test Period: 60 days
Model makes 60 predictions (one per day)

Predictions breakdown:
- 25 predicted HOLD → 15 correct (60% accuracy on HOLD)
- 20 predicted BUY  → 18 correct (90% accuracy on BUY)
- 15 predicted SELL → 13 correct (87% accuracy on SELL)

Overall: (15 + 18 + 13) / 60 = 46/60 = 76.7% accuracy
```

But wait, why is your actual test accuracy only 41%? Because the model is **terrible at predicting HOLD** (the majority class).

### **Win Rate** (91.7%)
**Question**: "Of EXECUTED trades only, how many were profitable?"

```
Test Period: 60 days
60 predictions made
Only 12 trades executed (high confidence + position requirements met)

Executed trades:
- 6 BUY trades → 5 profitable (83% win rate on BUYs)
- 6 SELL trades → 6 profitable (100% win rate on SELLs)

Overall: 11/12 = 91.7% win rate
```

**Key Insight**: The 12 executed trades are a **highly selective subset** of all 60 predictions. They're the ones where:
1. Model had HIGH confidence (>90%)
2. Position requirements were met (could actually execute)
3. Model was most "sure" about the signal

---

## Visual Breakdown

```
┌─────────────────────────────────────────────────────────────────┐
│  TEST PERIOD: 60 Days                                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Day 1-10: Predict HOLD (conf: 55-70%)                         │
│    → Actually mixed BUY/HOLD/SELL                               │
│    → 40% correct                                                │
│    → NO TRADES (low confidence or position locked)              │
│                                                                 │
│  Day 11: Predict BUY (conf: 94%) ✅                            │
│    → Actually BUY                                               │
│    → TRADE EXECUTED → Profitable ✅                            │
│                                                                 │
│  Day 12-20: Predict HOLD (conf: 60-75%)                        │
│    → Actually mixed                                             │
│    → 35% correct                                                │
│    → NO TRADES                                                  │
│                                                                 │
│  Day 21: Predict SELL (conf: 92%) ✅                           │
│    → Actually SELL                                              │
│    → TRADE EXECUTED → Profitable ✅                            │
│                                                                 │
│  [Pattern continues...]                                         │
│                                                                 │
│  RESULT:                                                        │
│  • Overall accuracy: 41% (many wrong HOLD predictions)          │
│  • Executed trades: 12 (only high-confidence signals)           │
│  • Win rate: 91.7% (high-confidence predictions are good!)      │
└─────────────────────────────────────────────────────────────────┘
```

---

## Why This Happens

### 1. **Confidence Filtering**
```python
# Model outputs probabilities
signal, probabilities = model.predict_proba(day_features)

# Example outputs:
Day 1:  [0.55, 0.30, 0.15] → Predict HOLD (55% conf) → Don't trade
Day 11: [0.02, 0.04, 0.94] → Predict BUY  (94% conf) → Execute trade!
Day 21: [0.92, 0.06, 0.02] → Predict SELL (92% conf) → Execute trade!
```

Only the high-confidence (>90%) predictions result in trades.

### 2. **Class Imbalance**
If test labels are:
- 60% HOLD (36 days)
- 25% BUY (15 days)
- 15% SELL (9 days)

Model might be:
- **Bad at HOLD**: 10/36 correct (28% accuracy) ← Drags down overall accuracy
- **Good at BUY**: 13/15 correct (87% accuracy)
- **Good at SELL**: 8/9 correct (89% accuracy)

**Overall**: (10 + 13 + 8) / 60 = 51.7% accuracy

But when it predicts BUY/SELL with high confidence, those are usually correct!

### 3. **Position Requirements**
```python
# Backtest only executes when conditions allow
if signal == BUY and no_position:
    execute_trade()  # Only happens 6 times in test period
elif signal == SELL and has_position:
    execute_trade()  # Only happens 6 times in test period
```

Even if model predicts 20 BUY signals, only 6 might be executable.

---

## The Real Problem: 100% Training Accuracy

**This is NOT normal!**

A healthy model should have:
- Training: 60-75%
- Test: 55-70%
- Gap: <15%

Your model:
- Training: **100%** 🚩 (Memorizing!)
- Test: 41.3%
- Gap: **58.7%** 🚩 (Severe overfitting)

### Why 100% Training Accuracy is Bad

The model has learned to perfectly classify every training example, which means:
1. **Memorization**: It's learned noise and specific patterns, not general rules
2. **Too complex**: XGBoost trees are too deep or too many
3. **Will fail on new data**: Even though win rate is 91% now, it might drop to 50% on future data

---

## What Should You Do?

### Immediate Actions:

1. **Don't trust the 91.7% win rate yet!**
   - It's based on only 12 trades (small sample)
   - Model is overfitting (100% train accuracy)
   - Might be getting lucky on high-confidence predictions

2. **Verify the results**
   - Run on different stocks (does win rate hold?)
   - Try different time periods
   - Use cross-validation (5-fold)

3. **Fix the overfitting**
   - Use 5Y timeframe (more data)
   - Add regularization (reduce max_depth, add penalties)
   - Feature selection (remove noise)

### Long-Term Validation:

```python
# Paper trading test
1. Train model on data up to Dec 2024
2. Make predictions for Jan-Mar 2025
3. Track actual trades and win rate
4. Compare: Does 91% win rate hold? Or drop to 60%?
```

---

## Scenarios Explained

### Scenario A: True Skill ✅
```
Model is genuinely good at high-confidence BUY/SELL predictions
Even though it sucks at HOLD (drags down accuracy)
→ Win rate will hold up on new data
→ System is profitable
```

### Scenario B: Lucky Overfitting 🚩
```
Model memorized training data
High-confidence predictions happened to be correct on this test set by luck
→ Win rate will drop to 50-60% on new data
→ System will fail
```

**Your case is likely Scenario B** because of the 100% training accuracy.

---

## How to Tell the Difference

### Run These Tests:

1. **Cross-Validation**
```python
# Train on different splits, check consistency
Split 1: Win rate = 91%
Split 2: Win rate = 45%  ← Red flag! Not consistent
Split 3: Win rate = 73%
Split 4: Win rate = 38%
Split 5: Win rate = 82%

Mean: 65.8% ± 22%  ← High variance = overfitting
```

2. **Different Stocks**
```python
AAPL: 91% win rate
GOOGL: 40% win rate  ← Model doesn't generalize
MSFT: 35% win rate
TSLA: 60% win rate
```

3. **Walk-Forward Test**
```python
Train on Jan-Jun 2024 → Test on Jul 2024 → Win rate: 85%
Train on Jan-Jul 2024 → Test on Aug 2024 → Win rate: 92%
Train on Jan-Aug 2024 → Test on Sep 2024 → Win rate: 38% ← Broke!
```

If win rate is consistent across different stocks/periods → True skill
If win rate varies wildly → Lucky overfitting

---

## Bottom Line

**Your question is spot-on!** The results don't make sense because:

1. ✅ **Test accuracy (41%)** tells you overall prediction quality → Poor
2. ✅ **Win rate (91.7%)** tells you executed trade quality → Excellent
3. 🚩 **Training accuracy (100%)** tells you the model is overfitting → Danger!

**Recommendation**: 
- The 91.7% win rate is encouraging but untrustworthy due to severe overfitting
- Fix the overfitting first (5Y data, regularization)
- Then validate on new data before trusting the system
- 12 trades is too small to be statistically significant

**Reality check**: A 91.7% win rate in trading is EXTREMELY rare. Most profitable systems have 55-65% win rates. If you're really getting 91%, you might be:
- Getting lucky on a small sample (12 trades)
- Overfitting to recent market conditions
- Or you've genuinely found something (but verify first!)

---

## Next Steps

1. Add confusion matrix to see where predictions are failing
2. Show confidence distribution for executed vs non-executed trades
3. Cross-validate on multiple time periods
4. Test on different stocks
5. Fix overfitting (use 5Y, add regularization)

---

**Created**: 2025-11-01  
**Topic**: Understanding the disconnect between accuracy and profitability

