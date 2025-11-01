# Backtesting & Portfolio Simulation Design

## Overview

The **$1000 Portfolio Simulation** feature allows users to see how the trading strategy would have performed with a $1000 initial investment over the selected timeframe.

This is implemented in **Phase 4.2** and visualized in **Phase 4.3**.

---

## Key Features

### 1. Realistic Trade Simulation 💰

Unlike basic backtests, ours includes:

**Transaction Costs**
- Commission: 0.1% per trade (typical for online brokers)
- Slippage: 0.05% (price movement during execution)
- Total cost: ~$1-2 per trade on $1000 position

**Position Sizing**
- Option 1: **Fixed Allocation** - Always use 100% of capital
- Option 2: **Kelly Criterion** - Optimal position size based on win rate
- Option 3: **Fixed Percentage** - Use 80% of capital, keep 20% cash buffer

**Risk Management**
- Stop-loss: Exit position if down 5% from entry
- Take-profit: Exit position if up 10% from entry (optional)
- Max position size: Never exceed 100% of capital

---

## Metrics Displayed

### Performance Summary

```
📊 BACKTESTING RESULTS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Initial Capital:    $1,000.00
Final Value:        $1,247.82
Total Return:       +24.78%
Period:             Jan 2024 - Oct 2025 (22 months)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📈 STRATEGY PERFORMANCE
vs Buy & Hold:      +8.2% outperformance
Annualized Return:  13.5%
Sharpe Ratio:       1.34 (risk-adjusted return)
Max Drawdown:       -12.4% (largest peak-to-trough decline)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

💼 TRADING ACTIVITY
Total Trades:       24
├─ Buy Signals:     12
├─ Sell Signals:    12
└─ Holds:           310 days (81% of period)

Win Rate:           58.33% (7 winners, 5 losers)
Avg Winning Trade:  +5.2%
Avg Losing Trade:   -2.8%
Profit Factor:      2.1 (gross profit / gross loss)

Best Trade:         +12.4% (Jun 15, 2024)
Worst Trade:        -5.2% (Aug 3, 2024)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

💸 COSTS & EFFICIENCY
Total Commission:   $4.80 (24 trades × $0.20 avg)
Total Slippage:     $2.40
Net Profit:         $247.82
```

---

## Visualizations

### 1. Portfolio Value Chart

**Interactive Plotly chart showing:**
- Portfolio value over time (blue line)
- Buy & Hold benchmark (gray dotted line)
- Buy markers (green triangles ▲)
- Sell markers (red triangles ▼)
- Drawdown shaded areas (pink)

```python
# Example chart
Portfolio Value: $1,000 → $1,248
Buy & Hold:      $1,000 → $1,142
Outperformance:  +$106 (+9.3%)
```

### 2. Trade History Table

**Expandable table with all trades:**

| # | Date | Action | Price | Shares | Value | P/L | P/L % |
|---|------|--------|-------|--------|-------|-----|-------|
| 1 | 2024-01-15 | BUY | $182.30 | 5.48 | $1,000 | - | - |
| 2 | 2024-01-22 | SELL | $189.45 | 5.48 | $1,038 | +$38 | +3.8% |
| 3 | 2024-02-10 | BUY | $185.20 | 5.60 | $1,038 | - | - |
| 4 | 2024-02-18 | SELL | $192.80 | 5.60 | $1,080 | +$42 | +4.0% |
| ... | ... | ... | ... | ... | ... | ... | ... |

**Color coding:**
- 🟢 Green rows = Profitable trades
- 🔴 Red rows = Losing trades
- ⚪ Gray rows = Entry points (no P/L yet)

### 3. Monthly Returns Heatmap (Optional)

```
         Jan   Feb   Mar   Apr   May   Jun   Jul   Aug   Sep   Oct   Nov   Dec
2024    +3.8% +4.0% -1.2% +5.6% +2.1% +7.3% -2.5% -3.1% +4.2% +1.8% +2.5% +3.1%
2025    +1.2% +2.8% +4.5% -1.8% +3.2% +5.1% +1.9% -0.5% +2.7% +3.4%   -     -
```

---

## Implementation Details

### File Structure

```
core/
  └── backtest/
      ├── __init__.py
      ├── backtest_core.py       # Main backtesting engine
      ├── portfolio.py           # Portfolio state management
      ├── trade_executor.py      # Trade execution logic
      └── metrics.py             # Performance metrics calculation

plots/
  └── performance/
      ├── __init__.py
      └── portfolio_plot.py      # Portfolio visualization
```

### Core Logic

**`backtest_core.py`**

```python
class Backtester:
    """
    Simulate trading strategy with realistic constraints
    """
    
    def __init__(self, initial_capital=1000, commission=0.001, slippage=0.0005):
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        self.portfolio = Portfolio(initial_capital)
        
    def run(self, signals, prices):
        """
        Execute backtest on historical signals
        
        Args:
            signals: Series of Buy(1)/Hold(0)/Sell(-1) predictions
            prices: Series of historical prices
            
        Returns:
            BacktestResults object with metrics and trade history
        """
        trades = []
        portfolio_values = []
        
        for date, signal in signals.items():
            price = prices[date]
            
            if signal == 1:  # Buy
                self._execute_buy(date, price, trades)
            elif signal == -1 and self.portfolio.has_position():  # Sell
                self._execute_sell(date, price, trades)
                
            portfolio_values.append(self.portfolio.get_value(price))
        
        return BacktestResults(
            trades=trades,
            portfolio_values=portfolio_values,
            initial_capital=self.initial_capital
        )
    
    def _execute_buy(self, date, price, trades):
        """Execute buy order with transaction costs"""
        effective_price = price * (1 + self.commission + self.slippage)
        shares = self.portfolio.cash / effective_price
        
        if shares > 0:
            self.portfolio.buy(shares, effective_price)
            trades.append(Trade(date, 'BUY', price, shares))
    
    def _execute_sell(self, date, price, trades):
        """Execute sell order with transaction costs"""
        effective_price = price * (1 - self.commission - self.slippage)
        cash = self.portfolio.shares * effective_price
        
        entry_price = trades[-1].price  # Last buy price
        pnl = (price - entry_price) / entry_price
        
        self.portfolio.sell(effective_price)
        trades.append(Trade(date, 'SELL', price, self.portfolio.shares, pnl))
```

**`metrics.py`**

```python
def calculate_metrics(results):
    """Calculate comprehensive performance metrics"""
    
    # Basic metrics
    total_return = (results.final_value - results.initial_capital) / results.initial_capital
    
    # Trade metrics
    winning_trades = [t for t in results.trades if t.pnl > 0]
    losing_trades = [t for t in results.trades if t.pnl < 0]
    win_rate = len(winning_trades) / len(results.trades) if results.trades else 0
    
    # Risk metrics
    returns = pd.Series(results.portfolio_values).pct_change()
    sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252)  # Annualized
    max_drawdown = calculate_max_drawdown(results.portfolio_values)
    
    return PerformanceMetrics(
        total_return=total_return,
        win_rate=win_rate,
        sharpe_ratio=sharpe_ratio,
        max_drawdown=max_drawdown,
        # ... more metrics
    )
```

---

## Comparison vs Hackathon Code

### What We're Taking From Them ✅

1. **Trade execution logic** - Buy/sell based on signals
2. **Portfolio value tracking** - Track equity over time
3. **Win rate calculation** - Profitable trades / total trades
4. **Trade log structure** - List of all buy/sell actions

### What We're Improving 🚀

1. **Out-of-sample testing** 
   - Theirs: Tests on training data (unrealistic)
   - Ours: Tests on unseen data (realistic)

2. **Transaction costs**
   - Theirs: Free trading (unrealistic)
   - Ours: Commission + slippage (realistic)

3. **Position sizing**
   - Theirs: Always 100% in/out (risky)
   - Ours: Smart sizing with Kelly Criterion

4. **Risk management**
   - Theirs: No stops
   - Ours: Stop-loss and take-profit

5. **Comparison benchmark**
   - Theirs: Just strategy results
   - Ours: vs Buy & Hold + market index

6. **Metrics depth**
   - Theirs: Total return + win rate
   - Ours: + Sharpe, drawdown, profit factor, etc.

---

## User Flow

### In the App

1. User selects ticker and timeframe (e.g., AAPL, 1Y)
2. Clicks "🔍 Analyze Stock"
3. App shows prediction (BUY/SELL/HOLD) for today
4. **New section appears below**: "📊 Strategy Performance"

**Strategy Performance Section:**

```
┌─────────────────────────────────────────────┐
│ 📊 Strategy Performance                     │
│                                             │
│ If you invested $1,000 at the start of     │
│ this period and followed our signals:       │
│                                             │
│ ╔═══════════════════════════════════╗      │
│ ║  Portfolio Value: $1,247.82       ║      │
│ ║  Total Return:    +24.78%         ║      │
│ ║  vs Buy & Hold:   +8.2% better    ║      │
│ ╚═══════════════════════════════════╝      │
│                                             │
│ [View Detailed Backtest Results]           │
└─────────────────────────────────────────────┘
```

5. User clicks "View Detailed Backtest Results"
6. Expands to show:
   - Portfolio value chart
   - Trade history table
   - All metrics
   - Comparison with buy & hold

---

## Configuration Options (Sidebar)

```
⚙️ Backtest Settings

Initial Capital:
[$1000] (custom amount)

Transaction Costs:
☑ Include commission (0.1%)
☑ Include slippage (0.05%)

Position Sizing:
○ Fixed 100%
● Kelly Criterion
○ Fixed 80%

Risk Management:
☑ Enable stop-loss (-5%)
☐ Enable take-profit (+10%)
```

---

## Future Enhancements (Phase 5)

1. **Multiple Starting Dates** - Test robustness with different entry points
2. **Monte Carlo Simulation** - Show range of possible outcomes
3. **Parameter Optimization** - Find best stop-loss/take-profit levels
4. **Compare Strategies** - Rule-based vs ML vs Buy & Hold side-by-side
5. **Export Results** - Download backtest report as PDF

---

## Acceptance Criteria

✅ Backtesting runs on out-of-sample data only  
✅ Transaction costs are included  
✅ Portfolio value is tracked over time  
✅ Win rate and total return are calculated  
✅ Results are compared to buy & hold benchmark  
✅ Interactive portfolio chart is displayed  
✅ Trade history table shows all trades  
✅ Max drawdown and Sharpe ratio are calculated  
✅ Results load in < 2 seconds for 1Y of data  

---

**Status:** Planned for Phase 4.2  
**Priority:** High (core feature)  
**Estimated Effort:** 4-6 hours implementation  

