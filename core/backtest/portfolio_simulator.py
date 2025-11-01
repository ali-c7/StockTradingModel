"""
Portfolio Simulation and Backtesting Engine
Simulates trading with risk management and position sizing
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from datetime import datetime


class PortfolioSimulator:
    """
    Simulate trading strategy with realistic constraints
    
    Features:
        - Kelly Criterion position sizing
        - Stop-loss and take-profit
        - Transaction costs
        - Portfolio tracking
        - Performance metrics
    """
    
    def __init__(
        self,
        initial_capital: float = 10000.0,
        transaction_cost: float = 0.001,  # 0.1% per trade
        position_sizing: str = 'kelly',  # 'kelly', 'fixed', 'equal'
        max_position_size: float = 0.95,  # Max 95% of capital in one position
        stop_loss_pct: float = 0.05,  # 5% stop loss
        take_profit_pct: float = 0.10,  # 10% take profit
        kelly_fraction: float = 0.5,  # Use half-Kelly for safety
    ):
        """
        Initialize portfolio simulator
        
        Parameters:
            initial_capital: Starting cash
            transaction_cost: Percentage cost per trade (default: 0.1%)
            position_sizing: 'kelly', 'fixed', or 'equal'
            max_position_size: Maximum fraction of capital per position
            stop_loss_pct: Stop loss as fraction of entry price
            take_profit_pct: Take profit as fraction of entry price
            kelly_fraction: Fraction of Kelly Criterion (0.5 = half-Kelly, safer)
        """
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.position_sizing = position_sizing
        self.max_position_size = max_position_size
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.kelly_fraction = kelly_fraction
        
        # Portfolio state
        self.cash = initial_capital
        self.shares = 0
        self.position_entry_price = None
        self.position_type = None  # 'LONG' or None
        
        # Track history
        self.trades = []
        self.portfolio_values = []
        self.signals_history = []
    
    def reset(self):
        """Reset portfolio to initial state"""
        self.cash = self.initial_capital
        self.shares = 0
        self.position_entry_price = None
        self.position_type = None
        self.trades = []
        self.portfolio_values = []
        self.signals_history = []
    
    def calculate_kelly_fraction(
        self,
        win_prob: float,
        avg_win: float,
        avg_loss: float
    ) -> float:
        """
        Calculate Kelly Criterion position size
        
        Kelly formula: f = (p * b - q) / b
        where:
            p = probability of win
            q = probability of loss (1-p)
            b = win/loss ratio (avg_win / avg_loss)
        
        Parameters:
            win_prob: Probability of winning trade (from model confidence)
            avg_win: Average win amount (historical or estimated)
            avg_loss: Average loss amount (historical or estimated)
        
        Returns:
            Fraction of capital to risk (0.0 to 1.0)
        """
        if avg_loss <= 0:
            return 0.0
        
        b = avg_win / avg_loss  # Win/loss ratio
        q = 1 - win_prob  # Loss probability
        
        # Kelly formula
        f = (win_prob * b - q) / b
        
        # Apply Kelly fraction (e.g., half-Kelly)
        f = f * self.kelly_fraction
        
        # Clamp between 0 and max_position_size
        f = max(0.0, min(f, self.max_position_size))
        
        return f
    
    def calculate_position_size(
        self,
        current_price: float,
        signal_confidence: float,
        win_rate: float = 0.5,
        avg_win_pct: float = 0.05,
        avg_loss_pct: float = 0.03
    ) -> float:
        """
        Calculate position size in dollars
        
        Parameters:
            current_price: Current stock price
            signal_confidence: Model confidence (0-1)
            win_rate: Historical win rate
            avg_win_pct: Average win as percentage
            avg_loss_pct: Average loss as percentage
        
        Returns:
            Dollar amount to invest
        """
        available_capital = self.cash
        
        if self.position_sizing == 'fixed':
            # Fixed 20% of capital
            position_value = available_capital * 0.20
        
        elif self.position_sizing == 'equal':
            # Equal weighting (100% / number of stocks, but we only trade 1)
            position_value = available_capital * 0.95
        
        elif self.position_sizing == 'kelly':
            # Kelly Criterion
            # Use signal confidence as win probability
            win_prob = min(max(signal_confidence, 0.4), 0.8)  # Clamp to reasonable range
            
            # Calculate Kelly fraction
            kelly_f = self.calculate_kelly_fraction(
                win_prob=win_prob,
                avg_win=avg_win_pct,
                avg_loss=avg_loss_pct
            )
            
            position_value = available_capital * kelly_f
        
        else:
            position_value = available_capital * 0.20  # Default
        
        # Ensure we can afford at least 1 share
        if position_value < current_price:
            return 0.0
        
        return position_value
    
    def execute_trade(
        self,
        date: pd.Timestamp,
        price: float,
        signal: int,
        signal_confidence: float,
        win_rate: float = 0.5
    ) -> Dict:
        """
        Execute a trade based on signal
        
        Parameters:
            date: Current date
            price: Current stock price
            signal: -1 (SELL) or 1 (BUY) - Binary classification only
            signal_confidence: Model confidence (0-1)
            win_rate: Historical win rate
        
        Returns:
            Dictionary with trade details
        """
        trade_info = {
            'date': date,
            'action': 'NONE',
            'price': price,
            'shares': 0,
            'value': 0,
            'cost': 0,
            'reason': ''
        }
        
        # Check for stop-loss or take-profit on existing position
        if self.position_type == 'LONG' and self.position_entry_price is not None:
            current_return = (price - self.position_entry_price) / self.position_entry_price
            
            # Stop-loss hit
            if current_return <= -self.stop_loss_pct:
                # Close position
                proceeds = self.shares * price * (1 - self.transaction_cost)
                self.cash += proceeds
                
                trade_info['action'] = 'STOP_LOSS'
                trade_info['shares'] = -self.shares
                trade_info['value'] = proceeds
                trade_info['cost'] = self.shares * price * self.transaction_cost
                trade_info['reason'] = f'Stop loss hit ({current_return:.1%})'
                
                self.trades.append(trade_info)
                self.shares = 0
                self.position_entry_price = None
                self.position_type = None
                
                return trade_info
            
            # Take-profit hit
            elif current_return >= self.take_profit_pct:
                # Close position
                proceeds = self.shares * price * (1 - self.transaction_cost)
                self.cash += proceeds
                
                trade_info['action'] = 'TAKE_PROFIT'
                trade_info['shares'] = -self.shares
                trade_info['value'] = proceeds
                trade_info['cost'] = self.shares * price * self.transaction_cost
                trade_info['reason'] = f'Take profit hit ({current_return:.1%})'
                
                self.trades.append(trade_info)
                self.shares = 0
                self.position_entry_price = None
                self.position_type = None
                
                return trade_info
        
        # Handle BUY signal
        if signal == 1 and self.position_type is None:
            # Calculate position size
            position_value = self.calculate_position_size(
                current_price=price,
                signal_confidence=signal_confidence,
                win_rate=win_rate
            )
            
            if position_value > 0:
                # Execute buy
                shares_to_buy = int(position_value / price)
                if shares_to_buy > 0:
                    cost = shares_to_buy * price
                    transaction_fee = cost * self.transaction_cost
                    total_cost = cost + transaction_fee
                    
                    if total_cost <= self.cash:
                        self.cash -= total_cost
                        self.shares += shares_to_buy
                        self.position_entry_price = price
                        self.position_type = 'LONG'
                        
                        trade_info['action'] = 'BUY'
                        trade_info['shares'] = shares_to_buy
                        trade_info['value'] = cost
                        trade_info['cost'] = transaction_fee
                        trade_info['reason'] = f'Buy signal (conf: {signal_confidence:.1%})'
                        
                        self.trades.append(trade_info)
        
        # Handle SELL signal
        elif signal == -1 and self.position_type == 'LONG':
            # Close position
            proceeds = self.shares * price * (1 - self.transaction_cost)
            self.cash += proceeds
            
            trade_info['action'] = 'SELL'
            trade_info['shares'] = -self.shares
            trade_info['value'] = proceeds
            trade_info['cost'] = self.shares * price * self.transaction_cost
            trade_info['reason'] = f'Sell signal (conf: {signal_confidence:.1%})'
            
            self.trades.append(trade_info)
            self.shares = 0
            self.position_entry_price = None
            self.position_type = None
        
        return trade_info
    
    def get_portfolio_value(self, current_price: float) -> float:
        """Calculate current portfolio value"""
        return self.cash + (self.shares * current_price)
    
    def run_backtest(
        self,
        df: pd.DataFrame,
        signals: pd.Series,
        confidences: pd.Series = None,
        verbose: bool = True
    ) -> Dict:
        """
        Run complete backtest
        
        Parameters:
            df: DataFrame with OHLCV data
            signals: Series with -1/0/1 signals
            confidences: Series with confidence scores (0-1)
            verbose: Print progress
        
        Returns:
            Dictionary with backtest results and metrics
        """
        self.reset()
        
        if confidences is None:
            confidences = pd.Series(0.6, index=signals.index)  # Default 60% confidence
        
        if verbose:
            print(f"\n📈 Running Backtest...")
            print(f"   Initial Capital: ${self.initial_capital:,.2f}")
            print(f"   Position Sizing: {self.position_sizing}")
            print(f"   Transaction Cost: {self.transaction_cost:.2%}")
            print(f"   Stop Loss: {self.stop_loss_pct:.1%}")
            print(f"   Take Profit: {self.take_profit_pct:.1%}")
        
        # Track win rate for Kelly Criterion
        wins = 0
        losses = 0
        
        # Simulate each day
        for idx in df.index:
            if idx not in signals.index:
                continue
            
            price = df.loc[idx, 'Close']
            signal = signals.loc[idx]
            confidence = confidences.loc[idx] if idx in confidences.index else 0.6
            
            # Calculate current win rate
            total_trades = wins + losses
            win_rate = wins / total_trades if total_trades > 0 else 0.5
            
            # Execute trade
            trade = self.execute_trade(
                date=idx,
                price=price,
                signal=signal,
                signal_confidence=confidence,
                win_rate=win_rate
            )
            
            # Update win/loss count for completed trades
            if trade['action'] in ['SELL', 'STOP_LOSS', 'TAKE_PROFIT'] and len(self.trades) >= 2:
                # Compare exit price with entry price of previous buy
                buy_trades = [t for t in self.trades if t['action'] == 'BUY']
                if buy_trades:
                    last_buy = buy_trades[-1]
                    pnl = trade['value'] - abs(last_buy['value'])
                    if pnl > 0:
                        wins += 1
                    else:
                        losses += 1
            
            # Track portfolio value
            portfolio_value = self.get_portfolio_value(price)
            self.portfolio_values.append({
                'date': idx,
                'value': portfolio_value,
                'cash': self.cash,
                'position_value': self.shares * price
            })
            
            self.signals_history.append({
                'date': idx,
                'signal': signal,
                'confidence': confidence,
                'price': price
            })
        
        # Calculate final metrics
        final_value = self.portfolio_values[-1]['value'] if self.portfolio_values else self.initial_capital
        total_return = (final_value - self.initial_capital) / self.initial_capital
        
        # Create portfolio DataFrame
        portfolio_df = pd.DataFrame(self.portfolio_values)
        portfolio_df.set_index('date', inplace=True)
        
        # Calculate metrics
        metrics = self.calculate_metrics(portfolio_df, df, verbose=verbose)
        
        return {
            'metrics': metrics,
            'portfolio_history': portfolio_df,
            'trades': self.trades,
            'final_value': final_value,
            'total_return': total_return
        }
    
    def calculate_metrics(
        self,
        portfolio_df: pd.DataFrame,
        price_df: pd.DataFrame,
        verbose: bool = True
    ) -> Dict:
        """Calculate comprehensive performance metrics"""
        
        # Basic returns
        final_value = portfolio_df['value'].iloc[-1]
        initial_value = self.initial_capital
        total_return = (final_value - initial_value) / initial_value
        
        # Buy & Hold comparison
        buy_hold_return = (price_df['Close'].iloc[-1] - price_df['Close'].iloc[0]) / price_df['Close'].iloc[0]
        
        # Daily returns
        portfolio_df['returns'] = portfolio_df['value'].pct_change()
        
        # Sharpe Ratio (annualized, assuming 252 trading days)
        if len(portfolio_df) > 1:
            mean_return = portfolio_df['returns'].mean()
            std_return = portfolio_df['returns'].std()
            sharpe_ratio = (mean_return / std_return) * np.sqrt(252) if std_return > 0 else 0
        else:
            sharpe_ratio = 0
        
        # Maximum Drawdown
        portfolio_df['cummax'] = portfolio_df['value'].cummax()
        portfolio_df['drawdown'] = (portfolio_df['value'] - portfolio_df['cummax']) / portfolio_df['cummax']
        max_drawdown = portfolio_df['drawdown'].min()
        
        # Trade statistics
        if self.trades:
            buy_trades = [t for t in self.trades if t['action'] == 'BUY']
            sell_trades = [t for t in self.trades if t['action'] in ['SELL', 'STOP_LOSS', 'TAKE_PROFIT']]
            
            total_trades = len(sell_trades)
            
            # Win rate
            profitable_trades = 0
            total_pnl = 0
            
            for i, sell in enumerate(sell_trades):
                if i < len(buy_trades):
                    buy = buy_trades[i]
                    pnl = sell['value'] - abs(buy['value'])
                    total_pnl += pnl
                    if pnl > 0:
                        profitable_trades += 1
            
            win_rate = profitable_trades / total_trades if total_trades > 0 else 0
        else:
            total_trades = 0
            win_rate = 0
            total_pnl = 0
        
        metrics = {
            'final_value': float(final_value),
            'total_return': float(total_return),
            'buy_hold_return': float(buy_hold_return),
            'alpha': float(total_return - buy_hold_return),
            'sharpe_ratio': float(sharpe_ratio),
            'max_drawdown': float(max_drawdown),
            'total_trades': int(total_trades),
            'win_rate': float(win_rate),
            'total_pnl': float(total_pnl),
        }
        
        if verbose:
            print(f"\n📊 Backtest Results:")
            print(f"   Final Value:      ${final_value:,.2f}")
            print(f"   Total Return:     {total_return:>+7.2%}")
            print(f"   Buy & Hold:       {buy_hold_return:>+7.2%}")
            print(f"   Alpha:            {metrics['alpha']:>+7.2%}")
            print(f"   Sharpe Ratio:     {sharpe_ratio:>7.2f}")
            print(f"   Max Drawdown:     {max_drawdown:>7.2%}")
            print(f"   Total Trades:     {total_trades}")
            print(f"   Win Rate:         {win_rate:>7.1%}")
        
        return metrics


# Example usage
if __name__ == "__main__":
    print("Portfolio Simulator Module")

