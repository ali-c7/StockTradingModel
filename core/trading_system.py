"""
Complete Trading System Pipeline
Ties together all components: data, features, models, backtesting
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple
from datetime import datetime

from data.stock.stock_data import fetch_stock_data
from core.features.technical_features import engineer_all_features, prepare_feature_matrix, get_feature_list
from core.labels.label_generator import generate_labels, print_label_distribution
from core.models.baseline_models import BaselineModelTrainer, compare_models
from core.backtest.portfolio_simulator import PortfolioSimulator


class TradingSystem:
    """
    Complete trading system for a single ticker
    
    Workflow:
        1. Fetch historical data
        2. Engineer technical features
        3. Generate labels
        4. Train model
        5. Generate signals
        6. Backtest strategy
    """
    
    def __init__(
        self,
        ticker: str,
        timeframe: str,
        model_type: str = 'xgboost',
        forward_days: int = 5,
        threshold: float = 0.02,
        train_split: float = 0.8,
        initial_capital: float = 10000.0,
        position_sizing: str = 'kelly'
    ):
        """
        Initialize trading system
        
        Parameters:
            ticker: Stock symbol (e.g., 'AAPL')
            timeframe: Data timeframe ('6M', '1Y', '2Y', '5Y')
            model_type: 'random_forest', 'xgboost', or 'lightgbm'
            forward_days: Days ahead for label generation
            threshold: Return threshold for buy/sell signals
            train_split: Train/test split ratio
            initial_capital: Starting capital for backtesting
            position_sizing: Position sizing method ('kelly', 'fixed', 'equal')
        """
        self.ticker = ticker
        self.timeframe = timeframe
        self.model_type = model_type
        self.forward_days = forward_days
        self.threshold = threshold
        self.train_split = train_split
        self.initial_capital = initial_capital
        self.position_sizing = position_sizing
        
        # Components
        self.raw_data = None
        self.feature_data = None
        self.labels = None
        self.label_distribution = None
        self.model = None
        self.backtest_results = None
        
        # Train/test split
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
    
    def fetch_data(self, verbose: bool = True) -> pd.DataFrame:
        """Step 1: Fetch historical data"""
        if verbose:
            print("\n" + "="*70)
            print(f"STEP 1: FETCHING DATA")
            print("="*70)
            print(f"   Ticker: {self.ticker}")
            print(f"   Timeframe: {self.timeframe}")
        
        self.raw_data = fetch_stock_data(self.ticker, self.timeframe)
        
        if verbose:
            print(f"   ✓ Fetched {len(self.raw_data)} rows")
            print(f"   Date range: {self.raw_data.index[0]} to {self.raw_data.index[-1]}")
        
        return self.raw_data
    
    def engineer_features(self, verbose: bool = True) -> pd.DataFrame:
        """Step 2: Engineer technical features"""
        if self.raw_data is None:
            raise ValueError("Must fetch data first!")
        
        if verbose:
            print("\n" + "="*70)
            print(f"STEP 2: ENGINEERING FEATURES")
            print("="*70)
        
        self.feature_data = engineer_all_features(self.raw_data, verbose=verbose)
        
        return self.feature_data
    
    def generate_labels(self, verbose: bool = True) -> pd.Series:
        """Step 3: Generate training labels"""
        if self.feature_data is None:
            raise ValueError("Must engineer features first!")
        
        if verbose:
            print("\n" + "="*70)
            print(f"STEP 3: GENERATING LABELS")
            print("="*70)
            print(f"   Forward days: {self.forward_days}")
            print(f"   Threshold: {self.threshold:.1%}")
        
        self.labels, self.label_distribution = generate_labels(
            self.feature_data,
            forward_days=self.forward_days,
            threshold=self.threshold,
            adaptive_threshold=True
        )
        
        if verbose:
            print_label_distribution(self.label_distribution)
        
        return self.labels
    
    def prepare_train_test_split(self, verbose: bool = True):
        """Step 4: Prepare train/test sets"""
        if self.labels is None:
            raise ValueError("Must generate labels first!")
        
        if verbose:
            print("\n" + "="*70)
            print(f"STEP 4: PREPARING TRAIN/TEST SPLIT")
            print("="*70)
        
        # Get feature matrix
        feature_list = get_feature_list('all')
        X = prepare_feature_matrix(self.feature_data, features=feature_list)
        
        # Align with labels (remove NaN)
        valid_idx = self.labels.dropna().index
        X = X.loc[valid_idx]
        y = self.labels.loc[valid_idx]
        
        # Time-series split (chronological, no shuffling)
        split_point = int(len(X) * self.train_split)
        
        self.X_train = X.iloc[:split_point]
        self.X_test = X.iloc[split_point:]
        self.y_train = y.iloc[:split_point]
        self.y_test = y.iloc[split_point:]
        
        if verbose:
            print(f"   Features: {X.shape[1]}")
            print(f"   Total samples: {len(X)}")
            print(f"   Train: {len(self.X_train)} samples ({split_point/len(X):.0%})")
            print(f"   Test:  {len(self.X_test)} samples ({(len(X)-split_point)/len(X):.0%})")
    
    def train_model(self, verbose: bool = True) -> Dict:
        """Step 5: Train ML model"""
        if self.X_train is None:
            raise ValueError("Must prepare train/test split first!")
        
        if verbose:
            print("\n" + "="*70)
            print(f"STEP 5: TRAINING MODEL")
            print("="*70)
        
        self.model = BaselineModelTrainer(self.model_type)
        metrics = self.model.train(
            self.X_train, self.y_train,
            self.X_test, self.y_test,
            verbose=verbose
        )
        
        return metrics
    
    def generate_signals(self, verbose: bool = True) -> Tuple[pd.Series, pd.Series]:
        """Step 6: Generate trading signals"""
        if self.model is None:
            raise ValueError("Must train model first!")
        
        if verbose:
            print("\n" + "="*70)
            print(f"STEP 6: GENERATING SIGNALS")
            print("="*70)
        
        # Get full feature matrix
        feature_list = get_feature_list('all')
        X = prepare_feature_matrix(self.feature_data, features=feature_list)
        
        # Generate predictions
        predictions = self.model.predict(X)
        probabilities = self.model.predict_proba(X)
        
        # Convert to Series
        signals = pd.Series(predictions, index=X.index)
        
        # Get confidence (max probability)
        confidences = pd.Series(probabilities.max(axis=1), index=X.index)
        
        if verbose:
            print(f"   Total signals: {len(signals)}")
            print(f"   BUY:  {(signals == 1).sum()} ({(signals == 1).sum()/len(signals)*100:.1f}%)")
            print(f"   HOLD: {(signals == 0).sum()} ({(signals == 0).sum()/len(signals)*100:.1f}%)")
            print(f"   SELL: {(signals == -1).sum()} ({(signals == -1).sum()/len(signals)*100:.1f}%)")
            print(f"   Avg confidence: {confidences.mean():.1%}")
        
        return signals, confidences
    
    def run_backtest(
        self,
        signals: pd.Series,
        confidences: pd.Series,
        initial_capital: float = 10000.0,
        position_sizing: str = 'kelly',
        verbose: bool = True
    ) -> Dict:
        """Step 7: Backtest trading strategy"""
        if signals is None:
            raise ValueError("Must generate signals first!")
        
        if verbose:
            print("\n" + "="*70)
            print(f"STEP 7: BACKTESTING STRATEGY")
            print("="*70)
        
        # Create simulator
        simulator = PortfolioSimulator(
            initial_capital=initial_capital,
            position_sizing=position_sizing,
            transaction_cost=0.001,  # 0.1%
            stop_loss_pct=0.05,      # 5%
            take_profit_pct=0.10,    # 10%
            kelly_fraction=0.5       # Half-Kelly
        )
        
        # Run backtest
        self.backtest_results = simulator.run_backtest(
            df=self.feature_data,
            signals=signals,
            confidences=confidences,
            verbose=verbose
        )
        
        return self.backtest_results
    
    def run_complete_pipeline(self, verbose: bool = True) -> Dict:
        """
        Run complete trading system pipeline
        
        Returns:
            Dictionary with all results
        """
        print("\n" + "="*70)
        print(f"🤖 TRADING SYSTEM: {self.ticker} ({self.timeframe})")
        print("="*70)
        
        # Step 1: Fetch data
        self.fetch_data(verbose=verbose)
        
        # Step 2: Engineer features
        self.engineer_features(verbose=verbose)
        
        # Step 3: Generate labels
        self.generate_labels(verbose=verbose)
        
        # Step 4: Prepare train/test
        self.prepare_train_test_split(verbose=verbose)
        
        # Step 5: Train model
        model_metrics = self.train_model(verbose=verbose)
        
        # Step 6: Generate signals
        signals, confidences = self.generate_signals(verbose=verbose)
        
        # Step 7: Backtest
        backtest_results = self.run_backtest(
            signals, 
            confidences, 
            initial_capital=self.initial_capital,
            position_sizing=self.position_sizing,
            verbose=verbose
        )
        
        # Summary
        print("\n" + "="*70)
        print("✅ PIPELINE COMPLETE")
        print("="*70)
        print(f"\n📊 Final Summary:")
        print(f"   Model Accuracy:   {model_metrics['test_accuracy']:.1%}")
        print(f"   Sharpe Ratio:     {backtest_results['metrics']['sharpe_ratio']:.2f}")
        print(f"   Total Return:     {backtest_results['metrics']['total_return']:+.2%}")
        print(f"   vs Buy & Hold:    {backtest_results['metrics']['alpha']:+.2%}")
        print(f"   Win Rate:         {backtest_results['metrics']['win_rate']:.1%}")
        print(f"   Max Drawdown:     {backtest_results['metrics']['max_drawdown']:.2%}")
        print("="*70)
        
        return {
            'ticker': self.ticker,
            'timeframe': self.timeframe,
            'model_metrics': model_metrics,
            'backtest_results': backtest_results,
            'signals': signals,
            'confidences': confidences,
        }
    
    def get_latest_signal(self) -> Dict:
        """Get the latest trading signal for current day"""
        if self.model is None:
            raise ValueError("Must train model first!")
        
        # Get latest data point
        feature_list = get_feature_list('all')
        X = prepare_feature_matrix(self.feature_data, features=feature_list)
        X_latest = X.iloc[[-1]]
        
        # Predict
        signal = self.model.predict(X_latest)[0]
        proba = self.model.predict_proba(X_latest)[0]
        confidence = proba.max()
        
        signal_names = {-1: 'SELL', 0: 'HOLD', 1: 'BUY'}
        
        return {
            'signal': signal,
            'signal_name': signal_names[signal],
            'confidence': confidence,
            'date': X_latest.index[0],
            'probabilities': {
                'SELL': proba[0],   # Index 0 = SELL probability
                'HOLD': proba[1],   # Index 1 = HOLD probability
                'BUY': proba[2],    # Index 2 = BUY probability
            }
        }


# Example usage
if __name__ == "__main__":
    print("Trading System Module")

