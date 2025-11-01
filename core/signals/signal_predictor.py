"""
Signal Predictor Module
Main API for generating Buy/Sell/Hold predictions
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple
from datetime import datetime

# Import our modules
from data.stock.stock_data import fetch_stock_data
from core.models.feature_engineer import prepare_feature_matrix, get_feature_list, validate_features, ML_FEATURES_LITE
from core.models.label_generator import generate_labels, analyze_label_distribution
from core.models.model_trainer import ModelTrainer
from core.utils.diagnostics import diagnose_model_performance
from core.validation.walk_forward import walk_forward_validation


class SignalPredictor:
    """
    Main prediction interface for Buy/Sell/Hold signals
    
    Usage:
        predictor = SignalPredictor("AAPL", "1Y")
        predictor.train()
        prediction = predictor.predict_latest()
    """
    
    def __init__(
        self,
        ticker: str,
        timeframe: str,
        forward_days: int = 3,
        threshold: float = 0.01,
        train_ratio: float = 0.8
    ):
        """
        Initialize predictor
        
        Parameters:
            ticker: Stock symbol (e.g., "AAPL")
            timeframe: Analysis timeframe ("1M", "6M", "1Y", etc.)
            forward_days: Days to look ahead for label generation (default: 3)
            threshold: Return threshold for Buy/Sell signals (default: 0.01 = 1%)
            train_ratio: Train/test split ratio (default: 0.8 = 80% train)
        """
        self.ticker = ticker
        self.timeframe = timeframe
        self.forward_days = forward_days
        self.threshold = threshold
        self.train_ratio = train_ratio
        
        # Will be populated during training
        self.data = None
        self.raw_data = None  # Alias for original OHLCV data (for visualization)
        self.features_df = None
        self.X = None
        self.y = None
        self.trainer = None
        self.metrics = None
        self.label_distribution = None
        self.feature_list = get_feature_list()
        
    def fetch_and_prepare_data(self) -> pd.DataFrame:
        """
        Fetch stock data and prepare features
        
        Returns:
            DataFrame with OHLCV data and all features
        """
        print(f"\nFetching data for {self.ticker} ({self.timeframe})...")
        
        # Fetch stock data
        self.data = fetch_stock_data(self.ticker, self.timeframe)
        
        if self.data is None or self.data.empty:
            raise ValueError(f"No data available for {self.ticker}")
        
        # Keep a copy of raw data for visualization
        self.raw_data = self.data.copy()
        
        print(f"✓ Fetched {len(self.data)} days of data")
        
        # Prepare features
        print("Preparing features...")
        self.features_df = prepare_feature_matrix(self.data)
        
        print(f"✓ Created {len(self.feature_list)} features")
        
        # Validate features
        validation = validate_features(self.features_df)
        if not validation['valid']:
            print(f"⚠️  Missing features: {validation['missing_features']}")
        
        print(f"✓ Ready: {validation['row_count']} samples after cleaning")
        
        return self.features_df
    
    def generate_training_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Generate X (features) and y (labels) for training
        
        Returns:
            Tuple of (X, y)
        """
        if self.features_df is None:
            raise ValueError("Must call fetch_and_prepare_data() first")
        
        print(f"\nGenerating labels (forward_days={self.forward_days}, threshold={self.threshold*100}%)...")
        
        # Generate labels
        labels, analysis = generate_labels(
            self.features_df,
            forward_days=self.forward_days,
            threshold=self.threshold
        )
        
        # Analyze distribution
        self.label_distribution = analyze_label_distribution(labels)
        print(f"✓ Label distribution:")
        print(f"    BUY:  {self.label_distribution['buy_count']:3d} ({self.label_distribution['buy_pct']:.1f}%)")
        print(f"    HOLD: {self.label_distribution['hold_count']:3d} ({self.label_distribution['hold_pct']:.1f}%)")
        print(f"    SELL: {self.label_distribution['sell_count']:3d} ({self.label_distribution['sell_pct']:.1f}%)")
        
        # Prepare X and y
        # Align features with labels (remove rows with NaN labels)
        valid_idx = labels.dropna().index
        
        # Auto-select feature set based on data size
        initial_samples = len(valid_idx)
        samples_per_feature = initial_samples / len(self.feature_list)
        
        # Need at least 20 samples per feature for good performance
        if samples_per_feature < 20:
            # Not enough data for full feature set
            print(f"\n⚠️  Limited data detected: {initial_samples} samples for {len(self.feature_list)} features")
            print(f"   Ratio: {samples_per_feature:.1f} samples/feature (need 20+)")
            print(f"   Switching to LITE feature set (11 features instead of {len(self.feature_list)})")
            
            # Use lite features
            self.feature_list = ML_FEATURES_LITE
            samples_per_feature = initial_samples / len(self.feature_list)
            print(f"   New ratio: {samples_per_feature:.1f} samples/feature")
            
            if samples_per_feature < 15:
                print(f"\n⚠️  WARNING: Still limited data even with LITE features!")
                print(f"   Recommendation: Use 5Y timeframe for better results")
        
        self.X = self.features_df.loc[valid_idx, self.feature_list]
        self.y = labels.loc[valid_idx]
        
        print(f"✓ Training dataset: {len(self.X)} samples, {len(self.feature_list)} features")
        
        return self.X, self.y
    
    def train(self, verbose: bool = True) -> Dict:
        """
        Complete training pipeline
        
        Parameters:
            verbose: Print detailed progress
        
        Returns:
            Dictionary with training metrics
        """
        # Step 1: Fetch and prepare data
        self.fetch_and_prepare_data()
        
        # Step 2: Generate training data
        self.generate_training_data()
        
        # Step 3: Train model
        self.trainer = ModelTrainer()
        self.metrics = self.trainer.train_and_evaluate(
            self.X,
            self.y,
            train_ratio=self.train_ratio,
            verbose=verbose
        )
        
        # Display comprehensive diagnostics if performance is poor
        if self.metrics['accuracy'] < 0.55:
            diagnostic_report = diagnose_model_performance(
                accuracy=self.metrics['accuracy'],
                training_samples=len(self.X),
                label_distribution=self.label_distribution,
                ticker=self.ticker,
                timeframe=self.timeframe
            )
            print(diagnostic_report)
        
        # Step 4: Retrain on all data for production
        self.trainer.retrain_on_all_data(self.X, self.y)
        
        return self.metrics
    
    def predict_latest(self) -> Dict:
        """
        Predict signal for the most recent day
        
        Returns:
            Dictionary with prediction details
        """
        if self.trainer is None or self.trainer.model is None:
            raise ValueError("Model not trained yet. Call train() first.")
        
        # Get latest data point
        latest_features = self.X.iloc[[-1]]  # Last row
        
        # Make prediction
        signal_numeric = self.trainer.predict(latest_features)[0]
        probabilities = self.trainer.predict_proba(latest_features)[0]
        
        # Map numeric to string
        signal_map = {-1: "SELL", 0: "HOLD", 1: "BUY"}
        signal = signal_map[signal_numeric]
        
        # Get probabilities for each class
        # RandomForest returns probabilities in order of classes: [-1, 0, 1]
        prob_dict = {
            "sell": float(probabilities[0]),
            "hold": float(probabilities[1]),
            "buy": float(probabilities[2])
        }
        
        # Confidence = probability of predicted class
        confidence = prob_dict[signal.lower()]
        
        # Generate reasoning
        reasoning = self._generate_reasoning(signal, latest_features, prob_dict)
        
        # Get feature importance
        top_features = self.trainer.get_feature_importance(top_n=5)
        
        return {
            "signal": signal,
            "confidence": confidence,
            "probabilities": prob_dict,
            "reasoning": reasoning,
            "model_accuracy": self.metrics.get('accuracy', 0),
            "feature_importance": top_features.to_dict('records'),
            "prediction_date": datetime.now().isoformat()
        }
    
    def predict_historical(self) -> pd.DataFrame:
        """
        Generate predictions for all historical data
        
        Useful for backtesting
        
        Returns:
            DataFrame with date, actual_signal, predicted_signal, probabilities
        """
        if self.trainer is None or self.trainer.model is None:
            raise ValueError("Model not trained yet. Call train() first.")
        
        # Predict on full dataset
        predictions = self.trainer.predict(self.X)
        probabilities = self.trainer.predict_proba(self.X)
        
        # Create results dataframe
        results = pd.DataFrame({
            'date': self.X.index,
            'predicted_signal': predictions,
            'buy_prob': probabilities[:, 2],
            'hold_prob': probabilities[:, 1],
            'sell_prob': probabilities[:, 0]
        })
        
        # Add actual signals if available
        if self.y is not None:
            results['actual_signal'] = self.y.values
        
        return results
    
    def _generate_reasoning(
        self,
        signal: str,
        features: pd.DataFrame,
        probabilities: Dict
    ) -> str:
        """
        Generate human-readable reasoning for the prediction
        
        Parameters:
            signal: Predicted signal (BUY/SELL/HOLD)
            features: Feature DataFrame (single row)
            probabilities: Probability dictionary
        
        Returns:
            String with reasoning
        """
        # Get feature values
        feature_values = features.iloc[0]
        
        # Build reasoning based on key indicators
        reasons = []
        
        # RSI
        if 'rsi' in feature_values:
            rsi = feature_values['rsi']
            if rsi < 30:
                reasons.append(f"RSI oversold ({rsi:.1f})")
            elif rsi > 70:
                reasons.append(f"RSI overbought ({rsi:.1f})")
            else:
                reasons.append(f"RSI neutral ({rsi:.1f})")
        
        # MACD
        if 'macd' in feature_values and 'macd_signal' in feature_values:
            macd = feature_values['macd']
            macd_signal = feature_values['macd_signal']
            if macd > macd_signal:
                reasons.append("MACD bullish crossover")
            else:
                reasons.append("MACD bearish crossover")
        
        # Bollinger Bands
        if 'bb_position' in feature_values:
            bb_pos = feature_values['bb_position']
            if bb_pos < 0.2:
                reasons.append("Price near lower Bollinger Band")
            elif bb_pos > 0.8:
                reasons.append("Price near upper Bollinger Band")
        
        # Construct final reasoning
        signal_emoji = {"BUY": "🟢", "HOLD": "🟡", "SELL": "🔴"}[signal]
        confidence_pct = probabilities[signal.lower()] * 100
        
        reasoning = f"{signal_emoji} {signal} signal (confidence: {confidence_pct:.1f}%)"
        
        if reasons:
            reasoning += f": {', '.join(reasons)}"
        
        return reasoning
    
    def get_metrics_summary(self) -> Dict:
        """
        Get training metrics summary
        
        Returns:
            Dictionary with key metrics
        """
        if self.metrics is None:
            return {
                "status": "not_trained",
                "message": "Model not trained yet"
            }
        
        return {
            "status": "trained",
            "accuracy": self.metrics['accuracy'],
            "test_size": self.metrics['test_size'],
            "per_class_accuracy": self.metrics['per_class_accuracy'],
            "ticker": self.ticker,
            "timeframe": self.timeframe
        }
    
    def save(self, filepath: str) -> None:
        """
        Save trained model to disk
        
        Parameters:
            filepath: Path to save model
        """
        if self.trainer is None:
            raise ValueError("No model to save")
        
        self.trainer.save_model(filepath)
    
    def load(self, filepath: str) -> None:
        """
        Load trained model from disk
        
        Parameters:
            filepath: Path to load model from
        """
        self.trainer = ModelTrainer()
        self.trainer.load_model(filepath)
    
    def run_comprehensive_validation(self, n_splits: int = 5) -> Dict:
        """
        Run comprehensive validation to test model robustness
        
        This goes beyond simple train/test split by testing on multiple
        time windows to ensure the model works consistently.
        
        Parameters:
            n_splits: Number of validation windows
        
        Returns:
            Dictionary with validation results
        """
        if self.X is None or self.y is None:
            raise ValueError("Must call fetch_and_prepare_data() and generate_training_data() first")
        
        print("\n" + "="*70)
        print("🔬 COMPREHENSIVE VALIDATION")
        print("="*70)
        print("Testing model across multiple time windows...")
        print("This helps verify the model works on ANY stock, not just this one.")
        print("="*70)
        
        # Get model parameters from trainer
        model_params = None
        if self.trainer:
            model_params = self.trainer.model_params
        
        # Run walk-forward validation
        validation_results = walk_forward_validation(
            self.X,
            self.y,
            n_splits=n_splits,
            test_size=0.2,
            model_params=model_params
        )
        
        # Add interpretation
        mean_acc = validation_results.get('mean_accuracy', 0)
        std_acc = validation_results.get('std_accuracy', 0)
        
        print("\n" + "="*70)
        print("🎯 VALIDATION VERDICT")
        print("="*70)
        
        if mean_acc < 0.40:
            print("❌ POOR: Model is not learning useful patterns")
            print("   Recommendation: Try longer timeframe or different stock")
        elif mean_acc < 0.50:
            print("⚠️  WEAK: Model has weak predictive power")
            print("   Recommendation: Consider longer timeframe")
        elif mean_acc < 0.60:
            print("✓ ACCEPTABLE: Model is learning useful patterns")
            print("   This is reasonable for stock prediction")
        elif mean_acc < 0.70:
            print("✅ GOOD: Model has strong predictive power")
            print("   Better than most stock prediction models")
        else:
            print("🎯 EXCELLENT: Model has exceptional predictive power")
            print("   This is rare for stock prediction!")
        
        if std_acc > 0.15:
            print("\n⚠️  HIGH VARIABILITY: Performance varies significantly across time periods")
            print("   The model may work well in some market conditions but not others")
        elif std_acc > 0.10:
            print("\n⚠️  MODERATE VARIABILITY: Some inconsistency across time periods")
        else:
            print("\n✓ CONSISTENT: Model performs consistently across time periods")
        
        print("="*70 + "\n")
        
        return validation_results


# Example usage
if __name__ == "__main__":
    print("Signal Predictor Module")
    print("\nExample usage:")
    print("  predictor = SignalPredictor('AAPL', '1Y')")
    print("  predictor.train()")
    print("  prediction = predictor.predict_latest()")

