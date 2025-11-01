"""
Model Training Module
Trains Random Forest classifier with walk-forward validation
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from typing import Tuple, Dict
import joblib
from datetime import datetime


class ModelTrainer:
    """
    Train and evaluate ML models with proper time-series validation
    """
    
    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 10,
        min_samples_split: int = 20,
        min_samples_leaf: int = 10,
        random_state: int = 42
    ):
        """
        Initialize model trainer
        
        Parameters:
            n_estimators: Number of trees in Random Forest
            max_depth: Maximum depth of trees (prevents overfitting)
            min_samples_split: Min samples required to split node
            min_samples_leaf: Min samples required in leaf node
            random_state: Random seed for reproducibility
        """
        self.model_params = {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'min_samples_split': min_samples_split,
            'min_samples_leaf': min_samples_leaf,
            'class_weight': 'balanced',  # Handle class imbalance
            'random_state': random_state,
            'n_jobs': -1  # Use all CPU cores
        }
        
        self.model = None
        self.feature_names = None
        self.training_metrics = {}
        
    def train_test_split_timeseries(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        train_ratio: float = 0.8
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Split data maintaining time order (no shuffling!)
        
        Parameters:
            X: Feature DataFrame
            y: Target Series
            train_ratio: Proportion of data for training (default: 80%)
        
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        
        Note: NEVER use sklearn's train_test_split() for time-series!
              It shuffles data, causing look-ahead bias
        """
        split_idx = int(len(X) * train_ratio)
        
        X_train = X.iloc[:split_idx]
        X_test = X.iloc[split_idx:]
        y_train = y.iloc[:split_idx]
        y_test = y.iloc[split_idx:]
        
        return X_train, X_test, y_train, y_test
    
    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series
    ) -> None:
        """
        Train Random Forest model
        
        Parameters:
            X_train: Training features
            y_train: Training labels
        """
        # Store feature names
        self.feature_names = X_train.columns.tolist()
        
        # Initialize model
        self.model = RandomForestClassifier(**self.model_params)
        
        # Train model
        print(f"Training Random Forest with {len(X_train)} samples...")
        self.model.fit(X_train, y_train)
        print("✓ Training complete")
        
    def evaluate(
        self,
        X_test: pd.DataFrame,
        y_test: pd.Series
    ) -> Dict:
        """
        Evaluate model on test set
        
        Parameters:
            X_test: Test features
            y_test: Test labels
        
        Returns:
            Dictionary with evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")
        
        # Make predictions
        y_pred = self.model.predict(X_test)
        
        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        
        # Get classification report
        report = classification_report(
            y_test,
            y_pred,
            target_names=['SELL', 'HOLD', 'BUY'],
            labels=[-1, 0, 1],
            output_dict=True,
            zero_division=0
        )
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred, labels=[-1, 0, 1])
        
        # Store metrics
        metrics = {
            'accuracy': accuracy,
            'test_size': len(X_test),
            'classification_report': report,
            'confusion_matrix': cm.tolist(),
            'per_class_accuracy': {
                'buy': report['BUY']['recall'] if 'BUY' in report else 0,
                'hold': report['HOLD']['recall'] if 'HOLD' in report else 0,
                'sell': report['SELL']['recall'] if 'SELL' in report else 0
            }
        }
        
        return metrics
    
    def train_and_evaluate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        train_ratio: float = 0.8,
        verbose: bool = True
    ) -> Dict:
        """
        Complete training and evaluation pipeline
        
        Parameters:
            X: Feature DataFrame
            y: Target Series
            train_ratio: Train/test split ratio
            verbose: Print detailed results
        
        Returns:
            Dictionary with all metrics
        """
        # Split data (time-series aware)
        X_train, X_test, y_train, y_test = self.train_test_split_timeseries(
            X, y, train_ratio
        )
        
        if verbose:
            print(f"\n{'='*60}")
            print("MODEL TRAINING")
            print(f"{'='*60}")
            print(f"Train set: {len(X_train)} samples ({train_ratio*100:.0f}%)")
            print(f"Test set:  {len(X_test)} samples ({(1-train_ratio)*100:.0f}%)")
            print(f"Features:  {len(self.feature_names) if self.feature_names else len(X.columns)}")
        
        # Train model
        self.train(X_train, y_train)
        
        # Evaluate on test set
        metrics = self.evaluate(X_test, y_test)
        
        # Store for later access
        self.training_metrics = metrics
        
        if verbose:
            print(f"\n{'='*60}")
            print("TEST SET PERFORMANCE")
            print(f"{'='*60}")
            print(f"Overall Accuracy: {metrics['accuracy']:.2%}")
            print(f"\nPer-Class Performance:")
            print(f"  BUY:  {metrics['per_class_accuracy']['buy']:.2%}")
            print(f"  HOLD: {metrics['per_class_accuracy']['hold']:.2%}")
            print(f"  SELL: {metrics['per_class_accuracy']['sell']:.2%}")
            print(f"{'='*60}\n")
        
        return metrics
    
    def retrain_on_all_data(
        self,
        X: pd.DataFrame,
        y: pd.Series
    ) -> None:
        """
        Retrain model on ALL available data for production use
        
        After validating on test set, we retrain on full dataset
        to get the best possible model for future predictions
        
        Parameters:
            X: Full feature DataFrame
            y: Full target Series
        """
        print("Retraining on full dataset for production...")
        self.train(X, y)
        print("✓ Production model ready")
    
    def get_feature_importance(self, top_n: int = 10) -> pd.DataFrame:
        """
        Get feature importance from trained model
        
        Parameters:
            top_n: Number of top features to return
        
        Returns:
            DataFrame with features and importance scores
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return importance_df.head(top_n)
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions
        
        Parameters:
            X: Feature DataFrame
        
        Returns:
            Array of predictions (Buy/Hold/Sell)
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        return self.model.predict(X)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Get prediction probabilities
        
        Parameters:
            X: Feature DataFrame
        
        Returns:
            Array of probabilities for each class
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        return self.model.predict_proba(X)
    
    def save_model(self, filepath: str) -> None:
        """
        Save trained model to disk
        
        Parameters:
            filepath: Path to save model (e.g., "models/aapl_model.pkl")
        """
        if self.model is None:
            raise ValueError("No model to save")
        
        model_data = {
            'model': self.model,
            'feature_names': self.feature_names,
            'params': self.model_params,
            'metrics': self.training_metrics,
            'trained_at': datetime.now().isoformat()
        }
        
        joblib.dump(model_data, filepath)
        print(f"✓ Model saved to {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """
        Load trained model from disk
        
        Parameters:
            filepath: Path to load model from
        """
        model_data = joblib.load(filepath)
        
        self.model = model_data['model']
        self.feature_names = model_data['feature_names']
        self.model_params = model_data['params']
        self.training_metrics = model_data.get('metrics', {})
        
        print(f"✓ Model loaded from {filepath}")
        print(f"  Trained at: {model_data.get('trained_at', 'Unknown')}")
        print(f"  Features: {len(self.feature_names)}")


# Example usage
if __name__ == "__main__":
    # Example: Train model on sample data
    print("Model Trainer Module")
    print("\nExample configuration:")
    print("  Random Forest: 100 trees, max_depth=10")
    print("  Train/Test Split: 80/20 (time-series aware)")
    print("  Class Weighting: Balanced (handles imbalance)")

