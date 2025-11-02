"""
Baseline Machine Learning Models
Random Forest, XGBoost, and LightGBM for trading signal prediction
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import lightgbm as lgb
import joblib
from typing import Dict, Tuple
import warnings
warnings.filterwarnings('ignore')


class BaselineModelTrainer:
    """
    Train and evaluate baseline ML models
    
    Models:
        - Random Forest: Ensemble of decision trees
        - XGBoost: Gradient boosted trees (fast, accurate)
        - LightGBM: Light gradient boosting (faster, less memory)
    """
    
    def __init__(self, model_type: str = 'xgboost'):
        """
        Initialize model trainer
        
        Parameters:
            model_type: 'random_forest', 'xgboost', or 'lightgbm'
        """
        self.model_type = model_type.lower()
        self.model = None
        self.feature_names = None
        self.metrics = None
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit([-1, 0, 1])  # 3-Class: SELL, HOLD, BUY
        
        # Initialize model
        if self.model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=20,
                min_samples_leaf=10,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            )
        elif self.model_type == 'xgboost':
            self.model = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=1.0,
                random_state=42,
                n_jobs=-1,
                eval_metric='mlogloss'
            )
        elif self.model_type == 'lightgbm':
            self.model = lgb.LGBMClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=1.0,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1,
                verbose=-1
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        verbose: bool = True
    ) -> Dict:
        """
        Train model and evaluate on test set
        
        Parameters:
            X_train, y_train: Training data
            X_test, y_test: Test data
            verbose: Print training progress
        
        Returns:
            Dictionary with performance metrics
        """
        if verbose:
            print(f"\n🤖 Training {self.model_type.upper()} model...")
            print(f"   Training samples: {len(X_train)}")
            print(f"   Test samples: {len(X_test)}")
            print(f"   Features: {X_train.shape[1]}")
        
        # Store feature names
        self.feature_names = list(X_train.columns)
        
        # Encode labels using LabelEncoder (handles missing classes properly)
        # -1 (SELL) -> 0, 0 (HOLD) -> 1, 1 (BUY) -> 2
        y_train_encoded = self.label_encoder.transform(y_train)
        y_test_encoded = self.label_encoder.transform(y_test)
        
        # Train model
        self.model.fit(X_train, y_train_encoded)
        
        if verbose:
            print(f"   ✓ Model trained")
        
        # Predictions (encoded as 0, 1, 2)
        y_pred_train_encoded = self.model.predict(X_train)
        y_pred_test_encoded = self.model.predict(X_test)
        
        # Probabilities
        y_prob_test = self.model.predict_proba(X_test)
        
        # Calculate metrics (using encoded labels)
        train_acc = accuracy_score(y_train_encoded, y_pred_train_encoded)
        test_acc = accuracy_score(y_test_encoded, y_pred_test_encoded)
        
        # Per-class metrics (handle case where some classes might be missing)
        classes = sorted(y_test.unique())
        
        metrics = {
            'model_type': self.model_type,
            'train_accuracy': float(train_acc),
            'test_accuracy': float(test_acc),
            'train_samples': int(len(X_train)),
            'test_samples': int(len(X_test)),
            'n_features': int(X_train.shape[1]),
            'classes': list(classes),
        }
        
        # Try to calculate precision/recall (might fail if some classes missing in predictions)
        try:
            metrics['precision'] = float(precision_score(y_test_encoded, y_pred_test_encoded, average='weighted', zero_division=0))
            metrics['recall'] = float(recall_score(y_test_encoded, y_pred_test_encoded, average='weighted', zero_division=0))
            metrics['f1'] = float(f1_score(y_test_encoded, y_pred_test_encoded, average='weighted', zero_division=0))
        except:
            metrics['precision'] = 0.0
            metrics['recall'] = 0.0
            metrics['f1'] = 0.0
        
        self.metrics = metrics
        
        if verbose:
            print(f"\n   📊 Performance Metrics:")
            print(f"      Train Accuracy: {train_acc:.1%}")
            print(f"      Test Accuracy:  {test_acc:.1%}")
            print(f"      Precision:      {metrics['precision']:.1%}")
            print(f"      Recall:         {metrics['recall']:.1%}")
            print(f"      F1 Score:       {metrics['f1']:.1%}")
            
            # Per-class breakdown
            print(f"\n   📋 Classification Report (3-Class):")
            print(classification_report(y_test_encoded, y_pred_test_encoded, target_names=['SELL', 'HOLD', 'BUY'], zero_division=0))
        
        return metrics
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions (returns original labels: -1, 0, 1)"""
        if self.model is None:
            raise ValueError("Model not trained yet!")
        # Predict with encoded labels (0, 1, 2)
        predictions_encoded = self.model.predict(X)
        # Convert back to original labels using label encoder
        predictions = self.label_encoder.inverse_transform(predictions_encoded)
        return predictions
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Get prediction probabilities (order: SELL, HOLD, BUY)"""
        if self.model is None:
            raise ValueError("Model not trained yet!")
        return self.model.predict_proba(X)
    
    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """
        Get feature importance rankings
        
        Parameters:
            top_n: Number of top features to return
        
        Returns:
            DataFrame with feature names and importance scores
        """
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        # Get importance scores
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
        else:
            return pd.DataFrame()
        
        # Create DataFrame
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        return importance_df.head(top_n)
    
    def save(self, filepath: str):
        """Save trained model to disk"""
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        joblib.dump({
            'model': self.model,
            'model_type': self.model_type,
            'feature_names': self.feature_names,
            'metrics': self.metrics,
            'label_encoder': self.label_encoder
        }, filepath)
        print(f"✓ Model saved to {filepath}")
    
    def load(self, filepath: str):
        """Load trained model from disk"""
        data = joblib.load(filepath)
        self.model = data['model']
        self.model_type = data['model_type']
        self.feature_names = data['feature_names']
        self.metrics = data['metrics']
        self.label_encoder = data.get('label_encoder', self.label_encoder)
        print(f"✓ Model loaded from {filepath}")


def compare_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    verbose: bool = True
) -> Dict[str, Dict]:
    """
    Train and compare all baseline models
    
    Returns:
        Dictionary mapping model_type to metrics
    """
    results = {}
    
    for model_type in ['random_forest', 'xgboost', 'lightgbm']:
        if verbose:
            print("\n" + "="*70)
        
        trainer = BaselineModelTrainer(model_type)
        metrics = trainer.train(X_train, y_train, X_test, y_test, verbose=verbose)
        results[model_type] = metrics
    
    if verbose:
        print("\n" + "="*70)
        print("🏆 MODEL COMPARISON")
        print("="*70)
        print(f"{'Model':<15} {'Train Acc':<12} {'Test Acc':<12} {'Precision':<12} {'F1 Score':<12}")
        print("-"*70)
        
        for model_type, metrics in results.items():
            print(f"{model_type:<15} "
                  f"{metrics['train_accuracy']:>10.1%}  "
                  f"{metrics['test_accuracy']:>10.1%}  "
                  f"{metrics['precision']:>10.1%}  "
                  f"{metrics['f1']:>10.1%}")
        
        # Recommend best model
        best_model = max(results.items(), key=lambda x: x[1]['test_accuracy'])
        print(f"\n🥇 Best Model: {best_model[0].upper()} ({best_model[1]['test_accuracy']:.1%} test accuracy)")
        print("="*70)
    
    return results


# Example usage
if __name__ == "__main__":
    print("Baseline Models Module")
    print("\nAvailable models:")
    print("  - Random Forest")
    print("  - XGBoost")
    print("  - LightGBM")

