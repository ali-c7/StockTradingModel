"""
Walk-Forward Validation Module
Tests model on multiple time windows to ensure robustness
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
from sklearn.ensemble import RandomForestClassifier


def walk_forward_validation(
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
    test_size: float = 0.2,
    model_params: dict = None
) -> Dict:
    """
    Perform walk-forward validation with expanding window
    
    This tests the model on multiple time periods to ensure it's robust
    and not just getting lucky on one specific test set.
    
    Parameters:
        X: Feature DataFrame
        y: Target Series
        n_splits: Number of validation windows
        test_size: Proportion of data for each test window
        model_params: RandomForest parameters
    
    Returns:
        Dictionary with validation results across all windows
    
    Example:
        Window 1: Train [0-200], Test [201-250]
        Window 2: Train [0-250], Test [251-300]
        Window 3: Train [0-300], Test [301-350]
        ...
    """
    if model_params is None:
        model_params = {
            'n_estimators': 100,
            'max_depth': 10,
            'min_samples_split': 20,
            'min_samples_leaf': 10,
            'class_weight': 'balanced',
            'random_state': 42,
            'n_jobs': -1
        }
    
    results = {
        'window_accuracies': [],
        'window_sizes': [],
        'mean_accuracy': 0.0,
        'std_accuracy': 0.0,
        'min_accuracy': 0.0,
        'max_accuracy': 0.0,
        'consistent': False
    }
    
    # Calculate window parameters
    total_samples = len(X)
    test_samples = int(total_samples * test_size)
    
    if test_samples < 20:
        print(f"⚠️  Warning: Test window too small ({test_samples} samples)")
        print("   Consider using fewer splits or more data")
    
    # Create expanding windows
    windows = []
    for i in range(n_splits):
        # Calculate split point
        # Start from minimum train size and expand
        min_train_size = int(total_samples * 0.5)  # At least 50% for training
        train_end = min_train_size + int((total_samples - min_train_size - test_samples) * i / (n_splits - 1))
        test_start = train_end
        test_end = min(test_start + test_samples, total_samples)
        
        if test_end - test_start < 20:  # Skip if test window too small
            continue
            
        windows.append((0, train_end, test_start, test_end))
    
    print(f"\n{'='*70}")
    print("WALK-FORWARD VALIDATION")
    print(f"{'='*70}")
    print(f"Total samples: {total_samples}")
    print(f"Number of windows: {len(windows)}")
    print(f"Test size per window: ~{test_samples} samples")
    
    # Test on each window
    for i, (train_start, train_end, test_start, test_end) in enumerate(windows, 1):
        print(f"\nWindow {i}/{len(windows)}:")
        print(f"  Train: [{train_start}:{train_end}] ({train_end - train_start} samples)")
        print(f"  Test:  [{test_start}:{test_end}] ({test_end - test_start} samples)")
        
        # Split data
        X_train = X.iloc[train_start:train_end]
        y_train = y.iloc[train_start:train_end]
        X_test = X.iloc[test_start:test_end]
        y_test = y.iloc[test_start:test_end]
        
        # Train model
        model = RandomForestClassifier(**model_params)
        model.fit(X_train, y_train)
        
        # Evaluate
        accuracy = model.score(X_test, y_test)
        
        print(f"  Accuracy: {accuracy:.1%}")
        
        results['window_accuracies'].append(accuracy)
        results['window_sizes'].append(test_end - test_start)
    
    # Calculate summary statistics
    if results['window_accuracies']:
        results['mean_accuracy'] = np.mean(results['window_accuracies'])
        results['std_accuracy'] = np.std(results['window_accuracies'])
        results['min_accuracy'] = np.min(results['window_accuracies'])
        results['max_accuracy'] = np.max(results['window_accuracies'])
        
        # Check if results are consistent (std < 10%)
        results['consistent'] = results['std_accuracy'] < 0.10
        
        print(f"\n{'='*70}")
        print("VALIDATION SUMMARY")
        print(f"{'='*70}")
        print(f"Mean Accuracy:   {results['mean_accuracy']:.1%}")
        print(f"Std Deviation:   {results['std_accuracy']:.1%}")
        print(f"Min Accuracy:    {results['min_accuracy']:.1%}")
        print(f"Max Accuracy:    {results['max_accuracy']:.1%}")
        print(f"Consistency:     {'✓ Consistent' if results['consistent'] else '⚠️  Variable'}")
        
        if results['std_accuracy'] > 0.15:
            print("\n⚠️  WARNING: High variability across windows")
            print("   Model performance is inconsistent over time")
            print("   This may indicate overfitting or regime changes")
        
        print(f"{'='*70}\n")
    
    return results


def time_period_validation(
    X: pd.DataFrame,
    y: pd.Series,
    model_params: dict = None
) -> Dict:
    """
    Test model on different time periods (e.g., different years)
    
    Parameters:
        X: Feature DataFrame (with datetime index)
        y: Target Series
        model_params: RandomForest parameters
    
    Returns:
        Dictionary with results per time period
    """
    if not isinstance(X.index, pd.DatetimeIndex):
        print("⚠️  Warning: X must have DatetimeIndex for time period validation")
        return {}
    
    if model_params is None:
        model_params = {
            'n_estimators': 100,
            'max_depth': 10,
            'class_weight': 'balanced',
            'random_state': 42,
            'n_jobs': -1
        }
    
    results = {
        'period_accuracies': {},
        'periods_tested': []
    }
    
    # Group by year
    years = X.index.year.unique()
    
    if len(years) < 2:
        print("⚠️  Not enough time periods for validation (need 2+ years)")
        return results
    
    print(f"\n{'='*70}")
    print("TIME PERIOD VALIDATION")
    print(f"{'='*70}")
    print(f"Testing across {len(years)} years: {list(years)}")
    
    # Train on all but one year, test on that year
    for test_year in years:
        train_mask = X.index.year != test_year
        test_mask = X.index.year == test_year
        
        X_train = X[train_mask]
        y_train = y[train_mask]
        X_test = X[test_mask]
        y_test = y[test_mask]
        
        if len(X_test) < 20:  # Skip if too few test samples
            continue
        
        print(f"\nTesting on year {test_year}:")
        print(f"  Train: {len(X_train)} samples (other years)")
        print(f"  Test:  {len(X_test)} samples ({test_year})")
        
        # Train and evaluate
        model = RandomForestClassifier(**model_params)
        model.fit(X_train, y_train)
        accuracy = model.score(X_test, y_test)
        
        print(f"  Accuracy: {accuracy:.1%}")
        
        results['period_accuracies'][test_year] = accuracy
        results['periods_tested'].append(test_year)
    
    if results['period_accuracies']:
        mean_acc = np.mean(list(results['period_accuracies'].values()))
        std_acc = np.std(list(results['period_accuracies'].values()))
        
        print(f"\n{'='*70}")
        print(f"Mean Accuracy Across Periods: {mean_acc:.1%} ± {std_acc:.1%}")
        print(f"{'='*70}\n")
        
        results['mean_accuracy'] = mean_acc
        results['std_accuracy'] = std_acc
    
    return results


def quick_validation_report(
    X: pd.DataFrame,
    y: pd.Series,
    model_params: dict = None
) -> str:
    """
    Generate a quick validation report with multiple test approaches
    
    Parameters:
        X: Feature DataFrame
        y: Target Series
        model_params: RandomForest parameters
    
    Returns:
        Formatted validation report
    """
    report = []
    report.append("\n" + "="*70)
    report.append("COMPREHENSIVE MODEL VALIDATION")
    report.append("="*70)
    
    # 1. Walk-forward validation
    report.append("\n1. Walk-Forward Validation (5 windows)")
    report.append("-" * 70)
    wf_results = walk_forward_validation(X, y, n_splits=5, model_params=model_params)
    
    if wf_results['window_accuracies']:
        report.append(f"   Mean: {wf_results['mean_accuracy']:.1%}")
        report.append(f"   Range: {wf_results['min_accuracy']:.1%} - {wf_results['max_accuracy']:.1%}")
        report.append(f"   Consistency: {'✓' if wf_results['consistent'] else '⚠️ Variable'}")
    
    # 2. Time period validation (if datetime index)
    if isinstance(X.index, pd.DatetimeIndex):
        report.append("\n2. Time Period Validation (by year)")
        report.append("-" * 70)
        tp_results = time_period_validation(X, y, model_params=model_params)
        
        if tp_results.get('period_accuracies'):
            for year, acc in tp_results['period_accuracies'].items():
                report.append(f"   {year}: {acc:.1%}")
    
    report.append("\n" + "="*70)
    
    return "\n".join(report)


# Example usage
if __name__ == "__main__":
    print("Walk-Forward Validation Module")
    print("\nThis module provides:")
    print("  1. Walk-forward validation (expanding window)")
    print("  2. Time period validation (test on different years)")
    print("  3. Comprehensive validation reports")

