"""
Label Generation Module
Creates Buy/Sell/Hold labels based on future returns
"""

import pandas as pd
import numpy as np
from typing import Tuple


def generate_labels(
    df: pd.DataFrame,
    forward_days: int = 3,
    threshold: float = 0.01
) -> Tuple[pd.Series, pd.DataFrame]:
    """
    Generate Buy/Sell/Hold labels based on future returns
    
    Parameters:
        df: DataFrame with OHLCV data (must have 'Close' column)
        forward_days: Number of days to look ahead (default: 3)
        threshold: Percentage threshold for buy/sell (default: 0.01 = 1%)
    
    Returns:
        Tuple of (labels_series, analysis_df)
        - labels_series: Series with Buy(1), Hold(0), Sell(-1) signals
        - analysis_df: DataFrame with future prices and returns for analysis
    
    Label Logic:
        - BUY (1):  Future return >= +threshold (price goes up)
        - HOLD (0): Future return between -threshold and +threshold
        - SELL (-1): Future return <= -threshold (price goes down)
    
    Example:
        If threshold=0.01 (1%):
        - Close today: $100
        - Close in 3 days: $102 → Return = +2% → Label = BUY (1)
        - Close in 3 days: $99  → Return = -1% → Label = SELL (-1)
        - Close in 3 days: $100.50 → Return = +0.5% → Label = HOLD (0)
    """
    # Validate input
    if 'Close' not in df.columns:
        raise ValueError("DataFrame must contain 'Close' column")
    
    if forward_days < 1:
        raise ValueError("forward_days must be >= 1")
    
    if threshold <= 0:
        raise ValueError("threshold must be positive")
    
    # Create a copy to avoid modifying original
    analysis_df = df[['Close']].copy()
    
    # Calculate future close price
    analysis_df['Future_Close'] = analysis_df['Close'].shift(-forward_days)
    
    # Calculate future return percentage
    analysis_df['Future_Return'] = (
        (analysis_df['Future_Close'] - analysis_df['Close']) / analysis_df['Close']
    )
    
    # Generate labels based on threshold
    labels = pd.Series(0, index=df.index, name='Signal')  # Default to HOLD (0)
    
    # BUY signal: Future return >= threshold
    labels[analysis_df['Future_Return'] >= threshold] = 1
    
    # SELL signal: Future return <= -threshold
    labels[analysis_df['Future_Return'] <= -threshold] = -1
    
    # HOLD signal: Already set to 0 by default
    
    # The last forward_days rows will have NaN (no future data)
    # Mark them as NaN instead of HOLD
    labels.iloc[-forward_days:] = np.nan
    
    return labels, analysis_df


def analyze_label_distribution(labels: pd.Series) -> dict:
    """
    Analyze the distribution of labels
    
    Parameters:
        labels: Series with Buy(1), Hold(0), Sell(-1) signals
    
    Returns:
        Dictionary with label counts and percentages
    """
    # Remove NaN values for counting
    valid_labels = labels.dropna()
    
    if len(valid_labels) == 0:
        return {
            'total': 0,
            'buy_count': 0,
            'hold_count': 0,
            'sell_count': 0,
            'buy_pct': 0.0,
            'hold_pct': 0.0,
            'sell_pct': 0.0
        }
    
    total = len(valid_labels)
    buy_count = (valid_labels == 1).sum()
    hold_count = (valid_labels == 0).sum()
    sell_count = (valid_labels == -1).sum()
    
    return {
        'total': total,
        'buy_count': int(buy_count),
        'hold_count': int(hold_count),
        'sell_count': int(sell_count),
        'buy_pct': (buy_count / total * 100),
        'hold_pct': (hold_count / total * 100),
        'sell_pct': (sell_count / total * 100)
    }


def optimize_threshold(
    df: pd.DataFrame,
    forward_days: int = 3,
    threshold_range: tuple = (0.005, 0.03),
    step: float = 0.005
) -> dict:
    """
    Find optimal threshold that balances label distribution
    
    Goal: Avoid extreme class imbalance (e.g., 90% HOLD, 5% BUY, 5% SELL)
    Ideal: Closer to 30-40% each for BUY/SELL, 20-40% HOLD
    
    Parameters:
        df: DataFrame with OHLCV data
        forward_days: Days to look ahead
        threshold_range: (min_threshold, max_threshold) to test
        step: Increment step for testing thresholds
    
    Returns:
        Dictionary with optimal threshold and distribution
    """
    min_thresh, max_thresh = threshold_range
    best_balance = float('inf')
    best_threshold = min_thresh
    best_distribution = None
    
    # Test different thresholds
    threshold = min_thresh
    while threshold <= max_thresh:
        labels, _ = generate_labels(df, forward_days, threshold)
        dist = analyze_label_distribution(labels)
        
        # Calculate imbalance score (lower is better)
        # Ideal is roughly equal distribution
        target_pct = 33.33  # Perfect balance would be 33.33% each
        imbalance = abs(dist['buy_pct'] - target_pct) + \
                   abs(dist['hold_pct'] - target_pct) + \
                   abs(dist['sell_pct'] - target_pct)
        
        if imbalance < best_balance:
            best_balance = imbalance
            best_threshold = threshold
            best_distribution = dist
        
        threshold += step
    
    return {
        'optimal_threshold': best_threshold,
        'distribution': best_distribution,
        'imbalance_score': best_balance
    }


def prepare_ml_dataset(
    df: pd.DataFrame,
    features: list,
    forward_days: int = 3,
    threshold: float = 0.01
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Prepare complete dataset for ML training
    
    Parameters:
        df: DataFrame with OHLCV data and technical indicators
        features: List of feature column names
        forward_days: Days to look ahead for labels
        threshold: Percentage threshold for buy/sell
    
    Returns:
        Tuple of (X, y)
        - X: Feature DataFrame (ready for ML)
        - y: Labels Series (Buy/Sell/Hold)
    
    Note: Rows with NaN values are automatically dropped
    """
    # Generate labels
    labels, _ = generate_labels(df, forward_days, threshold)
    
    # Add labels to dataframe
    df_ml = df.copy()
    df_ml['Signal'] = labels
    
    # Drop rows with NaN in features or labels
    df_ml = df_ml.dropna(subset=features + ['Signal'])
    
    # Separate features and target
    X = df_ml[features]
    y = df_ml['Signal']
    
    return X, y


# Example usage and testing
if __name__ == "__main__":
    # Example: Generate labels for sample data
    sample_prices = pd.DataFrame({
        'Close': [100, 102, 104, 103, 105, 110, 108, 107, 112, 115]
    })
    
    labels, analysis = generate_labels(sample_prices, forward_days=3, threshold=0.01)
    
    print("Sample Price Data:")
    print(sample_prices['Close'].values)
    print("\nGenerated Labels:")
    print(labels.values)
    print("\nLabel Distribution:")
    dist = analyze_label_distribution(labels)
    print(f"BUY:  {dist['buy_count']} ({dist['buy_pct']:.1f}%)")
    print(f"HOLD: {dist['hold_count']} ({dist['hold_pct']:.1f}%)")
    print(f"SELL: {dist['sell_count']} ({dist['sell_pct']:.1f}%)")

