"""
Label Generation for Trading Signals
Simple forward-return based labeling with adaptive thresholds
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict


def generate_labels(
    df: pd.DataFrame,
    forward_days: int = 5,
    threshold: float = 0.02,
    adaptive_threshold: bool = True
) -> Tuple[pd.Series, Dict]:
    """
    Generate Buy/Sell labels based on future returns (BINARY CLASSIFICATION)
    
    Parameters:
        df: DataFrame with OHLCV data (must have 'Close' column)
        forward_days: Days to look ahead (default: 5)
        threshold: Return threshold (used for adaptive adjustment only)
        adaptive_threshold: Adjust threshold based on stock volatility
    
    Returns:
        Tuple of (labels_series, metadata_dict)
        - labels_series: Buy(1) or Sell(-1) ONLY (no Hold)
        - metadata_dict: Label distribution and statistics
    
    Logic:
        - BUY (1): Future return >= 0 (price goes up)
        - SELL (-1): Future return < 0 (price goes down)
        - NO HOLD: Every signal is either BUY or SELL
    """
    if 'Close' not in df.columns:
        raise ValueError("DataFrame must contain 'Close' column")
    
    # Calculate forward return
    future_close = df['Close'].shift(-forward_days)
    forward_return = (future_close - df['Close']) / df['Close']
    
    # Adaptive threshold based on volatility
    if adaptive_threshold:
        volatility = df['Close'].pct_change().std()
        adjusted_threshold = max(threshold, volatility * 1.5)
        print(f"   Adaptive threshold: {adjusted_threshold:.2%} (base: {threshold:.2%}, volatility: {volatility:.2%})")
    else:
        adjusted_threshold = threshold
    
    # Generate labels - BINARY ONLY (no HOLD)
    # BUY if future return is positive, SELL if negative
    labels = pd.Series(0, index=df.index)
    labels[forward_return >= 0] = 1   # BUY (price goes up)
    labels[forward_return < 0] = -1   # SELL (price goes down)
    
    # Remove last forward_days rows (no future data)
    labels.iloc[-forward_days:] = np.nan
    
    # Calculate distribution
    valid_labels = labels.dropna()
    total = len(valid_labels)
    
    if total > 0:
        buy_count = (valid_labels == 1).sum()
        sell_count = (valid_labels == -1).sum()
        hold_count = (valid_labels == 0).sum()
        
        distribution = {
            'total': int(total),
            'buy': int(buy_count),
            'sell': int(sell_count),
            'hold': int(hold_count),
            'buy_pct': float(buy_count / total * 100),
            'sell_pct': float(sell_count / total * 100),
            'hold_pct': float(hold_count / total * 100),
            'threshold_used': float(adjusted_threshold),
            'forward_days': int(forward_days),
        }
    else:
        distribution = {
            'total': 0,
            'buy': 0,
            'sell': 0,
            'hold': 0,
            'buy_pct': 0.0,
            'sell_pct': 0.0,
            'hold_pct': 0.0,
            'threshold_used': float(adjusted_threshold),
            'forward_days': int(forward_days),
        }
    
    return labels, distribution


def print_label_distribution(distribution: Dict):
    """Print formatted label distribution"""
    print(f"\n   📊 Label Distribution (Binary: BUY vs SELL):")
    print(f"      Total: {distribution['total']}")
    print(f"      BUY:   {distribution['buy']:4d} ({distribution['buy_pct']:5.1f}%)")
    print(f"      SELL:  {distribution['sell']:4d} ({distribution['sell_pct']:5.1f}%)")
    print(f"      Forward days: {distribution['forward_days']}")
    
    # Warn about severe imbalance
    max_pct = max(distribution['buy_pct'], distribution['sell_pct'])
    if max_pct > 70:
        print(f"\n      ⚠️  Severe class imbalance! ({max_pct:.1f}% dominated)")
        print(f"      Consider adjusting forward_days")


# Example usage
if __name__ == "__main__":
    print("Label Generator Module")

