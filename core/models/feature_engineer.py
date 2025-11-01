"""
Feature Engineering Module
Transforms raw stock data + technical indicators into ML-ready features
"""

import pandas as pd
import numpy as np
from typing import List


# Feature list that will be used for ML
ML_FEATURES = [
    # Core Technical Indicators (from Phase 2)
    'rsi', 'macd', 'macd_signal', 'macd_histogram',
    'bb_upper', 'bb_middle', 'bb_lower', 'bb_width', 'bb_position',
    
    # Moving Averages
    'sma_20', 'sma_50', 'price_vs_sma20', 'price_vs_sma50',
    
    # Momentum
    'momentum_5', 'momentum_10', 'roc_5',
    
    # Volume
    'volume_sma_20', 'volume_ratio',
    
    # Volatility
    'atr', 'volatility_ratio',
    
    # Lagged Features (past values)
    'rsi_lag_1', 'macd_lag_1', 'close_lag_1',
    
    # Rate of Change
    'rsi_roc', 'macd_roc'
]

# Lite feature set for limited data (< 500 samples)
ML_FEATURES_LITE = [
    # Core indicators only
    'rsi', 'macd', 'macd_histogram',
    'bb_position', 'bb_width',
    
    # Price momentum
    'momentum_5', 'momentum_10',
    'price_vs_sma20',
    
    # Volume
    'volume_ratio',
    
    # Volatility
    'volatility_ratio'
]  # Only 11 features - needs ~110 samples


def engineer_features(df: pd.DataFrame, indicators: dict = None) -> pd.DataFrame:
    """
    Add engineered features to dataframe
    
    Parameters:
        df: DataFrame with OHLCV data
        indicators: Optional dict of pre-calculated indicators
    
    Returns:
        DataFrame with all engineered features added
    """
    # Create a copy to avoid modifying original
    df_features = df.copy()
    
    # If indicators provided, add them first
    if indicators:
        for key, value in indicators.items():
            if isinstance(value, (int, float)):
                # Scalar values (latest only) - not useful for ML
                # We need full time series
                pass
            elif isinstance(value, pd.Series):
                df_features[key] = value
    
    # 1. Moving Averages
    df_features = _add_moving_averages(df_features)
    
    # 2. Momentum Features
    df_features = _add_momentum_features(df_features)
    
    # 3. Volume Features
    df_features = _add_volume_features(df_features)
    
    # 4. Volatility Features
    df_features = _add_volatility_features(df_features)
    
    # 5. Bollinger Band Derived Features
    df_features = _add_bollinger_features(df_features)
    
    # 6. Lagged Features
    df_features = _add_lagged_features(df_features)
    
    # 7. Rate of Change Features
    df_features = _add_roc_features(df_features)
    
    return df_features


def _add_moving_averages(df: pd.DataFrame) -> pd.DataFrame:
    """Add moving average features"""
    # Simple Moving Averages
    df['sma_20'] = df['Close'].rolling(window=20).mean()
    df['sma_50'] = df['Close'].rolling(window=50).mean()
    
    # Price relative to SMAs (normalized)
    df['price_vs_sma20'] = (df['Close'] - df['sma_20']) / df['sma_20']
    df['price_vs_sma50'] = (df['Close'] - df['sma_50']) / df['sma_50']
    
    return df


def _add_momentum_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add momentum/price change features"""
    # Momentum over different periods
    df['momentum_5'] = (df['Close'] - df['Close'].shift(5)) / df['Close'].shift(5)
    df['momentum_10'] = (df['Close'] - df['Close'].shift(10)) / df['Close'].shift(10)
    
    # Rate of Change (5-day)
    df['roc_5'] = df['Close'].pct_change(periods=5)
    
    return df


def _add_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add volume-based features"""
    # Volume moving average
    df['volume_sma_20'] = df['Volume'].rolling(window=20).mean()
    
    # Volume ratio (current vs average)
    df['volume_ratio'] = df['Volume'] / df['volume_sma_20']
    
    return df


def _add_volatility_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add volatility features (ATR)"""
    # Average True Range (ATR)
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr'] = true_range.rolling(window=14).mean()
    
    # Normalized volatility (ATR / Price)
    df['volatility_ratio'] = df['atr'] / df['Close']
    
    return df


def _add_bollinger_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add Bollinger Band derived features (if BB columns exist)"""
    if all(col in df.columns for col in ['bb_upper', 'bb_middle', 'bb_lower']):
        # Band width (normalized by middle band)
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        
        # Price position within bands (0 = at lower, 1 = at upper)
        df['bb_position'] = (df['Close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # Clip position to [0, 1] range (price can sometimes be outside bands)
        df['bb_position'] = df['bb_position'].clip(0, 1)
    
    return df


def _add_lagged_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add lagged (previous day) features"""
    # Yesterday's values
    if 'rsi' in df.columns:
        df['rsi_lag_1'] = df['rsi'].shift(1)
    
    if 'macd' in df.columns:
        df['macd_lag_1'] = df['macd'].shift(1)
    
    df['close_lag_1'] = df['Close'].shift(1)
    
    return df


def _add_roc_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add rate of change for indicators"""
    # RSI rate of change
    if 'rsi' in df.columns:
        df['rsi_roc'] = df['rsi'].pct_change()
    
    # MACD rate of change
    if 'macd' in df.columns:
        df['macd_roc'] = df['macd'].pct_change()
    
    return df


def add_core_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate core technical indicators needed for features
    
    This replicates indicator calculations from data/indicators/indicators_data.py
    but returns full time series instead of just latest values
    
    Parameters:
        df: DataFrame with OHLCV data
    
    Returns:
        DataFrame with indicator columns added
    """
    from ta.momentum import RSIIndicator
    from ta.trend import MACD
    from ta.volatility import BollingerBands
    
    # RSI (14-period)
    rsi_indicator = RSIIndicator(close=df['Close'], window=14)
    df['rsi'] = rsi_indicator.rsi()
    
    # MACD (12/26/9)
    macd_indicator = MACD(close=df['Close'], window_slow=26, window_fast=12, window_sign=9)
    df['macd'] = macd_indicator.macd()
    df['macd_signal'] = macd_indicator.macd_signal()
    df['macd_histogram'] = macd_indicator.macd_diff()
    
    # Bollinger Bands (20-period, 2 std dev)
    bb_indicator = BollingerBands(close=df['Close'], window=20, window_dev=2)
    df['bb_upper'] = bb_indicator.bollinger_hband()
    df['bb_middle'] = bb_indicator.bollinger_mavg()
    df['bb_lower'] = bb_indicator.bollinger_lband()
    
    return df


def prepare_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Complete feature engineering pipeline
    
    Parameters:
        df: Raw DataFrame with OHLCV data
    
    Returns:
        DataFrame with all features ready for ML
    
    Usage:
        df = fetch_stock_data("AAPL", "1Y")
        df_ml = prepare_feature_matrix(df)
        X = df_ml[ML_FEATURES]  # Ready for model.fit(X, y)
    """
    # Step 1: Add core indicators (RSI, MACD, BB)
    df = add_core_indicators(df)
    
    # Step 2: Engineer additional features
    df = engineer_features(df)
    
    # Step 3: Drop rows with NaN values
    # (First ~50 rows will have NaN due to rolling windows)
    df = df.dropna()
    
    return df


def get_feature_list() -> List[str]:
    """
    Get the list of features used for ML
    
    Returns:
        List of feature column names
    """
    return ML_FEATURES.copy()


def validate_features(df: pd.DataFrame) -> dict:
    """
    Validate that all required features are present and valid
    
    Parameters:
        df: DataFrame with features
    
    Returns:
        Dictionary with validation results
    """
    results = {
        'valid': True,
        'missing_features': [],
        'features_with_nan': [],
        'feature_count': len(ML_FEATURES),
        'row_count': len(df)
    }
    
    # Check for missing features
    for feature in ML_FEATURES:
        if feature not in df.columns:
            results['missing_features'].append(feature)
            results['valid'] = False
    
    # Check for NaN values
    for feature in ML_FEATURES:
        if feature in df.columns and df[feature].isna().any():
            nan_count = df[feature].isna().sum()
            results['features_with_nan'].append({
                'feature': feature,
                'nan_count': int(nan_count),
                'nan_pct': float(nan_count / len(df) * 100)
            })
    
    return results


# Example usage
if __name__ == "__main__":
    # Example: Engineer features for sample data
    print("Feature Engineering Module")
    print(f"Total features: {len(ML_FEATURES)}")
    print("\nFeature list:")
    for i, feature in enumerate(ML_FEATURES, 1):
        print(f"  {i:2d}. {feature}")

