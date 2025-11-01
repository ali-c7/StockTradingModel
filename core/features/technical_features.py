"""
Technical Feature Engineering
Comprehensive technical indicators for stock prediction

Research-backed features organized by category:
- Trend: EMA, SMA, ADX, Ichimoku
- Momentum: RSI, Stochastic, CCI, Williams %R
- Volatility: ATR, Bollinger Bands, Keltner Channels
- Volume: OBV, Volume MA, VWAP
"""

import pandas as pd
import numpy as np
from ta.trend import ADXIndicator, EMAIndicator, SMAIndicator, IchimokuIndicator
from ta.momentum import RSIIndicator, StochasticOscillator, WilliamsRIndicator
from ta.volatility import AverageTrueRange, BollingerBands, KeltnerChannel
from ta.volume import OnBalanceVolumeIndicator, VolumeWeightedAveragePrice
from typing import Dict, List


# ============================================================================
# FEATURE LIST DEFINITION
# ============================================================================

TREND_FEATURES = [
    'ema_9', 'ema_21', 'ema_50', 'ema_200',
    'sma_20', 'sma_50', 'sma_200',
    'adx', 'adx_pos', 'adx_neg',
    'ichimoku_conv', 'ichimoku_base', 'ichimoku_a', 'ichimoku_b',
    'price_vs_ema50', 'price_vs_ema200',
    'ema_cross_50_200',
]

MOMENTUM_FEATURES = [
    'rsi_14',
    'stoch_k', 'stoch_d',
    'cci',
    'williams_r',
    'roc_10', 'roc_20',
    'momentum_10',
]

VOLATILITY_FEATURES = [
    'atr', 'atr_ratio',
    'bb_upper', 'bb_middle', 'bb_lower', 'bb_width', 'bb_position',
    'kc_upper', 'kc_middle', 'kc_lower', 'kc_width',
]

VOLUME_FEATURES = [
    'obv', 'obv_ema',
    'volume_sma_20', 'volume_ratio',
    'vwap', 'price_vs_vwap',
]

PRICE_FEATURES = [
    'Close', 'High', 'Low', 'Open', 'Volume',
    'returns', 'log_returns',
    'high_low_range', 'close_open_range',
]

# Complete feature list
ALL_FEATURES = (
    TREND_FEATURES + 
    MOMENTUM_FEATURES + 
    VOLATILITY_FEATURES + 
    VOLUME_FEATURES + 
    PRICE_FEATURES
)


# ============================================================================
# TREND INDICATORS
# ============================================================================

def add_trend_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add trend-following indicators
    
    References:
    - EMA: Exponential moving average (faster reaction than SMA)
    - ADX: Average Directional Index (trend strength)
    - Ichimoku Cloud: Japanese trend system
    """
    df = df.copy()
    close = df['Close']
    high = df['High']
    low = df['Low']
    
    # Exponential Moving Averages (multiple timeframes)
    for period in [9, 21, 50, 200]:
        df[f'ema_{period}'] = EMAIndicator(close, window=period).ema_indicator()
    
    # Simple Moving Averages
    for period in [20, 50, 200]:
        df[f'sma_{period}'] = SMAIndicator(close, window=period).sma_indicator()
    
    # Average Directional Index (trend strength)
    adx = ADXIndicator(high, low, close, window=14)
    df['adx'] = adx.adx()
    df['adx_pos'] = adx.adx_pos()  # Positive directional indicator
    df['adx_neg'] = adx.adx_neg()  # Negative directional indicator
    
    # Ichimoku Cloud components
    ichimoku = IchimokuIndicator(high, low)
    df['ichimoku_conv'] = ichimoku.ichimoku_conversion_line()
    df['ichimoku_base'] = ichimoku.ichimoku_base_line()
    df['ichimoku_a'] = ichimoku.ichimoku_a()
    df['ichimoku_b'] = ichimoku.ichimoku_b()
    
    # Relative position indicators
    df['price_vs_ema50'] = (close - df['ema_50']) / df['ema_50']
    df['price_vs_ema200'] = (close - df['ema_200']) / df['ema_200']
    
    # Golden/Death cross signal (EMA 50 vs 200)
    df['ema_cross_50_200'] = (df['ema_50'] > df['ema_200']).astype(int)
    
    return df


# ============================================================================
# MOMENTUM INDICATORS
# ============================================================================

def add_momentum_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add momentum oscillators
    
    References:
    - RSI: Relative Strength Index (overbought/oversold)
    - Stochastic: Fast/slow momentum indicator
    - CCI: Commodity Channel Index
    - Williams %R: Momentum indicator (similar to Stochastic)
    """
    df = df.copy()
    close = df['Close']
    high = df['High']
    low = df['Low']
    
    # RSI (14-period standard)
    df['rsi_14'] = RSIIndicator(close, window=14).rsi()
    
    # Stochastic Oscillator
    stoch = StochasticOscillator(high, low, close, window=14, smooth_window=3)
    df['stoch_k'] = stoch.stoch()  # %K line
    df['stoch_d'] = stoch.stoch_signal()  # %D line (signal)
    
    # CCI (Commodity Channel Index)
    typical_price = (high + low + close) / 3
    tp_sma = typical_price.rolling(window=20).mean()
    mean_deviation = typical_price.rolling(window=20).apply(
        lambda x: np.abs(x - x.mean()).mean(), raw=False
    )
    df['cci'] = (typical_price - tp_sma) / (0.015 * mean_deviation)
    
    # Williams %R
    df['williams_r'] = WilliamsRIndicator(high, low, close, lbp=14).williams_r()
    
    # Rate of Change (ROC)
    df['roc_10'] = close.pct_change(periods=10) * 100
    df['roc_20'] = close.pct_change(periods=20) * 100
    
    # Momentum (raw price change)
    df['momentum_10'] = close - close.shift(10)
    
    return df


# ============================================================================
# VOLATILITY INDICATORS
# ============================================================================

def add_volatility_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add volatility measures
    
    References:
    - ATR: Average True Range (volatility measure)
    - Bollinger Bands: Price envelope based on standard deviation
    - Keltner Channels: ATR-based envelope
    """
    df = df.copy()
    close = df['Close']
    high = df['High']
    low = df['Low']
    
    # Average True Range (ATR)
    atr = AverageTrueRange(high, low, close, window=14)
    df['atr'] = atr.average_true_range()
    df['atr_ratio'] = df['atr'] / close  # Normalized ATR
    
    # Bollinger Bands
    bb = BollingerBands(close, window=20, window_dev=2)
    df['bb_upper'] = bb.bollinger_hband()
    df['bb_middle'] = bb.bollinger_mavg()
    df['bb_lower'] = bb.bollinger_lband()
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
    df['bb_position'] = (close - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
    
    # Keltner Channels (similar to BB but uses ATR)
    kc = KeltnerChannel(high, low, close, window=20, window_atr=10)
    df['kc_upper'] = kc.keltner_channel_hband()
    df['kc_middle'] = kc.keltner_channel_mband()
    df['kc_lower'] = kc.keltner_channel_lband()
    df['kc_width'] = (df['kc_upper'] - df['kc_lower']) / df['kc_middle']
    
    return df


# ============================================================================
# VOLUME INDICATORS
# ============================================================================

def add_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add volume-based features
    
    References:
    - OBV: On-Balance Volume (accumulation/distribution)
    - VWAP: Volume Weighted Average Price (institutional benchmark)
    """
    df = df.copy()
    close = df['Close']
    high = df['High']
    low = df['Low']
    volume = df['Volume']
    
    # On-Balance Volume
    obv = OnBalanceVolumeIndicator(close, volume)
    df['obv'] = obv.on_balance_volume()
    df['obv_ema'] = df['obv'].ewm(span=20).mean()
    
    # Volume moving average
    df['volume_sma_20'] = volume.rolling(window=20).mean()
    df['volume_ratio'] = volume / df['volume_sma_20']
    
    # VWAP (Volume Weighted Average Price)
    vwap = VolumeWeightedAveragePrice(high, low, close, volume)
    df['vwap'] = vwap.volume_weighted_average_price()
    df['price_vs_vwap'] = (close - df['vwap']) / df['vwap']
    
    return df


# ============================================================================
# PRICE FEATURES
# ============================================================================

def add_price_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add basic price-derived features
    """
    df = df.copy()
    
    # Returns
    df['returns'] = df['Close'].pct_change()
    df['log_returns'] = np.log(df['Close'] / df['Close'].shift(1))
    
    # Intraday ranges
    df['high_low_range'] = (df['High'] - df['Low']) / df['Close']
    df['close_open_range'] = (df['Close'] - df['Open']) / df['Open']
    
    return df


# ============================================================================
# MAIN FEATURE ENGINEERING FUNCTION
# ============================================================================

def engineer_all_features(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Apply all feature engineering steps
    
    Parameters:
        df: DataFrame with OHLCV data
        verbose: Print progress messages
    
    Returns:
        DataFrame with all technical features
    """
    data_length = len(df)
    
    if verbose:
        print("\n[*] Engineering Technical Features...")
        print(f"   Input: {len(df)} rows, {len(df.columns)} columns")
    
    # Apply each category
    df = add_price_features(df)
    if verbose:
        print(f"   [+] Price features added")
    
    df = add_trend_features(df)
    if verbose:
        print(f"   [+] Trend indicators added (EMA, ADX, Ichimoku)")
    
    df = add_momentum_features(df)
    if verbose:
        print(f"   [+] Momentum indicators added (RSI, Stochastic, CCI)")
    
    df = add_volatility_features(df)
    if verbose:
        print(f"   [+] Volatility indicators added (ATR, BB, Keltner)")
    
    df = add_volume_features(df)
    if verbose:
        print(f"   [+] Volume indicators added (OBV, VWAP)")
    
    # For short datasets (<400 rows), drop long-period indicators to preserve samples
    if data_length < 400:
        long_period_cols = ['ema_200', 'sma_200', 'price_vs_ema200']
        cols_to_drop = [c for c in long_period_cols if c in df.columns]
        if cols_to_drop:
            df = df.drop(columns=cols_to_drop)
            if verbose:
                print(f"   [!] Dropped long-period indicators ({', '.join(cols_to_drop)}) to preserve data")
    
    # Drop NaN rows (from rolling windows)
    initial_rows = len(df)
    df = df.dropna()
    dropped = initial_rows - len(df)
    
    if verbose:
        print(f"\n   Output: {len(df)} rows, {len(df.columns)} columns")
        print(f"   Dropped {dropped} rows due to indicator lookback periods")
        print(f"   Features: {len(ALL_FEATURES)} total")
    
    return df


# ============================================================================
# FEATURE SELECTION
# ============================================================================

def get_feature_list(category: str = 'all') -> List[str]:
    """
    Get list of features by category
    
    Parameters:
        category: 'all', 'trend', 'momentum', 'volatility', 'volume', 'price'
    
    Returns:
        List of feature names
    """
    categories = {
        'all': ALL_FEATURES,
        'trend': TREND_FEATURES,
        'momentum': MOMENTUM_FEATURES,
        'volatility': VOLATILITY_FEATURES,
        'volume': VOLUME_FEATURES,
        'price': PRICE_FEATURES,
    }
    
    return categories.get(category.lower(), ALL_FEATURES)


def prepare_feature_matrix(df: pd.DataFrame, features: List[str] = None) -> pd.DataFrame:
    """
    Prepare final feature matrix for ML models
    
    Parameters:
        df: DataFrame with all features
        features: List of feature names to use (default: all)
    
    Returns:
        DataFrame with selected features, ready for training
    """
    if features is None:
        features = get_feature_list('all')
    
    # Only use features that exist in the DataFrame
    # (Some features may have been dropped for short datasets)
    available_features = [f for f in features if f in df.columns]
    
    # Select only available features
    X = df[available_features].copy()
    
    # Handle any remaining NaN (shouldn't be any, but safety check)
    X = X.fillna(method='ffill').fillna(method='bfill').fillna(0)
    
    return X


# ============================================================================
# FEATURE STATISTICS
# ============================================================================

def analyze_features(df: pd.DataFrame, verbose: bool = True) -> Dict:
    """
    Analyze feature quality and completeness
    
    Returns:
        Dictionary with feature statistics
    """
    stats = {
        'total_features': len(ALL_FEATURES),
        'trend_features': len(TREND_FEATURES),
        'momentum_features': len(MOMENTUM_FEATURES),
        'volatility_features': len(VOLATILITY_FEATURES),
        'volume_features': len(VOLUME_FEATURES),
        'price_features': len(PRICE_FEATURES),
        'samples': len(df),
        'nan_counts': {},
        'infinite_counts': {},
    }
    
    # Check for NaN and Inf
    for feature in ALL_FEATURES:
        if feature in df.columns:
            stats['nan_counts'][feature] = df[feature].isna().sum()
            stats['infinite_counts'][feature] = np.isinf(df[feature]).sum()
    
    if verbose:
        print("\n[*] Feature Analysis:")
        print(f"   Total Features: {stats['total_features']}")
        print(f"     - Trend: {stats['trend_features']}")
        print(f"     - Momentum: {stats['momentum_features']}")
        print(f"     - Volatility: {stats['volatility_features']}")
        print(f"     - Volume: {stats['volume_features']}")
        print(f"     - Price: {stats['price_features']}")
        print(f"   Samples: {stats['samples']}")
        
        # Report any data quality issues
        nan_features = [k for k, v in stats['nan_counts'].items() if v > 0]
        inf_features = [k for k, v in stats['infinite_counts'].items() if v > 0]
        
        if nan_features:
            print(f"   [!] Features with NaN: {len(nan_features)}")
        if inf_features:
            print(f"   [!] Features with Inf: {len(inf_features)}")
    
    return stats


# Example usage
if __name__ == "__main__":
    print("Technical Features Module")
    print(f"\nAvailable features: {len(ALL_FEATURES)}")
    print(f"  - Trend: {len(TREND_FEATURES)}")
    print(f"  - Momentum: {len(MOMENTUM_FEATURES)}")
    print(f"  - Volatility: {len(VOLATILITY_FEATURES)}")
    print(f"  - Volume: {len(VOLUME_FEATURES)}")
    print(f"  - Price: {len(PRICE_FEATURES)}")

