"""
Technical indicators calculation module
Uses the `ta` library for standard technical analysis indicators
"""

import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import MACD
from ta.volatility import BollingerBands


def calculate_rsi(df: pd.DataFrame, period: int = 14) -> dict:
    """
    Calculate Relative Strength Index
    
    Parameters:
        df: DataFrame with OHLCV data
        period: RSI period (default: 14 days)
    
    Returns:
        Dictionary with RSI value and status
    """
    try:
        if len(df) < period + 1:
            return {"value": 50.0, "status": "Neutral"}
        
        rsi_indicator = RSIIndicator(close=df['Close'], window=period)
        rsi_series = rsi_indicator.rsi()
        rsi_value = rsi_series.iloc[-1]
        
        # Determine status
        if rsi_value > 70:
            status = "Overbought"
        elif rsi_value < 30:
            status = "Oversold"
        else:
            status = "Neutral"
        
        return {
            "value": float(rsi_value),
            "status": status
        }
    
    except Exception:
        # Return neutral default on error
        return {"value": 50.0, "status": "Neutral"}


def calculate_macd(df: pd.DataFrame) -> dict:
    """
    Calculate MACD (Moving Average Convergence Divergence)
    
    Parameters:
        df: DataFrame with OHLCV data
    
    Returns:
        Dictionary with MACD values and trend
    """
    try:
        if len(df) < 26:  # Need at least 26 periods for MACD
            return {
                "macd": 0.0,
                "macd_signal": 0.0,
                "macd_diff": 0.0,
                "trend": "neutral"
            }
        
        macd_indicator = MACD(close=df['Close'])
        macd_line = macd_indicator.macd().iloc[-1]
        signal_line = macd_indicator.macd_signal().iloc[-1]
        macd_diff = macd_indicator.macd_diff().iloc[-1]
        
        # Determine trend
        if macd_line > signal_line:
            trend = "bullish"
        elif macd_line < signal_line:
            trend = "bearish"
        else:
            trend = "neutral"
        
        return {
            "macd": float(macd_line),
            "macd_signal": float(signal_line),
            "macd_diff": float(macd_diff),
            "trend": trend
        }
    
    except Exception:
        # Return neutral default on error
        return {
            "macd": 0.0,
            "macd_signal": 0.0,
            "macd_diff": 0.0,
            "trend": "neutral"
        }


def calculate_bollinger_bands(df: pd.DataFrame, period: int = 20, std_dev: int = 2) -> dict:
    """
    Calculate Bollinger Bands
    
    Parameters:
        df: DataFrame with OHLCV data
        period: Moving average period (default: 20)
        std_dev: Standard deviation multiplier (default: 2)
    
    Returns:
        Dictionary with Bollinger Bands values and position
    """
    try:
        if len(df) < period + 1:
            return {
                "bb_upper": 0.0,
                "bb_middle": 0.0,
                "bb_lower": 0.0,
                "bb_width": 0.0,
                "position": "Within Bands"
            }
        
        bb_indicator = BollingerBands(
            close=df['Close'], 
            window=period, 
            window_dev=std_dev
        )
        
        upper = bb_indicator.bollinger_hband().iloc[-1]
        middle = bb_indicator.bollinger_mavg().iloc[-1]
        lower = bb_indicator.bollinger_lband().iloc[-1]
        current_price = df['Close'].iloc[-1]
        
        # Determine position
        if current_price > upper:
            position = "Above Upper Band"
        elif current_price < lower:
            position = "Below Lower Band"
        else:
            position = "Within Bands"
        
        return {
            "bb_upper": float(upper),
            "bb_middle": float(middle),
            "bb_lower": float(lower),
            "bb_width": float(upper - lower),
            "position": position
        }
    
    except Exception:
        # Return default on error
        return {
            "bb_upper": 0.0,
            "bb_middle": 0.0,
            "bb_lower": 0.0,
            "bb_width": 0.0,
            "position": "Within Bands"
        }


def calculate_all_indicators(df: pd.DataFrame) -> dict:
    """
    Calculate all technical indicators at once
    
    Parameters:
        df: DataFrame with OHLCV data
    
    Returns:
        Dictionary with all indicator values
    """
    rsi_data = calculate_rsi(df)
    macd_data = calculate_macd(df)
    bb_data = calculate_bollinger_bands(df)
    
    return {
        "rsi": rsi_data["value"],
        "rsi_status": rsi_data["status"],
        "macd": macd_data["macd"],
        "macd_signal": macd_data["macd_signal"],
        "macd_diff": macd_data["macd_diff"],
        "macd_trend": macd_data["trend"],
        "bb_upper": bb_data["bb_upper"],
        "bb_middle": bb_data["bb_middle"],
        "bb_lower": bb_data["bb_lower"],
        "bb_width": bb_data["bb_width"],
        "bb_position": bb_data["position"]
    }


