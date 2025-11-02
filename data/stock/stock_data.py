"""
Stock data retrieval module
Fetches historical stock data from Yahoo Finance
"""

from datetime import datetime, timedelta
import pandas as pd
import yfinance as yf
import streamlit as st


def calculate_date_range(timeframe: str) -> tuple[str, str]:
    """
    Calculate start and end dates based on timeframe
    
    Parameters:
        timeframe: Time period ("1M", "3M", "6M", "1Y", "2Y", "5Y")
    
    Returns:
        Tuple of (start_date, end_date) as strings "YYYY-MM-DD"
    """
    end_date = datetime.now()
    
    timeframe_map = {
        "1M": timedelta(days=30),
        "3M": timedelta(days=90),
        "6M": timedelta(days=180),
        "1Y": timedelta(days=365),
        "2Y": timedelta(days=730),
        "5Y": timedelta(days=1825)
    }
    
    delta = timeframe_map.get(timeframe, timedelta(days=180))  # Default to 6 months
    start_date = end_date - delta
    
    return start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")


def validate_ticker_data(df: pd.DataFrame, ticker: str) -> bool:
    """
    Validate that fetched data is usable
    
    Parameters:
        df: DataFrame with stock data
        ticker: Ticker symbol for error messages
    
    Returns:
        True if data is valid, False otherwise
    """
    if df is None or df.empty:
        st.error(f"❌ No data available for ticker '{ticker}'. Please check the ticker symbol.")
        return False
    
    # Check minimum data points (at least 5 days)
    if len(df) < 5:
        st.error(f"❌ Insufficient data for '{ticker}'. Only {len(df)} data points available.")
        return False
    
    # Check for required columns
    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        st.error(f"❌ Missing required columns for '{ticker}': {', '.join(missing_cols)}")
        return False
    
    # Check for reasonable price values
    if (df['Close'] <= 0).any():
        st.error(f"❌ Invalid price data for '{ticker}'. Contains non-positive values.")
        return False
    
    return True


def fetch_stock_data(ticker: str, timeframe: str) -> pd.DataFrame | None:
    """
    Fetch historical OHLCV data from Yahoo Finance
    
    Parameters:
        ticker: Stock ticker symbol (e.g., "AAPL")
        timeframe: Analysis timeframe ("1M", "3M", "6M", "1Y", "2Y", "5Y")
    
    Returns:
        DataFrame with columns: Date (index), Open, High, Low, Close, Volume
        None if fetch fails
    """
    # Get today's date to include in cache key - ensures daily refresh
    today = datetime.now().strftime("%Y-%m-%d")
    return _fetch_stock_data_cached(ticker, timeframe, today)


@st.cache_data(ttl=3600)  # Cache for 1 hour
def _fetch_stock_data_cached(ticker: str, timeframe: str, cache_date: str) -> pd.DataFrame | None:
    """
    Internal cached function for fetching stock data
    Cache key includes current date to ensure daily refresh
    """
    try:
        start_date, end_date = calculate_date_range(timeframe)
        
        # Download data from Yahoo Finance
        # Note: Yahoo Finance daily data for today won't be available until after market close
        stock = yf.Ticker(ticker)
        df = stock.history(start=start_date, end=end_date)
        
        if df.empty:
            return None
        
        # Log the date range we received (for debugging)
        if not df.empty:
            last_date = df.index[-1].strftime("%Y-%m-%d")
            print(f"[DEBUG] {ticker}: Last data point available from Yahoo Finance: {last_date}")
        
        # Ensure we have required columns
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in df.columns for col in required_cols):
            return None
        
        # Return only required columns
        return df[required_cols]
    
    except Exception as e:
        st.error(f"❌ Error fetching data for '{ticker}': {str(e)}")
        return None


def get_current_price(ticker: str, df: pd.DataFrame) -> dict:
    """
    Get current price and change information
    
    Parameters:
        ticker: Stock ticker symbol
        df: DataFrame with historical stock data
    
    Returns:
        Dictionary with current price, change, and volume
    """
    if df is None or df.empty:
        return {
            "price": 0.0,
            "change": 0.0,
            "change_pct": 0.0,
            "volume": "0",
            "is_realtime": False
        }
    
    try:
        # Try to get real-time current price
        stock = yf.Ticker(ticker)
        info = stock.info
        
        # Try different fields for current price (Yahoo Finance uses different keys)
        current_price = None
        for key in ['currentPrice', 'regularMarketPrice', 'price']:
            if key in info and info[key] is not None:
                current_price = float(info[key])
                break
        
        # If real-time price available, use it
        if current_price is not None and current_price > 0:
            previous_close = info.get('previousClose', df['Close'].iloc[-1])
            change = current_price - previous_close
            change_pct = (change / previous_close) * 100
            
            # Get current volume (real-time or latest)
            current_volume = info.get('volume', df['Volume'].iloc[-1])
            
            is_realtime = True
        else:
            # Fall back to historical data (last close)
            current_price = df['Close'].iloc[-1]
            previous_close = df['Close'].iloc[-2] if len(df) > 1 else current_price
            change = current_price - previous_close
            change_pct = (change / previous_close) * 100 if previous_close != 0 else 0.0
            current_volume = df['Volume'].iloc[-1]
            is_realtime = False
    
    except Exception:
        # Fall back to historical data on any error
        current_price = df['Close'].iloc[-1]
        previous_close = df['Close'].iloc[-2] if len(df) > 1 else current_price
        change = current_price - previous_close
        change_pct = (change / previous_close) * 100 if previous_close != 0 else 0.0
        current_volume = df['Volume'].iloc[-1]
        is_realtime = False
    
    # Format volume (convert to M or K)
    if current_volume >= 1_000_000:
        volume_str = f"{current_volume / 1_000_000:.1f}M"
    elif current_volume >= 1_000:
        volume_str = f"{current_volume / 1_000:.1f}K"
    else:
        volume_str = str(int(current_volume))
    
    return {
        "price": float(current_price),
        "change": float(change),
        "change_pct": float(change_pct),
        "volume": volume_str,
        "is_realtime": is_realtime
    }


def get_stock_info(ticker: str) -> dict:
    """
    Get basic stock information (company name, etc.)
    
    Parameters:
        ticker: Stock ticker symbol
    
    Returns:
        Dictionary with stock info
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        return {
            "name": info.get("longName", ticker),
            "sector": info.get("sector", "N/A"),
            "industry": info.get("industry", "N/A")
        }
    except:
        return {
            "name": ticker,
            "sector": "N/A",
            "industry": "N/A"
        }

