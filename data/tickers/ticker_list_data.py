"""
Ticker list data module
Fetches and manages stock ticker lists from public sources
"""

import pandas as pd
import streamlit as st
import requests
from io import StringIO


@st.cache_data(ttl=86400)  # Cache for 24 hours
def load_ticker_list() -> list[dict]:
    """
    Fetch S&P 500 ticker list from Wikipedia
    
    Returns:
        List of dictionaries with ticker and company name
        Format: [{"ticker": "AAPL", "name": "Apple Inc."}, ...]
    """
    try:
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        
        # Add User-Agent header to avoid 403 Forbidden error
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        # Fetch HTML with proper headers
        response = requests.get(url, headers=headers)
        response.raise_for_status()  # Raise exception for bad status codes
        
        # Parse HTML tables
        tables = pd.read_html(StringIO(response.text))
        sp500_table = tables[0]  # First table contains the companies
        
        tickers = []
        for _, row in sp500_table.iterrows():
            tickers.append({
                "ticker": row['Symbol'],
                "name": row['Security']
            })
        
        # Sort alphabetically by ticker symbol
        return sorted(tickers, key=lambda x: x['ticker'])
    
    except Exception as e:
        # Show warning to user but don't crash the app
        st.warning(f"⚠️ Could not load ticker list: {str(e)}. You can still enter tickers manually.")
        return []


def get_ticker_options() -> list[str]:
    """
    Get formatted ticker options for selectbox display
    
    Returns:
        List of strings formatted as "TICKER - Company Name"
        Example: ["AAPL - Apple Inc.", "GOOGL - Alphabet Inc.", ...]
    """
    tickers = load_ticker_list()
    
    if not tickers:
        return []
    
    # Format as "TICKER - Company Name"
    return [f"{t['ticker']} - {t['name']}" for t in tickers]


def extract_ticker(selected_option: str) -> str:
    """
    Extract ticker symbol from formatted string
    
    Parameters:
        selected_option: Formatted string like "AAPL - Apple Inc."
    
    Returns:
        Ticker symbol (e.g., "AAPL")
    """
    if not selected_option or " - " not in selected_option:
        return ""
    
    # Split on " - " and take first part
    return selected_option.split(" - ")[0].strip()

