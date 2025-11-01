"""
Buy, Sell, Hold - Stock Prediction Application
Main Streamlit entry point
"""

import streamlit as st
import re
import time


def initialize_session_state():
    """Initialize session state variables"""
    if 'ticker' not in st.session_state:
        st.session_state.ticker = ''
    if 'timeframe' not in st.session_state:
        st.session_state.timeframe = '6M'
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    if 'prediction_result' not in st.session_state:
        st.session_state.prediction_result = None
    if 'analysis_triggered' not in st.session_state:
        st.session_state.analysis_triggered = False
    if 'last_ticker' not in st.session_state:
        st.session_state.last_ticker = ''
    if 'last_timeframe' not in st.session_state:
        st.session_state.last_timeframe = ''


def validate_ticker(ticker: str) -> tuple[bool, str]:
    """
    Validate stock ticker format
    
    Parameters:
        ticker: Stock ticker symbol string
    
    Returns:
        Tuple of (is_valid: bool, error_message: str)
    """
    if not ticker or ticker.strip() == '':
        return False, "Please enter a stock ticker symbol"
    
    ticker = ticker.strip()
    
    if len(ticker) > 10:
        return False, "Ticker symbol too long. Maximum 10 characters"
    
    # Check for valid characters (alphanumeric only)
    if not re.match(r'^[A-Z0-9]+$', ticker):
        return False, "Invalid ticker format. Use only letters and numbers"
    
    return True, ""


def main():
    """Main application function"""
    # Page configuration - must be first Streamlit command
    st.set_page_config(
        page_title="Buy, Sell, Hold - Stock Prediction",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize session state
    initialize_session_state()
    
    # Header section
    st.title("📈 Buy, Sell, Hold - Stock Prediction")
    st.markdown("""
    ### AI-Powered Trading Signal Generator
    
    This application analyzes stock data using technical indicators and predictive models 
    to generate **Buy**, **Sell**, or **Hold** recommendations.
    
    **Disclaimer:** This tool is for educational and informational purposes only. 
    Not financial advice. Always do your own research before making investment decisions.
    """)
    
    st.divider()
    
    # Input section (Phase 1.2)
    st.subheader("📊 Stock Analysis")
    
    input_container = st.container()
    with input_container:
        # Create three columns for layout
        col1, col2, col3 = st.columns([5, 3, 2])
        
        with col1:
            # Ticker input field
            ticker_input = st.text_input(
                "Stock Ticker Symbol",
                value=st.session_state.ticker,
                placeholder="e.g., AAPL, GOOGL, TSLA",
                max_chars=10,
                help="Enter a valid stock ticker symbol (e.g., AAPL for Apple Inc.)"
            ).upper()
        
        with col2:
            # Timeframe selector
            timeframe_options = {
                "1 Month": "1M",
                "3 Months": "3M",
                "6 Months": "6M",
                "1 Year": "1Y",
                "2 Years": "2Y",
                "5 Years": "5Y"
            }
            
            # Find the display name for current timeframe
            current_display = [k for k, v in timeframe_options.items() 
                             if v == st.session_state.timeframe][0]
            
            timeframe_display = st.selectbox(
                "Analysis Timeframe",
                options=list(timeframe_options.keys()),
                index=list(timeframe_options.keys()).index(current_display),
                help="Select the time period for historical data analysis"
            )
            
            timeframe_value = timeframe_options[timeframe_display]
        
        with col3:
            # Analyze button with vertical spacing to align with inputs
            st.write("")  # Add spacing
            st.write("")  # Add spacing
            analyze_button = st.button(
                "🔍 Analyze",
                type="primary",
                use_container_width=True,
                help="Click to analyze the stock"
            )
        
        # Validate inputs and handle analyze button click
        if analyze_button:
            is_valid, error_msg = validate_ticker(ticker_input)
            
            if not is_valid:
                st.error(f"❌ {error_msg}")
            else:
                # Update session state with valid inputs
                st.session_state.ticker = ticker_input
                st.session_state.timeframe = timeframe_value
                st.session_state.analysis_triggered = True
                st.session_state.last_ticker = ticker_input
                st.session_state.last_timeframe = timeframe_value
                
                # Show success message and mock loading
                st.success(f"✅ Analyzing **{ticker_input}** over **{timeframe_display}**")
                
                with st.spinner("🔄 Fetching data and generating prediction..."):
                    time.sleep(1.5)  # Simulate processing
                
                st.info("📊 **Phase 2 Coming Soon**: Data fetching and technical indicators will be implemented next!")
        
        # Show helpful hints when no analysis triggered
        if not st.session_state.analysis_triggered:
            st.info("💡 **Tip**: Enter a stock ticker (like AAPL, MSFT, TSLA) and select a timeframe to get started!")
    
    st.divider()
    
    # Results display section placeholders (Phase 1.3)
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📈 Stock Price & Indicators")
        chart_placeholder = st.empty()
        chart_placeholder.info("📊 Stock price chart will be displayed here")
    
    with col2:
        st.subheader("🎯 Prediction Results")
        prediction_placeholder = st.empty()
        prediction_placeholder.info("🔮 Buy/Sell/Hold signal will be displayed here")
    
    st.divider()
    
    # Technical indicators section placeholder
    st.subheader("🔧 Technical Indicators")
    indicators_container = st.container()
    with indicators_container:
        ind_col1, ind_col2, ind_col3 = st.columns(3)
        with ind_col1:
            st.info("📉 RSI visualization")
        with ind_col2:
            st.info("📊 MACD visualization")
        with ind_col3:
            st.info("📈 Bollinger Bands")
    
    # Footer
    st.divider()
    st.caption("Built with Streamlit | Data from Yahoo Finance | © 2025")


if __name__ == "__main__":
    main()

