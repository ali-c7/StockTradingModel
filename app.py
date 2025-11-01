"""
Buy, Sell, Hold - Stock Prediction Application
Main Streamlit entry point
"""

import streamlit as st
import re
import time
from datetime import datetime
import random


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
    
    # Display preferences
    if 'show_confidence' not in st.session_state:
        st.session_state.show_confidence = True
    if 'show_reasoning' not in st.session_state:
        st.session_state.show_reasoning = True
    if 'show_indicator_details' not in st.session_state:
        st.session_state.show_indicator_details = True
    if 'view_mode' not in st.session_state:
        st.session_state.view_mode = 'detailed'


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


def generate_mock_results(ticker: str, timeframe: str) -> dict:
    """
    Generate mock prediction results for demonstration
    
    Parameters:
        ticker: Stock ticker symbol
        timeframe: Analysis timeframe
    
    Returns:
        Dictionary containing mock results data
    """
    # Set random seed based on ticker for consistency
    random.seed(hash(ticker) % 1000)
    
    # Generate random signal
    signals = ["BUY", "SELL", "HOLD"]
    signal = random.choice(signals)
    
    # Generate mock data
    base_price = random.uniform(50, 500)
    price_change = random.uniform(-15, 15)
    price_change_pct = (price_change / base_price) * 100
    
    # Signal-specific reasoning
    reasoning_map = {
        "BUY": [
            "RSI indicates oversold conditions. MACD showing bullish crossover. Strong volume support.",
            "Price broke above key resistance. Positive momentum indicators. Volume trending up.",
            "Technical indicators align for upward movement. Support level holding strong."
        ],
        "SELL": [
            "RSI shows overbought conditions. MACD bearish divergence detected. Weakening volume.",
            "Price approaching resistance with negative momentum. Volume declining.",
            "Multiple indicators suggest downward pressure. Resistance level rejection observed."
        ],
        "HOLD": [
            "Mixed signals across indicators. Price consolidating in range. Wait for clearer direction.",
            "Neutral momentum. No strong buy or sell signals detected. Market indecision.",
            "Technical indicators inconclusive. Sideways trend observed. Patience recommended."
        ]
    }
    
    return {
        "signal": signal,
        "confidence": round(random.uniform(65, 92), 1),
        "current_price": round(base_price, 2),
        "price_change": round(price_change, 2),
        "price_change_pct": round(price_change_pct, 2),
        "volume": f"{random.randint(1, 50)}M",
        "rsi": round(random.uniform(20, 80), 1),
        "macd": round(random.uniform(-5, 5), 2),
        "bb_status": random.choice(["Above Upper Band", "Below Lower Band", "Within Bands"]),
        "reasoning": random.choice(reasoning_map[signal]),
        "accuracy": round(random.uniform(55, 75), 1),
        "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "signals_analyzed": random.randint(100, 500)
    }


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
    
    # Sidebar configuration
    with st.sidebar:
        st.title("⚙️ Settings")
        
        # App Information
        st.markdown("### 📊 About")
        st.markdown("""
        **Buy, Sell, Hold Predictor**  
        Version 1.0 (MVP)
        
        AI-powered stock analysis using technical indicators and predictive models.
        """)
        
        st.divider()
        
        # Display Preferences
        st.markdown("### 🎨 Display Preferences")
        
        st.session_state.show_confidence = st.checkbox(
            "Show confidence scores",
            value=st.session_state.show_confidence,
            help="Display model confidence percentages"
        )
        
        st.session_state.show_reasoning = st.checkbox(
            "Show signal reasoning",
            value=st.session_state.show_reasoning,
            help="Display explanation for each signal"
        )
        
        st.session_state.show_indicator_details = st.checkbox(
            "Show indicator details",
            value=st.session_state.show_indicator_details,
            help="Display detailed technical indicator information"
        )
        
        st.divider()
        
        # Advanced Settings
        with st.expander("🔧 Advanced Settings"):
            st.markdown("**Coming in Phase 3:**")
            st.markdown("- Model selection")
            st.markdown("- Signal sensitivity")
            st.markdown("- Risk tolerance")
            st.info("Advanced configuration will be available after Phase 2-3 implementation.")
        
        # Help & Guide
        with st.expander("❓ Help & Guide"):
            st.markdown("""
            **How to use:**
            1. Enter a stock ticker (e.g., AAPL)
            2. Select analysis timeframe
            3. Click "Analyze" button
            4. Review the prediction signal
            
            **Signal meanings:**
            - 🟢 **BUY**: Indicators suggest upward movement
            - 🔴 **SELL**: Indicators suggest downward pressure
            - 🟡 **HOLD**: Mixed signals, wait for clarity
            
            **Technical Indicators:**
            - **RSI**: Momentum indicator (oversold <30, overbought >70)
            - **MACD**: Trend direction and strength
            - **Bollinger Bands**: Volatility and price extremes
            """)
        
        # Documentation Links
        with st.expander("📚 Documentation"):
            st.markdown("""
            - [Product Brief](https://github.com/ali-c7/Buy-Sell-Hold-Predictive-Model/blob/main/docs/buy_sell_hold_product_brief.md)
            - [Features Plan](https://github.com/ali-c7/Buy-Sell-Hold-Predictive-Model/blob/main/docs/FEATURES_PLAN.md)
            - [Signal Visualization](https://github.com/ali-c7/Buy-Sell-Hold-Predictive-Model/blob/main/docs/signal_visualization_approach.md)
            """)
        
        st.divider()
        
        # Clear Analysis Button
        if st.button("🔄 Clear Analysis", use_container_width=True):
            st.session_state.analysis_triggered = False
            st.session_state.prediction_result = None
            st.session_state.ticker = ''
            st.session_state.last_ticker = ''
            st.session_state.last_timeframe = ''
            st.success("✅ Analysis cleared! Enter a new ticker to start.")
            st.rerun()
        
        st.divider()
        
        # Footer in sidebar
        st.caption("**Disclaimer:** Educational purposes only. Not financial advice.")
        st.caption("© 2025 - Built with Streamlit")
    
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
                    # Generate mock results
                    st.session_state.prediction_result = generate_mock_results(ticker_input, timeframe_value)
                
                st.success("✅ Analysis complete! See results below.")
        
        # Show helpful hints when no analysis triggered
        if not st.session_state.analysis_triggered:
            st.info("💡 **Tip**: Enter a stock ticker (like AAPL, MSFT, TSLA) and select a timeframe to get started!")
    
    st.divider()
    
    # Results display section (Phase 1.3)
    if st.session_state.analysis_triggered and st.session_state.prediction_result:
        results = st.session_state.prediction_result
        
        # Top section: Stock Info and Signal
        col1, col2 = st.columns([3, 2])
        
        with col1:
            st.subheader(f"📊 {st.session_state.ticker} Stock Information")
            
            # Stock metrics
            metric_col1, metric_col2, metric_col3 = st.columns(3)
            with metric_col1:
                st.metric(
                    label="Current Price",
                    value=f"${results['current_price']:.2f}",
                    delta=f"{results['price_change']:+.2f} ({results['price_change_pct']:+.2f}%)"
                )
            with metric_col2:
                st.metric(
                    label="Volume",
                    value=results['volume']
                )
            with metric_col3:
                st.metric(
                    label="Model Accuracy",
                    value=f"{results['accuracy']}%"
                )
        
        with col2:
            st.subheader("🎯 Trading Signal")
            
            # Color-coded signal display
            signal = results['signal']
            if signal == "BUY":
                signal_color = "#00CC00"
                signal_emoji = "🟢"
            elif signal == "SELL":
                signal_color = "#FF3333"
                signal_emoji = "🔴"
            else:  # HOLD
                signal_color = "#FFA500"
                signal_emoji = "🟡"
            
            # Display signal with custom styling
            if st.session_state.show_confidence:
                st.markdown(f"""
                <div style="background-color: {signal_color}; padding: 20px; border-radius: 10px; text-align: center;">
                    <h1 style="color: white; margin: 0;">{signal_emoji} {signal}</h1>
                    <p style="color: white; font-size: 18px; margin: 10px 0 0 0;">
                        Confidence: {results['confidence']}%
                    </p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style="background-color: {signal_color}; padding: 20px; border-radius: 10px; text-align: center;">
                    <h1 style="color: white; margin: 0;">{signal_emoji} {signal}</h1>
                </div>
                """, unsafe_allow_html=True)
            
            if st.session_state.show_reasoning:
                st.caption(f"**Analysis Reasoning:**")
                st.info(results['reasoning'])
        
        st.divider()
        
        # Price chart placeholder
        st.subheader("📈 Stock Price Chart & Historical Signals")
        st.info("📊 **Coming in Phase 2**: Interactive candlestick chart with historical price data, volume, and buy/sell signal markers will be displayed here. Historical signals will be shown as markers on the chart to visualize past recommendations.")
        
        st.divider()
        
        # Technical indicators section
        st.subheader("🔧 Technical Indicators")
        ind_col1, ind_col2, ind_col3 = st.columns(3)
        
        with ind_col1:
            st.metric(
                label="RSI (Relative Strength Index)",
                value=f"{results['rsi']:.1f}",
                help="RSI measures momentum. Values above 70 suggest overbought, below 30 suggest oversold."
            )
            if st.session_state.show_indicator_details:
                if results['rsi'] > 70:
                    st.caption("⚠️ Overbought territory")
                elif results['rsi'] < 30:
                    st.caption("💡 Oversold territory")
                else:
                    st.caption("✅ Neutral range")
        
        with ind_col2:
            st.metric(
                label="MACD",
                value=f"{results['macd']:.2f}",
                help="MACD shows trend direction and momentum. Positive values suggest bullish, negative suggest bearish."
            )
            if st.session_state.show_indicator_details:
                if results['macd'] > 0:
                    st.caption("📈 Bullish signal")
                else:
                    st.caption("📉 Bearish signal")
        
        with ind_col3:
            st.metric(
                label="Bollinger Bands",
                value=results['bb_status'],
                help="Bollinger Bands measure volatility. Price near bands suggests potential reversal."
            )
            if st.session_state.show_indicator_details:
                if "Above" in results['bb_status']:
                    st.caption("⚠️ Potential overbought")
                elif "Below" in results['bb_status']:
                    st.caption("💡 Potential oversold")
                else:
                    st.caption("✅ Normal range")
        
        if st.session_state.show_indicator_details:
            st.info("📊 **Coming in Phase 2**: Detailed indicator charts with historical trends will be displayed here.")
        
        st.divider()
        
        # Performance metrics
        st.subheader("📊 Model Performance")
        perf_col1, perf_col2, perf_col3 = st.columns(3)
        
        with perf_col1:
            st.metric("Model Accuracy", f"{results['accuracy']}%")
        with perf_col2:
            st.metric("Signals Analyzed", results['signals_analyzed'])
        with perf_col3:
            st.metric("Last Updated", results['last_updated'])
        
        st.caption("💡 **Note**: These are mock results for demonstration. Real analysis will be implemented in Phase 2-3.")
    
    else:
        # Show placeholders when no analysis has been run
        st.info("👆 Enter a stock ticker and click **Analyze** to see prediction results and visualizations.")
    
    # Footer
    st.divider()
    st.caption("Built with Streamlit | Data from Yahoo Finance | © 2025")


if __name__ == "__main__":
    main()

