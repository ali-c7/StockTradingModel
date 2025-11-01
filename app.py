"""
Buy, Sell, Hold - Stock Prediction Application
Main Streamlit entry point
"""

import streamlit as st
import re
import time
from datetime import datetime, timedelta
import random

# Import ticker list functions
from data.tickers.ticker_list_data import get_ticker_options, extract_ticker

# Import stock data functions
from data.stock.stock_data import fetch_stock_data, validate_ticker_data, get_current_price, get_stock_info

# Import plotting functions
from plots.stock.stock_plot import create_price_chart

# Import technical indicators
from data.indicators.indicators_data import calculate_all_indicators

# Import ML predictor
from core.signals.signal_predictor import SignalPredictor


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
    
    # Chart display timeframe (separate from analysis timeframe)
    if 'chart_timeframe' not in st.session_state:
        st.session_state.chart_timeframe = '6M'
    
    # Validation settings
    if 'run_validation' not in st.session_state:
        st.session_state.run_validation = False
    if 'validation_results' not in st.session_state:
        st.session_state.validation_results = None
    if 'show_label_viz' not in st.session_state:
        st.session_state.show_label_viz = False
    if 'label_viz_data' not in st.session_state:
        st.session_state.label_viz_data = None


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
    
    if len(ticker) > 15:
        return False, "Ticker symbol too long. Maximum 15 characters"
    
    # Allow alphanumeric, dashes, dots, equals, carets (for Yahoo Finance tickers)
    # Examples: BTC-USD, BRK.B, ^GSPC, 0700.HK
    if not re.match(r'^[A-Z0-9\.\-\=\^]+$', ticker):
        return False, "Invalid ticker format. Use only letters, numbers, and . - = ^"
    
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
        
        # Model Validation Settings
        st.markdown("### 🔬 Model Validation")
        
        st.session_state.run_validation = st.checkbox(
            "Run comprehensive validation",
            value=st.session_state.get('run_validation', False),
            help="Test model on 5 different time windows to verify consistency"
        )
        
        if st.session_state.run_validation:
            st.info("⏱️ Adds ~5-10 seconds to analysis")
            st.caption("Tests if the model works consistently over time.")
        
        st.session_state.show_label_viz = st.checkbox(
            "Show label visualization",
            value=st.session_state.get('show_label_viz', False),
            help="Visualize how Buy/Sell/Hold labels are distributed across time"
        )
        
        if st.session_state.show_label_viz:
            st.caption("📊 Shows price chart with label markers and distribution")
        
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
            1. Select a stock ticker or enter custom ticker
            2. Select **analysis timeframe** (data used for prediction)
            3. Click "Analyze" button
            4. Review the prediction signal
            5. Adjust **chart timeframe** to view different time periods
            
            **Timeframes Explained:**
            - **Analysis Timeframe**: Historical data used to generate the prediction
            - **Chart Timeframe**: Time period displayed on the price chart (can be adjusted independently)
            - Example: Analyze based on 6 months, but view 1 year on chart
            
            **Ticker Examples:**
            - **Stocks**: AAPL, GOOGL, TSLA
            - **Crypto**: BTC-USD, ETH-USD, DOGE-USD
            - **Indices**: ^GSPC (S&P 500), ^DJI (Dow Jones)
            - **Multi-class**: BRK.B (Berkshire Hathaway Class B)
            - **International**: 0700.HK (Tencent Hong Kong)
            
            **Signal meanings:**
            - 🟢 **BUY**: Indicators suggest upward movement
            - 🔴 **SELL**: Indicators suggest downward pressure
            - 🟡 **HOLD**: Mixed signals, wait for clarity
            
            **Technical Indicators:**
            - **RSI**: Momentum indicator (oversold <30, overbought >70)
            - **MACD**: Trend direction and strength
            - **Bollinger Bands**: Volatility and price extremes
            
            **Validating with TradingView:**
            1. Set TradingView to **1D (Daily)** chart
            2. Add indicators with same settings:
               - RSI: Length = 14
               - MACD: Fast=12, Slow=26, Signal=9
               - BB: Length=20, StdDev=2
            3. Check the **rightmost value** (latest day)
            4. Values should match (±0.1 due to rounding)
            
            **Model Validation:**
            - Enable "Run comprehensive validation" to test model on 5 time windows
            - Shows mean accuracy, variability, and consistency
            - Helps verify model isn't just lucky on recent data
            - Takes 5-10 extra seconds
            
            **Label Visualization:**
            - Enable "Show label visualization" to see training labels
            - Shows WHERE the model identifies Buy/Sell/Hold opportunities
            - Displays label distribution and future returns
            - Useful for debugging poor model performance
            - Can reveal class imbalance issues
            """)
        
        # Documentation Links
        with st.expander("📚 Documentation"):
            st.markdown("""
            - [Product Brief](https://github.com/ali-c7/Buy-Sell-Hold-Predictive-Model/blob/main/docs/buy_sell_hold_product_brief.md)
            - [Features Plan](https://github.com/ali-c7/Buy-Sell-Hold-Predictive-Model/blob/main/docs/FEATURES_PLAN.md)
            - [Signal Visualization](https://github.com/ali-c7/Buy-Sell-Hold-Predictive-Model/blob/main/docs/signal_visualization_approach.md)
            """)
        
        st.divider()
        
        # Action Buttons
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🔄 Clear Analysis", use_container_width=True):
                st.session_state.analysis_triggered = False
                st.session_state.prediction_result = None
                st.session_state.validation_results = None
                st.session_state.label_viz_data = None
                st.session_state.ticker = ''
                st.session_state.last_ticker = ''
                st.session_state.last_timeframe = ''
                st.success("✅ Analysis cleared! Enter a new ticker to start.")
                st.rerun()
        
        with col2:
            if st.button("🔃 Refresh Data", use_container_width=True, help="Clear cache and fetch latest data"):
                st.cache_data.clear()
                st.success("✅ Cache cleared! Click 'Analyze Stock' to fetch fresh data.")
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
            # Ticker selection with searchable dropdown
            ticker_options = get_ticker_options()
            
            # Add custom ticker option at the beginning
            if ticker_options:
                ticker_options.insert(0, "🔍 Type custom ticker...")
                
                selected_option = st.selectbox(
                    "Stock Ticker Symbol",
                    options=ticker_options,
                    help="Select a ticker or choose 'Type custom ticker...' to enter manually"
                )
                
                # Check if custom ticker was selected
                if selected_option == "🔍 Type custom ticker...":
                    ticker_input = st.text_input(
                        "Enter custom ticker",
                        max_chars=15,
                        placeholder="e.g., AAPL, BTC-USD, BRK.B",
                        help="Enter any valid ticker symbol (crypto, stocks, indices, forex)"
                    ).upper()
                else:
                    ticker_input = extract_ticker(selected_option)
            else:
                # Fallback to text input if ticker list couldn't be loaded
                ticker_input = st.text_input(
                    "Stock Ticker Symbol",
                    value=st.session_state.ticker,
                    placeholder="e.g., AAPL, BTC-USD, ^GSPC",
                    max_chars=15,
                    help="Enter any valid ticker symbol (stocks, crypto, indices)"
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
                help="Data period used for prediction analysis (chart display timeframe can be changed independently)"
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
                
                # Show progress message
                st.info(f"🔍 Fetching data for **{ticker_input}** over **{timeframe_display}**...")
                
                with st.spinner("📊 Downloading historical data from Yahoo Finance..."):
                    # Fetch real stock data
                    stock_df = fetch_stock_data(ticker_input, timeframe_value)
                
                # Validate the data
                if validate_ticker_data(stock_df, ticker_input):
                    # Update session state
                    st.session_state.analysis_triggered = True
                    st.session_state.last_ticker = ticker_input
                    st.session_state.last_timeframe = timeframe_value
                    # Initialize chart timeframe to match analysis timeframe
                    st.session_state.chart_timeframe = timeframe_value
                    
                    # Get current price info (real-time if available)
                    price_info = get_current_price(ticker_input, stock_df)
                    
                    # Get security name for display
                    stock_info = get_stock_info(ticker_input)
                    
                    with st.spinner("🤖 Calculating technical indicators..."):
                        # Calculate real technical indicators
                        indicators = calculate_all_indicators(stock_df)
                    
                    with st.spinner("🤖 Training ML model and generating prediction..."):
                        try:
                            # Train ML model and get prediction
                            predictor = SignalPredictor(ticker_input, timeframe_value)
                            metrics = predictor.train(verbose=False)
                            prediction = predictor.predict_latest()
                            
                            # Run comprehensive validation if enabled
                            if st.session_state.run_validation:
                                with st.spinner("🔬 Running comprehensive validation (5 time windows)..."):
                                    st.session_state.validation_results = predictor.run_comprehensive_validation(n_splits=5)
                            
                            # Capture label data for visualization if requested
                            if st.session_state.show_label_viz:
                                st.session_state.label_viz_data = {
                                    'df': predictor.raw_data.copy(),
                                    'labels': predictor.y.copy(),
                                    'ticker': ticker_input
                                }
                            
                            # Use real ML prediction
                            st.session_state.prediction_result = {
                                # Real ML signal!
                                "signal": prediction["signal"],
                                "confidence": prediction["confidence"],
                                "reasoning": prediction["reasoning"],
                                "accuracy": prediction["model_accuracy"],
                                "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                "signals_analyzed": len(predictor.X),  # Number of training samples
                            
                            # Real price data
                            "current_price": price_info["price"],
                            "price_change": price_info["change"],
                            "price_change_pct": price_info["change_pct"],
                            "volume": price_info["volume"],
                            "is_realtime": price_info["is_realtime"],
                            "security_name": stock_info["name"],
                            
                            # Real technical indicators
                            "rsi": indicators["rsi"],
                            "rsi_status": indicators["rsi_status"],
                            "macd": indicators["macd"],
                            "macd_diff": indicators["macd_diff"],
                            "macd_trend": indicators["macd_trend"],
                            "bb_status": indicators["bb_position"],
                            "bb_upper": indicators["bb_upper"],
                            "bb_middle": indicators["bb_middle"],
                            "bb_lower": indicators["bb_lower"],
                            
                            # OHLCV data for charting
                            "data": stock_df,
                        }
                        
                            st.success(f"✅ Analysis complete! Model trained with {metrics['accuracy']:.1%} accuracy.")
                        
                        except Exception as e:
                            st.error(f"❌ ML prediction failed: {str(e)}")
                            st.info("Showing data without ML prediction...")
                            
                            # Fallback: show data without prediction
                            st.session_state.prediction_result = {
                                "signal": "HOLD",
                                "confidence": 0.33,
                                "reasoning": "ML model training failed. Please try again.",
                                "accuracy": 0.0,
                                "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                "signals_analyzed": 0,
                                
                                # Real price data
                                "current_price": price_info["price"],
                                "price_change": price_info["change"],
                                "price_change_pct": price_info["change_pct"],
                                "volume": price_info["volume"],
                                "is_realtime": price_info["is_realtime"],
                                "security_name": stock_info["name"],
                                
                                # Real technical indicators
                                "rsi": indicators["rsi"],
                                "rsi_status": indicators["rsi_status"],
                                "macd": indicators["macd"],
                                "macd_diff": indicators["macd_diff"],
                                "macd_trend": indicators["macd_trend"],
                                "bb_status": indicators["bb_position"],
                                "bb_upper": indicators["bb_upper"],
                                "bb_middle": indicators["bb_middle"],
                                "bb_lower": indicators["bb_lower"],
                                
                                # OHLCV data for charting
                                "data": stock_df,
                            }
                else:
                    # Data validation failed, error already shown
                    st.session_state.analysis_triggered = False
        
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
            
            # Show company/security name if available
            if 'security_name' in results and results['security_name'] != st.session_state.ticker:
                st.caption(f"**{results['security_name']}**")
            
            # Show price update status
            if results.get('is_realtime', False):
                st.caption("🟢 Live Price (15-20 min delay)")
            else:
                st.caption("🟡 Last Close Price")
            
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
                    value=f"{results['accuracy']:.1%}"
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
                confidence_pct = results['confidence'] * 100  # Convert to percentage
                st.markdown(f"""
                <div style="background-color: {signal_color}; padding: 20px; border-radius: 10px; text-align: center;">
                    <h1 style="color: white; margin: 0;">{signal_emoji} {signal}</h1>
                    <p style="color: white; font-size: 18px; margin: 10px 0 0 0;">
                        Confidence: {confidence_pct:.1f}%
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
        
        # Price chart section
        chart_header_col1, chart_header_col2 = st.columns([3, 1])
        
        with chart_header_col1:
            st.subheader("📈 Stock Price Chart & Historical Signals")
        
        with chart_header_col2:
            # Chart timeframe selector (separate from analysis timeframe)
            if st.session_state.analysis_triggered:
                chart_timeframe_options = {
                    "1 Month": "1M",
                    "3 Months": "3M",
                    "6 Months": "6M",
                    "1 Year": "1Y",
                    "2 Years": "2Y",
                    "5 Years": "5Y"
                }
                
                # Find current display value
                current_chart_display = [k for k, v in chart_timeframe_options.items() 
                                        if v == st.session_state.chart_timeframe][0]
                
                chart_timeframe_display = st.selectbox(
                    "Chart Timeframe",
                    options=list(chart_timeframe_options.keys()),
                    index=list(chart_timeframe_options.keys()).index(current_chart_display),
                    help="Select timeframe to display on chart (independent of analysis timeframe)",
                    key="chart_timeframe_selector"
                )
                
                st.session_state.chart_timeframe = chart_timeframe_options[chart_timeframe_display]
        
        # Check if we have data to plot
        if st.session_state.analysis_triggered and st.session_state.ticker:
            try:
                # Fetch data for chart display (may be different from analysis timeframe)
                with st.spinner(f"📊 Loading {st.session_state.chart_timeframe} chart data..."):
                    chart_data = fetch_stock_data(st.session_state.ticker, st.session_state.chart_timeframe)
                
                if chart_data is not None and not chart_data.empty:
                    # Create and display the price chart
                    chart = create_price_chart(chart_data, st.session_state.ticker)
                    
                    if chart is not None:
                        st.plotly_chart(chart, use_container_width=True)
                        st.caption(f"💡 **Interactive**: Zoom (click & drag), Pan (shift + drag), Reset (double-click) | Showing **{st.session_state.chart_timeframe}** data")
                    else:
                        st.warning("Unable to create chart from available data")
                else:
                    st.warning(f"No chart data available for {st.session_state.chart_timeframe} timeframe")
            except Exception as e:
                st.error(f"Error creating chart: {str(e)}")
        else:
            st.info("📊 Chart will display after analyzing a stock")
        
        if st.session_state.analysis_triggered:
            st.info("🔮 **Coming in Phase 4**: Buy/Sell/Hold signal markers will be overlaid on this chart")
        
        st.divider()
        
        # Technical indicators section
        st.subheader("🔧 Technical Indicators")
        
        # Get last data date for display
        if 'data' in results and not results['data'].empty:
            last_date_obj = results['data'].index[-1]
            last_date = last_date_obj.strftime("%Y-%m-%d")
            today = datetime.now().date()
            data_date = last_date_obj.date()
            days_old = (today - data_date).days
            
            # Build caption with freshness indicator
            caption = f"📅 Calculated from **Daily** data | Last update: **{last_date}**"
            
            if days_old == 0:
                caption += " ✅ (Today)"
            elif days_old == 1:
                caption += " ⏳ (Yesterday - Today's data available after market close ~5pm ET)"
            elif days_old > 1:
                caption += f" ⚠️ ({days_old} days old - Click '🔃 Refresh Data' in sidebar)"
            
            caption += " | Compare with TradingView on **1D chart**"
            st.caption(caption)
        else:
            st.caption("📅 Calculated from **Daily** data | Compare with TradingView on **1D chart**")
        
        ind_col1, ind_col2, ind_col3 = st.columns(3)
        
        with ind_col1:
            st.metric(
                label="RSI",
                value=f"{results['rsi']:.1f}",
                help="RSI measures momentum. Values above 70 suggest overbought, below 30 suggest oversold."
            )
            st.caption("⚙️ 14-period")
            if st.session_state.show_indicator_details:
                rsi_status = results.get('rsi_status', 'Neutral')
                if rsi_status == "Overbought":
                    st.caption("⚠️ Overbought territory (>70)")
                elif rsi_status == "Oversold":
                    st.caption("💡 Oversold territory (<30)")
                else:
                    st.caption("✅ Neutral range (30-70)")
        
        with ind_col2:
            macd_value = results.get('macd', 0)
            macd_diff = results.get('macd_diff', 0)
            st.metric(
                label="MACD",
                value=f"{macd_value:.2f}",
                delta=f"{macd_diff:.2f}",
                help="MACD shows trend direction and momentum. Histogram shows strength."
            )
            st.caption("⚙️ 12/26/9 (Fast/Slow/Signal)")
            if st.session_state.show_indicator_details:
                macd_trend = results.get('macd_trend', 'neutral')
                if macd_trend == "bullish":
                    st.caption("📈 Bullish signal")
                elif macd_trend == "bearish":
                    st.caption("📉 Bearish signal")
                else:
                    st.caption("➡️ Neutral")
        
        with ind_col3:
            st.metric(
                label="Bollinger Bands",
                value=results['bb_status'],
                help="Bollinger Bands measure volatility. Price near bands suggests potential reversal."
            )
            st.caption("⚙️ 20-period, 2σ (StdDev)")
            if st.session_state.show_indicator_details:
                if "Above" in results['bb_status']:
                    st.caption("⚠️ Potential overbought")
                elif "Below" in results['bb_status']:
                    st.caption("💡 Potential oversold")
                else:
                    st.caption("✅ Normal range")
                
                # Show band values
                bb_upper = results.get('bb_upper', 0)
                bb_middle = results.get('bb_middle', 0)
                bb_lower = results.get('bb_lower', 0)
                if bb_upper > 0 and bb_lower > 0:
                    st.caption(f"Upper: ${bb_upper:.2f} | Middle: ${bb_middle:.2f} | Lower: ${bb_lower:.2f}")
        
        if st.session_state.show_indicator_details:
            st.info("📊 **Coming in Phase 2**: Detailed indicator charts with historical trends will be displayed here.")
        
        st.divider()
        
        # Performance metrics
        st.subheader("📊 Model Performance")
        perf_col1, perf_col2, perf_col3 = st.columns(3)
        
        with perf_col1:
            st.metric("Model Accuracy", f"{results['accuracy']:.1%}")
        with perf_col2:
            st.metric("Signals Analyzed", results['signals_analyzed'])
        with perf_col3:
            st.metric("Last Updated", results['last_updated'])
        
        # Display comprehensive validation results if available
        if st.session_state.validation_results is not None:
            st.divider()
            st.subheader("🔬 Validation Results")
            
            val_results = st.session_state.validation_results
            
            # Summary metrics
            val_col1, val_col2, val_col3, val_col4 = st.columns(4)
            
            with val_col1:
                st.metric(
                    "Mean Accuracy",
                    f"{val_results['mean_accuracy']:.1%}",
                    help="Average accuracy across all validation windows"
                )
            
            with val_col2:
                std_pct = val_results['std_accuracy'] * 100
                st.metric(
                    "Variability",
                    f"±{std_pct:.1f}%",
                    help="Standard deviation of accuracy across windows"
                )
            
            with val_col3:
                st.metric(
                    "Min Accuracy",
                    f"{val_results['min_accuracy']:.1%}",
                    help="Worst performance across all windows"
                )
            
            with val_col4:
                st.metric(
                    "Max Accuracy",
                    f"{val_results['max_accuracy']:.1%}",
                    help="Best performance across all windows"
                )
            
            # Verdict badge
            mean_acc = val_results['mean_accuracy']
            consistent = val_results.get('consistent', False)
            
            if mean_acc < 0.40:
                verdict = "❌ POOR"
                color = "#FF4B4B"
                message = "Model is not learning useful patterns. Try longer timeframe or different stock."
            elif mean_acc < 0.50:
                verdict = "⚠️ WEAK"
                color = "#FFA500"
                message = "Model has weak predictive power. Consider longer timeframe."
            elif mean_acc < 0.60:
                verdict = "✓ ACCEPTABLE"
                color = "#4CAF50"
                message = "Model is learning useful patterns. This is reasonable for stock prediction."
            elif mean_acc < 0.70:
                verdict = "✅ GOOD"
                color = "#4CAF50"
                message = "Model has strong predictive power. Better than most stock prediction models."
            else:
                verdict = "🎯 EXCELLENT"
                color = "#4CAF50"
                message = "Model has exceptional predictive power. This is rare for stock prediction!"
            
            consistency_msg = ""
            if not consistent:
                consistency_msg = " However, performance varies across time periods."
            
            st.markdown(f"""
            <div style="background-color: {color}; padding: 15px; border-radius: 5px; margin: 10px 0;">
                <h3 style="color: white; margin: 0;">{verdict}</h3>
                <p style="color: white; margin: 5px 0 0 0;">{message}{consistency_msg}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Show individual window accuracies
            with st.expander("📊 Accuracy by Validation Window"):
                if val_results['window_accuracies']:
                    import pandas as pd
                    window_df = pd.DataFrame({
                        'Window': [f"Window {i+1}" for i in range(len(val_results['window_accuracies']))],
                        'Accuracy': [f"{acc:.1%}" for acc in val_results['window_accuracies']],
                        'Samples': val_results['window_sizes']
                    })
                    st.dataframe(window_df, hide_index=True, use_container_width=True)
                    
                    st.caption("Each window tests the model on a different time period to verify consistency.")
        
        # Label visualization section
        if st.session_state.show_label_viz and st.session_state.label_viz_data is not None:
            st.divider()
            st.subheader("🎯 Label Visualization")
            
            from core.utils.label_visualizer import visualize_labels, create_label_summary
            
            viz_data = st.session_state.label_viz_data
            
            # Show label summary
            summary = create_label_summary(viz_data['labels'])
            
            sum_col1, sum_col2, sum_col3, sum_col4 = st.columns(4)
            
            with sum_col1:
                st.metric("Total Labels", summary['total'])
            with sum_col2:
                st.metric("BUY", f"{summary['buy']} ({summary['buy_pct']:.1f}%)")
            with sum_col3:
                st.metric("HOLD", f"{summary['hold']} ({summary['hold_pct']:.1f}%)")
            with sum_col4:
                st.metric("SELL", f"{summary['sell']} ({summary['sell_pct']:.1f}%)")
            
            # Check for severe imbalance
            if summary['imbalance_score'] > 100:  # More than 50% deviation from 33.33% ideal
                st.warning("⚠️ **Severe class imbalance detected!** This can significantly hurt model performance. "
                          "Try adjusting the threshold or forward_days parameters.")
            elif summary['imbalance_score'] > 60:
                st.info("ℹ️ **Moderate class imbalance.** The model may have difficulty learning some signal types.")
            
            # Create and display visualization
            with st.spinner("Creating label visualization..."):
                fig = visualize_labels(viz_data['df'], viz_data['labels'], viz_data['ticker'])
                st.plotly_chart(fig, use_container_width=True)
            
            st.caption("**How to read this chart:**")
            st.caption("• **Top panel**: Stock price with Buy (🟢), Sell (🔴), and Hold (🟠) markers")
            st.caption("• **Middle panel**: Label distribution over time (stacked bars)")
            st.caption("• **Bottom panel**: Future returns that generated the labels (threshold lines at ±1%)")
    
    else:
        # Show placeholders when no analysis has been run
        st.info("👆 Enter a stock ticker and click **Analyze** to see prediction results and visualizations.")
    
    # Footer
    st.divider()
    st.caption("Built with Streamlit | Data from Yahoo Finance | © 2025")


if __name__ == "__main__":
    main()

