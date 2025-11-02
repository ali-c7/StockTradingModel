"""
Buy, Sell, Hold - AI Trading System
Streamlit UI for the complete trading system
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime

# Import trading system
from core.trading_system import TradingSystem
from data.tickers.ticker_list_data import get_ticker_options, extract_ticker
from plots.stock.stock_plot import create_price_chart


# Page config
st.set_page_config(
    page_title="AI Trading System",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)


def initialize_session_state():
    """Initialize session state"""
    if 'trading_system' not in st.session_state:
        st.session_state.trading_system = None
    if 'results' not in st.session_state:
        st.session_state.results = None
    if 'analysis_complete' not in st.session_state:
        st.session_state.analysis_complete = False
    if 'all_model_results' not in st.session_state:
        st.session_state.all_model_results = None
    if 'chart_timeframe_display' not in st.session_state:
        st.session_state.chart_timeframe_display = '60D'


def create_equity_curve_chart(portfolio_history: pd.DataFrame, ticker: str) -> go.Figure:
    """Create portfolio equity curve chart"""
    fig = go.Figure()
    
    # Portfolio value line
    fig.add_trace(go.Scatter(
        x=portfolio_history.index,
        y=portfolio_history['value'],
        mode='lines',
        name='Portfolio Value',
        line=dict(color='#2E86AB', width=2),
        fill='tozeroy',
        fillcolor='rgba(46, 134, 171, 0.1)'
    ))
    
    # Initial capital line
    initial = portfolio_history['value'].iloc[0]
    fig.add_hline(
        y=initial,
        line_dash='dash',
        line_color='gray',
        annotation_text=f'Initial: ${initial:,.0f}'
    )
    
    fig.update_layout(
        title=f'{ticker} - Portfolio Equity Curve',
        xaxis_title='Date',
        yaxis_title='Portfolio Value ($)',
        hovermode='x unified',
        height=400
    )
    
    return fig


def create_predictions_chart(df: pd.DataFrame, signals: pd.Series, ticker: str, test_start_date=None) -> go.Figure:
    """Create price chart showing ALL model predictions (not just executed trades)"""
    fig = go.Figure()
    
    # Price line
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['Close'],
        mode='lines',
        name='Price',
        line=dict(color='gray', width=1.5)
    ))
    
    # Align signals with price data
    signals_aligned = signals.reindex(df.index)
    
    # Separate predictions by type and train/test
    if test_start_date:
        test_start_dt = pd.Timestamp(test_start_date)
        
        # Training predictions (small, faded)
        train_mask = df.index < test_start_dt
        train_buys = df[train_mask & (signals_aligned == 1)]
        train_sells = df[train_mask & (signals_aligned == -1)]
        
        # Test predictions (LARGE, bold)
        test_mask = df.index >= test_start_dt
        test_buys = df[test_mask & (signals_aligned == 1)]
        test_sells = df[test_mask & (signals_aligned == -1)]
        
        # Training predictions
        if not train_buys.empty:
            fig.add_trace(go.Scatter(
                x=train_buys.index,
                y=train_buys['Close'],
                mode='markers',
                name=f'Predict BUY - Train ({len(train_buys)})',
                marker=dict(symbol='triangle-up', size=6, color='lightgreen', opacity=0.2)
            ))
        
        if not train_sells.empty:
            fig.add_trace(go.Scatter(
                x=train_sells.index,
                y=train_sells['Close'],
                mode='markers',
                name=f'Predict SELL - Train ({len(train_sells)})',
                marker=dict(symbol='triangle-down', size=6, color='lightcoral', opacity=0.2)
            ))
        
        # Test predictions (what really matters!)
        if not test_buys.empty:
            fig.add_trace(go.Scatter(
                x=test_buys.index,
                y=test_buys['Close'],
                mode='markers',
                name=f'🔮 Predict BUY - TEST ({len(test_buys)})',
                marker=dict(symbol='triangle-up', size=10, color='lime', opacity=0.7)
            ))
        
        if not test_sells.empty:
            fig.add_trace(go.Scatter(
                x=test_sells.index,
                y=test_sells['Close'],
                mode='markers',
                name=f'🔮 Predict SELL - TEST ({len(test_sells)})',
                marker=dict(symbol='triangle-down', size=10, color='red', opacity=0.7)
            ))
    else:
        # No train/test split - show all predictions
        buy_predictions = df[signals_aligned == 1]
        sell_predictions = df[signals_aligned == -1]
        
        if not buy_predictions.empty:
            fig.add_trace(go.Scatter(
                x=buy_predictions.index,
                y=buy_predictions['Close'],
                mode='markers',
                name=f'Predict BUY ({len(buy_predictions)})',
                marker=dict(symbol='triangle-up', size=8, color='green', opacity=0.6)
            ))
        
        if not sell_predictions.empty:
            fig.add_trace(go.Scatter(
                x=sell_predictions.index,
                y=sell_predictions['Close'],
                mode='markers',
                name=f'Predict SELL ({len(sell_predictions)})',
                marker=dict(symbol='triangle-down', size=8, color='red', opacity=0.6)
            ))
    
    fig.update_layout(
        title=f'{ticker} - ALL Model Predictions (Raw Output)',
        xaxis_title='Date',
        yaxis_title='Price ($)',
        hovermode='x unified',
        height=400
    )
    
    return fig


def create_trade_markers_chart(df: pd.DataFrame, trades: list, ticker: str, test_start_date=None) -> go.Figure:
    """Create price chart with trade markers, highlighting test data trades"""
    fig = go.Figure()
    
    # Price line
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['Close'],
        mode='lines',
        name='Price',
        line=dict(color='gray', width=1)
    ))
    
    # If test_start_date provided, separate train vs test trades
    if test_start_date:
        test_start_dt = pd.Timestamp(test_start_date)
        
        # Separate trades by train/test
        train_buys = [t for t in trades if t['action'] == 'BUY' and pd.Timestamp(t['date']) < test_start_dt]
        test_buys = [t for t in trades if t['action'] == 'BUY' and pd.Timestamp(t['date']) >= test_start_dt]
        
        train_sells = [t for t in trades if t['action'] in ['SELL', 'TAKE_PROFIT'] and pd.Timestamp(t['date']) < test_start_dt]
        test_sells = [t for t in trades if t['action'] in ['SELL', 'TAKE_PROFIT'] and pd.Timestamp(t['date']) >= test_start_dt]
        
        train_stops = [t for t in trades if t['action'] == 'STOP_LOSS' and pd.Timestamp(t['date']) < test_start_dt]
        test_stops = [t for t in trades if t['action'] == 'STOP_LOSS' and pd.Timestamp(t['date']) >= test_start_dt]
        
        # Training data trades (smaller, transparent)
        if train_buys:
            fig.add_trace(go.Scatter(
                x=[t['date'] for t in train_buys],
                y=[t['price'] for t in train_buys],
                mode='markers',
                name=f'Buy - Train ({len(train_buys)})',
                marker=dict(symbol='triangle-up', size=8, color='green', opacity=0.3)
            ))
        
        if train_sells:
            fig.add_trace(go.Scatter(
                x=[t['date'] for t in train_sells],
                y=[t['price'] for t in train_sells],
                mode='markers',
                name=f'Sell - Train ({len(train_sells)})',
                marker=dict(symbol='triangle-down', size=8, color='red', opacity=0.3)
            ))
        
        if train_stops:
            fig.add_trace(go.Scatter(
                x=[t['date'] for t in train_stops],
                y=[t['price'] for t in train_stops],
                mode='markers',
                name=f'Stop - Train ({len(train_stops)})',
                marker=dict(symbol='x', size=7, color='orange', opacity=0.3)
            ))
        
        # Test data trades (LARGER, BOLD - these are what matter!)
        if test_buys:
            fig.add_trace(go.Scatter(
                x=[t['date'] for t in test_buys],
                y=[t['price'] for t in test_buys],
                mode='markers',
                name=f'🎯 Buy - TEST ({len(test_buys)})',
                marker=dict(symbol='triangle-up', size=15, color='lime', line=dict(width=2, color='darkgreen'))
            ))
        
        if test_sells:
            fig.add_trace(go.Scatter(
                x=[t['date'] for t in test_sells],
                y=[t['price'] for t in test_sells],
                mode='markers',
                name=f'🎯 Sell - TEST ({len(test_sells)})',
                marker=dict(symbol='triangle-down', size=15, color='red', line=dict(width=2, color='darkred'))
            ))
        
        if test_stops:
            fig.add_trace(go.Scatter(
                x=[t['date'] for t in test_stops],
                y=[t['price'] for t in test_stops],
                mode='markers',
                name=f'🎯 Stop - TEST ({len(test_stops)})',
                marker=dict(symbol='x', size=12, color='orange', line=dict(width=2))
            ))
    
    else:
        # No test split info - show all trades normally
        buys = [t for t in trades if t['action'] == 'BUY']
        sells = [t for t in trades if t['action'] in ['SELL', 'TAKE_PROFIT']]
        stops = [t for t in trades if t['action'] == 'STOP_LOSS']
        
        if buys:
            buy_dates = [t['date'] for t in buys]
            buy_prices = [t['price'] for t in buys]
            fig.add_trace(go.Scatter(
                x=buy_dates,
                y=buy_prices,
                mode='markers',
                name=f'Buy ({len(buys)})',
                marker=dict(symbol='triangle-up', size=12, color='green')
            ))
        
        if sells:
            sell_dates = [t['date'] for t in sells]
            sell_prices = [t['price'] for t in sells]
            fig.add_trace(go.Scatter(
                x=sell_dates,
                y=sell_prices,
                mode='markers',
                name=f'Sell ({len(sells)})',
                marker=dict(symbol='triangle-down', size=12, color='red')
            ))
        
        if stops:
            stop_dates = [t['date'] for t in stops]
            stop_prices = [t['price'] for t in stops]
            fig.add_trace(go.Scatter(
                x=stop_dates,
                y=stop_prices,
                mode='markers',
                name=f'Stop Loss ({len(stops)})',
                marker=dict(symbol='x', size=10, color='orange')
            ))
    
    fig.update_layout(
        title=f'{ticker} - Executed Trades Only',
        xaxis_title='Date',
        yaxis_title='Price ($)',
        hovermode='x unified',
        height=400
    )
    
    return fig


def main():
    initialize_session_state()
    
    # Header
    st.title("📈 AI Trading System")
    st.markdown("*Research-driven algorithmic trading with ML, Kelly Criterion, and risk management*")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Ticker selection
        st.subheader("1️⃣ Select Stock")
        ticker_options = get_ticker_options()
        
        if ticker_options:
            ticker_options.insert(0, "🔍 Type custom ticker...")
            selected_option = st.selectbox(
                "Stock Ticker",
                options=ticker_options,
                help="Select from S&P 500 or enter custom ticker"
            )
            
            if selected_option == "🔍 Type custom ticker...":
                ticker = st.text_input(
                    "Enter ticker",
                    value="AAPL",
                    max_chars=15,
                    help="e.g., AAPL, GOOGL, BTC-USD"
                ).upper()
            else:
                ticker = extract_ticker(selected_option)
        else:
            ticker = st.text_input(
                "Stock Ticker",
                value="AAPL",
                max_chars=15
            ).upper()
        
        # Timeframe
        st.subheader("2️⃣ Select Timeframe")
        timeframe = st.selectbox(
            "Analysis Timeframe",
            options=['6M', '1Y', '2Y', '5Y'],
            index=2,  # Default to 2Y (better accuracy)
            help="Historical data period for training. 2Y+ recommended for reliable results."
        )
        
        # Warning for short timeframes
        if timeframe in ['6M', '1Y']:
            st.info("💡 **Tip:** 2Y or 5Y timeframe gives more reliable results (more training data)")
        
        # Model selection
        st.subheader("3️⃣ Select Model")
        run_all_models = st.checkbox(
            "🔄 Compare All 3 Models",
            value=True,
            help="Run XGBoost, Random Forest, and LightGBM simultaneously"
        )
        
        if not run_all_models:
            model_type = st.selectbox(
                "ML Model",
                options=['xgboost', 'random_forest', 'lightgbm'],
                index=0,
                help="XGBoost usually performs best"
            )
        else:
            model_type = 'xgboost'  # Will be overridden
        
        # Advanced settings
        with st.expander("🔧 Advanced Settings"):
            forward_days = st.slider(
                "Forward Days (Label)",
                min_value=3,
                max_value=10,
                value=5,
                help="Days ahead for prediction"
            )
            
            threshold = st.slider(
                "Signal Threshold",
                min_value=0.01,
                max_value=0.05,
                value=0.02,
                step=0.005,
                format="%.2f%%",
                help="Return threshold for buy/sell signals"
            )
            
            train_split = st.slider(
                "Train/Test Split",
                min_value=0.60,
                max_value=0.90,
                value=0.80,
                step=0.05,
                format="%.0f%%",
                help="Percentage of data used for training (e.g., 80% = train on first 80%, test on last 20%)"
            )
            
            # Show what this means
            st.caption(f"📊 Split: {train_split:.0%} training, {1-train_split:.0%} testing")
            
            initial_capital = st.number_input(
                "Initial Capital ($)",
                min_value=1000,
                max_value=100000,
                value=10000,
                step=1000
            )
            
            position_sizing = st.selectbox(
                "Position Sizing",
                options=['kelly', 'fixed', 'equal'],
                index=0,
                help="Kelly Criterion usually best"
            )
        
        st.divider()
        
        # Analyze button
        if st.button("🚀 Run Analysis", type="primary", use_container_width=True):
            st.session_state.analysis_complete = False
            st.session_state.all_model_results = None
            
            try:
                if run_all_models:
                    # Run all 3 models
                    with st.spinner(f"🤖 Analyzing {ticker} with 3 models... This may take 2-4 minutes..."):
                        all_results = {}
                        
                        for idx, model_name in enumerate(['xgboost', 'random_forest', 'lightgbm']):
                            st.text(f"Training {model_name.upper()}... ({idx+1}/3)")
                            
                            system = TradingSystem(
                                ticker=ticker,
                                timeframe=timeframe,
                                model_type=model_name,
                                forward_days=forward_days,
                                threshold=threshold,
                                train_split=train_split,
                                initial_capital=initial_capital,
                                position_sizing=position_sizing
                            )
                            
                            results = system.run_complete_pipeline(verbose=False)
                            all_results[model_name] = {
                                'system': system,
                                'results': results
                            }
                        
                        # Store all results
                        st.session_state.all_model_results = all_results
                        # Use XGBoost as default display
                        st.session_state.trading_system = all_results['xgboost']['system']
                        st.session_state.results = all_results['xgboost']['results']
                        st.session_state.analysis_complete = True
                        
                        st.success("✅ All 3 models analyzed!")
                        st.rerun()
                else:
                    # Single model
                    with st.spinner(f"🤖 Analyzing {ticker}... This may take 1-2 minutes..."):
                        system = TradingSystem(
                            ticker=ticker,
                            timeframe=timeframe,
                            model_type=model_type,
                            forward_days=forward_days,
                            threshold=threshold,
                            train_split=train_split,
                            initial_capital=initial_capital,
                            position_sizing=position_sizing
                        )
                        
                        results = system.run_complete_pipeline(verbose=False)
                        
                        st.session_state.trading_system = system
                        st.session_state.results = results
                        st.session_state.all_model_results = None
                        st.session_state.analysis_complete = True
                        
                        st.success("✅ Analysis complete!")
                        st.rerun()
                    
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                st.exception(e)
        
        # Clear button
        if st.button("🔄 Clear Results", use_container_width=True):
            st.session_state.trading_system = None
            st.session_state.results = None
            st.session_state.analysis_complete = False
            st.rerun()
        
        st.divider()
        
        # Info
        st.caption("**Built with:**")
        st.caption("• 50+ technical indicators")
        st.caption("• XGBoost/RF/LightGBM")
        st.caption("• Kelly Criterion sizing")
        st.caption("• Stop-loss & take-profit")
    
    # Main content
    if not st.session_state.analysis_complete:
        # Show instructions
        st.info("👈 Configure settings in the sidebar and click **Run Analysis** to start")
        
        st.markdown("""
        ### 🎯 What This System Does
        
        1. **Fetches Data**: Downloads historical stock data from Yahoo Finance
        2. **Engineers Features**: Calculates 50+ technical indicators (RSI, MACD, ADX, etc.)
        3. **Trains Model**: Uses Random Forest, XGBoost, or LightGBM
        4. **Generates Signals**: Predicts **BUY, HOLD, or SELL** (3-class classification)
        5. **Backtests Strategy**: Simulates trading with Kelly Criterion position sizing
        6. **Analyzes Performance**: Calculates Sharpe ratio, returns, win rate, etc.
        
        ### 📊 Expected Performance (3-Class Classification)
        - **Model Accuracy**: 40-60% (3 classes harder than 2!)
        - **Sharpe Ratio**: 0.5-2.0 (> 1.5 is excellent)
        - **Win Rate**: 50-70% (HOLD reduces bad trades)
        
        ### 💡 Why 3-Class (BUY, HOLD, SELL)?
        - **More Realistic**: Not every day requires action
        - **Reduces Overtrading**: HOLD = "don't trade" (saves on fees)
        - **Better Risk Management**: Only trade when there's a clear signal
        
        ### ⏱️ Processing Time
        - First run: ~1-2 minutes (downloading data + training)
        - Subsequent runs: ~30 seconds (cached data)
        """)
    
    else:
        # Display results
        system = st.session_state.trading_system
        results = st.session_state.results
        
        model_metrics = results['model_metrics']
        backtest_results = results['backtest_results']
        metrics = backtest_results['metrics']
        
        # Header
        st.header(f"📊 Results: {system.ticker} ({system.timeframe})")
        
        # Latest signal (prominent)
        latest_signal = system.get_latest_signal()
        signal_name = latest_signal['signal_name']
        confidence = latest_signal['confidence']
        
        signal_colors = {
            'BUY': '#28A745',
            'HOLD': '#FFC107',
            'SELL': '#DC3545'
        }
        
        col1, col2, col3 = st.columns([2, 2, 3])
        
        with col1:
            st.markdown(f"""
            <div style="background-color: {signal_colors[signal_name]}; padding: 20px; border-radius: 10px; text-align: center;">
                <h2 style="color: white; margin: 0;">SIGNAL: {signal_name}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.metric("Confidence", f"{confidence:.1%}")
        
        with col3:
            st.caption("**Signal Probabilities:**")
            for sig, prob in latest_signal['probabilities'].items():
                st.caption(f"{sig}: {prob:.1%}")
        
        st.divider()
        
        # Key metrics in 4 columns
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            # Show test samples in help text
            test_samples_info = f"Based on {model_metrics.get('test_samples', 'N/A')} test samples"
            st.metric(
                "Model Accuracy",
                f"{model_metrics['test_accuracy']:.1%}",
                help=f"ML model classification accuracy. {test_samples_info}"
            )
        
        with col2:
            st.metric(
                "Sharpe Ratio",
                f"{metrics['sharpe_ratio']:.2f}",
                help="Risk-adjusted return (> 1.5 is excellent)"
            )
        
        with col3:
            st.metric(
                "Total Return",
                f"{metrics['total_return']:+.1%}",
                delta=f"{metrics['alpha']:+.1%} vs Buy&Hold",
                help="Portfolio return vs buy-and-hold"
            )
        
        with col4:
            st.metric(
                "Win Rate",
                f"{metrics['win_rate']:.1%}",
                help="Percentage of profitable trades"
            )
        
        # Additional metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Max Drawdown", f"{metrics['max_drawdown']:.1%}")
        
        with col2:
            st.metric("Total Trades", metrics['total_trades'])
        
        with col3:
            st.metric("Final Value", f"${metrics['final_value']:,.0f}")
        
        with col4:
            buy_hold = metrics['buy_hold_return']
            st.metric("Buy & Hold", f"{buy_hold:+.1%}")
        
        st.divider()
        
        # Warning for suspiciously high accuracy with small test set
        test_samples = model_metrics.get('test_samples', 0)
        if model_metrics['test_accuracy'] >= 0.95 and test_samples < 50:
            st.warning(
                f"⚠️ **Small Test Set Warning**: Only {test_samples} test samples. "
                f"High accuracy ({model_metrics['test_accuracy']:.1%}) may be due to chance with small data. "
                f"Consider using a longer timeframe (2Y or 5Y) for more reliable results."
            )
        
        # Model Comparison (if all 3 models were run)
        if st.session_state.all_model_results is not None:
            st.subheader("🔄 Model Comparison")
            
            comparison_data = []
            for model_name, model_data in st.session_state.all_model_results.items():
                m_results = model_data['results']
                m_metrics = m_results['model_metrics']
                b_metrics = m_results['backtest_results']['metrics']
                
                signal = model_data['system'].get_latest_signal()
                
                comparison_data.append({
                    'Model': model_name.upper(),
                    'Signal': signal['signal_name'],
                    'Confidence': f"{signal['confidence']:.1%}",
                    'Accuracy': f"{m_metrics['test_accuracy']:.1%}",
                    'Sharpe': f"{b_metrics['sharpe_ratio']:.2f}",
                    'Return': f"{b_metrics['total_return']:+.1%}",
                    'Win Rate': f"{b_metrics['win_rate']:.1%}",
                    'Max DD': f"{b_metrics['max_drawdown']:.1%}",
                    'Alpha': f"{b_metrics['alpha']:+.1%}"
                })
            
            comparison_df = pd.DataFrame(comparison_data)
            
            # Style the dataframe
            st.dataframe(
                comparison_df,
                hide_index=True,
                use_container_width=True,
                height=150
            )
            
            # Agreement indicator
            signals = [d['Signal'] for d in comparison_data]
            if len(set(signals)) == 1:
                st.success(f"✅ **All models agree:** {signals[0]}")
            else:
                st.warning(f"⚠️ **Models disagree:** {', '.join(signals)}")
            
            st.caption("💡 **Tip:** When models agree, the signal is more reliable!")
            
            st.divider()
        
        # Stock Data & Indicators Section
        with st.expander("📊 Stock Data & Technical Indicators", expanded=True):
            st.subheader(f"{system.ticker} - Current Price & Indicators")
            
            # Get latest data
            latest_data = system.feature_data.iloc[-1]
            latest_date = system.feature_data.index[-1]
            
            # Price info
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Close Price", f"${latest_data['Close']:.2f}")
            with col2:
                st.metric("High", f"${latest_data['High']:.2f}")
            with col3:
                st.metric("Low", f"${latest_data['Low']:.2f}")
            with col4:
                st.metric("Volume", f"{latest_data['Volume']:,.0f}")
            
            st.caption(f"Data as of: {latest_date.strftime('%Y-%m-%d')}")
            
            st.divider()
            
            # Technical Indicators
            st.markdown("**📈 Technical Indicators:**")
            
            ind_col1, ind_col2, ind_col3 = st.columns(3)
            
            with ind_col1:
                st.markdown("**Momentum:**")
                rsi = latest_data.get('rsi_14', 0)
                rsi_status = "Overbought" if rsi > 70 else "Oversold" if rsi < 30 else "Neutral"
                st.write(f"• RSI (14): {rsi:.1f} - {rsi_status}")
                st.write(f"• Stochastic %K: {latest_data.get('stoch_k', 0):.1f}")
                st.write(f"• Williams %R: {latest_data.get('williams_r', 0):.1f}")
                st.write(f"• CCI: {latest_data.get('cci', 0):.1f}")
            
            with ind_col2:
                st.markdown("**Trend:**")
                st.write(f"• EMA 50: ${latest_data.get('ema_50', 0):.2f}")
                st.write(f"• EMA 200: ${latest_data.get('ema_200', 0):.2f}")
                st.write(f"• ADX: {latest_data.get('adx', 0):.1f}")
                
                price_vs_ema50 = latest_data.get('price_vs_ema50', 0) * 100
                trend = "Above" if price_vs_ema50 > 0 else "Below"
                st.write(f"• Price vs EMA50: {trend} ({price_vs_ema50:+.1f}%)")
            
            with ind_col3:
                st.markdown("**Volatility & Volume:**")
                st.write(f"• ATR: ${latest_data.get('atr', 0):.2f}")
                st.write(f"• BB Upper: ${latest_data.get('bb_upper', 0):.2f}")
                st.write(f"• BB Lower: ${latest_data.get('bb_lower', 0):.2f}")
                
                vol_ratio = latest_data.get('volume_ratio', 0)
                vol_status = "High" if vol_ratio > 1.5 else "Low" if vol_ratio < 0.5 else "Normal"
                st.write(f"• Volume Ratio: {vol_ratio:.2f}x - {vol_status}")
            
            # Price chart with indicators
            st.divider()
            
            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown("**📉 Price Chart:**")
            with col2:
                chart_options = {
                    '30D': 30,
                    '60D': 60,
                    '90D': 90,
                    '6M': 120,
                    '1Y': 252,
                    'All': len(system.feature_data)
                }
                chart_display = st.selectbox(
                    "Timeframe",
                    options=list(chart_options.keys()),
                    index=1,  # Default to 60D
                    key='chart_timeframe_selector'
                )
                days_to_show = chart_options[chart_display]
            
            recent_data = system.feature_data.tail(days_to_show)
            
            fig_price = go.Figure()
            
            # Candlestick
            fig_price.add_trace(go.Candlestick(
                x=recent_data.index,
                open=recent_data['Open'],
                high=recent_data['High'],
                low=recent_data['Low'],
                close=recent_data['Close'],
                name='Price'
            ))
            
            # EMA 50 (always available)
            if 'ema_50' in recent_data.columns:
                fig_price.add_trace(go.Scatter(
                    x=recent_data.index,
                    y=recent_data['ema_50'],
                    mode='lines',
                    name='EMA 50',
                    line=dict(color='blue', width=1)
                ))
            
            # EMA 200 (only if available - may be dropped for short datasets)
            if 'ema_200' in recent_data.columns:
                fig_price.add_trace(go.Scatter(
                    x=recent_data.index,
                    y=recent_data['ema_200'],
                    mode='lines',
                    name='EMA 200',
                    line=dict(color='red', width=1)
                ))
            
            fig_price.update_layout(
                xaxis_rangeslider_visible=False,
                height=400,
                hovermode='x unified'
            )
            
            st.plotly_chart(fig_price, use_container_width=True)
        
        st.divider()
        
        # Simple/Advanced view toggle
        view_mode = st.radio(
            "View Mode:",
            options=["Simple", "Advanced"],
            index=0,
            horizontal=True,
            help="Simple: key metrics only. Advanced: detailed analysis"
        )
        
        st.divider()
        
        # Charts in tabs
        if view_mode == "Simple":
            tab1, tab2, tab_pred = st.tabs(["📈 Equity Curve", "🎯 Executed Trades", "🔮 Model Predictions"])
        else:
            tab1, tab2, tab_pred, tab3, tab4 = st.tabs(["📈 Equity Curve", "🎯 Executed Trades", "🔮 Model Predictions", "🏆 Feature Importance", "📋 Trade Log"])
        
        with tab1:
            st.subheader("Portfolio Equity Curve")
            portfolio_df = backtest_results['portfolio_history']
            fig_equity = create_equity_curve_chart(portfolio_df, system.ticker)
            st.plotly_chart(fig_equity, use_container_width=True)
            
            st.caption(f"Started with ${metrics['final_value'] - metrics['total_pnl']:,.0f}, ended with ${metrics['final_value']:,.0f}")
        
        with tab2:
            st.subheader("Executed Trades Only")
            st.caption("⚠️ **Important:** These are ACTUAL TRADES executed by the backtest, not all model predictions!")
            
            # Train/Test split info - USE ACTUAL MODEL SPLIT, NOT FEATURE_DATA
            # (feature_data includes NaN labels that were dropped during training)
            train_end_date = system.X_train.index[-1]
            test_start_date = system.X_test.index[0]
            
            # Show split info
            st.info(f"📅 **Actual train/test split:** Training ends {train_end_date.strftime('%Y-%m-%d')}, Testing starts {test_start_date.strftime('%Y-%m-%d')}")
            
            col1, col2, col3 = st.columns([1, 1, 2])
            with col1:
                st.metric("Training Period", f"{len(system.X_train)} days")
            with col2:
                st.metric("Testing Period", f"{len(system.X_test)} days") 
            with col3:
                # Calculate how many test days have actually occurred
                today = pd.Timestamp.now(tz=test_start_date.tz)
                test_days_elapsed = min(len(system.X_test), max(0, (today - test_start_date).days))
                st.metric("Test Days Elapsed", f"{test_days_elapsed}/{len(system.X_test)} days")
            
            st.info(f"🎯 **Test split:** {test_start_date.strftime('%Y-%m-%d')} to {system.X_test.index[-1].strftime('%Y-%m-%d')} (model has NOT seen this data during training!)")
            
            trades = backtest_results['trades']
            
            # Count test trades
            test_start_dt_ts = pd.Timestamp(test_start_date)
            test_trades = [t for t in trades if pd.Timestamp(t['date']) >= test_start_dt_ts]
            st.info(f"📊 **Executed {len(test_trades)} trades in test period** (BUY only works if no position, SELL only works if position open)")
            
            # Create enhanced chart with train/test split visualization
            fig_trades = create_trade_markers_chart(system.feature_data, trades, system.ticker, test_start_date=test_start_date)
            
            # Convert pandas Timestamps to Python datetime for Plotly compatibility
            test_start_dt = test_start_date.to_pydatetime()
            train_end_dt = train_end_date.to_pydatetime()
            data_start_dt = system.feature_data.index[0].to_pydatetime()
            data_end_dt = system.feature_data.index[-1].to_pydatetime()
            
            # Add shaded regions (without annotations to avoid datetime arithmetic issues)
            fig_trades.add_vrect(
                x0=data_start_dt,
                x1=train_end_dt,
                fillcolor="blue",
                opacity=0.05,
                layer="below",
                line_width=0
            )
            
            fig_trades.add_vrect(
                x0=test_start_dt,
                x1=data_end_dt,
                fillcolor="green",
                opacity=0.05,
                layer="below",
                line_width=0
            )
            
            # Add vertical line to show train/test split (without annotation)
            fig_trades.add_vline(
                x=test_start_dt,
                line_dash="dash",
                line_color="orange",
                line_width=2
            )
            
            # Add text annotations manually to avoid datetime arithmetic issues
            fig_trades.add_annotation(
                x=test_start_dt,
                y=1,
                yref="paper",
                text="← Test Data Starts",
                showarrow=False,
                font=dict(size=12, color="orange"),
                xshift=5,
                yshift=-10
            )
            
            st.plotly_chart(fig_trades, use_container_width=True)
            
            st.caption(f"**Legend:** Small faded markers = Training data | 🎯 Large bold markers = TEST data (what matters!) | 🔶 Orange line = Train/Test split")
        
        with tab_pred:
            st.subheader("ALL Model Predictions (Raw Output)")
            st.caption("🔮 **This shows EVERY prediction the model makes**, regardless of whether trades were executed")
            
            # Train/Test split info - USE ACTUAL MODEL SPLIT
            train_end_date = system.X_train.index[-1]
            test_start_date = system.X_test.index[0]
            
            # Get signals from results
            signals = results['signals']
            
            # Count test predictions
            test_mask = signals.index >= test_start_date
            test_signals = signals[test_mask]
            test_buys = (test_signals == 1).sum()
            test_sells = (test_signals == -1).sum()
            test_holds = (test_signals == 0).sum()
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Test BUY Predictions", test_buys)
            with col2:
                st.metric("Test SELL Predictions", test_sells)
            with col3:
                st.metric("Test HOLD Predictions", test_holds)
            with col4:
                st.metric("Total Test Days", len(test_signals))
            
            st.warning(f"⚠️ **Why so few executed trades?** Even though the model predicts **{test_buys + test_sells} actionable signals** in the test period, only trades that meet position requirements (have/don't have open position) are executed!")
            
            # Create predictions chart
            fig_predictions = create_predictions_chart(system.feature_data, signals, system.ticker, test_start_date=test_start_date)
            
            # Add shaded regions and train/test split line
            test_start_dt = test_start_date.to_pydatetime()
            train_end_dt = train_end_date.to_pydatetime()
            data_start_dt = system.feature_data.index[0].to_pydatetime()
            data_end_dt = system.feature_data.index[-1].to_pydatetime()
            
            fig_predictions.add_vrect(
                x0=data_start_dt,
                x1=train_end_dt,
                fillcolor="blue",
                opacity=0.05,
                layer="below",
                line_width=0
            )
            
            fig_predictions.add_vrect(
                x0=test_start_dt,
                x1=data_end_dt,
                fillcolor="green",
                opacity=0.05,
                layer="below",
                line_width=0
            )
            
            fig_predictions.add_vline(
                x=test_start_dt,
                line_dash="dash",
                line_color="orange",
                line_width=2
            )
            
            fig_predictions.add_annotation(
                x=test_start_dt,
                y=1,
                yref="paper",
                text="← Test Data Starts",
                showarrow=False,
                font=dict(size=12, color="orange"),
                xshift=5,
                yshift=-10
            )
            
            st.plotly_chart(fig_predictions, use_container_width=True)
            
            st.caption("**Legend:** Small faded = Training predictions | 🔮 Large = TEST predictions (raw model output) | Note: HOLD signals (0) are not shown")
        
        if view_mode == "Advanced":
            with tab3:
                st.subheader("Top 20 Most Important Features")
                importance_df = system.model.get_feature_importance(top_n=20)
                
                fig_importance = go.Figure(go.Bar(
                    x=importance_df['importance'],
                    y=importance_df['feature'],
                    orientation='h',
                    marker_color='#2E86AB'
                ))
                
                fig_importance.update_layout(
                    xaxis_title='Importance',
                    yaxis_title='Feature',
                    height=600,
                    yaxis={'categoryorder': 'total ascending'}
                )
                
                st.plotly_chart(fig_importance, use_container_width=True)
            
            with tab4:
                st.subheader("Trade History")
                trades_df = pd.DataFrame(trades)
                
                if len(trades_df) > 0:
                    trades_df['date'] = pd.to_datetime(trades_df['date'])
                    
                    # Add TRAIN/TEST column
                    test_start_dt = system.X_test.index[0]
                    trades_df['period'] = trades_df['date'].apply(
                        lambda x: '🎯 TEST' if pd.Timestamp(x) >= test_start_dt else 'TRAIN'
                    )
                    
                    trades_df = trades_df.sort_values('date', ascending=False)
                    
                    # Style function for color-coding
                    def highlight_trades(row):
                        if row['action'] == 'BUY':
                            return ['background-color: #d4edda'] * len(row)  # Light green
                        elif row['action'] == 'SELL':
                            return ['background-color: #f8d7da'] * len(row)  # Light red
                        elif row['action'] == 'TAKE_PROFIT':
                            return ['background-color: #d1ecf1'] * len(row)  # Light blue
                        elif row['action'] == 'STOP_LOSS':
                            return ['background-color: #fff3cd'] * len(row)  # Light yellow/orange
                        else:
                            return [''] * len(row)
                    
                    # Apply styling
                    styled_df = trades_df[['period', 'date', 'action', 'price', 'shares', 'value', 'reason']].style.apply(
                        highlight_trades, axis=1
                    )
                    
                    st.dataframe(
                        styled_df,
                        hide_index=True,
                        use_container_width=True
                    )
                    
                    # Show breakdown and legend
                    train_count = (trades_df['period'] == 'TRAIN').sum()
                    test_count = (trades_df['period'] == '🎯 TEST').sum()
                    st.caption(f"**Total trades: {len(trades_df)}** | Training: {train_count} | 🎯 Testing: {test_count}")
                    st.caption("🟢 Green = BUY | 🔴 Red = SELL | 🔵 Blue = TAKE_PROFIT | 🟡 Yellow = STOP_LOSS")
                else:
                    st.info("No trades executed in backtest")
        
        st.divider()
        
        # Performance summary (Advanced mode only)
        if view_mode == "Advanced":
            with st.expander("📊 Detailed Performance Metrics", expanded=False):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Model Metrics:**")
                    st.write(f"• Training Accuracy: {model_metrics['train_accuracy']:.1%}")
                    st.write(f"• Test Accuracy: {model_metrics['test_accuracy']:.1%}")
                    st.write(f"• Precision: {model_metrics['precision']:.1%}")
                    st.write(f"• Recall: {model_metrics['recall']:.1%}")
                    st.write(f"• F1 Score: {model_metrics['f1']:.1%}")
                
                with col2:
                    st.markdown("**Trading Metrics:**")
                    st.write(f"• Initial Capital: ${initial_capital:,.0f}")
                    st.write(f"• Final Value: ${metrics['final_value']:,.0f}")
                    st.write(f"• Total Return: {metrics['total_return']:+.2%}")
                    st.write(f"• Buy & Hold Return: {metrics['buy_hold_return']:+.2%}")
                    st.write(f"• Alpha: {metrics['alpha']:+.2%}")
                    st.write(f"• Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
                    st.write(f"• Max Drawdown: {metrics['max_drawdown']:.2%}")
                    st.write(f"• Win Rate: {metrics['win_rate']:.1%}")
                    st.write(f"• Total Trades: {metrics['total_trades']}")


if __name__ == "__main__":
    main()

