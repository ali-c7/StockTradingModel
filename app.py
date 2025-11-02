"""
Simplified Trading Signal Predictor
Focus: Train model, predict signals, visualize on test data
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime

# Core imports
from data.stock.stock_data import fetch_stock_data
from core.features.technical_features import engineer_all_features, prepare_feature_matrix, get_feature_list
from core.labels.label_generator import generate_labels, print_label_distribution
from core.models.baseline_models import BaselineModelTrainer


st.set_page_config(
    page_title="Simple Trading Signals",
    page_icon="📊",
    layout="wide"
)


def simulate_portfolio(feature_data: pd.DataFrame, predictions: pd.Series, 
                      initial_capital: float, position_size: float) -> dict:
    """
    Simulate portfolio performance on test data
    
    Args:
        feature_data: OHLCV data for test period
        predictions: Model predictions (1=BUY, 0=HOLD, -1=SELL)
        initial_capital: Starting cash
        position_size: % of capital to use per trade (0.0-1.0)
    
    Returns:
        Dictionary with portfolio metrics, trades, and history
    """
    cash = initial_capital
    shares = 0
    position = None  # None, 'LONG', or 'SHORT' (we'll only do LONG for simplicity)
    
    trades = []
    portfolio_history = []
    
    prices = feature_data['Close']
    
    for date in feature_data.index:
        price = prices[date]
        signal = predictions.get(date, 0)  # 0 = HOLD if missing
        
        # Current portfolio value
        portfolio_value = cash + (shares * price if shares > 0 else 0)
        
        # Execute trades based on signal
        if signal == 1 and position is None:  # BUY signal, not currently in position
            # Enter LONG position
            invest_amount = cash * position_size
            shares_to_buy = invest_amount / price
            
            if cash >= invest_amount:
                cash -= invest_amount
                shares += shares_to_buy
                position = 'LONG'
                
                trades.append({
                    'date': date,
                    'action': 'BUY',
                    'price': price,
                    'shares': shares_to_buy,
                    'value': invest_amount,
                    'cash_after': cash,
                    'shares_after': shares
                })
        
        elif signal == -1 and position == 'LONG':  # SELL signal, currently in position
            # Exit LONG position
            sell_value = shares * price
            cash += sell_value
            
            trades.append({
                'date': date,
                'action': 'SELL',
                'price': price,
                'shares': shares,
                'value': sell_value,
                'cash_after': cash,
                'shares_after': 0
            })
            
            shares = 0
            position = None
        
        # Track portfolio value
        portfolio_value = cash + (shares * price if shares > 0 else 0)
        portfolio_history.append({
            'date': date,
            'portfolio_value': portfolio_value,
            'cash': cash,
            'shares': shares,
            'price': price
        })
    
    # Final portfolio value
    final_price = prices.iloc[-1]
    final_value = cash + (shares * final_price)
    
    # Calculate Buy & Hold for comparison
    start_price = prices.iloc[0]
    end_price = prices.iloc[-1]
    buy_hold_shares = initial_capital / start_price
    buy_hold_value = buy_hold_shares * end_price
    
    # Calculate returns
    strategy_return = (final_value - initial_capital) / initial_capital
    buy_hold_return = (buy_hold_value - initial_capital) / initial_capital
    
    return {
        'initial_capital': initial_capital,
        'final_value': final_value,
        'total_return': strategy_return,
        'total_return_pct': strategy_return * 100,
        'profit_loss': final_value - initial_capital,
        'buy_hold_value': buy_hold_value,
        'buy_hold_return': buy_hold_return,
        'buy_hold_return_pct': buy_hold_return * 100,
        'vs_buy_hold': strategy_return - buy_hold_return,
        'vs_buy_hold_pct': (strategy_return - buy_hold_return) * 100,
        'num_trades': len(trades),
        'trades': trades,
        'portfolio_history': pd.DataFrame(portfolio_history),
        'final_cash': cash,
        'final_shares': shares
    }


def main():
    st.title("💰 ML Trading Strategy Simulator")
    st.markdown("**Train a model → Simulate portfolio on test data → See if you would have made money!**")
    st.info("📌 Simulates starting with your chosen capital on the test start date and following the model's BUY/SELL signals")
    
    st.divider()
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Ticker
        ticker = st.text_input(
            "Stock Ticker",
            value="AAPL",
            max_chars=10,
            help="Enter stock symbol (e.g., AAPL, TSLA, NVDA)"
        ).upper()
        
        # Timeframe
        timeframe = st.selectbox(
            "Timeframe",
            options=['6M', '1Y', '2Y', '5Y'],
            index=2,
            help="Historical data to fetch"
        )
        
        # Model
        model_type = st.selectbox(
            "ML Model",
            options=['xgboost', 'random_forest', 'lightgbm'],
            index=0,
            help="Machine learning algorithm"
        )
        
        # Train/Test Split
        train_split = st.slider(
            "Train/Test Split",
            min_value=0.5,
            max_value=0.9,
            value=0.8,
            step=0.05,
            format="%.0f%%",
            help="% of data for training (rest is for testing)"
        )
        
        st.divider()
        
        # Portfolio Simulation Settings
        st.subheader("💰 Portfolio Settings")
        
        initial_capital = st.number_input(
            "Initial Capital ($)",
            min_value=1000,
            max_value=1000000,
            value=10000,
            step=1000,
            help="Starting investment amount on test start date"
        )
        
        trade_size = st.selectbox(
            "Position Size",
            options=['100%', '50%', '33%', '25%'],
            index=1,
            help="% of capital to use per trade (50% recommended)"
        )
        
        st.divider()
        
        # Run button
        run_button = st.button("🚀 Run Analysis", type="primary", use_container_width=True)
        
        if st.button("Clear", use_container_width=True):
            st.session_state.clear()
            st.rerun()
    
    # Main content
    if run_button:
        with st.spinner(f"Analyzing {ticker}..."):
            try:
                # Step 1: Fetch data
                st.info(f"📥 Fetching {ticker} data ({timeframe})...")
                raw_data = fetch_stock_data(ticker, timeframe)
                
                if raw_data is None or len(raw_data) < 100:
                    st.error("Not enough data. Try a different ticker or timeframe.")
                    return
                
                st.success(f"✓ Fetched {len(raw_data)} days of data")
                
                # Step 2: Engineer features
                st.info("🔧 Calculating technical indicators...")
                feature_data = engineer_all_features(raw_data, verbose=False)
                st.success(f"✓ Engineered {len(feature_data.columns)} features")
                
                # Step 3: Generate labels
                st.info("🏷️ Generating labels...")
                labels, label_dist = generate_labels(
                    feature_data,
                    forward_days=5,
                    threshold=0.02,
                    adaptive_threshold=True
                )
                st.success(f"✓ Generated {label_dist['total']} labels (BUY: {label_dist['buy']}, HOLD: {label_dist['hold']}, SELL: {label_dist['sell']})")
                
                # Step 4: Prepare train/test split
                st.info("📊 Splitting data...")
                feature_list = get_feature_list('all')
                X = prepare_feature_matrix(feature_data, features=feature_list)
                
                # Align with labels
                valid_idx = labels.dropna().index
                X = X.loc[valid_idx]
                y = labels.loc[valid_idx]
                
                # Time-series split
                split_point = int(len(X) * train_split)
                X_train = X.iloc[:split_point]
                X_test = X.iloc[split_point:]
                y_train = y.iloc[:split_point]
                y_test = y.iloc[split_point:]
                
                train_end_date = X_train.index[-1]
                test_start_date = X_test.index[0]
                
                st.success(f"✓ Train: {len(X_train)} samples | Test: {len(X_test)} samples")
                st.info(f"🎯 **Test data starts: {test_start_date.strftime('%Y-%m-%d')}** (model will NOT see this during training!)")
                
                # Step 5: Train model
                st.info(f"🤖 Training {model_type.upper()} model...")
                model = BaselineModelTrainer(model_type)
                metrics = model.train(X_train, y_train, X_test, y_test, verbose=False)
                
                st.success(f"✓ Model trained! Test Accuracy: {metrics['test_accuracy']:.1%}")
                
                # Step 6: Generate predictions for ALL data
                st.info("🔮 Generating predictions...")
                predictions = model.predict(X)
                predictions_series = pd.Series(predictions, index=X.index)
                
                # Step 7: Portfolio Simulation on TEST data
                st.info("💰 Simulating portfolio on test data...")
                
                # Parse trade size
                trade_size_pct = float(trade_size.rstrip('%')) / 100.0
                
                portfolio_results = simulate_portfolio(
                    feature_data=feature_data.loc[test_start_date:],
                    predictions=predictions_series.loc[test_start_date:],
                    initial_capital=initial_capital,
                    position_size=trade_size_pct
                )
                
                st.success(f"✓ Portfolio simulation complete! Final value: ${portfolio_results['final_value']:,.2f}")
                
                # Store in session state
                st.session_state.results = {
                    'ticker': ticker,
                    'timeframe': timeframe,
                    'model_type': model_type,
                    'feature_data': feature_data,
                    'X': X,
                    'y': y,
                    'predictions': predictions_series,
                    'train_split': train_split,
                    'split_point': split_point,
                    'train_end_date': train_end_date,
                    'test_start_date': test_start_date,
                    'metrics': metrics,
                    'portfolio': portfolio_results,
                    'initial_capital': initial_capital,
                    'trade_size': trade_size
                }
                
                st.success("✅ Analysis complete!")
                st.rerun()
                
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                st.exception(e)
    
    # Display results
    if 'results' in st.session_state:
        results = st.session_state.results
        
        # Check if portfolio simulation exists (backward compatibility)
        if 'portfolio' not in results:
            st.warning("⚠️ Old results detected. Please click **Clear** and run the analysis again.")
            return
        
        portfolio = results['portfolio']
        
        st.divider()
        
        # Portfolio Performance Metrics (Primary Focus)
        st.subheader("💰 Portfolio Performance on Test Data")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric(
                "Initial Capital", 
                f"${portfolio['initial_capital']:,.0f}",
                help="Starting investment amount"
            )
        with col2:
            profit_loss = portfolio['profit_loss']
            st.metric(
                "Final Value", 
                f"${portfolio['final_value']:,.2f}",
                delta=f"${profit_loss:,.2f}",
                delta_color="normal",
                help="Portfolio value at end of test period"
            )
        with col3:
            st.metric(
                "Total Return", 
                f"{portfolio['total_return_pct']:.2f}%",
                help="Profit/loss as % of initial capital"
            )
        with col4:
            vs_buy_hold = portfolio['vs_buy_hold_pct']
            st.metric(
                "vs Buy & Hold", 
                f"{vs_buy_hold:+.2f}%",
                delta=f"{vs_buy_hold:+.2f}%",
                delta_color="normal",
                help="Outperformance vs simply buying and holding"
            )
        
        # Secondary Metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Ticker", results['ticker'])
        with col2:
            st.metric("Model", results['model_type'].upper())
        with col3:
            st.metric("Test Accuracy", f"{results['metrics']['test_accuracy']:.1%}",
                     help="How often model predicted correctly")
        with col4:
            st.metric("Trades Executed", portfolio['num_trades'],
                     help="Number of BUY/SELL trades")
        
        st.divider()
        
        # Visualization
        st.subheader("📈 Price Chart with Predicted Signals")
        
        # Info box
        col1, col2 = st.columns([1, 1])
        with col1:
            st.info(f"🔵 **Blue region** = Training data ({int(results['train_split']*100)}%)")
        with col2:
            st.success(f"🟢 **Green region** = Test data ({int((1-results['train_split'])*100)}%) - Model predictions on UNSEEN data!")
        
        # Create chart
        fig = create_signal_chart(
            feature_data=results['feature_data'],
            predictions=results['predictions'],
            actual_labels=results['y'],
            train_end_date=results['train_end_date'],
            test_start_date=results['test_start_date'],
            ticker=results['ticker']
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Legend
        st.caption("**Legend:** Small faded = Training predictions | 🎯 Large bold = TEST predictions (unseen data!) | Orange line = Train/Test split")
        
        st.divider()
        
        # Portfolio Value Over Time
        st.subheader("📊 Portfolio Value Over Time (Test Period)")
        
        fig_portfolio = create_portfolio_chart(
            portfolio_history=portfolio['portfolio_history'],
            initial_capital=portfolio['initial_capital'],
            buy_hold_value=portfolio['buy_hold_value'],
            ticker=results['ticker'],
            test_start_date=results['test_start_date']
        )
        
        st.plotly_chart(fig_portfolio, use_container_width=True)
        
        st.caption("**Green line** = ML strategy portfolio | **Gray dashed** = Buy & Hold strategy")
        
        st.divider()
        
        # Trade Log
        st.subheader("📋 Trade History (Test Period)")
        
        if portfolio['num_trades'] > 0:
            trades_df = pd.DataFrame(portfolio['trades'])
            trades_df['date'] = trades_df['date'].dt.strftime('%Y-%m-%d')
            trades_df['price'] = trades_df['price'].apply(lambda x: f"${x:.2f}")
            trades_df['shares'] = trades_df['shares'].apply(lambda x: f"{x:.4f}")
            trades_df['value'] = trades_df['value'].apply(lambda x: f"${x:,.2f}")
            trades_df['cash_after'] = trades_df['cash_after'].apply(lambda x: f"${x:,.2f}")
            
            st.dataframe(
                trades_df[['date', 'action', 'price', 'shares', 'value', 'cash_after']],
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info("No trades executed in test period (model predicted mostly HOLD)")
        
        st.divider()
        
        # Signal breakdown
        st.subheader("📊 Signal Breakdown")
        
        # Separate train and test predictions
        train_preds = results['predictions'].iloc[:results['split_point']]
        test_preds = results['predictions'].iloc[results['split_point']:]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Training Predictions:**")
            train_buy = (train_preds == 1).sum()
            train_hold = (train_preds == 0).sum()
            train_sell = (train_preds == -1).sum()
            st.write(f"- BUY: {train_buy} ({train_buy/len(train_preds)*100:.1f}%)")
            st.write(f"- HOLD: {train_hold} ({train_hold/len(train_preds)*100:.1f}%)")
            st.write(f"- SELL: {train_sell} ({train_sell/len(train_preds)*100:.1f}%)")
        
        with col2:
            st.markdown("**🎯 Test Predictions (Unseen):**")
            test_buy = (test_preds == 1).sum()
            test_hold = (test_preds == 0).sum()
            test_sell = (test_preds == -1).sum()
            st.write(f"- BUY: {test_buy} ({test_buy/len(test_preds)*100:.1f}%)")
            st.write(f"- HOLD: {test_hold} ({test_hold/len(test_preds)*100:.1f}%)")
            st.write(f"- SELL: {test_sell} ({test_sell/len(test_preds)*100:.1f}%)")
        
        # Performance Interpretation
        st.divider()
        st.subheader("🧠 Performance Analysis")
        
        total_return = portfolio['total_return_pct']
        vs_buy_hold = portfolio['vs_buy_hold_pct']
        test_accuracy = results['metrics']['test_accuracy']
        
        # Overall Verdict
        st.markdown("### 📊 Overall Result:")
        
        if vs_buy_hold > 5:
            st.success(f"✅ **PROFITABLE** - Strategy beat Buy & Hold by {vs_buy_hold:.2f}%")
            st.write(f"- Starting with **${portfolio['initial_capital']:,.0f}** on {results['test_start_date'].strftime('%Y-%m-%d')}")
            st.write(f"- Ending with **${portfolio['final_value']:,.2f}** ({total_return:+.2f}%)")
            st.write(f"- Buy & Hold would have: **${portfolio['buy_hold_value']:,.2f}** ({portfolio['buy_hold_return_pct']:+.2f}%)")
            st.write("✨ **The model's predictions added value!**")
        elif vs_buy_hold > -5:
            st.info(f"📊 **NEUTRAL** - Strategy performed similarly to Buy & Hold ({vs_buy_hold:+.2f}%)")
            st.write(f"- Starting with **${portfolio['initial_capital']:,.0f}** on {results['test_start_date'].strftime('%Y-%m-%d')}")
            st.write(f"- Ending with **${portfolio['final_value']:,.2f}** ({total_return:+.2f}%)")
            st.write(f"- Buy & Hold would have: **${portfolio['buy_hold_value']:,.2f}** ({portfolio['buy_hold_return_pct']:+.2f}%)")
            st.write("💡 Model isn't hurting, but not helping much either")
        else:
            st.error(f"❌ **UNDERPERFORMED** - Strategy lost to Buy & Hold by {abs(vs_buy_hold):.2f}%")
            st.write(f"- Starting with **${portfolio['initial_capital']:,.0f}** on {results['test_start_date'].strftime('%Y-%m-%d')}")
            st.write(f"- Ending with **${portfolio['final_value']:,.2f}** ({total_return:+.2f}%)")
            st.write(f"- Buy & Hold would have: **${portfolio['buy_hold_value']:,.2f}** ({portfolio['buy_hold_return_pct']:+.2f}%)")
            st.write("⚠️ **You would have been better off just buying and holding!**")
        
        st.markdown("---")
        
        # Recommendations
        st.markdown("### 💡 What To Do:")
        
        if vs_buy_hold > 5:
            st.success("**✅ Strategy worked! To improve further:**")
            st.write("- Test on different time periods to ensure consistency")
            st.write("- Try different stocks to see if strategy generalizes")
            st.write("- Consider paper trading before using real money")
            st.write("- Add risk management (stop-losses, position limits)")
        elif vs_buy_hold > -5:
            st.warning("**⚠️ Strategy is break-even. To improve:**")
            st.write("- Try a **longer timeframe** (more training data)")
            st.write("- Try a **different stock** (some are more predictable)")
            st.write("- Try a **different model** (XGBoost vs Random Forest)")
            st.write("- Adjust **position size** (try 33% or 25%)")
        else:
            st.error("**❌ Strategy underperformed. Consider:**")
            st.write("- **Don't trade with this model** - it's losing money!")
            st.write("- Try a **much longer timeframe** (e.g., 5Y)")
            st.write("- Try a **completely different stock** (maybe crypto?)")
            st.write("- The stock may not be predictable with technical indicators")
        
        st.markdown("---")
        
        # Additional Metrics
        st.markdown("### 📈 Model Stats:")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write(f"**Test Accuracy:** {test_accuracy:.1%}")
            st.write(f"**Trades Executed:** {portfolio['num_trades']}")
            st.write(f"**Final Cash:** ${portfolio['final_cash']:,.2f}")
        
        with col2:
            st.write(f"**Final Shares:** {portfolio['final_shares']:.4f}")
            st.write(f"**Test Period:** {len(portfolio['portfolio_history'])} days")
            if portfolio['num_trades'] > 0:
                st.write(f"**Avg Days Per Trade:** {len(portfolio['portfolio_history']) / portfolio['num_trades']:.1f}")
        
        # Model metrics
        with st.expander("🔍 Model Performance Details"):
            metrics = results['metrics']
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"**Training Accuracy:** {metrics['train_accuracy']:.1%}")
                st.write(f"**Test Accuracy:** {metrics['test_accuracy']:.1%}")
                st.write(f"**Precision:** {metrics['precision']:.1%}")
            
            with col2:
                st.write(f"**Recall:** {metrics['recall']:.1%}")
                st.write(f"**F1 Score:** {metrics['f1']:.1%}")
                st.write(f"**Total Features:** {metrics['n_features']}")


def create_signal_chart(
    feature_data: pd.DataFrame,
    predictions: pd.Series,
    actual_labels: pd.Series,
    train_end_date,
    test_start_date,
    ticker: str
) -> go.Figure:
    """Create price chart with predicted signals"""
    
    fig = go.Figure()
    
    # Price line
    fig.add_trace(go.Scatter(
        x=feature_data.index,
        y=feature_data['Close'],
        mode='lines',
        name='Price',
        line=dict(color='gray', width=1.5)
    ))
    
    # Train/test split line
    test_start_dt = test_start_date.to_pydatetime()
    train_end_dt = train_end_date.to_pydatetime()
    data_start_dt = feature_data.index[0].to_pydatetime()
    data_end_dt = feature_data.index[-1].to_pydatetime()
    
    # Shaded regions
    fig.add_vrect(
        x0=data_start_dt,
        x1=train_end_dt,
        fillcolor="blue",
        opacity=0.05,
        layer="below",
        line_width=0
    )
    
    fig.add_vrect(
        x0=test_start_dt,
        x1=data_end_dt,
        fillcolor="green",
        opacity=0.1,
        layer="below",
        line_width=0
    )
    
    # Split line
    fig.add_vline(
        x=test_start_dt,
        line_dash="dash",
        line_color="orange",
        line_width=2
    )
    
    # Add annotation
    fig.add_annotation(
        x=test_start_dt,
        y=1,
        yref="paper",
        text="← Test Data Starts (Unseen)",
        showarrow=False,
        font=dict(size=12, color="orange"),
        xshift=5,
        yshift=-10
    )
    
    # Get predictions aligned with feature_data
    aligned_preds = predictions.reindex(feature_data.index)
    
    # Separate train vs test predictions
    train_mask = feature_data.index < test_start_date
    test_mask = feature_data.index >= test_start_date
    
    # Training predictions (small, faded)
    train_buys = feature_data.index[train_mask & (aligned_preds == 1)]
    train_sells = feature_data.index[train_mask & (aligned_preds == -1)]
    
    if len(train_buys) > 0:
        fig.add_trace(go.Scatter(
            x=train_buys,
            y=feature_data.loc[train_buys, 'Close'],
            mode='markers',
            name=f'Buy - Train ({len(train_buys)})',
            marker=dict(symbol='triangle-up', size=6, color='green', opacity=0.3)
        ))
    
    if len(train_sells) > 0:
        fig.add_trace(go.Scatter(
            x=train_sells,
            y=feature_data.loc[train_sells, 'Close'],
            mode='markers',
            name=f'Sell - Train ({len(train_sells)})',
            marker=dict(symbol='triangle-down', size=6, color='red', opacity=0.3)
        ))
    
    # Test predictions (large, bold) - THESE ARE WHAT MATTER!
    test_buys = feature_data.index[test_mask & (aligned_preds == 1)]
    test_sells = feature_data.index[test_mask & (aligned_preds == -1)]
    
    if len(test_buys) > 0:
        fig.add_trace(go.Scatter(
            x=test_buys,
            y=feature_data.loc[test_buys, 'Close'],
            mode='markers',
            name=f'🎯 BUY - TEST ({len(test_buys)})',
            marker=dict(symbol='triangle-up', size=14, color='lime', line=dict(width=2, color='darkgreen'))
        ))
    
    if len(test_sells) > 0:
        fig.add_trace(go.Scatter(
            x=test_sells,
            y=feature_data.loc[test_sells, 'Close'],
            mode='markers',
            name=f'🎯 SELL - TEST ({len(test_sells)})',
            marker=dict(symbol='triangle-down', size=14, color='red', line=dict(width=2, color='darkred'))
        ))
    
    fig.update_layout(
        title=f'{ticker} - Predicted Trading Signals',
        xaxis_title='Date',
        yaxis_title='Price ($)',
        hovermode='x unified',
        height=600,
        showlegend=True
    )
    
    return fig


def create_portfolio_chart(
    portfolio_history: pd.DataFrame,
    initial_capital: float,
    buy_hold_value: float,
    ticker: str,
    test_start_date
) -> go.Figure:
    """Create portfolio value chart comparing ML strategy vs Buy & Hold"""
    
    fig = go.Figure()
    
    # ML Strategy Portfolio Value
    fig.add_trace(go.Scatter(
        x=portfolio_history['date'],
        y=portfolio_history['portfolio_value'],
        mode='lines',
        name='ML Strategy',
        line=dict(color='green', width=2.5)
    ))
    
    # Buy & Hold Strategy (straight line from initial to final)
    start_date = portfolio_history['date'].iloc[0]
    end_date = portfolio_history['date'].iloc[-1]
    
    fig.add_trace(go.Scatter(
        x=[start_date, end_date],
        y=[initial_capital, buy_hold_value],
        mode='lines',
        name='Buy & Hold',
        line=dict(color='gray', width=2, dash='dash')
    ))
    
    # Initial capital line
    fig.add_hline(
        y=initial_capital,
        line_dash="dot",
        line_color="blue",
        opacity=0.5,
        annotation_text=f"Initial: ${initial_capital:,.0f}",
        annotation_position="right"
    )
    
    fig.update_layout(
        title=f'{ticker} - Portfolio Value Over Time (Test Period)',
        xaxis_title='Date',
        yaxis_title='Portfolio Value ($)',
        hovermode='x unified',
        height=500,
        showlegend=True
    )
    
    return fig


if __name__ == "__main__":
    main()

