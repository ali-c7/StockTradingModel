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


def main():
    st.title("📊 Simple Trading Signal Predictor")
    st.markdown("**Train a model → Predict signals → Visualize on unseen test data**")
    
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
                    'metrics': metrics
                }
                
                st.success("✅ Analysis complete!")
                st.rerun()
                
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                st.exception(e)
    
    # Display results
    if 'results' in st.session_state:
        results = st.session_state.results
        
        st.divider()
        
        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Ticker", results['ticker'])
        with col2:
            st.metric("Model", results['model_type'].upper())
        with col3:
            st.metric("Test Accuracy", f"{results['metrics']['test_accuracy']:.1%}")
        with col4:
            train_samples = int(len(results['X']) * results['train_split'])
            test_samples = len(results['X']) - train_samples
            st.metric("Test Samples", test_samples)
        
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


if __name__ == "__main__":
    main()

