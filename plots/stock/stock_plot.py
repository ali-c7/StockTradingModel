"""
Stock price visualization module
Creates interactive charts using Plotly
"""

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def create_price_chart(df: pd.DataFrame, ticker: str) -> go.Figure:
    """
    Create interactive candlestick chart with volume subplot
    
    Parameters:
        df: DataFrame with OHLCV data (columns: Open, High, Low, Close, Volume)
        ticker: Stock ticker symbol for chart title
    
    Returns:
        Plotly Figure object with candlestick and volume charts
    """
    if df is None or df.empty:
        return None
    
    # Create figure with 2 subplots (price and volume)
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.7, 0.3],
        subplot_titles=(f'{ticker} Price Chart', 'Volume')
    )
    
    # Add candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name='Price',
            increasing_line_color='#26a69a',  # Teal green for up
            decreasing_line_color='#ef5350'   # Red for down
        ),
        row=1, col=1
    )
    
    # Determine volume bar colors (green for up days, red for down days)
    colors = []
    for i in range(len(df)):
        if df['Close'].iloc[i] >= df['Open'].iloc[i]:
            colors.append('#26a69a')  # Green (up)
        else:
            colors.append('#ef5350')  # Red (down)
    
    # Add volume bars
    fig.add_trace(
        go.Bar(
            x=df.index,
            y=df['Volume'],
            name='Volume',
            marker_color=colors,
            showlegend=False
        ),
        row=2, col=1
    )
    
    # Update layout
    fig.update_layout(
        xaxis_rangeslider_visible=False,  # Remove range slider
        height=600,
        hovermode='x unified',
        showlegend=False,
        margin=dict(l=50, r=50, t=50, b=50),
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    
    # Update x-axes
    fig.update_xaxes(
        gridcolor='lightgray',
        showgrid=True,
        row=2, col=1
    )
    
    # Update y-axes
    fig.update_yaxes(
        title_text="Price ($)",
        gridcolor='lightgray',
        showgrid=True,
        row=1, col=1
    )
    
    fig.update_yaxes(
        title_text="Volume",
        gridcolor='lightgray',
        showgrid=True,
        row=2, col=1
    )
    
    return fig


def create_simple_line_chart(df: pd.DataFrame, ticker: str) -> go.Figure:
    """
    Create simple line chart (fallback option)
    
    Parameters:
        df: DataFrame with OHLCV data
        ticker: Stock ticker symbol
    
    Returns:
        Plotly Figure object with line chart
    """
    if df is None or df.empty:
        return None
    
    # Create figure with 2 subplots
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.7, 0.3],
        subplot_titles=(f'{ticker} Close Price', 'Volume')
    )
    
    # Add line chart for closing price
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df['Close'],
            mode='lines',
            name='Close Price',
            line=dict(color='#2196F3', width=2)
        ),
        row=1, col=1
    )
    
    # Add volume bars
    fig.add_trace(
        go.Bar(
            x=df.index,
            y=df['Volume'],
            name='Volume',
            marker_color='#90CAF9',
            showlegend=False
        ),
        row=2, col=1
    )
    
    # Update layout
    fig.update_layout(
        height=600,
        hovermode='x unified',
        showlegend=False,
        margin=dict(l=50, r=50, t=50, b=50),
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    
    # Update axes
    fig.update_xaxes(gridcolor='lightgray', showgrid=True, row=2, col=1)
    fig.update_yaxes(title_text="Price ($)", gridcolor='lightgray', showgrid=True, row=1, col=1)
    fig.update_yaxes(title_text="Volume", gridcolor='lightgray', showgrid=True, row=2, col=1)
    
    return fig

