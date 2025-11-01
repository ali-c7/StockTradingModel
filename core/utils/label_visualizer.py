"""
Label Visualization Tool
Visualize how labels are distributed and if they make sense
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Tuple


def visualize_labels(
    df: pd.DataFrame,
    labels: pd.Series,
    ticker: str = "Stock"
) -> go.Figure:
    """
    Create visualization of stock prices with Buy/Sell/Hold labels
    
    Parameters:
        df: DataFrame with OHLCV data
        labels: Series with Buy(1)/Hold(0)/Sell(-1) labels
        ticker: Stock ticker for title
    
    Returns:
        Plotly figure
    """
    # Align data
    valid_idx = labels.dropna().index
    df_plot = df.loc[valid_idx].copy()
    labels_plot = labels.loc[valid_idx]
    
    # Separate by signal type
    buy_idx = labels_plot[labels_plot == 1].index
    sell_idx = labels_plot[labels_plot == -1].index
    hold_idx = labels_plot[labels_plot == 0].index
    
    # Count labels
    buy_count = len(buy_idx)
    sell_count = len(sell_idx)
    hold_count = len(hold_idx)
    total = len(labels_plot)
    
    # Create subplots
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.5, 0.3, 0.2],
        subplot_titles=(
            f'{ticker} - Price Chart with Labels',
            'Label Distribution Over Time',
            'Future Returns (Used to Create Labels)'
        )
    )
    
    # 1. Price chart with markers
    fig.add_trace(
        go.Scatter(
            x=df_plot.index,
            y=df_plot['Close'],
            mode='lines',
            name='Price',
            line=dict(color='gray', width=1),
            showlegend=True
        ),
        row=1, col=1
    )
    
    # BUY markers (green triangles up)
    if buy_count > 0:
        fig.add_trace(
            go.Scatter(
                x=buy_idx,
                y=df_plot.loc[buy_idx, 'Close'],
                mode='markers',
                name=f'BUY ({buy_count}, {buy_count/total*100:.1f}%)',
                marker=dict(
                    symbol='triangle-up',
                    size=8,
                    color='green',
                    line=dict(color='darkgreen', width=1)
                ),
                showlegend=True
            ),
            row=1, col=1
        )
    
    # SELL markers (red triangles down)
    if sell_count > 0:
        fig.add_trace(
            go.Scatter(
                x=sell_idx,
                y=df_plot.loc[sell_idx, 'Close'],
                mode='markers',
                name=f'SELL ({sell_count}, {sell_count/total*100:.1f}%)',
                marker=dict(
                    symbol='triangle-down',
                    size=8,
                    color='red',
                    line=dict(color='darkred', width=1)
                ),
                showlegend=True
            ),
            row=1, col=1
        )
    
    # HOLD markers (small gray dots - sample only 10% to avoid clutter)
    if hold_count > 0:
        # Sample holds to avoid clutter
        sample_size = max(int(hold_count * 0.1), 10)
        sample_size = min(sample_size, hold_count)
        hold_sample = np.random.choice(hold_idx, size=sample_size, replace=False)
        
        fig.add_trace(
            go.Scatter(
                x=hold_sample,
                y=df_plot.loc[hold_sample, 'Close'],
                mode='markers',
                name=f'HOLD ({hold_count}, {hold_count/total*100:.1f}%) [10% shown]',
                marker=dict(
                    symbol='circle',
                    size=3,
                    color='orange',
                    opacity=0.3
                ),
                showlegend=True
            ),
            row=1, col=1
        )
    
    # 2. Label distribution over time (bar chart)
    # Group by label and count per month
    df_plot['Label'] = labels_plot
    df_plot['YearMonth'] = df_plot.index.to_period('M')
    
    label_dist = df_plot.groupby(['YearMonth', 'Label']).size().unstack(fill_value=0)
    label_dist.index = label_dist.index.to_timestamp()
    
    if -1 in label_dist.columns:
        fig.add_trace(
            go.Bar(
                x=label_dist.index,
                y=label_dist[-1],
                name='SELL',
                marker_color='red',
                showlegend=False
            ),
            row=2, col=1
        )
    
    if 0 in label_dist.columns:
        fig.add_trace(
            go.Bar(
                x=label_dist.index,
                y=label_dist[0],
                name='HOLD',
                marker_color='orange',
                showlegend=False
            ),
            row=2, col=1
        )
    
    if 1 in label_dist.columns:
        fig.add_trace(
            go.Bar(
                x=label_dist.index,
                y=label_dist[1],
                name='BUY',
                marker_color='green',
                showlegend=False
            ),
            row=2, col=1
        )
    
    # 3. Future returns (what created the labels)
    from core.models.label_generator import generate_labels
    _, analysis_df = generate_labels(df, forward_days=3, threshold=0.01)
    
    future_returns = analysis_df.loc[valid_idx, 'Future_Return'] * 100  # Convert to %
    
    # Color by label
    colors = []
    for idx in valid_idx:
        if labels_plot[idx] == 1:
            colors.append('green')
        elif labels_plot[idx] == -1:
            colors.append('red')
        else:
            colors.append('orange')
    
    fig.add_trace(
        go.Scatter(
            x=valid_idx,
            y=future_returns,
            mode='markers',
            name='Future Return',
            marker=dict(
                color=colors,
                size=4,
                opacity=0.6
            ),
            showlegend=False
        ),
        row=3, col=1
    )
    
    # Add threshold lines
    fig.add_hline(y=1.0, line_dash="dash", line_color="green", opacity=0.5, row=3, col=1)
    fig.add_hline(y=-1.0, line_dash="dash", line_color="red", opacity=0.5, row=3, col=1)
    fig.add_hline(y=0, line_dash="solid", line_color="gray", opacity=0.3, row=3, col=1)
    
    # Update layout
    fig.update_layout(
        height=900,
        showlegend=True,
        hovermode='x unified',
        title=f'{ticker} - Label Analysis'
    )
    
    fig.update_xaxes(title_text="Date", row=3, col=1)
    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig.update_yaxes(title_text="Count", row=2, col=1)
    fig.update_yaxes(title_text="Future Return (%)", row=3, col=1)
    
    return fig


def create_label_summary(labels: pd.Series) -> dict:
    """
    Create summary statistics for labels
    
    Parameters:
        labels: Series with Buy/Sell/Hold labels
    
    Returns:
        Dictionary with summary stats
    """
    valid_labels = labels.dropna()
    
    if len(valid_labels) == 0:
        return {
            'total': 0,
            'buy': 0,
            'sell': 0,
            'hold': 0,
            'buy_pct': 0,
            'sell_pct': 0,
            'hold_pct': 0,
            'imbalance_score': 0
        }
    
    total = len(valid_labels)
    buy = (valid_labels == 1).sum()
    sell = (valid_labels == -1).sum()
    hold = (valid_labels == 0).sum()
    
    buy_pct = buy / total * 100
    sell_pct = sell / total * 100
    hold_pct = hold / total * 100
    
    # Calculate imbalance (ideal is 33.33% each)
    target = 33.33
    imbalance = abs(buy_pct - target) + abs(sell_pct - target) + abs(hold_pct - target)
    
    return {
        'total': int(total),
        'buy': int(buy),
        'sell': int(sell),
        'hold': int(hold),
        'buy_pct': float(buy_pct),
        'sell_pct': float(sell_pct),
        'hold_pct': float(hold_pct),
        'imbalance_score': float(imbalance)
    }


# Example usage
if __name__ == "__main__":
    print("Label Visualizer Module")
    print("\nThis module provides:")
    print("  1. Visual chart of price with Buy/Sell/Hold markers")
    print("  2. Label distribution over time")
    print("  3. Future returns that created the labels")

