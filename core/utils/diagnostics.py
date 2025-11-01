"""
Diagnostic utilities for troubleshooting model performance
"""

import pandas as pd
from typing import Dict


def diagnose_model_performance(
    accuracy: float,
    training_samples: int,
    label_distribution: Dict,
    ticker: str,
    timeframe: str
) -> str:
    """
    Generate diagnostic report for low model accuracy
    
    Parameters:
        accuracy: Model accuracy (0-1)
        training_samples: Number of training samples
        label_distribution: Dict with buy/hold/sell counts
        ticker: Stock symbol
        timeframe: Analysis timeframe
    
    Returns:
        Formatted diagnostic string
    """
    report = []
    report.append("\n" + "="*70)
    report.append("🔍 MODEL PERFORMANCE DIAGNOSTICS")
    report.append("="*70)
    
    # Overall assessment
    if accuracy < 0.35:
        status = "❌ CRITICAL"
        assessment = "Model is essentially random guessing"
    elif accuracy < 0.45:
        status = "⚠️  WARNING"
        assessment = "Model accuracy is very low"
    elif accuracy < 0.55:
        status = "⚠️  MARGINAL"
        assessment = "Model is learning but performance is weak"
    elif accuracy < 0.65:
        status = "✓ ACCEPTABLE"
        assessment = "Model performance is acceptable for stock prediction"
    else:
        status = "✅ GOOD"
        assessment = "Model performance is good"
    
    report.append(f"\nStatus: {status}")
    report.append(f"Accuracy: {accuracy:.1%} (Random baseline: 33.3%)")
    report.append(f"Assessment: {assessment}")
    
    # Check training data size
    report.append(f"\n{'─'*70}")
    report.append("DATA ANALYSIS")
    report.append(f"{'─'*70}")
    report.append(f"Ticker: {ticker}")
    report.append(f"Timeframe: {timeframe}")
    report.append(f"Training samples: {training_samples}")
    
    if training_samples < 150:
        report.append("\n❌ INSUFFICIENT DATA")
        report.append(f"   You have {training_samples} samples, but need 200+ for good results")
        report.append(f"   Solution: Use longer timeframe (1Y or 2Y)")
    elif training_samples < 250:
        report.append("\n⚠️  LIMITED DATA")
        report.append(f"   {training_samples} samples is marginal. More data would help.")
        report.append(f"   Recommendation: Use 1Y+ timeframe")
    else:
        report.append(f"\n✓ Data quantity is adequate ({training_samples} samples)")
    
    # Check label distribution
    report.append(f"\n{'─'*70}")
    report.append("LABEL DISTRIBUTION")
    report.append(f"{'─'*70}")
    
    total = label_distribution['total']
    buy_pct = label_distribution['buy_pct']
    hold_pct = label_distribution['hold_pct']
    sell_pct = label_distribution['sell_pct']
    
    report.append(f"BUY:  {label_distribution['buy_count']:3d} ({buy_pct:.1f}%)")
    report.append(f"HOLD: {label_distribution['hold_count']:3d} ({hold_pct:.1f}%)")
    report.append(f"SELL: {label_distribution['sell_count']:3d} ({sell_pct:.1f}%)")
    
    # Check for severe imbalance
    max_pct = max(buy_pct, hold_pct, sell_pct)
    min_pct = min(buy_pct, hold_pct, sell_pct)
    
    if max_pct > 70:
        report.append("\n⚠️  SEVERE CLASS IMBALANCE")
        report.append(f"   One class dominates ({max_pct:.1f}%)")
        report.append(f"   This makes learning difficult")
        if hold_pct > 70:
            report.append(f"   → Stock is too sideways/stable (mostly HOLD signals)")
            report.append(f"   → Try: More volatile stock or adjust threshold")
    elif max_pct > 60:
        report.append("\n⚠️  CLASS IMBALANCE DETECTED")
        report.append(f"   Distribution is uneven (max: {max_pct:.1f}%, min: {min_pct:.1f}%)")
    else:
        report.append("\n✓ Label distribution is reasonable")
    
    # Recommendations
    report.append(f"\n{'─'*70}")
    report.append("RECOMMENDATIONS")
    report.append(f"{'─'*70}")
    
    recommendations = []
    
    if training_samples < 250:
        recommendations.append("1. Use longer timeframe (1Y, 2Y, or 5Y)")
    
    if accuracy < 0.45:
        if ticker.endswith("-USD"):
            recommendations.append("2. Try traditional stocks instead of crypto (e.g., AAPL, MSFT)")
        else:
            recommendations.append("2. Try large-cap stable stocks (AAPL, MSFT, JNJ)")
        recommendations.append("3. Adjust label parameters:")
        recommendations.append("   - Increase forward_days from 3 to 5 or 10")
        recommendations.append("   - Adjust threshold based on stock volatility")
    
    if max_pct > 70:
        recommendations.append(f"4. Stock may be too stable - try more volatile ticker")
        recommendations.append(f"   OR adjust threshold to capture smaller movements")
    
    if not recommendations:
        recommendations.append("Continue monitoring performance")
        recommendations.append("Consider backtesting to validate predictions")
    
    for rec in recommendations:
        report.append(f"  {rec}")
    
    # Expected performance
    report.append(f"\n{'─'*70}")
    report.append("EXPECTED PERFORMANCE")
    report.append(f"{'─'*70}")
    report.append("Stock prediction is inherently difficult:")
    report.append("  • 33.3% = Random guessing (baseline)")
    report.append("  • 40-50% = Weak signal, barely predictive")
    report.append("  • 50-60% = Acceptable for swing trading")
    report.append("  • 60-70% = Good performance")
    report.append("  • 70%+ = Excellent (rare for stocks)")
    
    report.append("\n" + "="*70 + "\n")
    
    return "\n".join(report)


def suggest_timeframe(current_accuracy: float, current_samples: int) -> str:
    """
    Suggest optimal timeframe based on current performance
    
    Parameters:
        current_accuracy: Current model accuracy
        current_samples: Current number of training samples
    
    Returns:
        Recommended timeframe
    """
    if current_samples < 150:
        return "1Y or 2Y (need significantly more data)"
    elif current_samples < 250:
        return "1Y (need more data)"
    elif current_accuracy < 0.45:
        return "2Y or 5Y (more data may help)"
    else:
        return "Current timeframe is adequate"


def suggest_ticker_alternatives(ticker: str) -> list:
    """
    Suggest alternative tickers that may be easier to predict
    
    Parameters:
        ticker: Current ticker
    
    Returns:
        List of alternative ticker suggestions
    """
    # Crypto usually harder
    if ticker.endswith("-USD"):
        return [
            "AAPL - Apple (large cap tech, stable trends)",
            "MSFT - Microsoft (large cap tech, stable)",
            "JNJ - Johnson & Johnson (defensive, stable)",
            "SPY - S&P 500 ETF (diversified, smooth)"
        ]
    
    # Generic suggestions
    return [
        "AAPL - Apple (large cap, good liquidity)",
        "MSFT - Microsoft (stable tech leader)",
        "GOOGL - Google (consistent growth)",
        "SPY - S&P 500 ETF (market index)"
    ]

