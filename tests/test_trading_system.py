"""
Test Complete Trading System
End-to-end validation of the full pipeline
"""

import sys
sys.path.append('.')

from core.trading_system import TradingSystem


def test_full_pipeline(ticker='AAPL', timeframe='1Y', model_type='xgboost'):
    """Test complete trading system pipeline"""
    
    print("\n" + "="*70)
    print("TESTING COMPLETE TRADING SYSTEM")
    print("="*70)
    
    # Create trading system
    system = TradingSystem(
        ticker=ticker,
        timeframe=timeframe,
        model_type=model_type,
        forward_days=5,
        threshold=0.02,
        train_split=0.8
    )
    
    # Run complete pipeline
    results = system.run_complete_pipeline(verbose=True)
    
    # Get latest signal
    latest_signal = system.get_latest_signal()
    
    print(f"\n[SIGNAL] Latest Signal for {ticker}:")
    print(f"   Signal:     {latest_signal['signal_name']}")
    print(f"   Confidence: {latest_signal['confidence']:.1%}")
    print(f"   Date:       {latest_signal['date']}")
    print(f"\n   Probabilities:")
    for signal_name, prob in latest_signal['probabilities'].items():
        print(f"      {signal_name}: {prob:.1%}")
    
    # Feature importance
    print(f"\n[TOP 10] Most Important Features:")
    importance = system.model.get_feature_importance(top_n=10)
    for idx, row in importance.iterrows():
        print(f"      {row['feature']:20s}: {row['importance']:.4f}")
    
    print("\n" + "="*70)
    print("[PASS] TEST PASSED!")
    print("="*70)
    
    return system, results


def compare_all_models(ticker='GOOGL', timeframe='1Y'):
    """Compare all three baseline models"""
    
    print("\n" + "="*70)
    print(f"COMPARING ALL MODELS: {ticker} ({timeframe})")
    print("="*70)
    
    results_by_model = {}
    
    for model_type in ['random_forest', 'xgboost', 'lightgbm']:
        print(f"\n{'='*70}")
        print(f"Testing: {model_type.upper()}")
        print(f"{'='*70}")
        
        system = TradingSystem(
            ticker=ticker,
            timeframe=timeframe,
            model_type=model_type,
            forward_days=5,
            threshold=0.02
        )
        
        results = system.run_complete_pipeline(verbose=False)
        results_by_model[model_type] = results
    
    # Print comparison
    print("\n" + "="*70)
    print("[COMPARISON] MODEL COMPARISON SUMMARY")
    print("="*70)
    print(f"\n{'Model':<15} {'Accuracy':<12} {'Sharpe':<10} {'Return':<12} {'Alpha':<12} {'Win Rate':<10}")
    print("-"*70)
    
    for model_type, results in results_by_model.items():
        metrics = results['model_metrics']
        backtest = results['backtest_results']['metrics']
        
        print(f"{model_type:<15} "
              f"{metrics['test_accuracy']:>10.1%}  "
              f"{backtest['sharpe_ratio']:>8.2f}  "
              f"{backtest['total_return']:>10.1%}  "
              f"{backtest['alpha']:>10.1%}  "
              f"{backtest['win_rate']:>8.1%}")
    
    # Find best by Sharpe ratio
    best_model = max(results_by_model.items(), 
                     key=lambda x: x[1]['backtest_results']['metrics']['sharpe_ratio'])
    
    print(f"\n[BEST] Best Model (by Sharpe): {best_model[0].upper()}")
    print(f"   Sharpe Ratio: {best_model[1]['backtest_results']['metrics']['sharpe_ratio']:.2f}")
    print("="*70)
    
    return results_by_model


if __name__ == "__main__":
    # Test 1: Full pipeline with single model
    system, results = test_full_pipeline('AAPL', '1Y', 'xgboost')
    
    # Test 2: Compare all models
    # comparison = compare_all_models('GOOGL', '1Y')

