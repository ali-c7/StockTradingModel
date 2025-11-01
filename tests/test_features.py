"""
Test feature engineering module
Quick validation that all indicators work correctly
"""

import sys
sys.path.append('.')

from data.stock.stock_data import fetch_stock_data
from core.features.technical_features import (
    engineer_all_features,
    get_feature_list,
    analyze_features,
    prepare_feature_matrix
)


def test_feature_engineering():
    """Test the complete feature engineering pipeline"""
    print("="*70)
    print("FEATURE ENGINEERING TEST")
    print("="*70)
    
    # Fetch sample data
    print("\n[1] Fetching test data (AAPL, 1Y)...")
    df = fetch_stock_data("AAPL", "1Y")
    print(f"   Loaded: {len(df)} rows")
    print(f"   Columns: {list(df.columns)}")
    
    # Engineer features
    print("\n[2] Engineering features...")
    df_features = engineer_all_features(df, verbose=True)
    
    # Analyze results
    print("\n[3] Analyzing features...")
    stats = analyze_features(df_features, verbose=True)
    
    # Prepare feature matrix
    print("\n[4] Preparing feature matrix...")
    X = prepare_feature_matrix(df_features)
    print(f"   Shape: {X.shape}")
    print(f"   Features: {X.shape[1]}")
    print(f"   Samples: {X.shape[0]}")
    
    # Show sample of features
    print("\n[5] Sample of engineered features:")
    print(X.tail(3))
    
    # Feature categories
    print("\n[6] Feature counts by category:")
    for category in ['trend', 'momentum', 'volatility', 'volume', 'price']:
        features = get_feature_list(category)
        print(f"   {category.capitalize():12s}: {len(features)} features")
    
    print("\n" + "="*70)
    print("[PASS] Feature engineering test PASSED!")
    print("="*70)
    
    return df_features, X


if __name__ == "__main__":
    df, X = test_feature_engineering()

