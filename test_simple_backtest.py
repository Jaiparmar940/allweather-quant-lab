#!/usr/bin/env python3
"""
Test script for simple backtesting
"""

import requests
import json
import numpy as np
import pandas as pd

def test_simple_backtest():
    """Test with a very simple backtest."""
    print("Testing simple backtest...")
    
    # Very simple data
    returns = [[0.01, 0.02], [0.02, 0.01], [0.01, 0.01], [0.02, 0.02]]
    asset_names = ["Asset1", "Asset2"]
    dates = ["2023-01-01", "2023-01-02", "2023-01-03", "2023-01-04"]
    
    request_data = {
        "returns": returns,
        "asset_names": asset_names,
        "dates": dates,
        "objective": "gmv",
        "initial_capital": 100000.0,
        "train_months": 1,
        "test_months": 1,
        "step_months": 1,
        "rebalance_frequency": "daily",
        "transaction_costs": 0.0,
        "slippage": 0.0,
        "long_only": True,
        "bounds": (0.0, 1.0)
    }
    
    try:
        response = requests.post("http://localhost:8000/backtest", json=request_data, timeout=10)
        print(f"Response status: {response.status_code}")
        print(f"Response: {response.text[:500]}...")
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Simple backtest successful!")
            return True
        else:
            print(f"❌ Simple backtest failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_walk_forward_directly():
    """Test the walk-forward engine directly."""
    print("\nTesting walk-forward engine directly...")
    
    try:
        # Import the engine directly
        import sys
        sys.path.append('/Users/jaivir/omega')
        
        from engine.backtest import WalkForwardEngine
        import pandas as pd
        import numpy as np
        
        # Create simple data
        np.random.seed(42)
        returns = pd.DataFrame(
            np.random.normal(0.001, 0.02, (100, 3)),
            columns=['Asset1', 'Asset2', 'Asset3'],
            index=pd.date_range('2023-01-01', periods=100, freq='D')
        )
        
        # Simple optimizer function
        def simple_optimizer(returns):
            # Equal weights
            return pd.Series(1.0/len(returns.columns), index=returns.columns)
        
        # Test walk-forward engine
        engine = WalkForwardEngine(train_months=2, test_months=1, step_months=1)
        results = engine.run_walk_forward(returns, simple_optimizer)
        
        if results and 'all_returns' in results:
            print("✅ Walk-forward engine works!")
            print(f"   Overall return: {results['overall_return']:.2%}")
            print(f"   Overall volatility: {results['overall_volatility']:.2%}")
            print(f"   Number of periods: {len(results.get('period_results', []))}")
            return True
        else:
            print("❌ Walk-forward engine failed to produce results")
            return False
            
    except Exception as e:
        print(f"❌ Error testing walk-forward engine: {e}")
        return False

def main():
    """Run the tests."""
    print("🔧 Testing Backtesting Issues")
    print("=" * 40)
    
    # Test simple backtest
    simple_success = test_simple_backtest()
    
    # Test walk-forward engine directly
    engine_success = test_walk_forward_directly()
    
    print("\n" + "=" * 40)
    if simple_success:
        print("🎉 Simple backtest works!")
    elif engine_success:
        print("🎉 Walk-forward engine works directly!")
        print("   The issue might be in the API integration")
    else:
        print("⚠️  Both tests failed. There's a deeper issue.")

if __name__ == "__main__":
    main()
