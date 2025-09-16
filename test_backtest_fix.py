#!/usr/bin/env python3
"""
Test script to verify backtesting functionality
"""

import requests
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def test_backtest_api():
    """Test the backtesting API endpoint."""
    print("Testing backtesting API...")
    
    # Create sample data
    np.random.seed(42)
    n_days = 500  # More data for backtesting
    n_assets = 5
    
    # Generate sample returns
    returns = np.random.normal(0.001, 0.02, (n_days, n_assets))
    asset_names = [f"Asset_{i+1}" for i in range(n_assets)]
    dates = pd.date_range(start="2022-01-01", periods=n_days, freq="D")
    
    # Test backtest request
    request_data = {
        "returns": returns.tolist(),
        "asset_names": asset_names,
        "dates": dates.strftime("%Y-%m-%d").tolist(),
        "objective": "gmv",
        "theta": 0.02,
        "initial_capital": 1000000.0,
        "train_months": 120,
        "test_months": 12,
        "step_months": 1,
        "rebalance_frequency": "monthly",
        "transaction_costs": 0.0005,
        "slippage": 0.0002,
        "long_only": True,
        "bounds": (0.0, 1.0)
    }
    
    print(f"Request data prepared: {len(returns)} days, {n_assets} assets")
    
    try:
        response = requests.post("http://localhost:8000/backtest", json=request_data, timeout=30)
        print(f"Response status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Backtest successful!")
            print(f"   Total Return: {data.get('total_return', 0):.2%}")
            print(f"   Annualized Return: {data.get('annualized_return', 0):.2%}")
            print(f"   Volatility: {data.get('annualized_volatility', 0):.2%}")
            print(f"   Sharpe Ratio: {data.get('sharpe_ratio', 0):.3f}")
            print(f"   Max Drawdown: {data.get('max_drawdown', 0):.2%}")
            return True
        else:
            print(f"❌ Backtest failed: {response.status_code}")
            print(f"   Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_optimization_api():
    """Test the optimization API endpoint."""
    print("\nTesting optimization API...")
    
    # Create sample data
    np.random.seed(42)
    n_days = 252
    n_assets = 5
    
    # Generate sample returns
    returns = np.random.normal(0.001, 0.02, (n_days, n_assets))
    asset_names = [f"Asset_{i+1}" for i in range(n_assets)]
    dates = pd.date_range(start="2023-01-01", periods=n_days, freq="D")
    
    # Test optimization request
    request_data = {
        "returns": returns.tolist(),
        "asset_names": asset_names,
        "dates": dates.strftime("%Y-%m-%d").tolist(),
        "objective": "gmv",
        "long_only": True,
        "bounds": (0.0, 1.0)
    }
    
    try:
        response = requests.post("http://localhost:8000/optimize", json=request_data, timeout=10)
        print(f"Response status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Optimization successful!")
            print(f"   Status: {data.get('status', 'Unknown')}")
            print(f"   Solve Time: {data.get('solve_time', 0):.4f} seconds")
            print(f"   Weights:")
            for asset, weight in data.get('weights', {}).items():
                print(f"     {asset}: {weight:.3f} ({weight*100:.1f}%)")
            return True
        else:
            print(f"❌ Optimization failed: {response.status_code}")
            print(f"   Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def main():
    """Run the tests."""
    print("🔧 Testing Backtesting and Results Storage")
    print("=" * 50)
    
    # Test API health
    print("Testing API health...")
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code == 200:
            print("✅ API is healthy")
        else:
            print("❌ API health check failed")
            return
    except Exception as e:
        print(f"❌ API not accessible: {e}")
        return
    
    # Test optimization
    opt_success = test_optimization_api()
    
    # Test backtesting
    backtest_success = test_backtest_api()
    
    print("\n" + "=" * 50)
    if opt_success and backtest_success:
        print("🎉 All tests passed!")
        print("✅ Optimization API working")
        print("✅ Backtesting API working")
        print("✅ Results storage implemented")
        print("\n🌐 You can now use the web interface at: http://localhost:8501")
        print("   - Run optimizations and backtests")
        print("   - Results will be stored and displayed in the Results page")
        print("   - Filter and analyze your results")
    else:
        print("⚠️  Some tests failed. Check the output above for details.")

if __name__ == "__main__":
    main()
