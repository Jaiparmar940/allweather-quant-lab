#!/usr/bin/env python3
"""
Test script for the Omega Portfolio Engine API
"""

import requests
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# API base URL
BASE_URL = "http://localhost:8000"

def test_health():
    """Test the health endpoint."""
    print("Testing health endpoint...")
    response = requests.get(f"{BASE_URL}/health")
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Health check passed: {data['status']}")
        print(f"   Version: {data['version']}")
        print(f"   Memory usage: {data['memory_usage']:.1f} MB")
        print(f"   CPU usage: {data['cpu_usage']:.1f}%")
        return True
    else:
        print(f"❌ Health check failed: {response.status_code}")
        return False

def test_gmv_optimization():
    """Test GMV optimization."""
    print("\nTesting GMV optimization...")
    
    # Create sample data
    np.random.seed(42)
    returns = np.random.normal(0.001, 0.02, (100, 5))
    asset_names = ["US_Equity", "Intl_Equity", "EM_Equity", "Bonds", "REITs"]
    dates = pd.date_range('2023-01-01', periods=100, freq='D').strftime('%Y-%m-%d').tolist()
    
    payload = {
        "returns": returns.tolist(),
        "asset_names": asset_names,
        "dates": dates,
        "objective": "gmv",
        "long_only": True
    }
    
    response = requests.post(f"{BASE_URL}/optimize", json=payload)
    if response.status_code == 200:
        data = response.json()
        print(f"✅ GMV optimization successful")
        print(f"   Status: {data['status']}")
        print(f"   Solve time: {data['solve_time']:.4f} seconds")
        print(f"   Weights:")
        for asset, weight in data['weights'].items():
            print(f"     {asset}: {weight:.3f} ({weight*100:.1f}%)")
        return True
    else:
        print(f"❌ GMV optimization failed: {response.status_code}")
        print(f"   Error: {response.text}")
        return False

def test_omega_optimization():
    """Test Omega optimization."""
    print("\nTesting Omega optimization...")
    
    # Create sample data
    np.random.seed(42)
    returns = np.random.normal(0.001, 0.02, (100, 5))
    asset_names = ["US_Equity", "Intl_Equity", "EM_Equity", "Bonds", "REITs"]
    dates = pd.date_range('2023-01-01', periods=100, freq='D').strftime('%Y-%m-%d').tolist()
    
    payload = {
        "returns": returns.tolist(),
        "asset_names": asset_names,
        "dates": dates,
        "objective": "omega",
        "theta": 0.02,
        "long_only": True
    }
    
    response = requests.post(f"{BASE_URL}/optimize", json=payload)
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Omega optimization successful")
        print(f"   Status: {data['status']}")
        print(f"   Solve time: {data['solve_time']:.4f} seconds")
        print(f"   Weights:")
        for asset, weight in data['weights'].items():
            print(f"     {asset}: {weight:.3f} ({weight*100:.1f}%)")
        return True
    else:
        print(f"❌ Omega optimization failed: {response.status_code}")
        print(f"   Error: {response.text}")
        return False

def test_regime_detection():
    """Test regime detection."""
    print("\nTesting regime detection...")
    
    # Create sample features
    np.random.seed(42)
    features = np.random.normal(0, 1, (50, 5))
    feature_names = ["volatility", "momentum", "value", "quality", "size"]
    dates = pd.date_range('2023-01-01', periods=50, freq='D').strftime('%Y-%m-%d').tolist()
    
    payload = {
        "features": features.tolist(),
        "feature_names": feature_names,
        "dates": dates,
        "method": "hmm",
        "n_regimes": 3
    }
    
    response = requests.post(f"{BASE_URL}/regime", json=payload)
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Regime detection successful")
        print(f"   Method: {data['method']}")
        print(f"   Number of regimes: {data['n_regimes']}")
        print(f"   Observations: {data['n_observations']}")
        print(f"   Regime distribution:")
        for regime, stats in data['regime_characteristics'].items():
            print(f"     {regime}: {stats['count']} observations ({stats['percentage']:.1f}%)")
        return True
    else:
        print(f"❌ Regime detection failed: {response.status_code}")
        print(f"   Error: {response.text}")
        return False

def test_backtest():
    """Test backtesting."""
    print("\nTesting backtesting...")
    
    # Create sample data
    np.random.seed(42)
    returns = np.random.normal(0.001, 0.02, (100, 5))
    asset_names = ["US_Equity", "Intl_Equity", "EM_Equity", "Bonds", "REITs"]
    dates = pd.date_range('2023-01-01', periods=100, freq='D').strftime('%Y-%m-%d').tolist()
    
    payload = {
        "returns": returns.tolist(),
        "asset_names": asset_names,
        "dates": dates,
        "objective": "gmv",
        "initial_capital": 1000000.0,
        "train_months": 6,
        "test_months": 1,
        "step_months": 1,
        "transaction_costs": 0.0005,
        "slippage": 0.0002,
        "rebalance_frequency": "monthly",
        "long_only": True
    }
    
    response = requests.post(f"{BASE_URL}/backtest", json=payload)
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Backtesting successful")
        print(f"   Total return: {data['total_return']:.2%}")
        print(f"   Annualized return: {data['annualized_return']:.2%}")
        print(f"   Annualized volatility: {data['annualized_volatility']:.2%}")
        print(f"   Sharpe ratio: {data['sharpe_ratio']:.3f}")
        print(f"   Max drawdown: {data['max_drawdown']:.2%}")
        return True
    else:
        print(f"❌ Backtesting failed: {response.status_code}")
        print(f"   Error: {response.text}")
        return False

def main():
    """Run all API tests."""
    print("🚀 Omega Portfolio Engine API Test Suite")
    print("=" * 50)
    
    tests = [
        test_health,
        test_gmv_optimization,
        test_omega_optimization,
        test_regime_detection,
        test_backtest
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The API is working correctly.")
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
    
    print("\n🌐 Access the web interface at: http://localhost:8501")
    print("📚 API documentation at: http://localhost:8000/docs")

if __name__ == "__main__":
    main()
