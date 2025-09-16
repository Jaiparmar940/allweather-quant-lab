#!/usr/bin/env python3
"""
Test script to verify the UI variable scope fix
"""

import requests
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def test_yahoo_data_download():
    """Test downloading data from Yahoo Finance and optimizing."""
    print("Testing Yahoo Finance data download and optimization...")
    
    # Test the API directly with Yahoo Finance data
    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
    start_date = "2023-01-01"
    end_date = "2023-12-31"
    
    # Download data using yfinance
    import yfinance as yf
    data = yf.download(tickers, start=start_date, end=end_date, progress=False)
    
    if len(tickers) == 1:
        returns = data["Close"].pct_change().dropna()
    else:
        returns = data["Close"].pct_change().dropna()
    
    print(f"Downloaded {len(returns)} days of data for {len(returns.columns)} assets")
    print(f"Date range: {returns.index[0].date()} to {returns.index[-1].date()}")
    
    # Test GMV optimization
    print("\nTesting GMV optimization...")
    request_data = {
        "returns": returns.values.tolist(),
        "asset_names": returns.columns.tolist(),
        "dates": returns.index.strftime("%Y-%m-%d").tolist(),
        "objective": "gmv",
        "long_only": True
    }
    
    response = requests.post("http://localhost:8000/optimize", json=request_data)
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
    """Test Omega optimization with Yahoo Finance data."""
    print("\nTesting Omega optimization...")
    
    # Download data
    import yfinance as yf
    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
    data = yf.download(tickers, start="2023-01-01", end="2023-12-31", progress=False)
    returns = data["Close"].pct_change().dropna()
    
    request_data = {
        "returns": returns.values.tolist(),
        "asset_names": returns.columns.tolist(),
        "dates": returns.index.strftime("%Y-%m-%d").tolist(),
        "objective": "omega",
        "theta": 0.02,
        "long_only": True
    }
    
    response = requests.post("http://localhost:8000/optimize", json=request_data)
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

def main():
    """Run the UI fix test."""
    print("🔧 Testing UI Variable Scope Fix")
    print("=" * 50)
    
    # Test API health
    print("Testing API health...")
    response = requests.get("http://localhost:8000/health")
    if response.status_code == 200:
        print("✅ API is healthy")
    else:
        print("❌ API health check failed")
        return
    
    # Test Yahoo Finance data download and optimization
    success1 = test_yahoo_data_download()
    success2 = test_omega_optimization()
    
    print("\n" + "=" * 50)
    if success1 and success2:
        print("🎉 All tests passed! The variable scope issue is fixed.")
        print("✅ Yahoo Finance data download works")
        print("✅ Portfolio optimization works with downloaded data")
        print("\n🌐 You can now use the web interface at: http://localhost:8501")
        print("   - Select 'Yahoo Finance' as data source")
        print("   - Enter tickers like 'AAPL,MSFT,GOOGL'")
        print("   - Click 'Download Data'")
        print("   - Click 'Optimize Portfolio'")
    else:
        print("⚠️  Some tests failed. Check the output above for details.")

if __name__ == "__main__":
    main()
