#!/usr/bin/env python3
"""
Download sample data for the Omega Portfolio Engine demo.
"""

import os
import sys
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

def download_sample_data():
    """Download sample data for demonstration."""
    
    print("Downloading sample data for Omega Portfolio Engine...")
    
    # Create data directories
    data_dir = Path("data/processed")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Define sample universe
    tickers = [
        'VTI',   # Vanguard Total Stock Market ETF
        'SPY',   # SPDR S&P 500 ETF Trust
        'QQQ',   # Invesco QQQ Trust
        'IWM',   # iShares Russell 2000 ETF
        'VEA',   # Vanguard FTSE Developed Markets ETF
        'VWO',   # Vanguard FTSE Emerging Markets ETF
        'SHY',   # iShares 1-3 Year Treasury Bond ETF
        'IEF',   # iShares 7-10 Year Treasury Bond ETF
        'TLT',   # iShares 20+ Year Treasury Bond ETF
        'TIP',   # iShares TIPS Bond ETF
        'GLD',   # SPDR Gold Shares
        'SLV',   # iShares Silver Trust
        'VNQ',   # Vanguard Real Estate ETF
    ]
    
    # Set date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=1000)  # About 3 years of data
    
    print(f"Downloading data for {len(tickers)} assets from {start_date.date()} to {end_date.date()}")
    
    try:
        # Download price data
        data = yf.download(tickers, start=start_date, end=end_date, progress=True)
        
        # Extract adjusted close prices
        if len(tickers) == 1:
            prices = data['Adj Close']
        else:
            prices = data['Adj Close']
        
        # Calculate returns
        returns = prices.pct_change().dropna()
        
        # Save data
        prices.to_parquet(data_dir / "sample_prices.parquet")
        returns.to_parquet(data_dir / "sample_returns.parquet")
        
        print(f"Data saved to:")
        print(f"  - {data_dir / 'sample_prices.parquet'}")
        print(f"  - {data_dir / 'sample_returns.parquet'}")
        print(f"  - Shape: {prices.shape}")
        print(f"  - Date range: {prices.index[0].date()} to {prices.index[-1].date()}")
        
        # Download some macro data
        macro_tickers = ['^VIX', '^TNX', '^FVX', '^TYX']  # VIX, 10Y, 5Y, 30Y yields
        macro_data = yf.download(macro_tickers, start=start_date, end=end_date, progress=True)
        
        if len(macro_tickers) == 1:
            macro_prices = macro_data['Adj Close']
        else:
            macro_prices = macro_data['Adj Close']
        
        macro_prices.to_parquet(data_dir / "sample_macro.parquet")
        print(f"  - {data_dir / 'sample_macro.parquet'}")
        
        print("\nSample data download completed successfully!")
        
    except Exception as e:
        print(f"Error downloading data: {e}")
        sys.exit(1)

if __name__ == "__main__":
    download_sample_data()
