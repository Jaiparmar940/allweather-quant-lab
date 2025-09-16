#!/usr/bin/env python3
"""
Omega Portfolio Engine Demo

This script demonstrates the core functionality of the Omega Portfolio Engine.
"""

import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from engine import GMVOptimizer, OmegaOptimizer
from engine.signals import HMMRegimeDetector
from engine.data import DataLoader
from engine.backtest import BacktestSimulator, WalkForwardEngine


def create_sample_data():
    """Create sample data for demonstration."""
    print("Creating sample data...")
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Create sample returns
    n_days = 252  # One year of trading days
    n_assets = 5
    
    # Generate correlated returns
    mean_returns = np.array([0.08, 0.10, 0.12, 0.06, 0.04])  # Annual returns
    volatilities = np.array([0.15, 0.20, 0.25, 0.10, 0.08])  # Annual volatilities
    
    # Create correlation matrix
    corr_matrix = np.array([
        [1.00, 0.70, 0.60, 0.30, 0.20],
        [0.70, 1.00, 0.80, 0.40, 0.30],
        [0.60, 0.80, 1.00, 0.50, 0.40],
        [0.30, 0.40, 0.50, 1.00, 0.60],
        [0.20, 0.30, 0.40, 0.60, 1.00]
    ])
    
    # Convert to covariance matrix
    cov_matrix = np.outer(volatilities, volatilities) * corr_matrix
    
    # Generate returns
    returns = np.random.multivariate_normal(
        mean_returns / 252,  # Daily returns
        cov_matrix / 252,    # Daily covariance
        n_days
    )
    
    # Create DataFrame
    asset_names = ['US_Equity', 'Intl_Equity', 'EM_Equity', 'Bonds', 'REITs']
    dates = pd.date_range(start='2023-01-01', periods=n_days, freq='D')
    
    returns_df = pd.DataFrame(returns, index=dates, columns=asset_names)
    
    print(f"Generated {n_days} days of returns for {n_assets} assets")
    print(f"Date range: {dates[0].date()} to {dates[-1].date()}")
    
    return returns_df


def demo_gmv_optimization():
    """Demonstrate GMV optimization."""
    print("\n" + "="*60)
    print("GMV OPTIMIZATION DEMO")
    print("="*60)
    
    # Create sample data
    returns = create_sample_data()
    
    # Initialize GMV optimizer
    gmv_optimizer = GMVOptimizer()
    
    # Solve GMV optimization
    print("\nSolving GMV optimization...")
    result = gmv_optimizer.solve_gmv_with_returns(returns)
    
    if result['status'] == 'optimal':
        print(f"✓ GMV optimization successful!")
        print(f"  Objective value: {result['objective_value']:.4f}")
        print(f"  Solve time: {result['solve_time']:.4f} seconds")
        print(f"  Weights:")
        for asset, weight in zip(returns.columns, result['weights']):
            print(f"    {asset:12s}: {weight:6.3f} ({weight*100:5.1f}%)")
        
        # Calculate portfolio statistics
        portfolio_return = np.dot(result['weights'], returns.mean()) * 252
        portfolio_vol = np.sqrt(np.dot(result['weights'], np.dot(returns.cov() * 252, result['weights'])))
        sharpe_ratio = portfolio_return / portfolio_vol
        
        print(f"\n  Portfolio Statistics:")
        print(f"    Annual Return: {portfolio_return:.2%}")
        print(f"    Annual Volatility: {portfolio_vol:.2%}")
        print(f"    Sharpe Ratio: {sharpe_ratio:.3f}")
        
    else:
        print(f"✗ GMV optimization failed: {result['status']}")


def demo_omega_optimization():
    """Demonstrate Omega optimization."""
    print("\n" + "="*60)
    print("OMEGA OPTIMIZATION DEMO")
    print("="*60)
    
    # Create sample data
    returns = create_sample_data()
    
    # Initialize Omega optimizer
    omega_optimizer = OmegaOptimizer()
    
    # Solve Omega optimization
    print("\nSolving Omega optimization...")
    result = omega_optimizer.solve_omega_with_returns(returns, theta=0.02)
    
    if result['status'] == 'optimal':
        print(f"✓ Omega optimization successful!")
        print(f"  Objective value: {result['objective_value']:.4f}")
        print(f"  Solve time: {result['solve_time']:.4f} seconds")
        print(f"  Weights:")
        for asset, weight in zip(returns.columns, result['weights']):
            print(f"    {asset:12s}: {weight:6.3f} ({weight*100:5.1f}%)")
        
        # Calculate portfolio statistics
        portfolio_return = np.dot(result['weights'], returns.mean()) * 252
        portfolio_vol = np.sqrt(np.dot(result['weights'], np.dot(returns.cov() * 252, result['weights'])))
        sharpe_ratio = portfolio_return / portfolio_vol
        
        print(f"\n  Portfolio Statistics:")
        print(f"    Annual Return: {portfolio_return:.2%}")
        print(f"    Annual Volatility: {portfolio_vol:.2%}")
        print(f"    Sharpe Ratio: {sharpe_ratio:.3f}")
        
    else:
        print(f"✗ Omega optimization failed: {result['status']}")


def demo_regime_detection():
    """Demonstrate regime detection."""
    print("\n" + "="*60)
    print("REGIME DETECTION DEMO")
    print("="*60)
    
    # Create sample data
    returns = create_sample_data()
    
    # Initialize regime detector
    regime_detector = HMMRegimeDetector(n_regimes=3)
    
    # Detect regimes
    print("\nDetecting market regimes...")
    regime_labels = regime_detector.fit_predict(returns)
    
    print(f"✓ Regime detection completed!")
    print(f"  Number of regimes: {len(np.unique(regime_labels))}")
    print(f"  Regime distribution:")
    for regime in np.unique(regime_labels):
        count = np.sum(regime_labels == regime)
        percentage = count / len(regime_labels) * 100
        print(f"    Regime {regime}: {count:3d} days ({percentage:5.1f}%)")
    
    # Analyze regime characteristics
    print(f"\n  Regime Characteristics:")
    for regime in np.unique(regime_labels):
        regime_returns = returns[regime_labels == regime]
        regime_mean = regime_returns.mean().mean() * 252
        regime_vol = regime_returns.std().mean() * np.sqrt(252)
        regime_sharpe = regime_mean / regime_vol
        
        print(f"    Regime {regime}:")
        print(f"      Mean Return: {regime_mean:.2%}")
        print(f"      Volatility:  {regime_vol:.2%}")
        print(f"      Sharpe:      {regime_sharpe:.3f}")


def demo_backtesting():
    """Demonstrate backtesting."""
    print("\n" + "="*60)
    print("BACKTESTING DEMO")
    print("="*60)
    
    # Create sample data
    returns = create_sample_data()
    
    # Define a simple equal-weight strategy
    def equal_weight_strategy(returns):
        n_assets = len(returns.columns)
        return pd.Series(1.0 / n_assets, index=returns.columns)
    
    # Initialize walk-forward engine
    walk_forward_engine = WalkForwardEngine(
        train_months=6,
        test_months=1,
        step_months=1
    )
    
    # Run walk-forward backtest
    print("\nRunning walk-forward backtest...")
    results = walk_forward_engine.run_walk_forward(returns, equal_weight_strategy)
    
    print(f"✓ Walk-forward backtest completed!")
    
    if 'period_results' in results and len(results['period_results']) > 0:
        print(f"  Number of periods: {len(results['period_results'])}")
        print(f"  Overall return: {results['overall_return']:.2%}")
        print(f"  Overall volatility: {results['overall_volatility']:.2%}")
        print(f"  Overall Sharpe: {results['overall_sharpe']:.3f}")
        print(f"  Max drawdown: {results['overall_max_drawdown']:.2%}")
    else:
        print(f"  No periods completed (insufficient training data)")
        print(f"  This is expected for the short demo dataset")


def main():
    """Run the complete demo."""
    print("OMEGA PORTFOLIO ENGINE DEMO")
    print("="*60)
    print("This demo showcases the core functionality of the Omega Portfolio Engine.")
    print("It demonstrates portfolio optimization, regime detection, and backtesting.")
    
    try:
        # Run demos
        demo_gmv_optimization()
        demo_omega_optimization()
        demo_regime_detection()
        demo_backtesting()
        
        print("\n" + "="*60)
        print("DEMO COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("The Omega Portfolio Engine is working correctly.")
        print("You can now use the API or UI to explore more features.")
        
    except Exception as e:
        print(f"\n✗ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
