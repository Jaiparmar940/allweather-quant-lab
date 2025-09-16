"""
Tests for data loading and transformation modules.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

from engine.data.loaders import DataLoader, PriceDataLoader, MacroDataLoader
from engine.data.transforms import DataTransformer, ReturnCalculator
from engine.data.universe import UniverseManager


class TestDataLoader:
    """Test DataLoader class."""
    
    def test_data_loader_init(self):
        """Test DataLoader initialization."""
        loader = DataLoader()
        assert loader.cache_dir == Path("./data/processed")
        assert loader.cache_ttl_hours == 24
    
    def test_data_loader_init_with_params(self):
        """Test DataLoader initialization with parameters."""
        cache_dir = Path("./test_cache")
        loader = DataLoader(cache_dir=cache_dir, cache_ttl_hours=12)
        assert loader.cache_dir == cache_dir
        assert loader.cache_ttl_hours == 12


class TestPriceDataLoader:
    """Test PriceDataLoader class."""
    
    def test_price_loader_init(self):
        """Test PriceDataLoader initialization."""
        loader = PriceDataLoader()
        assert loader.cache_dir == Path("./data/processed")
        assert loader.cache_ttl_hours == 24
    
    def test_price_loader_init_with_params(self):
        """Test PriceDataLoader initialization with parameters."""
        cache_dir = Path("./test_cache")
        loader = PriceDataLoader(cache_dir=cache_dir, cache_ttl_hours=12)
        assert loader.cache_dir == cache_dir
        assert loader.cache_ttl_hours == 12


class TestMacroDataLoader:
    """Test MacroDataLoader class."""
    
    def test_macro_loader_init(self):
        """Test MacroDataLoader initialization."""
        loader = MacroDataLoader()
        assert loader.cache_dir == Path("./data/processed")
        assert loader.cache_ttl_hours == 24
        assert loader.fred is None
    
    def test_macro_loader_init_with_api_key(self):
        """Test MacroDataLoader initialization with API key."""
        loader = MacroDataLoader(api_key="test_key")
        assert loader.api_key == "test_key"
        assert loader.fred is not None


class TestDataTransformer:
    """Test DataTransformer class."""
    
    def test_validate_data(self):
        """Test data validation."""
        # Create test data with missing values
        data = pd.DataFrame({
            'A': [1, 2, np.nan, 4, 5],
            'B': [1, 2, 3, 4, 5],
            'C': [1, 2, 3, np.nan, np.nan]
        })
        
        # Test with default max_missing_pct
        result = DataTransformer.validate_data(data)
        assert len(result.columns) == 2  # Column C should be dropped
        assert result.isnull().sum().sum() == 0  # No missing values after cleaning
    
    def test_validate_data_high_missing(self):
        """Test data validation with high missing percentage."""
        # Create test data with high missing values
        data = pd.DataFrame({
            'A': [1, 2, 3, 4, 5],
            'B': [1, 2, 3, 4, 5],
            'C': [np.nan, np.nan, np.nan, np.nan, np.nan]
        })
        
        # Test with low max_missing_pct
        result = DataTransformer.validate_data(data, max_missing_pct=0.1)
        assert len(result.columns) == 2  # Column C should be dropped
    
    def test_detect_outliers(self):
        """Test outlier detection."""
        # Create test data with outliers
        data = pd.Series([1, 2, 3, 4, 5, 100, 6, 7, 8, 9, 10])
        
        outliers = DataTransformer.detect_outliers(data, threshold=3.0)
        assert outliers.sum() == 1  # One outlier detected
        assert outliers.iloc[5] == True  # The outlier at index 5
    
    def test_winsorize_data(self):
        """Test data winsorization."""
        # Create test data with extreme values
        data = pd.Series([1, 2, 3, 4, 5, 100, 6, 7, 8, 9, 10])
        
        winsorized = DataTransformer.winsorize_data(data, limits=(0.1, 0.1))
        assert winsorized.max() < data.max()  # Maximum value should be reduced
        assert winsorized.min() > data.min()  # Minimum value should be increased
    
    def test_standardize_data(self):
        """Test data standardization."""
        # Create test data
        data = pd.DataFrame({
            'A': [1, 2, 3, 4, 5],
            'B': [10, 20, 30, 40, 50]
        })
        
        # Test z-score standardization
        standardized = DataTransformer.standardize_data(data, method="zscore")
        assert np.isclose(standardized.mean().sum(), 0, atol=1e-10)
        assert np.isclose(standardized.std().sum(), 2, atol=1e-10)  # Two columns
    
    def test_detrend_data(self):
        """Test data detrending."""
        # Create test data with trend
        data = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        
        detrended = DataTransformer.detrend_data(data, method="linear")
        assert np.isclose(detrended.mean(), 0, atol=1e-10)


class TestReturnCalculator:
    """Test ReturnCalculator class."""
    
    def test_calculate_returns_log(self):
        """Test log return calculation."""
        # Create test price data
        prices = pd.DataFrame({
            'A': [100, 110, 121, 133.1, 146.41],
            'B': [50, 55, 60.5, 66.55, 73.205]
        })
        
        returns = ReturnCalculator.calculate_returns(prices, method="log")
        
        # Check that returns are approximately equal to expected log returns
        expected_returns = np.log(prices / prices.shift(1))
        assert np.allclose(returns, expected_returns, atol=1e-10)
    
    def test_calculate_returns_simple(self):
        """Test simple return calculation."""
        # Create test price data
        prices = pd.DataFrame({
            'A': [100, 110, 121, 133.1, 146.41],
            'B': [50, 55, 60.5, 66.55, 73.205]
        })
        
        returns = ReturnCalculator.calculate_returns(prices, method="simple")
        
        # Check that returns are approximately equal to expected simple returns
        expected_returns = (prices / prices.shift(1)) - 1
        assert np.allclose(returns, expected_returns, atol=1e-10)
    
    def test_calculate_volatility(self):
        """Test volatility calculation."""
        # Create test returns
        returns = pd.DataFrame({
            'A': [0.01, 0.02, -0.01, 0.03, 0.01],
            'B': [0.02, 0.01, 0.02, -0.01, 0.02]
        })
        
        vol = ReturnCalculator.calculate_volatility(returns, window=3, annualize=True)
        
        # Check that volatility is calculated correctly
        assert len(vol) == len(returns)
        assert vol.iloc[0].isna().all()  # First row should be NaN
        assert not vol.iloc[2].isna().any()  # Third row should have values
    
    def test_calculate_correlation(self):
        """Test correlation calculation."""
        # Create test returns
        returns = pd.DataFrame({
            'A': [0.01, 0.02, -0.01, 0.03, 0.01],
            'B': [0.02, 0.01, 0.02, -0.01, 0.02]
        })
        
        corr = ReturnCalculator.calculate_correlation(returns, window=3)
        
        # Check that correlation is calculated correctly
        assert len(corr) == len(returns)
        assert corr.iloc[0].isna().all().all()  # First row should be NaN
    
    def test_calculate_sharpe_ratio(self):
        """Test Sharpe ratio calculation."""
        # Create test returns
        returns = pd.Series([0.01, 0.02, -0.01, 0.03, 0.01])
        
        sharpe = ReturnCalculator.calculate_sharpe_ratio(returns, risk_free_rate=0.02, annualize=True)
        
        # Check that Sharpe ratio is calculated correctly
        assert isinstance(sharpe, float)
        assert not np.isnan(sharpe)
    
    def test_calculate_max_drawdown(self):
        """Test maximum drawdown calculation."""
        # Create test returns
        returns = pd.Series([0.01, 0.02, -0.01, 0.03, 0.01])
        
        max_dd = ReturnCalculator.calculate_max_drawdown(returns)
        
        # Check that maximum drawdown is calculated correctly
        assert isinstance(max_dd, float)
        assert max_dd <= 0  # Drawdown should be negative or zero
    
    def test_calculate_omega_ratio(self):
        """Test Omega ratio calculation."""
        # Create test returns
        returns = pd.Series([0.01, 0.02, -0.01, 0.03, 0.01])
        
        omega = ReturnCalculator.calculate_omega_ratio(returns, threshold=0.0)
        
        # Check that Omega ratio is calculated correctly
        assert isinstance(omega, float)
        assert omega > 0  # Omega ratio should be positive
    
    def test_calculate_var(self):
        """Test VaR calculation."""
        # Create test returns
        returns = pd.Series([0.01, 0.02, -0.01, 0.03, 0.01])
        
        var = ReturnCalculator.calculate_var(returns, confidence_level=0.05)
        
        # Check that VaR is calculated correctly
        assert isinstance(var, float)
        assert var <= 0  # VaR should be negative or zero
    
    def test_calculate_cvar(self):
        """Test CVaR calculation."""
        # Create test returns
        returns = pd.Series([0.01, 0.02, -0.01, 0.03, 0.01])
        
        cvar = ReturnCalculator.calculate_cvar(returns, confidence_level=0.05)
        
        # Check that CVaR is calculated correctly
        assert isinstance(cvar, float)
        assert cvar <= 0  # CVaR should be negative or zero


class TestUniverseManager:
    """Test UniverseManager class."""
    
    def test_universe_manager_init(self):
        """Test UniverseManager initialization."""
        manager = UniverseManager()
        assert manager.config_path == Path("configs/universe.yaml")
        assert isinstance(manager.config, dict)
    
    def test_get_all_tickers(self):
        """Test getting all tickers."""
        manager = UniverseManager()
        tickers = manager.get_all_tickers()
        
        assert isinstance(tickers, list)
        assert len(tickers) > 0
    
    def test_get_tickers_by_category(self):
        """Test getting tickers by category."""
        manager = UniverseManager()
        tickers = manager.get_tickers_by_category("us_equity")
        
        assert isinstance(tickers, list)
    
    def test_get_tickers_by_sector(self):
        """Test getting tickers by sector."""
        manager = UniverseManager()
        tickers = manager.get_tickers_by_sector("technology")
        
        assert isinstance(tickers, list)
    
    def test_get_universe_for_year(self):
        """Test getting universe for a specific year."""
        manager = UniverseManager()
        tickers = manager.get_universe_for_year(2023)
        
        assert isinstance(tickers, list)
        assert len(tickers) > 0
    
    def test_get_universe_for_date(self):
        """Test getting universe for a specific date."""
        manager = UniverseManager()
        date = datetime(2023, 6, 15)
        tickers = manager.get_universe_for_date(date)
        
        assert isinstance(tickers, list)
        assert len(tickers) > 0
    
    def test_validate_tickers(self):
        """Test ticker validation."""
        manager = UniverseManager()
        all_tickers = manager.get_all_tickers()
        
        # Test with valid tickers
        valid_tickers, invalid_tickers = manager.validate_tickers(all_tickers[:5])
        assert len(valid_tickers) == 5
        assert len(invalid_tickers) == 0
        
        # Test with invalid tickers
        invalid_ticker_list = ["INVALID1", "INVALID2", "INVALID3"]
        valid_tickers, invalid_tickers = manager.validate_tickers(invalid_ticker_list)
        assert len(valid_tickers) == 0
        assert len(invalid_tickers) == 3
    
    def test_get_benchmark_ticker(self):
        """Test getting benchmark ticker."""
        manager = UniverseManager()
        benchmark = manager.get_benchmark_ticker()
        
        assert isinstance(benchmark, str)
        assert benchmark == "ALLW"
    
    def test_get_rebalancing_schedule(self):
        """Test getting rebalancing schedule."""
        manager = UniverseManager()
        schedule = manager.get_rebalancing_schedule()
        
        assert isinstance(schedule, dict)
        assert "frequency" in schedule
    
    def test_get_liquidity_requirements(self):
        """Test getting liquidity requirements."""
        manager = UniverseManager()
        liquidity = manager.get_liquidity_requirements()
        
        assert isinstance(liquidity, dict)
    
    def test_get_universe_freeze_policy(self):
        """Test getting universe freeze policy."""
        manager = UniverseManager()
        policy = manager.get_universe_freeze_policy()
        
        assert isinstance(policy, dict)
        assert "freeze_date" in policy


if __name__ == "__main__":
    pytest.main([__file__])
