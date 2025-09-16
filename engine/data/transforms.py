"""
Data transformation utilities for portfolio optimization.
"""

import logging
from typing import Dict, List, Optional, Union, Tuple
import pandas as pd
import numpy as np
from scipy import stats
import structlog

logger = structlog.get_logger(__name__)


class DataTransformer:
    """Data transformation utilities."""
    
    @staticmethod
    def validate_data(data: pd.DataFrame, max_missing_pct: float = 0.05) -> pd.DataFrame:
        """Validate data quality and handle missing values."""
        
        logger.info("Validating data quality", shape=data.shape, max_missing_pct=max_missing_pct)
        
        # Check for excessive missing data
        missing_pct = data.isnull().sum() / len(data)
        high_missing_cols = missing_pct[missing_pct > max_missing_pct].index.tolist()
        
        if high_missing_cols:
            logger.warning("Columns with high missing data", columns=high_missing_cols, missing_pct=missing_pct[high_missing_cols].to_dict())
            # Drop columns with too much missing data
            data = data.drop(columns=high_missing_cols)
        
        # Fill remaining missing values
        data = data.fillna(method='ffill').fillna(method='bfill')
        
        # Check for infinite values
        inf_cols = data.isin([np.inf, -np.inf]).any()
        if inf_cols.any():
            logger.warning("Columns with infinite values", columns=inf_cols[inf_cols].index.tolist())
            data = data.replace([np.inf, -np.inf], np.nan)
            data = data.fillna(method='ffill').fillna(method='bfill')
        
        logger.info("Data validation completed", final_shape=data.shape)
        return data
    
    @staticmethod
    def detect_outliers(data: pd.Series, threshold: float = 3.0) -> pd.Series:
        """Detect outliers using Z-score method."""
        
        z_scores = np.abs(stats.zscore(data.dropna()))
        outliers = z_scores > threshold
        
        if outliers.any():
            logger.warning("Outliers detected", count=outliers.sum(), threshold=threshold)
        
        return outliers
    
    @staticmethod
    def winsorize_data(data: pd.Series, limits: Tuple[float, float] = (0.01, 0.01)) -> pd.Series:
        """Winsorize data to limit extreme values."""
        
        from scipy.stats import mstats
        
        winsorized = mstats.winsorize(data.dropna(), limits=limits)
        result = data.copy()
        result.loc[data.notna()] = winsorized
        
        logger.info("Data winsorized", limits=limits, original_std=data.std(), winsorized_std=result.std())
        return result
    
    @staticmethod
    def standardize_data(data: pd.DataFrame, method: str = "zscore") -> pd.DataFrame:
        """Standardize data using specified method."""
        
        if method == "zscore":
            return (data - data.mean()) / data.std()
        elif method == "minmax":
            return (data - data.min()) / (data.max() - data.min())
        elif method == "robust":
            return (data - data.median()) / data.mad()
        else:
            raise ValueError(f"Unknown standardization method: {method}")
    
    @staticmethod
    def detrend_data(data: pd.Series, method: str = "linear") -> pd.Series:
        """Remove trend from time series data."""
        
        if method == "linear":
            from scipy import signal
            detrended = signal.detrend(data.dropna())
            result = data.copy()
            result.loc[data.notna()] = detrended
            return result
        elif method == "hp":
            # Hodrick-Prescott filter
            from statsmodels.tsa.filters.hp_filter import hpfilter
            cycle, trend = hpfilter(data.dropna(), lamb=1600)
            return cycle
        else:
            raise ValueError(f"Unknown detrending method: {method}")


class ReturnCalculator:
    """Calculate various types of returns and risk metrics."""
    
    @staticmethod
    def calculate_returns(
        prices: pd.DataFrame,
        method: str = "log",
        frequency: str = "daily"
    ) -> pd.DataFrame:
        """Calculate returns from price data."""
        
        logger.info("Calculating returns", method=method, frequency=frequency, shape=prices.shape)
        
        if method == "log":
            returns = np.log(prices / prices.shift(1))
        elif method == "simple":
            returns = (prices / prices.shift(1)) - 1
        else:
            raise ValueError(f"Unknown return method: {method}")
        
        # Handle frequency conversion
        if frequency == "monthly":
            returns = returns.resample("M").sum()
        elif frequency == "quarterly":
            returns = returns.resample("Q").sum()
        elif frequency == "annual":
            returns = returns.resample("Y").sum()
        
        # Remove first row (NaN)
        returns = returns.dropna()
        
        logger.info("Returns calculated successfully", shape=returns.shape)
        return returns
    
    @staticmethod
    def calculate_volatility(
        returns: pd.DataFrame,
        window: int = 252,
        annualize: bool = True
    ) -> pd.DataFrame:
        """Calculate rolling volatility."""
        
        vol = returns.rolling(window=window).std()
        
        if annualize:
            vol = vol * np.sqrt(252)  # Annualize daily volatility
        
        return vol
    
    @staticmethod
    def calculate_correlation(
        returns: pd.DataFrame,
        window: int = 252
    ) -> pd.DataFrame:
        """Calculate rolling correlation matrix."""
        
        return returns.rolling(window=window).corr()
    
    @staticmethod
    def calculate_covariance(
        returns: pd.DataFrame,
        window: int = 252,
        annualize: bool = True
    ) -> pd.DataFrame:
        """Calculate rolling covariance matrix."""
        
        cov = returns.rolling(window=window).cov()
        
        if annualize:
            cov = cov * 252  # Annualize daily covariance
        
        return cov
    
    @staticmethod
    def calculate_sharpe_ratio(
        returns: pd.Series,
        risk_free_rate: float = 0.02,
        annualize: bool = True
    ) -> float:
        """Calculate Sharpe ratio."""
        
        excess_returns = returns - risk_free_rate / 252  # Daily risk-free rate
        
        if annualize:
            return excess_returns.mean() * 252 / (returns.std() * np.sqrt(252))
        else:
            return excess_returns.mean() / returns.std()
    
    @staticmethod
    def calculate_sortino_ratio(
        returns: pd.Series,
        risk_free_rate: float = 0.02,
        annualize: bool = True
    ) -> float:
        """Calculate Sortino ratio."""
        
        excess_returns = returns - risk_free_rate / 252  # Daily risk-free rate
        downside_returns = excess_returns[excess_returns < 0]
        
        if len(downside_returns) == 0:
            return np.inf
        
        downside_std = downside_returns.std()
        
        if annualize:
            return excess_returns.mean() * 252 / (downside_std * np.sqrt(252))
        else:
            return excess_returns.mean() / downside_std
    
    @staticmethod
    def calculate_max_drawdown(returns: pd.Series) -> float:
        """Calculate maximum drawdown."""
        
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        
        return drawdown.min()
    
    @staticmethod
    def calculate_calmar_ratio(
        returns: pd.Series,
        annualize: bool = True
    ) -> float:
        """Calculate Calmar ratio."""
        
        if annualize:
            annual_return = returns.mean() * 252
        else:
            annual_return = returns.mean()
        
        max_dd = abs(ReturnCalculator.calculate_max_drawdown(returns))
        
        if max_dd == 0:
            return np.inf
        
        return annual_return / max_dd
    
    @staticmethod
    def calculate_omega_ratio(
        returns: pd.Series,
        threshold: float = 0.0,
        annualize: bool = True
    ) -> float:
        """Calculate Omega ratio."""
        
        if annualize:
            threshold = threshold / 252  # Convert annual threshold to daily
        
        excess_returns = returns - threshold
        
        positive_returns = excess_returns[excess_returns > 0]
        negative_returns = excess_returns[excess_returns < 0]
        
        if len(negative_returns) == 0:
            return np.inf
        
        positive_sum = positive_returns.sum()
        negative_sum = abs(negative_returns.sum())
        
        return positive_sum / negative_sum
    
    @staticmethod
    def calculate_var(
        returns: pd.Series,
        confidence_level: float = 0.05
    ) -> float:
        """Calculate Value at Risk (VaR)."""
        
        return returns.quantile(confidence_level)
    
    @staticmethod
    def calculate_cvar(
        returns: pd.Series,
        confidence_level: float = 0.05
    ) -> float:
        """Calculate Conditional Value at Risk (CVaR)."""
        
        var = ReturnCalculator.calculate_var(returns, confidence_level)
        return returns[returns <= var].mean()
    
    @staticmethod
    def calculate_tail_ratio(returns: pd.Series) -> float:
        """Calculate tail ratio (95th percentile / 5th percentile)."""
        
        p95 = returns.quantile(0.95)
        p5 = returns.quantile(0.05)
        
        if p5 == 0:
            return np.inf
        
        return p95 / abs(p5)
    
    @staticmethod
    def calculate_hit_rate(
        returns: pd.Series,
        threshold: float = 0.0
    ) -> float:
        """Calculate hit rate (percentage of positive returns above threshold)."""
        
        return (returns > threshold).mean()
    
    @staticmethod
    def calculate_turnover(weights: pd.DataFrame) -> pd.Series:
        """Calculate portfolio turnover."""
        
        return weights.diff().abs().sum(axis=1)
    
    @staticmethod
    def calculate_cost_drag(
        turnover: pd.Series,
        cost_bps: float = 5.0
    ) -> float:
        """Calculate cost drag from turnover."""
        
        return (turnover * cost_bps / 10000).mean()
    
    @staticmethod
    def calculate_tax_drag(
        turnover: pd.Series,
        tax_rate: float = 0.20
    ) -> float:
        """Calculate tax drag from turnover."""
        
        return (turnover * tax_rate).mean()
