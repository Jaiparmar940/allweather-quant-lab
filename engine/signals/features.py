"""
Feature extraction for regime detection and portfolio optimization.
"""

import logging
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy import stats
import structlog

logger = structlog.get_logger(__name__)


class FeatureExtractor:
    """Extract features for regime detection and portfolio optimization."""
    
    def __init__(self, lookback_window: int = 252):
        self.lookback_window = lookback_window
    
    def extract_market_features(
        self,
        returns: pd.DataFrame,
        prices: pd.DataFrame,
        macro_data: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """Extract comprehensive market features."""
        
        logger.info("Extracting market features", n_assets=len(returns.columns))
        
        features = pd.DataFrame(index=returns.index)
        
        # Price-based features
        price_features = self._extract_price_features(returns, prices)
        features = pd.concat([features, price_features], axis=1)
        
        # Volatility features
        vol_features = self._extract_volatility_features(returns)
        features = pd.concat([features, vol_features], axis=1)
        
        # Correlation features
        corr_features = self._extract_correlation_features(returns)
        features = pd.concat([features, corr_features], axis=1)
        
        # Momentum features
        momentum_features = self._extract_momentum_features(returns, prices)
        features = pd.concat([features, momentum_features], axis=1)
        
        # Macro features
        if macro_data is not None:
            macro_features = self._extract_macro_features(macro_data)
            features = pd.concat([features, macro_features], axis=1)
        
        # Clean features
        features = features.fillna(method='ffill').fillna(method='bfill')
        
        logger.info("Market features extracted", n_features=len(features.columns))
        return features
    
    def _extract_price_features(
        self,
        returns: pd.DataFrame,
        prices: pd.DataFrame
    ) -> pd.DataFrame:
        """Extract price-based features."""
        
        features = pd.DataFrame(index=returns.index)
        
        # Realized volatility
        features['realized_vol'] = returns.rolling(window=21).std() * np.sqrt(252)
        
        # Price momentum
        features['momentum_1m'] = returns.rolling(window=21).sum()
        features['momentum_3m'] = returns.rolling(window=63).sum()
        features['momentum_6m'] = returns.rolling(window=126).sum()
        features['momentum_12m'] = returns.rolling(window=252).sum()
        
        # Price trends
        features['trend_short'] = (prices / prices.rolling(window=21).mean() - 1).mean(axis=1)
        features['trend_medium'] = (prices / prices.rolling(window=63).mean() - 1).mean(axis=1)
        features['trend_long'] = (prices / prices.rolling(window=252).mean() - 1).mean(axis=1)
        
        # Price levels
        features['price_level'] = (prices / prices.rolling(window=252).mean() - 1).mean(axis=1)
        
        return features
    
    def _extract_volatility_features(self, returns: pd.DataFrame) -> pd.DataFrame:
        """Extract volatility-based features."""
        
        features = pd.DataFrame(index=returns.index)
        
        # Rolling volatility
        features['vol_1m'] = returns.rolling(window=21).std() * np.sqrt(252)
        features['vol_3m'] = returns.rolling(window=63).std() * np.sqrt(252)
        features['vol_6m'] = returns.rolling(window=126).std() * np.sqrt(252)
        features['vol_12m'] = returns.rolling(window=252).std() * np.sqrt(252)
        
        # Volatility of volatility
        features['vol_of_vol'] = features['vol_1m'].rolling(window=21).std()
        
        # Volatility regime
        features['vol_regime'] = (features['vol_1m'] > features['vol_1m'].rolling(window=63).mean()).astype(int)
        
        # Volatility clustering
        features['vol_clustering'] = features['vol_1m'].rolling(window=21).skew()
        
        return features
    
    def _extract_correlation_features(self, returns: pd.DataFrame) -> pd.DataFrame:
        """Extract correlation-based features."""
        
        features = pd.DataFrame(index=returns.index)
        
        # Average correlation
        features['avg_correlation'] = returns.rolling(window=63).corr().groupby(level=0).mean().mean(axis=1)
        
        # Correlation regime
        features['corr_regime'] = (features['avg_correlation'] > features['avg_correlation'].rolling(window=63).mean()).astype(int)
        
        # Correlation stability
        features['corr_stability'] = features['avg_correlation'].rolling(window=21).std()
        
        # Cross-asset correlation
        if len(returns.columns) > 1:
            # Calculate pairwise correlations
            corr_matrix = returns.rolling(window=63).corr()
            
            # Extract upper triangle correlations
            upper_tri_indices = np.triu_indices(len(returns.columns), k=1)
            corr_values = []
            
            for i, j in zip(upper_tri_indices[0], upper_tri_indices[1]):
                corr_series = corr_matrix.loc[(slice(None), returns.columns[i]), returns.columns[j]]
                corr_values.append(corr_series)
            
            if corr_values:
                features['cross_corr_mean'] = pd.concat(corr_values, axis=1).mean(axis=1)
                features['cross_corr_std'] = pd.concat(corr_values, axis=1).std(axis=1)
        
        return features
    
    def _extract_momentum_features(
        self,
        returns: pd.DataFrame,
        prices: pd.DataFrame
    ) -> pd.DataFrame:
        """Extract momentum-based features."""
        
        features = pd.DataFrame(index=returns.index)
        
        # Price momentum
        features['momentum_1m'] = returns.rolling(window=21).sum()
        features['momentum_3m'] = returns.rolling(window=63).sum()
        features['momentum_6m'] = returns.rolling(window=126).sum()
        features['momentum_12m'] = returns.rolling(window=252).sum()
        
        # Momentum consistency
        features['momentum_consistency'] = (
            (features['momentum_1m'] > 0).astype(int) +
            (features['momentum_3m'] > 0).astype(int) +
            (features['momentum_6m'] > 0).astype(int) +
            (features['momentum_12m'] > 0).astype(int)
        ) / 4
        
        # Momentum strength
        features['momentum_strength'] = (
            features['momentum_1m'] * 0.4 +
            features['momentum_3m'] * 0.3 +
            features['momentum_6m'] * 0.2 +
            features['momentum_12m'] * 0.1
        )
        
        # Momentum reversal
        features['momentum_reversal'] = (
            (features['momentum_1m'] * features['momentum_3m'] < 0).astype(int)
        )
        
        return features
    
    def _extract_macro_features(self, macro_data: pd.DataFrame) -> pd.DataFrame:
        """Extract macroeconomic features."""
        
        features = pd.DataFrame(index=macro_data.index)
        
        # Interest rate features
        if '10Y_Treasury' in macro_data.columns:
            features['yield_10y'] = macro_data['10Y_Treasury']
            features['yield_curve'] = macro_data['10Y_Treasury'] - macro_data.get('3M_Treasury', 0)
        
        # Inflation features
        if 'CPI' in macro_data.columns:
            features['inflation'] = macro_data['CPI'].pct_change(12) * 100  # Annual inflation
            features['inflation_trend'] = features['inflation'].rolling(window=12).mean()
        
        # Economic indicators
        if 'Unemployment_Rate' in macro_data.columns:
            features['unemployment'] = macro_data['Unemployment_Rate']
            features['unemployment_trend'] = features['unemployment'].rolling(window=12).mean()
        
        # Market indicators
        if 'VIX' in macro_data.columns:
            features['vix'] = macro_data['VIX']
            features['vix_regime'] = (features['vix'] > features['vix'].rolling(window=63).mean()).astype(int)
        
        # Currency features
        if 'USD_EUR' in macro_data.columns:
            features['usd_eur'] = macro_data['USD_EUR']
            features['usd_eur_trend'] = features['usd_eur'].pct_change(21)
        
        return features
    
    def extract_regime_features(
        self,
        returns: pd.DataFrame,
        prices: pd.DataFrame,
        macro_data: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """Extract features specifically for regime detection."""
        
        logger.info("Extracting regime features", n_assets=len(returns.columns))
        
        features = pd.DataFrame(index=returns.index)
        
        # Market regime indicators
        features['market_return'] = returns.mean(axis=1)
        features['market_vol'] = returns.std(axis=1) * np.sqrt(252)
        
        # Volatility regime
        features['vol_regime'] = (features['market_vol'] > features['market_vol'].rolling(window=63).mean()).astype(int)
        
        # Correlation regime
        features['corr_regime'] = (returns.rolling(window=63).corr().groupby(level=0).mean().mean(axis=1) > 
                                  returns.rolling(window=63).corr().groupby(level=0).mean().mean(axis=1).rolling(window=63).mean()).astype(int)
        
        # Trend regime
        features['trend_regime'] = (features['market_return'].rolling(window=63).sum() > 0).astype(int)
        
        # Risk-on/risk-off indicators
        features['risk_on'] = (
            (features['market_return'] > 0).astype(int) +
            (features['vol_regime'] == 0).astype(int) +
            (features['corr_regime'] == 0).astype(int)
        ) / 3
        
        # Macro regime indicators
        if macro_data is not None:
            if 'VIX' in macro_data.columns:
                features['vix_regime'] = (macro_data['VIX'] > macro_data['VIX'].rolling(window=63).mean()).astype(int)
            
            if '10Y_Treasury' in macro_data.columns:
                features['yield_regime'] = (macro_data['10Y_Treasury'] > macro_data['10Y_Treasury'].rolling(window=63).mean()).astype(int)
            
            if 'CPI' in macro_data.columns:
                inflation = macro_data['CPI'].pct_change(12) * 100
                features['inflation_regime'] = (inflation > inflation.rolling(window=63).mean()).astype(int)
        
        # Clean features
        features = features.fillna(method='ffill').fillna(method='bfill')
        
        logger.info("Regime features extracted", n_features=len(features.columns))
        return features


class TechnicalIndicators:
    """Technical indicators for portfolio optimization."""
    
    @staticmethod
    def sma(prices: pd.Series, window: int) -> pd.Series:
        """Simple Moving Average."""
        return prices.rolling(window=window).mean()
    
    @staticmethod
    def ema(prices: pd.Series, window: int) -> pd.Series:
        """Exponential Moving Average."""
        return prices.ewm(span=window).mean()
    
    @staticmethod
    def rsi(prices: pd.Series, window: int = 14) -> pd.Series:
        """Relative Strength Index."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    @staticmethod
    def macd(prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
        """MACD indicator."""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal).mean()
        histogram = macd_line - signal_line
        
        return pd.DataFrame({
            'macd': macd_line,
            'signal': signal_line,
            'histogram': histogram
        })
    
    @staticmethod
    def bollinger_bands(prices: pd.Series, window: int = 20, num_std: float = 2) -> pd.DataFrame:
        """Bollinger Bands."""
        sma = prices.rolling(window=window).mean()
        std = prices.rolling(window=window).std()
        upper = sma + (std * num_std)
        lower = sma - (std * num_std)
        
        return pd.DataFrame({
            'upper': upper,
            'middle': sma,
            'lower': lower
        })
    
    @staticmethod
    def stochastic(prices: pd.Series, window: int = 14) -> pd.DataFrame:
        """Stochastic oscillator."""
        low_min = prices.rolling(window=window).min()
        high_max = prices.rolling(window=window).max()
        k_percent = 100 * ((prices - low_min) / (high_max - low_min))
        d_percent = k_percent.rolling(window=3).mean()
        
        return pd.DataFrame({
            'k_percent': k_percent,
            'd_percent': d_percent
        })
    
    @staticmethod
    def williams_r(prices: pd.Series, window: int = 14) -> pd.Series:
        """Williams %R."""
        low_min = prices.rolling(window=window).min()
        high_max = prices.rolling(window=window).max()
        return -100 * ((high_max - prices) / (high_max - low_min))
    
    @staticmethod
    def atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
        """Average True Range."""
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(window=window).mean()
    
    @staticmethod
    def adx(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
        """Average Directional Index."""
        # Calculate directional movement
        dm_plus = high.diff()
        dm_minus = -low.diff()
        
        # Set negative values to zero
        dm_plus = dm_plus.where(dm_plus > 0, 0)
        dm_minus = dm_minus.where(dm_minus > 0, 0)
        
        # Calculate true range
        tr = TechnicalIndicators.atr(high, low, close, window)
        
        # Calculate directional indicators
        di_plus = 100 * (dm_plus.rolling(window=window).mean() / tr)
        di_minus = 100 * (dm_minus.rolling(window=window).mean() / tr)
        
        # Calculate ADX
        dx = 100 * (di_plus - di_minus).abs() / (di_plus + di_minus)
        adx = dx.rolling(window=window).mean()
        
        return adx
