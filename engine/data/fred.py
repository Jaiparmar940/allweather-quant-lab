"""
FRED (Federal Reserve Economic Data) loader and utilities.
"""

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
import pandas as pd
import numpy as np
from fredapi import Fred
import structlog

logger = structlog.get_logger(__name__)


class FREDLoader:
    """Enhanced FRED data loader with caching and data validation."""
    
    def __init__(self, api_key: str, cache_dir: Optional[Path] = None, cache_ttl_hours: int = 24):
        self.api_key = api_key
        self.fred = Fred(api_key=api_key)
        self.cache_dir = cache_dir or Path("./data/processed/fred")
        self.cache_ttl_hours = cache_ttl_hours
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _is_cache_valid(self, cache_file: Path) -> bool:
        """Check if cache file is still valid."""
        if not cache_file.exists():
            return False
        
        cache_age = datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)
        return cache_age < timedelta(hours=self.cache_ttl_hours)
    
    def _save_to_cache(self, data: pd.DataFrame, cache_file: Path) -> None:
        """Save data to cache file."""
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        data.to_parquet(cache_file)
        logger.info("FRED data saved to cache", cache_file=str(cache_file))
    
    def _load_from_cache(self, cache_file: Path) -> Optional[pd.DataFrame]:
        """Load data from cache file."""
        if self._is_cache_valid(cache_file):
            try:
                data = pd.read_parquet(cache_file)
                logger.info("FRED data loaded from cache", cache_file=str(cache_file))
                return data
            except Exception as e:
                logger.warning("Failed to load FRED data from cache", error=str(e))
        return None
    
    def get_series(
        self,
        series_id: str,
        start_date: Union[str, datetime],
        end_date: Union[str, datetime],
        use_cache: bool = True
    ) -> pd.Series:
        """Get a single FRED series with caching."""
        
        # Convert to datetime if needed
        if isinstance(start_date, str):
            start_date = pd.to_datetime(start_date)
        if isinstance(end_date, str):
            end_date = pd.to_datetime(end_date)
        
        # Create cache file name
        cache_file = self.cache_dir / f"{series_id}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.parquet"
        
        # Try to load from cache first
        if use_cache:
            cached_data = self._load_from_cache(cache_file)
            if cached_data is not None:
                return cached_data.iloc[:, 0]  # Return first column as Series
        
        # Load data from FRED
        logger.info("Loading FRED series", series_id=series_id, start_date=start_date, end_date=end_date)
        
        try:
            data = self.fred.get_series(series_id, start_date, end_date)
            
            if data.empty:
                logger.warning("Empty series returned from FRED", series_id=series_id)
                return pd.Series(dtype=float, name=series_id)
            
            # Save to cache
            if use_cache:
                data_df = pd.DataFrame({series_id: data})
                self._save_to_cache(data_df, cache_file)
            
            logger.info("FRED series loaded successfully", series_id=series_id, length=len(data))
            return data
            
        except Exception as e:
            logger.error("Failed to load FRED series", series_id=series_id, error=str(e))
            raise
    
    def get_multiple_series(
        self,
        series_ids: List[str],
        start_date: Union[str, datetime],
        end_date: Union[str, datetime],
        use_cache: bool = True
    ) -> pd.DataFrame:
        """Get multiple FRED series with caching."""
        
        # Convert to datetime if needed
        if isinstance(start_date, str):
            start_date = pd.to_datetime(start_date)
        if isinstance(end_date, str):
            end_date = pd.to_datetime(end_date)
        
        # Create cache file name
        cache_file = self.cache_dir / f"multi_{'-'.join(sorted(series_ids))}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.parquet"
        
        # Try to load from cache first
        if use_cache:
            cached_data = self._load_from_cache(cache_file)
            if cached_data is not None:
                return cached_data
        
        # Load data from FRED
        logger.info("Loading multiple FRED series", series_ids=series_ids, start_date=start_date, end_date=end_date)
        
        try:
            series_data = {}
            for series_id in series_ids:
                try:
                    data = self.fred.get_series(series_id, start_date, end_date)
                    if not data.empty:
                        series_data[series_id] = data
                    else:
                        logger.warning("Empty series returned from FRED", series_id=series_id)
                except Exception as e:
                    logger.warning("Failed to load FRED series", series_id=series_id, error=str(e))
            
            if not series_data:
                return pd.DataFrame()
            
            # Combine all series
            combined_data = pd.concat(series_data.values(), axis=1, keys=series_data.keys())
            combined_data = combined_data.fillna(method='ffill').fillna(method='bfill')
            
            # Save to cache
            if use_cache:
                self._save_to_cache(combined_data, cache_file)
            
            logger.info("Multiple FRED series loaded successfully", shape=combined_data.shape)
            return combined_data
            
        except Exception as e:
            logger.error("Failed to load multiple FRED series", error=str(e), series_ids=series_ids)
            raise
    
    def get_interest_rates(
        self,
        start_date: Union[str, datetime],
        end_date: Union[str, datetime],
        use_cache: bool = True
    ) -> pd.DataFrame:
        """Get interest rate data (3M, 1Y, 3Y, 10Y, 30Y Treasury rates)."""
        
        series_ids = [
            "DGS3MO",  # 3-Month Treasury Bill
            "DGS1",    # 1-Year Treasury Rate
            "DGS3",    # 3-Year Treasury Rate
            "DGS10",   # 10-Year Treasury Rate
            "DGS30",   # 30-Year Treasury Rate
        ]
        
        data = self.get_multiple_series(series_ids, start_date, end_date, use_cache)
        
        # Rename columns for clarity
        column_mapping = {
            "DGS3MO": "3M_Treasury",
            "DGS1": "1Y_Treasury",
            "DGS3": "3Y_Treasury",
            "DGS10": "10Y_Treasury",
            "DGS30": "30Y_Treasury",
        }
        
        data = data.rename(columns=column_mapping)
        return data
    
    def get_inflation_data(
        self,
        start_date: Union[str, datetime],
        end_date: Union[str, datetime],
        use_cache: bool = True
    ) -> pd.DataFrame:
        """Get inflation data (CPI, Core CPI, Breakeven Inflation)."""
        
        series_ids = [
            "CPIAUCSL",  # Consumer Price Index
            "CPILFESL",  # Core CPI
            "T10YIE",    # 10-Year Breakeven Inflation Rate
        ]
        
        data = self.get_multiple_series(series_ids, start_date, end_date, use_cache)
        
        # Rename columns for clarity
        column_mapping = {
            "CPIAUCSL": "CPI",
            "CPILFESL": "Core_CPI",
            "T10YIE": "Breakeven_Inflation_10Y",
        }
        
        data = data.rename(columns=column_mapping)
        return data
    
    def get_economic_indicators(
        self,
        start_date: Union[str, datetime],
        end_date: Union[str, datetime],
        use_cache: bool = True
    ) -> pd.DataFrame:
        """Get key economic indicators."""
        
        series_ids = [
            "UNRATE",    # Unemployment Rate
            "GDPC1",     # Real GDP
            "FEDFUNDS",  # Federal Funds Rate
        ]
        
        data = self.get_multiple_series(series_ids, start_date, end_date, use_cache)
        
        # Rename columns for clarity
        column_mapping = {
            "UNRATE": "Unemployment_Rate",
            "GDPC1": "Real_GDP",
            "FEDFUNDS": "Fed_Funds_Rate",
        }
        
        data = data.rename(columns=column_mapping)
        return data
    
    def get_market_indicators(
        self,
        start_date: Union[str, datetime],
        end_date: Union[str, datetime],
        use_cache: bool = True
    ) -> pd.DataFrame:
        """Get market indicators (VIX, Exchange Rates)."""
        
        series_ids = [
            "VIXCLS",   # VIX
            "DEXUSEU",  # USD/EUR Exchange Rate
            "DEXJPUS",  # USD/JPY Exchange Rate
        ]
        
        data = self.get_multiple_series(series_ids, start_date, end_date, use_cache)
        
        # Rename columns for clarity
        column_mapping = {
            "VIXCLS": "VIX",
            "DEXUSEU": "USD_EUR",
            "DEXJPUS": "USD_JPY",
        }
        
        data = data.rename(columns=column_mapping)
        return data
    
    def get_all_macro_data(
        self,
        start_date: Union[str, datetime],
        end_date: Union[str, datetime],
        use_cache: bool = True
    ) -> pd.DataFrame:
        """Get all macro data categories combined."""
        
        logger.info("Loading all macro data from FRED", start_date=start_date, end_date=end_date)
        
        try:
            # Load all categories
            interest_rates = self.get_interest_rates(start_date, end_date, use_cache)
            inflation_data = self.get_inflation_data(start_date, end_date, use_cache)
            economic_indicators = self.get_economic_indicators(start_date, end_date, use_cache)
            market_indicators = self.get_market_indicators(start_date, end_date, use_cache)
            
            # Combine all data
            all_data = pd.concat([
                interest_rates,
                inflation_data,
                economic_indicators,
                market_indicators
            ], axis=1)
            
            # Fill missing values
            all_data = all_data.fillna(method='ffill').fillna(method='bfill')
            
            logger.info("All macro data loaded successfully", shape=all_data.shape)
            return all_data
            
        except Exception as e:
            logger.error("Failed to load all macro data", error=str(e))
            raise
