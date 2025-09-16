"""
Data loaders for price and macro data.
"""

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
import pandas as pd
import numpy as np
import yfinance as yf
from fredapi import Fred
import requests
from functools import lru_cache
import structlog

logger = structlog.get_logger(__name__)


class DataLoader:
    """Base class for data loaders."""
    
    def __init__(self, cache_dir: Optional[Path] = None, cache_ttl_hours: int = 24):
        self.cache_dir = cache_dir or Path("./data/processed")
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
        logger.info("Data saved to cache", cache_file=str(cache_file))
    
    def _load_from_cache(self, cache_file: Path) -> Optional[pd.DataFrame]:
        """Load data from cache file."""
        if self._is_cache_valid(cache_file):
            try:
                data = pd.read_parquet(cache_file)
                logger.info("Data loaded from cache", cache_file=str(cache_file))
                return data
            except Exception as e:
                logger.warning("Failed to load from cache", error=str(e))
        return None


class PriceDataLoader(DataLoader):
    """Loader for price data from Yahoo Finance."""
    
    def __init__(self, cache_dir: Optional[Path] = None, cache_ttl_hours: int = 24):
        super().__init__(cache_dir, cache_ttl_hours)
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    def load_prices(
        self,
        tickers: List[str],
        start_date: Union[str, datetime],
        end_date: Union[str, datetime],
        use_cache: bool = True
    ) -> pd.DataFrame:
        """Load price data for given tickers and date range."""
        
        # Convert to datetime if needed
        if isinstance(start_date, str):
            start_date = pd.to_datetime(start_date)
        if isinstance(end_date, str):
            end_date = pd.to_datetime(end_date)
        
        # Create cache file name
        cache_file = self.cache_dir / f"prices_{'-'.join(sorted(tickers))}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.parquet"
        
        # Try to load from cache first
        if use_cache:
            cached_data = self._load_from_cache(cache_file)
            if cached_data is not None:
                return cached_data
        
        # Load data from Yahoo Finance
        logger.info("Loading price data from Yahoo Finance", tickers=tickers, start_date=start_date, end_date=end_date)
        
        try:
            # Download data
            data = yf.download(
                tickers,
                start=start_date,
                end=end_date,
                group_by='ticker',
                progress=False,
                session=self.session
            )
            
            # Handle single ticker case
            if len(tickers) == 1:
                data = data.rename(columns=lambda x: f"{tickers[0]}_{x}")
                data.columns = pd.MultiIndex.from_tuples([(tickers[0], col.split('_', 1)[1]) for col in data.columns])
            
            # Flatten multi-level columns
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = [f"{ticker}_{col}" for ticker, col in data.columns]
            
            # Ensure we have the expected columns
            expected_cols = []
            for ticker in tickers:
                expected_cols.extend([f"{ticker}_Open", f"{ticker}_High", f"{ticker}_Low", f"{ticker}_Close", f"{ticker}_Volume"])
            
            # Filter to only have expected columns
            available_cols = [col for col in expected_cols if col in data.columns]
            if not available_cols:
                raise ValueError(f"No price data found for tickers: {tickers}")
            
            data = data[available_cols]
            
            # Clean data
            data = data.dropna(how='all')
            data = data.fillna(method='ffill').fillna(method='bfill')
            
            # Save to cache
            if use_cache:
                self._save_to_cache(data, cache_file)
            
            logger.info("Price data loaded successfully", shape=data.shape, columns=list(data.columns))
            return data
            
        except Exception as e:
            logger.error("Failed to load price data", error=str(e), tickers=tickers)
            raise
    
    def load_dividends(
        self,
        tickers: List[str],
        start_date: Union[str, datetime],
        end_date: Union[str, datetime],
        use_cache: bool = True
    ) -> pd.DataFrame:
        """Load dividend data for given tickers and date range."""
        
        # Convert to datetime if needed
        if isinstance(start_date, str):
            start_date = pd.to_datetime(start_date)
        if isinstance(end_date, str):
            end_date = pd.to_datetime(end_date)
        
        # Create cache file name
        cache_file = self.cache_dir / f"dividends_{'-'.join(sorted(tickers))}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.parquet"
        
        # Try to load from cache first
        if use_cache:
            cached_data = self._load_from_cache(cache_file)
            if cached_data is not None:
                return cached_data
        
        # Load dividend data
        logger.info("Loading dividend data from Yahoo Finance", tickers=tickers, start_date=start_date, end_date=end_date)
        
        try:
            dividends_data = {}
            for ticker in tickers:
                try:
                    stock = yf.Ticker(ticker, session=self.session)
                    div_data = stock.dividends
                    if not div_data.empty:
                        div_data = div_data[(div_data.index >= start_date) & (div_data.index <= end_date)]
                        dividends_data[ticker] = div_data
                except Exception as e:
                    logger.warning("Failed to load dividends for ticker", ticker=ticker, error=str(e))
            
            if not dividends_data:
                return pd.DataFrame()
            
            # Combine all dividend data
            combined_data = pd.concat(dividends_data.values(), axis=1, keys=dividends_data.keys())
            combined_data = combined_data.fillna(0)
            
            # Save to cache
            if use_cache:
                self._save_to_cache(combined_data, cache_file)
            
            logger.info("Dividend data loaded successfully", shape=combined_data.shape)
            return combined_data
            
        except Exception as e:
            logger.error("Failed to load dividend data", error=str(e), tickers=tickers)
            raise


class MacroDataLoader(DataLoader):
    """Loader for macroeconomic data from FRED."""
    
    def __init__(self, api_key: Optional[str] = None, cache_dir: Optional[Path] = None, cache_ttl_hours: int = 24):
        super().__init__(cache_dir, cache_ttl_hours)
        self.api_key = api_key
        if api_key:
            self.fred = Fred(api_key=api_key)
        else:
            self.fred = None
            logger.warning("No FRED API key provided, macro data loading will be limited")
    
    def load_macro_data(
        self,
        series_ids: List[str],
        start_date: Union[str, datetime],
        end_date: Union[str, datetime],
        use_cache: bool = True
    ) -> pd.DataFrame:
        """Load macroeconomic data from FRED."""
        
        if not self.fred:
            raise ValueError("FRED API key is required for macro data loading")
        
        # Convert to datetime if needed
        if isinstance(start_date, str):
            start_date = pd.to_datetime(start_date)
        if isinstance(end_date, str):
            end_date = pd.to_datetime(end_date)
        
        # Create cache file name
        cache_file = self.cache_dir / f"macro_{'-'.join(sorted(series_ids))}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.parquet"
        
        # Try to load from cache first
        if use_cache:
            cached_data = self._load_from_cache(cache_file)
            if cached_data is not None:
                return cached_data
        
        # Load data from FRED
        logger.info("Loading macro data from FRED", series_ids=series_ids, start_date=start_date, end_date=end_date)
        
        try:
            macro_data = {}
            for series_id in series_ids:
                try:
                    data = self.fred.get_series(series_id, start_date, end_date)
                    if not data.empty:
                        macro_data[series_id] = data
                except Exception as e:
                    logger.warning("Failed to load macro series", series_id=series_id, error=str(e))
            
            if not macro_data:
                return pd.DataFrame()
            
            # Combine all macro data
            combined_data = pd.concat(macro_data.values(), axis=1, keys=macro_data.keys())
            combined_data = combined_data.fillna(method='ffill').fillna(method='bfill')
            
            # Save to cache
            if use_cache:
                self._save_to_cache(combined_data, cache_file)
            
            logger.info("Macro data loaded successfully", shape=combined_data.shape)
            return combined_data
            
        except Exception as e:
            logger.error("Failed to load macro data", error=str(e), series_ids=series_ids)
            raise


class DataLoader:
    """Main data loader that coordinates price and macro data loading."""
    
    def __init__(
        self,
        fred_api_key: Optional[str] = None,
        cache_dir: Optional[Path] = None,
        cache_ttl_hours: int = 24
    ):
        self.cache_dir = cache_dir or Path("./data/processed")
        self.price_loader = PriceDataLoader(cache_dir, cache_ttl_hours)
        self.macro_loader = MacroDataLoader(fred_api_key, cache_dir, cache_ttl_hours)
    
    def load_universe_data(
        self,
        tickers: List[str],
        macro_series: List[str],
        start_date: Union[str, datetime],
        end_date: Union[str, datetime],
        use_cache: bool = True
    ) -> Dict[str, pd.DataFrame]:
        """Load both price and macro data for the universe."""
        
        logger.info("Loading universe data", tickers=tickers, macro_series=macro_series, start_date=start_date, end_date=end_date)
        
        results = {}
        
        # Load price data
        if tickers:
            try:
                price_data = self.price_loader.load_prices(tickers, start_date, end_date, use_cache)
                results['prices'] = price_data
            except Exception as e:
                logger.error("Failed to load price data", error=str(e))
                results['prices'] = pd.DataFrame()
        
        # Load macro data
        if macro_series:
            try:
                macro_data = self.macro_loader.load_macro_data(macro_series, start_date, end_date, use_cache)
                results['macro'] = macro_data
            except Exception as e:
                logger.error("Failed to load macro data", error=str(e))
                results['macro'] = pd.DataFrame()
        
        return results
