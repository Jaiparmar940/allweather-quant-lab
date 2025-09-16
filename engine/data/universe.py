"""
Universe management for portfolio optimization.
"""

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Union, Set, Tuple
import pandas as pd
import numpy as np
import yaml
import structlog

logger = structlog.get_logger(__name__)


class UniverseManager:
    """Manages the investable universe of assets."""
    
    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        self.config_path = config_path or Path("configs/universe.yaml")
        self.config = self._load_config()
        self.universe_cache: Dict[str, List[str]] = {}
    
    def _load_config(self) -> Dict:
        """Load universe configuration from YAML file."""
        
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info("Universe configuration loaded", config_path=str(self.config_path))
            return config
        except Exception as e:
            logger.error("Failed to load universe configuration", error=str(e), config_path=str(self.config_path))
            raise
    
    def get_all_tickers(self) -> List[str]:
        """Get all tickers from the universe configuration."""
        
        all_tickers = []
        
        # Add tickers from each asset category
        for category, tickers in self.config.get('assets', {}).items():
            all_tickers.extend(tickers)
        
        # Remove duplicates while preserving order
        unique_tickers = list(dict.fromkeys(all_tickers))
        
        logger.info("All tickers retrieved", count=len(unique_tickers), categories=list(self.config.get('assets', {}).keys()))
        return unique_tickers
    
    def get_tickers_by_category(self, category: str) -> List[str]:
        """Get tickers for a specific asset category."""
        
        tickers = self.config.get('assets', {}).get(category, [])
        logger.info("Tickers retrieved for category", category=category, count=len(tickers))
        return tickers
    
    def get_tickers_by_sector(self, sector: str) -> List[str]:
        """Get tickers for a specific sector."""
        
        tickers = self.config.get('sectors', {}).get(sector, [])
        logger.info("Tickers retrieved for sector", sector=sector, count=len(tickers))
        return tickers
    
    def get_universe_for_year(self, year: int) -> List[str]:
        """Get frozen universe for a specific year (point-in-time)."""
        
        cache_key = f"universe_{year}"
        
        if cache_key in self.universe_cache:
            return self.universe_cache[cache_key]
        
        # Get all tickers
        all_tickers = self.get_all_tickers()
        
        # Apply liquidity filters if available
        liquidity_config = self.config.get('liquidity', {})
        if liquidity_config:
            # In a real implementation, you would check actual liquidity data
            # For now, we'll use the configured tickers
            filtered_tickers = all_tickers
        else:
            filtered_tickers = all_tickers
        
        # Cache the result
        self.universe_cache[cache_key] = filtered_tickers
        
        logger.info("Universe retrieved for year", year=year, count=len(filtered_tickers))
        return filtered_tickers
    
    def get_universe_for_date(self, date: Union[str, datetime]) -> List[str]:
        """Get frozen universe for a specific date."""
        
        if isinstance(date, str):
            date = pd.to_datetime(date)
        
        year = date.year
        return self.get_universe_for_year(year)
    
    def get_asset_categories(self) -> List[str]:
        """Get list of asset categories."""
        
        categories = list(self.config.get('assets', {}).keys())
        logger.info("Asset categories retrieved", categories=categories)
        return categories
    
    def get_sectors(self) -> List[str]:
        """Get list of sectors."""
        
        sectors = list(self.config.get('sectors', {}).keys())
        logger.info("Sectors retrieved", sectors=sectors)
        return sectors
    
    def get_ticker_category(self, ticker: str) -> Optional[str]:
        """Get the category for a specific ticker."""
        
        for category, tickers in self.config.get('assets', {}).items():
            if ticker in tickers:
                return category
        
        return None
    
    def get_ticker_sectors(self, ticker: str) -> List[str]:
        """Get the sectors for a specific ticker."""
        
        sectors = []
        for sector, tickers in self.config.get('sectors', {}).items():
            if ticker in tickers:
                sectors.append(sector)
        
        return sectors
    
    def validate_tickers(self, tickers: List[str]) -> Tuple[List[str], List[str]]:
        """Validate tickers against the universe configuration."""
        
        all_tickers = self.get_all_tickers()
        valid_tickers = [t for t in tickers if t in all_tickers]
        invalid_tickers = [t for t in tickers if t not in all_tickers]
        
        if invalid_tickers:
            logger.warning("Invalid tickers found", invalid_tickers=invalid_tickers)
        
        logger.info("Ticker validation completed", valid_count=len(valid_tickers), invalid_count=len(invalid_tickers))
        return valid_tickers, invalid_tickers
    
    def get_benchmark_ticker(self) -> str:
        """Get the benchmark ticker (ALLW)."""
        
        benchmark_config = self.config.get('benchmark', {})
        ticker = benchmark_config.get('ticker', 'ALLW')
        
        logger.info("Benchmark ticker retrieved", ticker=ticker)
        return ticker
    
    def get_rebalancing_schedule(self) -> Dict[str, Union[str, int]]:
        """Get rebalancing schedule configuration."""
        
        schedule = self.config.get('rebalancing', {})
        logger.info("Rebalancing schedule retrieved", schedule=schedule)
        return schedule
    
    def get_liquidity_requirements(self) -> Dict[str, float]:
        """Get liquidity requirements configuration."""
        
        liquidity = self.config.get('liquidity', {})
        logger.info("Liquidity requirements retrieved", liquidity=liquidity)
        return liquidity
    
    def get_universe_freeze_policy(self) -> Dict[str, Union[str, int]]:
        """Get universe freeze policy configuration."""
        
        policy = self.config.get('universe_freeze', {})
        logger.info("Universe freeze policy retrieved", policy=policy)
        return policy
    
    def apply_liquidity_filter(
        self,
        tickers: List[str],
        price_data: pd.DataFrame,
        min_adv_usd: Optional[float] = None,
        min_market_cap_usd: Optional[float] = None
    ) -> List[str]:
        """Apply liquidity filters to tickers based on actual data."""
        
        if not price_data.empty:
            # Calculate average daily volume in USD
            volume_cols = [col for col in price_data.columns if col.endswith('_Volume')]
            close_cols = [col for col in price_data.columns if col.endswith('_Close')]
            
            if volume_cols and close_cols:
                # Match volume and close columns
                volume_data = price_data[volume_cols]
                close_data = price_data[close_cols]
                
                # Calculate ADV in USD
                adv_usd = (volume_data * close_data).mean()
                
                # Apply minimum ADV filter
                if min_adv_usd:
                    liquid_tickers = adv_usd[adv_usd >= min_adv_usd].index.tolist()
                    # Extract ticker names from column names
                    liquid_tickers = [col.split('_')[0] for col in liquid_tickers]
                    tickers = [t for t in tickers if t in liquid_tickers]
        
        logger.info("Liquidity filter applied", original_count=len(tickers), filtered_count=len(tickers))
        return tickers
    
    def get_asset_weights_constraints(
        self,
        tickers: List[str],
        policy_config: Optional[Dict] = None
    ) -> Dict[str, Tuple[float, float]]:
        """Get asset weight constraints based on policy configuration."""
        
        if not policy_config:
            return {}
        
        constraints = {}
        asset_constraints = policy_config.get('asset_constraints', {})
        
        for ticker in tickers:
            category = self.get_ticker_category(ticker)
            if category:
                min_weight = asset_constraints.get(f'min_{category}', 0.0)
                max_weight = asset_constraints.get(f'max_{category}', 1.0)
                constraints[ticker] = (min_weight, max_weight)
        
        logger.info("Asset weight constraints retrieved", count=len(constraints))
        return constraints
    
    def get_sector_weights_constraints(
        self,
        tickers: List[str],
        policy_config: Optional[Dict] = None
    ) -> Dict[str, Tuple[float, float]]:
        """Get sector weight constraints based on policy configuration."""
        
        if not policy_config:
            return {}
        
        constraints = {}
        sector_constraints = policy_config.get('sector_constraints', {})
        
        for ticker in tickers:
            sectors = self.get_ticker_sectors(ticker)
            for sector in sectors:
                min_weight = sector_constraints.get(f'min_{sector}', 0.0)
                max_weight = sector_constraints.get(f'max_{sector}', 1.0)
                constraints[sector] = (min_weight, max_weight)
        
        logger.info("Sector weight constraints retrieved", count=len(constraints))
        return constraints
    
    def get_esg_constraints(
        self,
        tickers: List[str],
        policy_config: Optional[Dict] = None
    ) -> Dict[str, Union[bool, float, List[str]]]:
        """Get ESG constraints based on policy configuration."""
        
        if not policy_config:
            return {}
        
        esg_constraints = policy_config.get('esg_constraints', {})
        
        constraints = {
            'enabled': esg_constraints.get('enabled', False),
            'min_esg_score': esg_constraints.get('min_esg_score', 0.0),
            'exclude_controversial': esg_constraints.get('exclude_controversial', False),
            'exclude_sectors': esg_constraints.get('exclude_sectors', [])
        }
        
        logger.info("ESG constraints retrieved", constraints=constraints)
        return constraints
    
    def update_universe(self, new_config: Dict) -> None:
        """Update universe configuration."""
        
        self.config.update(new_config)
        
        # Clear cache
        self.universe_cache.clear()
        
        # Save updated config
        with open(self.config_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)
        
        logger.info("Universe configuration updated", config_path=str(self.config_path))
    
    def add_ticker(self, ticker: str, category: str, sectors: Optional[List[str]] = None) -> None:
        """Add a new ticker to the universe."""
        
        # Add to assets
        if category not in self.config['assets']:
            self.config['assets'][category] = []
        
        if ticker not in self.config['assets'][category]:
            self.config['assets'][category].append(ticker)
        
        # Add to sectors
        if sectors:
            for sector in sectors:
                if sector not in self.config['sectors']:
                    self.config['sectors'][sector] = []
                
                if ticker not in self.config['sectors'][sector]:
                    self.config['sectors'][sector].append(ticker)
        
        # Clear cache
        self.universe_cache.clear()
        
        # Save updated config
        with open(self.config_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)
        
        logger.info("Ticker added to universe", ticker=ticker, category=category, sectors=sectors)
    
    def remove_ticker(self, ticker: str) -> None:
        """Remove a ticker from the universe."""
        
        # Remove from assets
        for category, tickers in self.config['assets'].items():
            if ticker in tickers:
                tickers.remove(ticker)
        
        # Remove from sectors
        for sector, tickers in self.config['sectors'].items():
            if ticker in tickers:
                tickers.remove(ticker)
        
        # Clear cache
        self.universe_cache.clear()
        
        # Save updated config
        with open(self.config_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)
        
        logger.info("Ticker removed from universe", ticker=ticker)
