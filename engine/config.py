"""
Configuration management for the Omega Portfolio Engine.
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional, Union
import yaml
from pydantic import BaseModel, Field, validator
from dotenv import load_dotenv


class DataConfig(BaseModel):
    """Data source configuration."""
    
    sources: Dict[str, Any] = Field(default_factory=dict)
    validation: Dict[str, Any] = Field(default_factory=dict)
    processing: Dict[str, Any] = Field(default_factory=dict)
    macro_indicators: Dict[str, Any] = Field(default_factory=dict)
    storage: Dict[str, Any] = Field(default_factory=dict)
    cache: Dict[str, Any] = Field(default_factory=dict)


class UniverseConfig(BaseModel):
    """Universe configuration."""
    
    assets: Dict[str, list] = Field(default_factory=dict)
    sectors: Dict[str, list] = Field(default_factory=dict)
    liquidity: Dict[str, Any] = Field(default_factory=dict)
    rebalancing: Dict[str, Any] = Field(default_factory=dict)
    data_sources: Dict[str, str] = Field(default_factory=dict)
    universe_freeze: Dict[str, Any] = Field(default_factory=dict)


class BacktestConfig(BaseModel):
    """Backtesting configuration."""
    
    walk_forward: Dict[str, Any] = Field(default_factory=dict)
    rebalancing: Dict[str, Any] = Field(default_factory=dict)
    transaction_costs: Dict[str, Any] = Field(default_factory=dict)
    tax_settings: Dict[str, Any] = Field(default_factory=dict)
    risk_management: Dict[str, Any] = Field(default_factory=dict)
    benchmark: Dict[str, Any] = Field(default_factory=dict)
    evaluation: Dict[str, Any] = Field(default_factory=dict)
    robustness: Dict[str, Any] = Field(default_factory=dict)
    output: Dict[str, Any] = Field(default_factory=dict)


class PolicyConfig(BaseModel):
    """Client policy configuration."""
    
    client_profile: Dict[str, Any] = Field(default_factory=dict)
    return_requirements: Dict[str, Any] = Field(default_factory=dict)
    risk_constraints: Dict[str, Any] = Field(default_factory=dict)
    asset_constraints: Dict[str, Any] = Field(default_factory=dict)
    sector_constraints: Dict[str, Any] = Field(default_factory=dict)
    geographic_constraints: Dict[str, Any] = Field(default_factory=dict)
    esg_constraints: Dict[str, Any] = Field(default_factory=dict)
    liquidity_constraints: Dict[str, Any] = Field(default_factory=dict)
    turnover_constraints: Dict[str, Any] = Field(default_factory=dict)
    tax_settings: Dict[str, Any] = Field(default_factory=dict)
    optimization: Dict[str, Any] = Field(default_factory=dict)
    rebalancing: Dict[str, Any] = Field(default_factory=dict)
    reporting: Dict[str, Any] = Field(default_factory=dict)


class Config(BaseModel):
    """Main configuration class."""
    
    data: DataConfig = Field(default_factory=DataConfig)
    universe: UniverseConfig = Field(default_factory=UniverseConfig)
    backtest: BacktestConfig = Field(default_factory=BacktestConfig)
    policy: PolicyConfig = Field(default_factory=PolicyConfig)
    
    # Environment variables
    fred_api_key: Optional[str] = None
    quandl_api_key: Optional[str] = None
    mlflow_tracking_uri: str = "sqlite:///mlruns/mlflow.db"
    mlflow_experiment_name: str = "omega_portfolio_engine"
    
    # Paths
    data_root: Path = Path("./data")
    raw_data_path: Path = Path("./data/raw")
    processed_data_path: Path = Path("./data/processed")
    results_path: Path = Path("./results")
    
    # Logging
    log_level: str = "INFO"
    log_format: str = "json"
    
    @validator('data_root', 'raw_data_path', 'processed_data_path', 'results_path', pre=True)
    def convert_to_path(cls, v):
        if isinstance(v, str):
            return Path(v)
        return v
    
    @classmethod
    def from_files(
        cls,
        data_config_path: Optional[Union[str, Path]] = None,
        universe_config_path: Optional[Union[str, Path]] = None,
        backtest_config_path: Optional[Union[str, Path]] = None,
        policy_config_path: Optional[Union[str, Path]] = None,
        env_file: Optional[Union[str, Path]] = None,
    ) -> "Config":
        """Load configuration from YAML files and environment variables."""
        
        # Load environment variables
        if env_file:
            load_dotenv(env_file)
        else:
            load_dotenv()
        
        # Default config paths
        config_dir = Path("configs")
        data_config_path = data_config_path or config_dir / "data.yaml"
        universe_config_path = universe_config_path or config_dir / "universe.yaml"
        backtest_config_path = backtest_config_path or config_dir / "backtest.yaml"
        policy_config_path = policy_config_path or config_dir / "policy.examples" / "balanced.yaml"
        
        # Load YAML configurations
        config_data = {}
        
        for config_path, config_key in [
            (data_config_path, "data"),
            (universe_config_path, "universe"),
            (backtest_config_path, "backtest"),
            (policy_config_path, "policy"),
        ]:
            if Path(config_path).exists():
                with open(config_path, 'r') as f:
                    config_data[config_key] = yaml.safe_load(f)
        
        # Load environment variables
        env_vars = {
            "fred_api_key": os.getenv("FRED_API_KEY"),
            "quandl_api_key": os.getenv("QUANDL_API_KEY"),
            "mlflow_tracking_uri": os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlruns/mlflow.db"),
            "mlflow_experiment_name": os.getenv("MLFLOW_EXPERIMENT_NAME", "omega_portfolio_engine"),
            "log_level": os.getenv("LOG_LEVEL", "INFO"),
            "log_format": os.getenv("LOG_FORMAT", "json"),
        }
        
        # Merge configurations
        config_data.update(env_vars)
        
        return cls(**config_data)


def load_config(
    data_config_path: Optional[Union[str, Path]] = None,
    universe_config_path: Optional[Union[str, Path]] = None,
    backtest_config_path: Optional[Union[str, Path]] = None,
    policy_config_path: Optional[Union[str, Path]] = None,
    env_file: Optional[Union[str, Path]] = None,
) -> Config:
    """Load configuration from files and environment variables."""
    return Config.from_files(
        data_config_path=data_config_path,
        universe_config_path=universe_config_path,
        backtest_config_path=backtest_config_path,
        policy_config_path=policy_config_path,
        env_file=env_file,
    )


# Global config instance
_config: Optional[Config] = None


def get_config() -> Config:
    """Get the global configuration instance."""
    global _config
    if _config is None:
        _config = load_config()
    return _config


def set_config(config: Config) -> None:
    """Set the global configuration instance."""
    global _config
    _config = config
