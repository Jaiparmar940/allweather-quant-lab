"""
Omega Portfolio Engine

A regime-aware, client-customizable portfolio optimization engine that:
- Detects market regimes using HMM or LSTM
- Optimizes portfolios for Global Minimum Variance and Omega ratio
- Benchmarks against Bridgewater's All Weather ETF
- Provides comprehensive backtesting and evaluation
"""

__version__ = "0.1.0"
__author__ = "Omega Team"

from .config import Config, load_config
from .data import DataLoader, UniverseManager
from .optimize import GMVOptimizer, OmegaOptimizer
from .backtest import BacktestSimulator, WalkForwardEngine
from .signals import RegimeDetector
from .report import ClientMemoGenerator, ChartGenerator

__all__ = [
    "Config",
    "load_config",
    "DataLoader",
    "UniverseManager",
    "GMVOptimizer",
    "OmegaOptimizer",
    "BacktestSimulator",
    "WalkForwardEngine",
    "RegimeDetector",
    "ClientMemoGenerator",
    "ChartGenerator",
]
