"""
Input/Output utilities for the Omega Portfolio Engine.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
import pandas as pd
import numpy as np
import json
import pickle
import yaml
import structlog

logger = structlog.get_logger(__name__)


class IOUtils:
    """Utility class for input/output operations."""
    
    @staticmethod
    def save_dataframe(
        df: pd.DataFrame,
        filepath: Union[str, Path],
        format: str = "parquet",
        **kwargs
    ) -> None:
        """Save DataFrame to file in specified format."""
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        if format == "parquet":
            df.to_parquet(filepath, **kwargs)
        elif format == "csv":
            df.to_csv(filepath, **kwargs)
        elif format == "excel":
            df.to_excel(filepath, **kwargs)
        elif format == "json":
            df.to_json(filepath, **kwargs)
        elif format == "pickle":
            df.to_pickle(filepath, **kwargs)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        logger.debug("DataFrame saved", filepath=str(filepath), format=format, shape=df.shape)
    
    @staticmethod
    def load_dataframe(
        filepath: Union[str, Path],
        format: str = "parquet",
        **kwargs
    ) -> pd.DataFrame:
        """Load DataFrame from file in specified format."""
        
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        
        if format == "parquet":
            df = pd.read_parquet(filepath, **kwargs)
        elif format == "csv":
            df = pd.read_csv(filepath, **kwargs)
        elif format == "excel":
            df = pd.read_excel(filepath, **kwargs)
        elif format == "json":
            df = pd.read_json(filepath, **kwargs)
        elif format == "pickle":
            df = pd.read_pickle(filepath, **kwargs)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        logger.debug("DataFrame loaded", filepath=str(filepath), format=format, shape=df.shape)
        return df
    
    @staticmethod
    def save_dict(
        data: Dict[str, Any],
        filepath: Union[str, Path],
        format: str = "json",
        **kwargs
    ) -> None:
        """Save dictionary to file in specified format."""
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        if format == "json":
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2, **kwargs)
        elif format == "yaml":
            with open(filepath, 'w') as f:
                yaml.dump(data, f, **kwargs)
        elif format == "pickle":
            with open(filepath, 'wb') as f:
                pickle.dump(data, f, **kwargs)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        logger.debug("Dictionary saved", filepath=str(filepath), format=format)
    
    @staticmethod
    def load_dict(
        filepath: Union[str, Path],
        format: str = "json",
        **kwargs
    ) -> Dict[str, Any]:
        """Load dictionary from file in specified format."""
        
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        
        if format == "json":
            with open(filepath, 'r') as f:
                data = json.load(f, **kwargs)
        elif format == "yaml":
            with open(filepath, 'r') as f:
                data = yaml.safe_load(f, **kwargs)
        elif format == "pickle":
            with open(filepath, 'rb') as f:
                data = pickle.load(f, **kwargs)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        logger.debug("Dictionary loaded", filepath=str(filepath), format=format)
        return data
    
    @staticmethod
    def save_numpy_array(
        array: np.ndarray,
        filepath: Union[str, Path],
        format: str = "npy",
        **kwargs
    ) -> None:
        """Save NumPy array to file in specified format."""
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        if format == "npy":
            np.save(filepath, array, **kwargs)
        elif format == "npz":
            np.savez(filepath, array=array, **kwargs)
        elif format == "txt":
            np.savetxt(filepath, array, **kwargs)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        logger.debug("NumPy array saved", filepath=str(filepath), format=format, shape=array.shape)
    
    @staticmethod
    def load_numpy_array(
        filepath: Union[str, Path],
        format: str = "npy",
        **kwargs
    ) -> np.ndarray:
        """Load NumPy array from file in specified format."""
        
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        
        if format == "npy":
            array = np.load(filepath, **kwargs)
        elif format == "npz":
            data = np.load(filepath, **kwargs)
            array = data['array']
        elif format == "txt":
            array = np.loadtxt(filepath, **kwargs)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        logger.debug("NumPy array loaded", filepath=str(filepath), format=format, shape=array.shape)
        return array
    
    @staticmethod
    def save_backtest_results(
        results: Dict[str, Any],
        output_dir: Union[str, Path],
        prefix: str = "backtest"
    ) -> Dict[str, Path]:
        """Save backtest results to files."""
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        saved_files = {}
        
        # Save performance history
        if "performance_history" in results:
            perf_path = output_dir / f"{prefix}_performance_history.parquet"
            IOUtils.save_dataframe(results["performance_history"], perf_path)
            saved_files["performance_history"] = perf_path
        
        # Save weights history
        if "weights_history" in results:
            weights_path = output_dir / f"{prefix}_weights_history.parquet"
            IOUtils.save_dataframe(results["weights_history"], weights_path)
            saved_files["weights_history"] = weights_path
        
        # Save trade history
        if "trade_history" in results:
            trades_path = output_dir / f"{prefix}_trade_history.parquet"
            IOUtils.save_dataframe(pd.DataFrame(results["trade_history"]), trades_path)
            saved_files["trade_history"] = trades_path
        
        # Save metrics
        if "metrics" in results:
            metrics_path = output_dir / f"{prefix}_metrics.json"
            IOUtils.save_dict(results["metrics"], metrics_path)
            saved_files["metrics"] = metrics_path
        
        # Save configuration
        if "config" in results:
            config_path = output_dir / f"{prefix}_config.yaml"
            IOUtils.save_dict(results["config"], config_path)
            saved_files["config"] = config_path
        
        logger.info("Backtest results saved", output_dir=str(output_dir), files=saved_files)
        return saved_files
    
    @staticmethod
    def load_backtest_results(
        input_dir: Union[str, Path],
        prefix: str = "backtest"
    ) -> Dict[str, Any]:
        """Load backtest results from files."""
        
        input_dir = Path(input_dir)
        
        if not input_dir.exists():
            raise FileNotFoundError(f"Directory not found: {input_dir}")
        
        results = {}
        
        # Load performance history
        perf_path = input_dir / f"{prefix}_performance_history.parquet"
        if perf_path.exists():
            results["performance_history"] = IOUtils.load_dataframe(perf_path)
        
        # Load weights history
        weights_path = input_dir / f"{prefix}_weights_history.parquet"
        if weights_path.exists():
            results["weights_history"] = IOUtils.load_dataframe(weights_path)
        
        # Load trade history
        trades_path = input_dir / f"{prefix}_trade_history.parquet"
        if trades_path.exists():
            results["trade_history"] = IOUtils.load_dataframe(trades_path).to_dict('records')
        
        # Load metrics
        metrics_path = input_dir / f"{prefix}_metrics.json"
        if metrics_path.exists():
            results["metrics"] = IOUtils.load_dict(metrics_path)
        
        # Load configuration
        config_path = input_dir / f"{prefix}_config.yaml"
        if config_path.exists():
            results["config"] = IOUtils.load_dict(config_path)
        
        logger.info("Backtest results loaded", input_dir=str(input_dir), files=list(results.keys()))
        return results
    
    @staticmethod
    def save_portfolio_weights(
        weights: Union[pd.DataFrame, Dict[str, float]],
        filepath: Union[str, Path],
        format: str = "csv"
    ) -> None:
        """Save portfolio weights to file."""
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        if isinstance(weights, dict):
            weights = pd.Series(weights)
        
        if format == "csv":
            weights.to_csv(filepath, header=True)
        elif format == "json":
            weights.to_json(filepath, indent=2)
        elif format == "excel":
            weights.to_excel(filepath, header=True)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        logger.debug("Portfolio weights saved", filepath=str(filepath), format=format)
    
    @staticmethod
    def load_portfolio_weights(
        filepath: Union[str, Path],
        format: str = "csv"
    ) -> pd.Series:
        """Load portfolio weights from file."""
        
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        
        if format == "csv":
            weights = pd.read_csv(filepath, index_col=0, squeeze=True)
        elif format == "json":
            weights = pd.read_json(filepath, typ='series')
        elif format == "excel":
            weights = pd.read_excel(filepath, index_col=0, squeeze=True)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        logger.debug("Portfolio weights loaded", filepath=str(filepath), format=format)
        return weights
    
    @staticmethod
    def save_regime_results(
        regime_results: Dict[str, Any],
        output_dir: Union[str, Path],
        prefix: str = "regime"
    ) -> Dict[str, Path]:
        """Save regime detection results to files."""
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        saved_files = {}
        
        # Save regime labels
        if "regime_labels" in regime_results:
            labels_path = output_dir / f"{prefix}_labels.npy"
            IOUtils.save_numpy_array(regime_results["regime_labels"], labels_path)
            saved_files["regime_labels"] = labels_path
        
        # Save regime probabilities
        if "regime_probs" in regime_results:
            probs_path = output_dir / f"{prefix}_probabilities.npy"
            IOUtils.save_numpy_array(regime_results["regime_probs"], probs_path)
            saved_files["regime_probs"] = probs_path
        
        # Save regime characteristics
        if "regime_characteristics" in regime_results:
            char_path = output_dir / f"{prefix}_characteristics.json"
            IOUtils.save_dict(regime_results["regime_characteristics"], char_path)
            saved_files["regime_characteristics"] = char_path
        
        # Save features
        if "features" in regime_results:
            features_path = output_dir / f"{prefix}_features.parquet"
            IOUtils.save_dataframe(regime_results["features"], features_path)
            saved_files["features"] = features_path
        
        logger.info("Regime results saved", output_dir=str(output_dir), files=saved_files)
        return saved_files
    
    @staticmethod
    def load_regime_results(
        input_dir: Union[str, Path],
        prefix: str = "regime"
    ) -> Dict[str, Any]:
        """Load regime detection results from files."""
        
        input_dir = Path(input_dir)
        
        if not input_dir.exists():
            raise FileNotFoundError(f"Directory not found: {input_dir}")
        
        results = {}
        
        # Load regime labels
        labels_path = input_dir / f"{prefix}_labels.npy"
        if labels_path.exists():
            results["regime_labels"] = IOUtils.load_numpy_array(labels_path)
        
        # Load regime probabilities
        probs_path = input_dir / f"{prefix}_probabilities.npy"
        if probs_path.exists():
            results["regime_probs"] = IOUtils.load_numpy_array(probs_path)
        
        # Load regime characteristics
        char_path = input_dir / f"{prefix}_characteristics.json"
        if char_path.exists():
            results["regime_characteristics"] = IOUtils.load_dict(char_path)
        
        # Load features
        features_path = input_dir / f"{prefix}_features.parquet"
        if features_path.exists():
            results["features"] = IOUtils.load_dataframe(features_path)
        
        logger.info("Regime results loaded", input_dir=str(input_dir), files=list(results.keys()))
        return results
    
    @staticmethod
    def create_directory_structure(base_dir: Union[str, Path]) -> None:
        """Create standard directory structure for the project."""
        
        base_dir = Path(base_dir)
        
        directories = [
            "data/raw",
            "data/interim",
            "data/processed",
            "results",
            "logs",
            "configs",
            "configs/policy.examples",
            "research",
            "tests",
            "mlruns"
        ]
        
        for directory in directories:
            dir_path = base_dir / directory
            dir_path.mkdir(parents=True, exist_ok=True)
        
        logger.info("Directory structure created", base_dir=str(base_dir))
    
    @staticmethod
    def cleanup_old_files(
        directory: Union[str, Path],
        pattern: str = "*.tmp",
        max_age_days: int = 7
    ) -> None:
        """Clean up old temporary files."""
        
        directory = Path(directory)
        
        if not directory.exists():
            return
        
        import time
        from datetime import datetime, timedelta
        
        cutoff_time = time.time() - (max_age_days * 24 * 60 * 60)
        
        for file_path in directory.glob(pattern):
            if file_path.is_file() and file_path.stat().st_mtime < cutoff_time:
                file_path.unlink()
                logger.debug("Old file removed", filepath=str(file_path))
        
        logger.info("Old files cleaned up", directory=str(directory), pattern=pattern)
