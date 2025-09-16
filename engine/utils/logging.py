"""
Logging utilities for the Omega Portfolio Engine.
"""

import logging
import sys
from pathlib import Path
from typing import Optional, Dict, Any, Union
import structlog
from structlog.stdlib import LoggerFactory


def setup_logging(
    log_level: str = "INFO",
    log_format: str = "json",
    log_file: Optional[Union[str, Path]] = None,
    log_dir: Union[str, Path] = "./logs"
) -> None:
    """
    Setup logging configuration for the Omega Portfolio Engine.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_format: Log format (json, text)
        log_file: Log file path (optional)
        log_dir: Log directory path
    """
    
    # Create log directory
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Configure structlog
    if log_format == "json":
        structlog.configure(
            processors=[
                structlog.stdlib.filter_by_level,
                structlog.stdlib.add_logger_name,
                structlog.stdlib.add_log_level,
                structlog.stdlib.PositionalArgumentsFormatter(),
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                structlog.processors.UnicodeDecoder(),
                structlog.processors.JSONRenderer()
            ],
            context_class=dict,
            logger_factory=LoggerFactory(),
            wrapper_class=structlog.stdlib.BoundLogger,
            cache_logger_on_first_use=True,
        )
    else:
        structlog.configure(
            processors=[
                structlog.stdlib.filter_by_level,
                structlog.stdlib.add_logger_name,
                structlog.stdlib.add_log_level,
                structlog.stdlib.PositionalArgumentsFormatter(),
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                structlog.processors.UnicodeDecoder(),
                structlog.dev.ConsoleRenderer()
            ],
            context_class=dict,
            logger_factory=LoggerFactory(),
            wrapper_class=structlog.stdlib.BoundLogger,
            cache_logger_on_first_use=True,
        )
    
    # Configure standard library logging
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(message)s",
        stream=sys.stdout
    )
    
    # Add file handler if specified
    if log_file:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, log_level.upper()))
        
        if log_format == "json":
            file_handler.setFormatter(logging.Formatter('%(message)s'))
        else:
            file_handler.setFormatter(logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            ))
        
        logging.getLogger().addHandler(file_handler)
    
    # Create logger
    logger = structlog.get_logger()
    logger.info("Logging configured", level=log_level, format=log_format)


def get_logger(name: str) -> structlog.BoundLogger:
    """Get a logger instance."""
    return structlog.get_logger(name)


def log_function_call(func):
    """Decorator to log function calls."""
    def wrapper(*args, **kwargs):
        logger = structlog.get_logger(func.__module__)
        logger.info("Function called", function=func.__name__, args=args, kwargs=kwargs)
        try:
            result = func(*args, **kwargs)
            logger.info("Function completed", function=func.__name__, success=True)
            return result
        except Exception as e:
            logger.error("Function failed", function=func.__name__, error=str(e))
            raise
    return wrapper


def log_performance(func):
    """Decorator to log function performance."""
    import time
    
    def wrapper(*args, **kwargs):
        logger = structlog.get_logger(func.__module__)
        start_time = time.time()
        
        try:
            result = func(*args, **kwargs)
            end_time = time.time()
            duration = end_time - start_time
            
            logger.info("Function performance", 
                       function=func.__name__, 
                       duration=duration,
                       success=True)
            return result
        except Exception as e:
            end_time = time.time()
            duration = end_time - start_time
            
            logger.error("Function performance", 
                        function=func.__name__, 
                        duration=duration,
                        error=str(e))
            raise
    return wrapper


def log_data_quality(data: Any, name: str) -> None:
    """Log data quality metrics."""
    logger = structlog.get_logger()
    
    if hasattr(data, 'shape'):
        logger.info("Data quality", 
                   name=name,
                   shape=data.shape,
                   dtype=str(data.dtype))
    elif hasattr(data, '__len__'):
        logger.info("Data quality", 
                   name=name,
                   length=len(data),
                   type=type(data).__name__)
    else:
        logger.info("Data quality", 
                   name=name,
                   type=type(data).__name__)


def log_optimization_result(result: Dict[str, Any]) -> None:
    """Log optimization result."""
    logger = structlog.get_logger()
    
    logger.info("Optimization result",
               status=result.get("status", "unknown"),
               objective_value=result.get("objective_value", None),
               solve_time=result.get("solve_time", None))


def log_backtest_result(result: Dict[str, Any]) -> None:
    """Log backtest result."""
    logger = structlog.get_logger()
    
    logger.info("Backtest result",
               total_return=result.get("total_return", None),
               sharpe_ratio=result.get("sharpe_ratio", None),
               max_drawdown=result.get("max_drawdown", None),
               final_value=result.get("final_value", None))


def log_regime_detection_result(result: Dict[str, Any]) -> None:
    """Log regime detection result."""
    logger = structlog.get_logger()
    
    logger.info("Regime detection result",
               n_regimes=result.get("n_regimes", None),
               n_observations=result.get("n_observations", None),
               method=result.get("method", None))


def log_portfolio_weights(weights: Dict[str, float], name: str = "portfolio") -> None:
    """Log portfolio weights."""
    logger = structlog.get_logger()
    
    logger.info("Portfolio weights",
               name=name,
               n_assets=len(weights),
               total_weight=sum(weights.values()),
               weights=weights)


def log_error(error: Exception, context: str = "") -> None:
    """Log error with context."""
    logger = structlog.get_logger()
    
    logger.error("Error occurred",
                context=context,
                error_type=type(error).__name__,
                error_message=str(error),
                exc_info=True)


def log_warning(message: str, context: str = "") -> None:
    """Log warning with context."""
    logger = structlog.get_logger()
    
    logger.warning("Warning",
                  context=context,
                  message=message)


def log_info(message: str, context: str = "", **kwargs) -> None:
    """Log info message with context."""
    logger = structlog.get_logger()
    
    logger.info("Info",
               context=context,
               message=message,
               **kwargs)


def log_debug(message: str, context: str = "", **kwargs) -> None:
    """Log debug message with context."""
    logger = structlog.get_logger()
    
    logger.debug("Debug",
                context=context,
                message=message,
                **kwargs)
