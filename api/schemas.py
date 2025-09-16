"""
Pydantic schemas for the Omega Portfolio Engine API.
"""

from typing import Dict, List, Optional, Union, Any
from datetime import datetime
from pydantic import BaseModel, Field, validator
import pandas as pd
import numpy as np


class PortfolioWeights(BaseModel):
    """Portfolio weights schema."""
    
    weights: Dict[str, float] = Field(..., description="Asset weights")
    total_weight: float = Field(..., description="Total weight (should be 1.0)")
    
    @validator('total_weight')
    def validate_total_weight(cls, v):
        if not np.isclose(v, 1.0, atol=1e-6):
            raise ValueError("Total weight must be 1.0")
        return v


class OptimizationRequest(BaseModel):
    """Portfolio optimization request schema."""
    
    # Data
    returns: List[List[float]] = Field(..., description="Historical returns matrix")
    asset_names: List[str] = Field(..., description="Asset names")
    dates: List[str] = Field(..., description="Date strings")
    
    # Optimization parameters
    objective: str = Field("gmv", description="Optimization objective (gmv, omega)")
    theta: float = Field(0.02, description="Omega threshold")
    bounds: Optional[Tuple[float, float]] = Field(None, description="Weight bounds")
    long_only: bool = Field(True, description="Long-only constraint")
    
    # Constraints
    max_weight: Optional[float] = Field(None, description="Maximum single asset weight")
    min_weight: Optional[float] = Field(None, description="Minimum single asset weight")
    sector_bounds: Optional[Dict[str, Tuple[float, float]]] = Field(None, description="Sector bounds")
    
    # Risk constraints
    cvar_cap: Optional[float] = Field(None, description="CVaR constraint cap")
    max_volatility: Optional[float] = Field(None, description="Maximum volatility")
    max_drawdown: Optional[float] = Field(None, description="Maximum drawdown")
    
    # Turnover
    turnover_penalty: float = Field(0.0, description="Turnover penalty")
    w_prev: Optional[List[float]] = Field(None, description="Previous weights")
    
    @validator('returns')
    def validate_returns(cls, v):
        if not v or not v[0]:
            raise ValueError("Returns matrix cannot be empty")
        return v
    
    @validator('asset_names')
    def validate_asset_names(cls, v):
        if not v:
            raise ValueError("Asset names cannot be empty")
        return v
    
    @validator('dates')
    def validate_dates(cls, v):
        if not v:
            raise ValueError("Dates cannot be empty")
        return v


class OptimizationResponse(BaseModel):
    """Portfolio optimization response schema."""
    
    weights: Dict[str, float] = Field(..., description="Optimal portfolio weights")
    objective_value: float = Field(..., description="Objective function value")
    status: str = Field(..., description="Optimization status")
    solve_time: float = Field(..., description="Solve time in seconds")
    error: Optional[str] = Field(None, description="Error message if failed")


class BacktestRequest(BaseModel):
    """Portfolio backtest request schema."""
    
    # Data
    returns: List[List[float]] = Field(..., description="Historical returns matrix")
    asset_names: List[str] = Field(..., description="Asset names")
    dates: List[str] = Field(..., description="Date strings")
    
    # Backtest parameters
    initial_capital: float = Field(1000000.0, description="Initial capital")
    rebalance_frequency: str = Field("monthly", description="Rebalancing frequency")
    transaction_costs: float = Field(0.0005, description="Transaction costs")
    slippage: float = Field(0.0002, description="Slippage")
    
    # Walk-forward parameters
    train_months: int = Field(120, description="Training window in months")
    test_months: int = Field(12, description="Test window in months")
    step_months: int = Field(1, description="Step size in months")
    
    # Optimization parameters
    objective: str = Field("gmv", description="Optimization objective")
    theta: float = Field(0.02, description="Omega threshold")
    bounds: Optional[Tuple[float, float]] = Field(None, description="Weight bounds")
    long_only: bool = Field(True, description="Long-only constraint")
    
    # Constraints
    max_weight: Optional[float] = Field(None, description="Maximum single asset weight")
    min_weight: Optional[float] = Field(None, description="Minimum single asset weight")
    sector_bounds: Optional[Dict[str, Tuple[float, float]]] = Field(None, description="Sector bounds")
    
    # Risk constraints
    cvar_cap: Optional[float] = Field(None, description="CVaR constraint cap")
    max_volatility: Optional[float] = Field(None, description="Maximum volatility")
    max_drawdown: Optional[float] = Field(None, description="Maximum drawdown")
    
    # Turnover
    turnover_penalty: float = Field(0.0, description="Turnover penalty")
    
    @validator('returns')
    def validate_returns(cls, v):
        if not v or not v[0]:
            raise ValueError("Returns matrix cannot be empty")
        return v
    
    @validator('asset_names')
    def validate_asset_names(cls, v):
        if not v:
            raise ValueError("Asset names cannot be empty")
        return v
    
    @validator('dates')
    def validate_dates(cls, v):
        if not v:
            raise ValueError("Dates cannot be empty")
        return v


class BacktestResponse(BaseModel):
    """Portfolio backtest response schema."""
    
    # Performance metrics
    total_return: float = Field(..., description="Total return")
    annualized_return: float = Field(..., description="Annualized return")
    annualized_volatility: float = Field(..., description="Annualized volatility")
    sharpe_ratio: float = Field(..., description="Sharpe ratio")
    sortino_ratio: float = Field(..., description="Sortino ratio")
    calmar_ratio: float = Field(..., description="Calmar ratio")
    max_drawdown: float = Field(..., description="Maximum drawdown")
    
    # Risk metrics
    var_95: float = Field(..., description="Value at Risk (95%)")
    cvar_95: float = Field(..., description="Conditional Value at Risk (95%)")
    
    # Turnover metrics
    annual_turnover: float = Field(..., description="Annual turnover")
    cost_drag: float = Field(..., description="Cost drag")
    
    # Portfolio composition
    final_weights: Dict[str, float] = Field(..., description="Final portfolio weights")
    
    # Performance history
    performance_history: List[Dict[str, Any]] = Field(..., description="Performance history")
    
    # Trade history
    trade_history: List[Dict[str, Any]] = Field(..., description="Trade history")
    
    # Metadata
    start_date: str = Field(..., description="Backtest start date")
    end_date: str = Field(..., description="Backtest end date")
    n_observations: int = Field(..., description="Number of observations")
    solve_time: float = Field(..., description="Total solve time in seconds")


class RegimeDetectionRequest(BaseModel):
    """Regime detection request schema."""
    
    # Data
    features: List[List[float]] = Field(..., description="Feature matrix")
    feature_names: List[str] = Field(..., description="Feature names")
    dates: List[str] = Field(..., description="Date strings")
    
    # Model parameters
    method: str = Field("hmm", description="Regime detection method (hmm, lstm, gmm)")
    n_regimes: int = Field(3, description="Number of regimes")
    
    # HMM parameters
    n_iter: int = Field(100, description="Number of iterations")
    random_state: int = Field(42, description="Random state")
    
    # LSTM parameters
    sequence_length: int = Field(60, description="Sequence length")
    hidden_units: int = Field(64, description="Hidden units")
    dropout: float = Field(0.2, description="Dropout rate")
    epochs: int = Field(100, description="Number of epochs")
    batch_size: int = Field(32, description="Batch size")
    
    @validator('features')
    def validate_features(cls, v):
        if not v or not v[0]:
            raise ValueError("Feature matrix cannot be empty")
        return v
    
    @validator('feature_names')
    def validate_feature_names(cls, v):
        if not v:
            raise ValueError("Feature names cannot be empty")
        return v
    
    @validator('dates')
    def validate_dates(cls, v):
        if not v:
            raise ValueError("Dates cannot be empty")
        return v


class RegimeDetectionResponse(BaseModel):
    """Regime detection response schema."""
    
    # Regime labels
    regime_labels: List[int] = Field(..., description="Regime labels")
    regime_probs: List[List[float]] = Field(..., description="Regime probabilities")
    
    # Regime characteristics
    regime_characteristics: Dict[str, Any] = Field(..., description="Regime characteristics")
    
    # Model information
    method: str = Field(..., description="Detection method used")
    n_regimes: int = Field(..., description="Number of regimes")
    n_observations: int = Field(..., description="Number of observations")
    
    # Performance metrics
    accuracy: Optional[float] = Field(None, description="Model accuracy")
    log_likelihood: Optional[float] = Field(None, description="Log likelihood")
    
    # Metadata
    fit_time: float = Field(..., description="Fit time in seconds")
    predict_time: float = Field(..., description="Predict time in seconds")


class JobStatus(BaseModel):
    """Job status schema."""
    
    job_id: str = Field(..., description="Job ID")
    status: str = Field(..., description="Job status (pending, running, completed, failed)")
    progress: float = Field(0.0, description="Progress percentage (0-100)")
    message: Optional[str] = Field(None, description="Status message")
    created_at: datetime = Field(..., description="Job creation time")
    started_at: Optional[datetime] = Field(None, description="Job start time")
    completed_at: Optional[datetime] = Field(None, description="Job completion time")
    error: Optional[str] = Field(None, description="Error message if failed")
    result: Optional[Dict[str, Any]] = Field(None, description="Job result if completed")


class JobRequest(BaseModel):
    """Job request schema."""
    
    job_type: str = Field(..., description="Job type (optimize, backtest, regime)")
    parameters: Dict[str, Any] = Field(..., description="Job parameters")
    priority: int = Field(0, description="Job priority (higher = more priority)")
    timeout: Optional[int] = Field(None, description="Timeout in seconds")


class JobResponse(BaseModel):
    """Job response schema."""
    
    job_id: str = Field(..., description="Job ID")
    status: str = Field(..., description="Job status")
    message: str = Field(..., description="Response message")
    created_at: datetime = Field(..., description="Job creation time")


class HealthCheck(BaseModel):
    """Health check response schema."""
    
    status: str = Field(..., description="Service status")
    version: str = Field(..., description="Service version")
    uptime: float = Field(..., description="Uptime in seconds")
    memory_usage: float = Field(..., description="Memory usage in MB")
    cpu_usage: float = Field(..., description="CPU usage percentage")
    active_jobs: int = Field(..., description="Number of active jobs")
    total_jobs: int = Field(..., description="Total number of jobs")


class ErrorResponse(BaseModel):
    """Error response schema."""
    
    error: str = Field(..., description="Error message")
    error_type: str = Field(..., description="Error type")
    error_code: int = Field(..., description="Error code")
    timestamp: datetime = Field(..., description="Error timestamp")
    request_id: Optional[str] = Field(None, description="Request ID")
    details: Optional[Dict[str, Any]] = Field(None, description="Error details")
