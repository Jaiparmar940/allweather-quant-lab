"""
FastAPI application for the Omega Portfolio Engine.
"""

import logging
import uuid
import asyncio
from datetime import datetime
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import structlog

from .schemas import (
    OptimizationRequest, OptimizationResponse,
    BacktestRequest, BacktestResponse,
    RegimeDetectionRequest, RegimeDetectionResponse,
    JobRequest, JobResponse, JobStatus,
    HealthCheck, ErrorResponse
)
from engine.optimize import GMVOptimizer, OmegaOptimizer
from engine.backtest import BacktestSimulator, WalkForwardEngine
from engine.signals import RegimeDetector
from engine.utils.logging import setup_logging

# Setup logging
setup_logging()

# Create FastAPI app
app = FastAPI(
    title="Omega Portfolio Engine API",
    description="Regime-aware portfolio optimization and backtesting API",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory job storage (in production, use Redis or database)
jobs: Dict[str, JobStatus] = {}

# Global optimizers
gmv_optimizer = GMVOptimizer()
omega_optimizer = OmegaOptimizer()

# Global backtest simulator
backtest_simulator = BacktestSimulator()

# Global walk-forward engine
walk_forward_engine = WalkForwardEngine()

# Global regime detector
regime_detector = RegimeDetector()

logger = structlog.get_logger(__name__)


@app.get("/health", response_model=HealthCheck)
async def health_check():
    """Health check endpoint."""
    
    import psutil
    import time
    
    # Get system metrics
    memory_usage = psutil.virtual_memory().used / 1024 / 1024  # MB
    cpu_usage = psutil.cpu_percent()
    
    # Count active jobs
    active_jobs = sum(1 for job in jobs.values() if job.status == "running")
    total_jobs = len(jobs)
    
    return HealthCheck(
        status="healthy",
        version="0.1.0",
        uptime=time.time(),
        memory_usage=memory_usage,
        cpu_usage=cpu_usage,
        active_jobs=active_jobs,
        total_jobs=total_jobs
    )


@app.post("/optimize", response_model=OptimizationResponse)
async def optimize_portfolio(request: OptimizationRequest):
    """Optimize portfolio weights."""
    
    try:
        logger.info("Portfolio optimization requested", objective=request.objective)
        
        # Convert data to pandas DataFrame
        returns_df = pd.DataFrame(
            request.returns,
            columns=request.asset_names,
            index=pd.to_datetime(request.dates)
        )
        
        # Set bounds
        bounds = request.bounds or (0.0, 1.0)
        
        # Optimize based on objective
        if request.objective == "gmv":
            result = gmv_optimizer.solve_gmv_with_returns(
                returns=returns_df,
                bounds=bounds,
                long_only=request.long_only
            )
        elif request.objective == "omega":
            result = omega_optimizer.solve_omega_with_returns(
                returns=returns_df,
                theta=request.theta,
                bounds=bounds,
                long_only=request.long_only,
                cvar_cap=request.cvar_cap,
                turnover_penalty=request.turnover_penalty,
                w_prev=np.array(request.w_prev) if request.w_prev else None
            )
        else:
            raise HTTPException(status_code=400, detail=f"Unknown objective: {request.objective}")
        
        # Convert weights to dictionary
        if result["weights"] is not None:
            weights_dict = dict(zip(request.asset_names, result["weights"]))
        else:
            weights_dict = {}
        
        return OptimizationResponse(
            weights=weights_dict,
            objective_value=result["objective_value"],
            status=result["status"],
            solve_time=result["solve_time"],
            error=result.get("error")
        )
        
    except Exception as e:
        logger.error("Portfolio optimization failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/backtest", response_model=BacktestResponse)
async def run_backtest(request: BacktestRequest):
    """Run portfolio backtest."""
    
    try:
        logger.info("Portfolio backtest requested", objective=request.objective)
        
        # Convert data to pandas DataFrame
        returns_df = pd.DataFrame(
            request.returns,
            columns=request.asset_names,
            index=pd.to_datetime(request.dates)
        )
        
        # Set bounds
        bounds = request.bounds or (0.0, 1.0)
        
        # Run walk-forward backtest
        def optimizer_func(returns):
            if request.objective == "gmv":
                result = gmv_optimizer.solve_gmv_with_returns(
                    returns=returns,
                    bounds=bounds,
                    long_only=request.long_only
                )
            elif request.objective == "omega":
                result = omega_optimizer.solve_omega_with_returns(
                    returns=returns,
                    theta=request.theta,
                    bounds=bounds,
                    long_only=request.long_only,
                    cvar_cap=request.cvar_cap,
                    turnover_penalty=request.turnover_penalty
                )
            else:
                raise ValueError(f"Unknown objective: {request.objective}")
            
            if result["weights"] is not None:
                return pd.Series(result["weights"], index=returns.columns)
            else:
                return pd.Series(0.0, index=returns.columns)
        
        # Run walk-forward backtest
        try:
            walk_forward_results = walk_forward_engine.run_walk_forward(
                returns=returns_df,
                optimizer_func=optimizer_func
            )
        except Exception as e:
            logger.error("Walk-forward backtest failed", error=str(e))
            raise HTTPException(status_code=500, detail=f"Walk-forward backtest failed: {str(e)}")
        
        # Extract results
        if walk_forward_results and "all_returns" in walk_forward_results and len(walk_forward_results["all_returns"]) > 0:
            all_returns = walk_forward_results["all_returns"]
            
            # Calculate metrics
            total_return = (1 + all_returns).prod() - 1
            annualized_return = (1 + total_return) ** (252 / len(all_returns)) - 1
            annualized_vol = all_returns.std() * np.sqrt(252)
            sharpe_ratio = annualized_return / annualized_vol if annualized_vol > 0 else 0
            
            # Calculate maximum drawdown
            cumulative = (1 + all_returns).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            max_drawdown = drawdown.min()
            
            # Calculate VaR and CVaR
            var_95 = all_returns.quantile(0.05)
            cvar_95 = all_returns[all_returns <= var_95].mean()
            
            # Calculate turnover (simplified)
            annual_turnover = 0.0  # Would need weights history for accurate calculation
            cost_drag = annual_turnover * request.transaction_costs
            
            # Create performance history
            performance_history = []
            for i, (date, ret) in enumerate(all_returns.items()):
                performance_history.append({
                    "date": date.isoformat(),
                    "return": ret,
                    "cumulative_return": cumulative.iloc[i]
                })
            
            # Create trade history (simplified)
            trade_history = []
            
            return BacktestResponse(
                total_return=total_return,
                annualized_return=annualized_return,
                annualized_volatility=annualized_vol,
                sharpe_ratio=sharpe_ratio,
                sortino_ratio=0.0,  # Would need to calculate
                calmar_ratio=0.0,   # Would need to calculate
                max_drawdown=max_drawdown,
                var_95=var_95,
                cvar_95=cvar_95,
                annual_turnover=annual_turnover,
                cost_drag=cost_drag,
                final_weights={},  # Would need to extract from results
                performance_history=performance_history,
                trade_history=trade_history,
                start_date=request.dates[0],
                end_date=request.dates[-1],
                n_observations=len(all_returns),
                solve_time=0.0  # Would need to track
            )
        else:
            # Fallback: Simple single-period backtest
            logger.warning("Walk-forward backtest produced no results, falling back to simple backtest")
            
            # Use all data for a simple backtest
            try:
                # Optimize on all data
                weights = optimizer_func(returns_df)
                
                # Calculate portfolio returns
                portfolio_returns = (returns_df * weights).sum(axis=1)
                
                # Calculate metrics
                total_return = (1 + portfolio_returns).prod() - 1
                annualized_return = (1 + total_return) ** (252 / len(portfolio_returns)) - 1
                annualized_vol = portfolio_returns.std() * np.sqrt(252)
                sharpe_ratio = annualized_return / annualized_vol if annualized_vol > 0 else 0
                
                # Calculate maximum drawdown
                cumulative = (1 + portfolio_returns).cumprod()
                running_max = cumulative.expanding().max()
                drawdown = (cumulative - running_max) / running_max
                max_drawdown = drawdown.min()
                
                # Calculate VaR and CVaR
                var_95 = portfolio_returns.quantile(0.05)
                cvar_95 = portfolio_returns[portfolio_returns <= var_95].mean()
                
                # Create performance history
                performance_history = []
                for i, (date, ret) in enumerate(portfolio_returns.items()):
                    performance_history.append({
                        "date": date.isoformat(),
                        "return": ret,
                        "cumulative_return": cumulative.iloc[i]
                    })
                
                return BacktestResponse(
                    total_return=total_return,
                    annualized_return=annualized_return,
                    annualized_volatility=annualized_vol,
                    sharpe_ratio=sharpe_ratio,
                    sortino_ratio=0.0,
                    calmar_ratio=0.0,
                    max_drawdown=max_drawdown,
                    var_95=var_95,
                    cvar_95=cvar_95,
                    annual_turnover=0.0,
                    cost_drag=0.0,
                    final_weights=weights.to_dict(),
                    performance_history=performance_history,
                    trade_history=[],
                    start_date=request.dates[0],
                    end_date=request.dates[-1],
                    n_observations=len(portfolio_returns),
                    solve_time=0.0
                )
                
            except Exception as e:
                logger.error("Simple backtest also failed", error=str(e))
                raise HTTPException(status_code=500, detail=f"Backtest failed: {str(e)}")
        
    except Exception as e:
        logger.error("Portfolio backtest failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/regime", response_model=RegimeDetectionResponse)
async def detect_regimes(request: RegimeDetectionRequest):
    """Detect market regimes."""
    
    try:
        logger.info("Regime detection requested", method=request.method)
        
        # Convert data to pandas DataFrame
        features_df = pd.DataFrame(
            request.features,
            columns=request.feature_names,
            index=pd.to_datetime(request.dates)
        )
        
        # Configure regime detector
        regime_detector.method = request.method
        regime_detector.n_regimes = request.n_regimes
        
        # Fit regime detector
        regime_detector.fit(features_df)
        
        # Predict regimes
        regime_labels = regime_detector.predict(features_df)
        regime_probs = regime_detector.predict_proba(features_df)
        
        # Get regime characteristics
        regime_characteristics = regime_detector.get_regime_characteristics(features_df)
        
        return RegimeDetectionResponse(
            regime_labels=regime_labels.tolist(),
            regime_probs=regime_probs.tolist(),
            regime_characteristics=regime_characteristics,
            method=request.method,
            n_regimes=request.n_regimes,
            n_observations=len(features_df),
            accuracy=None,  # Would need to calculate
            log_likelihood=None,  # Would need to calculate
            fit_time=0.0,  # Would need to track
            predict_time=0.0  # Would need to track
        )
        
    except Exception as e:
        logger.error("Regime detection failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/jobs", response_model=JobResponse)
async def create_job(request: JobRequest, background_tasks: BackgroundTasks):
    """Create a new job."""
    
    try:
        # Generate job ID
        job_id = str(uuid.uuid4())
        
        # Create job status
        job_status = JobStatus(
            job_id=job_id,
            status="pending",
            progress=0.0,
            message="Job created",
            created_at=datetime.now()
        )
        
        # Store job
        jobs[job_id] = job_status
        
        # Start job in background
        background_tasks.add_task(run_job, job_id, request)
        
        return JobResponse(
            job_id=job_id,
            status="pending",
            message="Job created successfully",
            created_at=job_status.created_at
        )
        
    except Exception as e:
        logger.error("Job creation failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/jobs/{job_id}", response_model=JobStatus)
async def get_job_status(job_id: str):
    """Get job status."""
    
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return jobs[job_id]


@app.get("/jobs/{job_id}/result")
async def get_job_result(job_id: str):
    """Get job result."""
    
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[job_id]
    
    if job.status != "completed":
        raise HTTPException(status_code=400, detail="Job not completed")
    
    return job.result


@app.delete("/jobs/{job_id}")
async def delete_job(job_id: str):
    """Delete a job."""
    
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    del jobs[job_id]
    
    return {"message": "Job deleted successfully"}


async def run_job(job_id: str, request: JobRequest):
    """Run a job in the background."""
    
    try:
        # Update job status
        jobs[job_id].status = "running"
        jobs[job_id].started_at = datetime.now()
        jobs[job_id].progress = 0.0
        jobs[job_id].message = "Job started"
        
        # Run job based on type
        if request.job_type == "optimize":
            result = await run_optimization_job(request.parameters)
        elif request.job_type == "backtest":
            result = await run_backtest_job(request.parameters)
        elif request.job_type == "regime":
            result = await run_regime_job(request.parameters)
        else:
            raise ValueError(f"Unknown job type: {request.job_type}")
        
        # Update job status
        jobs[job_id].status = "completed"
        jobs[job_id].completed_at = datetime.now()
        jobs[job_id].progress = 100.0
        jobs[job_id].message = "Job completed successfully"
        jobs[job_id].result = result
        
    except Exception as e:
        # Update job status
        jobs[job_id].status = "failed"
        jobs[job_id].completed_at = datetime.now()
        jobs[job_id].message = "Job failed"
        jobs[job_id].error = str(e)
        
        logger.error("Job failed", job_id=job_id, error=str(e))


async def run_optimization_job(parameters: Dict[str, Any]) -> Dict[str, Any]:
    """Run optimization job."""
    
    # This would contain the actual optimization logic
    # For now, return a placeholder
    return {"message": "Optimization job completed"}


async def run_backtest_job(parameters: Dict[str, Any]) -> Dict[str, Any]:
    """Run backtest job."""
    
    # This would contain the actual backtest logic
    # For now, return a placeholder
    return {"message": "Backtest job completed"}


async def run_regime_job(parameters: Dict[str, Any]) -> Dict[str, Any]:
    """Run regime detection job."""
    
    # This would contain the actual regime detection logic
    # For now, return a placeholder
    return {"message": "Regime detection job completed"}


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler."""
    
    logger.error("Unhandled exception", error=str(exc), exc_info=True)
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "error_type": type(exc).__name__,
            "error_code": 500,
            "timestamp": datetime.now().isoformat(),
            "details": {"message": str(exc)}
        }
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
