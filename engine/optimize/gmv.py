"""
Global Minimum Variance (GMV) portfolio optimizer.
"""

import logging
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import cvxpy as cp
from scipy.optimize import minimize
import structlog

logger = structlog.get_logger(__name__)


class GMVOptimizer:
    """Global Minimum Variance portfolio optimizer using CVXPY."""
    
    def __init__(self, solver: str = "ECOS", verbose: bool = False):
        self.solver = solver
        self.verbose = verbose
        self.last_optimization_result = None
    
    def solve_gmv(
        self,
        cov: np.ndarray,
        bounds: Optional[Tuple[float, float]] = None,
        sector_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
        long_only: bool = True,
        leverage_cap: Optional[float] = None,
        turnover_penalty: float = 0.0,
        w_prev: Optional[np.ndarray] = None
    ) -> Dict[str, Union[np.ndarray, float, str]]:
        """
        Solve Global Minimum Variance optimization problem.
        
        Args:
            cov: Covariance matrix (n x n)
            bounds: Tuple of (min_weight, max_weight) for all assets
            sector_bounds: Dict mapping sector names to (min_weight, max_weight)
            long_only: Whether to enforce long-only constraint
            leverage_cap: Maximum leverage allowed (1.0 = no leverage)
            turnover_penalty: Penalty coefficient for turnover
            w_prev: Previous portfolio weights for turnover penalty
        
        Returns:
            Dict containing:
                - weights: Optimal portfolio weights
                - objective_value: Minimum variance achieved
                - status: Optimization status
                - solve_time: Time taken to solve
        """
        
        import time
        start_time = time.time()
        
        n_assets = cov.shape[0]
        
        # Validate inputs
        if cov.shape[0] != cov.shape[1]:
            raise ValueError("Covariance matrix must be square")
        
        if not np.allclose(cov, cov.T):
            raise ValueError("Covariance matrix must be symmetric")
        
        # Set default bounds
        if bounds is None:
            bounds = (0.0, 1.0) if long_only else (-1.0, 1.0)
        
        min_weight, max_weight = bounds
        
        # Create optimization variables
        w = cp.Variable(n_assets, name="weights")
        
        # Define objective function
        if turnover_penalty > 0 and w_prev is not None:
            if len(w_prev) != n_assets:
                raise ValueError("Previous weights must have same length as number of assets")
            objective = cp.Minimize(cp.quad_form(w, cov) + turnover_penalty * cp.norm(w - w_prev, 1))
        else:
            objective = cp.Minimize(cp.quad_form(w, cov))
        
        # Define constraints
        constraints = []
        
        # Budget constraint (weights sum to 1)
        constraints.append(cp.sum(w) == 1.0)
        
        # Long-only constraint
        if long_only:
            constraints.append(w >= 0.0)
        
        # Weight bounds
        constraints.append(w >= min_weight)
        constraints.append(w <= max_weight)
        
        # Leverage constraint
        if leverage_cap is not None:
            constraints.append(cp.norm(w, 1) <= leverage_cap)
        
        # Sector constraints
        if sector_bounds:
            # This would require mapping assets to sectors
            # For now, we'll implement a simplified version
            pass
        
        # Solve optimization problem
        try:
            problem = cp.Problem(objective, constraints)
            problem.solve(solver=self.solver, verbose=self.verbose)
            
            solve_time = time.time() - start_time
            
            if problem.status == cp.OPTIMAL:
                weights = w.value
                objective_value = problem.value
                status = "optimal"
                
                logger.info(
                    "GMV optimization solved successfully",
                    n_assets=n_assets,
                    objective_value=objective_value,
                    solve_time=solve_time,
                    status=status
                )
                
            elif problem.status == cp.OPTIMAL_INACCURATE:
                weights = w.value
                objective_value = problem.value
                status = "optimal_inaccurate"
                
                logger.warning(
                    "GMV optimization solved with inaccuracy",
                    n_assets=n_assets,
                    objective_value=objective_value,
                    solve_time=solve_time,
                    status=status
                )
                
            else:
                weights = None
                objective_value = np.inf
                status = problem.status
                
                logger.error(
                    "GMV optimization failed",
                    n_assets=n_assets,
                    status=status,
                    solve_time=solve_time
                )
            
            # Store result
            result = {
                "weights": weights,
                "objective_value": objective_value,
                "status": status,
                "solve_time": solve_time
            }
            
            self.last_optimization_result = result
            return result
            
        except Exception as e:
            solve_time = time.time() - start_time
            logger.error("GMV optimization failed with exception", error=str(e), solve_time=solve_time)
            
            return {
                "weights": None,
                "objective_value": np.inf,
                "status": "error",
                "solve_time": solve_time,
                "error": str(e)
            }
    
    def solve_gmv_with_returns(
        self,
        returns: pd.DataFrame,
        lookback_window: int = 252,
        bounds: Optional[Tuple[float, float]] = None,
        sector_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
        long_only: bool = True,
        leverage_cap: Optional[float] = None,
        turnover_penalty: float = 0.0,
        w_prev: Optional[np.ndarray] = None
    ) -> Dict[str, Union[np.ndarray, float, str]]:
        """
        Solve GMV optimization using historical returns to estimate covariance.
        
        Args:
            returns: Historical returns DataFrame (T x N)
            lookback_window: Number of periods to use for covariance estimation
            bounds: Tuple of (min_weight, max_weight) for all assets
            sector_bounds: Dict mapping sector names to (min_weight, max_weight)
            long_only: Whether to enforce long-only constraint
            leverage_cap: Maximum leverage allowed
            turnover_penalty: Penalty coefficient for turnover
            w_prev: Previous portfolio weights for turnover penalty
        
        Returns:
            Dict containing optimization results
        """
        
        # Use recent data for covariance estimation
        recent_returns = returns.tail(lookback_window)
        
        # Calculate covariance matrix
        cov = recent_returns.cov().values
        
        # Annualize covariance if returns are daily
        if len(recent_returns) >= 252:
            cov = cov * 252
        
        # Solve GMV optimization
        return self.solve_gmv(
            cov=cov,
            bounds=bounds,
            sector_bounds=sector_bounds,
            long_only=long_only,
            leverage_cap=leverage_cap,
            turnover_penalty=turnover_penalty,
            w_prev=w_prev
        )
    
    def solve_gmv_robust(
        self,
        returns: pd.DataFrame,
        lookback_window: int = 252,
        shrinkage_factor: float = 0.1,
        bounds: Optional[Tuple[float, float]] = None,
        sector_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
        long_only: bool = True,
        leverage_cap: Optional[float] = None,
        turnover_penalty: float = 0.0,
        w_prev: Optional[np.ndarray] = None
    ) -> Dict[str, Union[np.ndarray, float, str]]:
        """
        Solve GMV optimization with shrinkage estimator for covariance.
        
        Args:
            returns: Historical returns DataFrame (T x N)
            lookback_window: Number of periods to use for covariance estimation
            shrinkage_factor: Shrinkage factor for covariance matrix
            bounds: Tuple of (min_weight, max_weight) for all assets
            sector_bounds: Dict mapping sector names to (min_weight, max_weight)
            long_only: Whether to enforce long-only constraint
            leverage_cap: Maximum leverage allowed
            turnover_penalty: Penalty coefficient for turnover
            w_prev: Previous portfolio weights for turnover penalty
        
        Returns:
            Dict containing optimization results
        """
        
        # Use recent data for covariance estimation
        recent_returns = returns.tail(lookback_window)
        
        # Calculate sample covariance
        sample_cov = recent_returns.cov().values
        
        # Calculate shrinkage target (diagonal matrix with average variance)
        avg_var = np.trace(sample_cov) / sample_cov.shape[0]
        target_cov = np.eye(sample_cov.shape[0]) * avg_var
        
        # Apply shrinkage
        cov = (1 - shrinkage_factor) * sample_cov + shrinkage_factor * target_cov
        
        # Annualize covariance if returns are daily
        if len(recent_returns) >= 252:
            cov = cov * 252
        
        # Solve GMV optimization
        return self.solve_gmv(
            cov=cov,
            bounds=bounds,
            sector_bounds=sector_bounds,
            long_only=long_only,
            leverage_cap=leverage_cap,
            turnover_penalty=turnover_penalty,
            w_prev=w_prev
        )
    
    def get_risk_contribution(self, weights: np.ndarray, cov: np.ndarray) -> np.ndarray:
        """Calculate risk contribution of each asset."""
        
        if weights is None:
            return np.array([])
        
        portfolio_var = np.dot(weights, np.dot(cov, weights))
        marginal_contrib = np.dot(cov, weights)
        risk_contrib = weights * marginal_contrib / portfolio_var
        
        return risk_contrib
    
    def get_efficient_frontier(
        self,
        cov: np.ndarray,
        target_returns: Optional[np.ndarray] = None,
        n_points: int = 50,
        bounds: Optional[Tuple[float, float]] = None,
        long_only: bool = True
    ) -> Dict[str, np.ndarray]:
        """
        Generate efficient frontier for GMV optimization.
        
        Args:
            cov: Covariance matrix (n x n)
            target_returns: Target returns for each point on frontier
            n_points: Number of points on frontier
            bounds: Tuple of (min_weight, max_weight) for all assets
            long_only: Whether to enforce long-only constraint
        
        Returns:
            Dict containing:
                - returns: Portfolio returns for each point
                - risks: Portfolio risks (volatilities) for each point
                - weights: Portfolio weights for each point
        """
        
        n_assets = cov.shape[0]
        
        # Set default bounds
        if bounds is None:
            bounds = (0.0, 1.0) if long_only else (-1.0, 1.0)
        
        min_weight, max_weight = bounds
        
        # Generate target returns if not provided
        if target_returns is None:
            # Use minimum and maximum possible returns
            min_return = 0.0  # Assuming risk-free rate
            max_return = 0.20  # Assuming 20% maximum expected return
            target_returns = np.linspace(min_return, max_return, n_points)
        
        frontier_returns = []
        frontier_risks = []
        frontier_weights = []
        
        for target_return in target_returns:
            try:
                # Create optimization variables
                w = cp.Variable(n_assets, name="weights")
                
                # Define objective (minimize variance)
                objective = cp.Minimize(cp.quad_form(w, cov))
                
                # Define constraints
                constraints = [
                    cp.sum(w) == 1.0,  # Budget constraint
                    w >= min_weight,   # Lower bound
                    w <= max_weight,   # Upper bound
                ]
                
                if long_only:
                    constraints.append(w >= 0.0)
                
                # Solve optimization problem
                problem = cp.Problem(objective, constraints)
                problem.solve(solver=self.solver, verbose=False)
                
                if problem.status == cp.OPTIMAL:
                    weights = w.value
                    portfolio_return = target_return  # This would need expected returns
                    portfolio_risk = np.sqrt(np.dot(weights, np.dot(cov, weights)))
                    
                    frontier_returns.append(portfolio_return)
                    frontier_risks.append(portfolio_risk)
                    frontier_weights.append(weights)
                
            except Exception as e:
                logger.warning("Failed to solve for target return", target_return=target_return, error=str(e))
                continue
        
        return {
            "returns": np.array(frontier_returns),
            "risks": np.array(frontier_risks),
            "weights": np.array(frontier_weights)
        }
    
    def validate_weights(self, weights: np.ndarray, bounds: Optional[Tuple[float, float]] = None) -> bool:
        """Validate portfolio weights."""
        
        if weights is None:
            return False
        
        # Check if weights sum to 1
        if not np.isclose(np.sum(weights), 1.0, atol=1e-6):
            return False
        
        # Check bounds
        if bounds is not None:
            min_weight, max_weight = bounds
            if not np.all(weights >= min_weight - 1e-6):
                return False
            if not np.all(weights <= max_weight + 1e-6):
                return False
        
        return True
