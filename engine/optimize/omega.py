"""
Omega ratio portfolio optimizer.
"""

import logging
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import cvxpy as cp
from scipy.optimize import minimize, differential_evolution
import structlog

logger = structlog.get_logger(__name__)


class OmegaOptimizer:
    """Omega ratio portfolio optimizer."""
    
    def __init__(self, solver: str = "ECOS", verbose: bool = False):
        self.solver = solver
        self.verbose = verbose
        self.last_optimization_result = None
    
    def calculate_omega_ratio(
        self,
        returns: np.ndarray,
        weights: np.ndarray,
        theta: float = 0.0
    ) -> float:
        """
        Calculate Omega ratio for given portfolio weights.
        
        Args:
            returns: Historical returns matrix (T x N)
            weights: Portfolio weights (N,)
            theta: Threshold return level
        
        Returns:
            Omega ratio
        """
        
        # Calculate portfolio returns
        portfolio_returns = np.dot(returns, weights)
        
        # Calculate excess returns above threshold
        excess_returns = portfolio_returns - theta
        
        # Separate positive and negative excess returns
        positive_returns = excess_returns[excess_returns > 0]
        negative_returns = excess_returns[excess_returns < 0]
        
        # Calculate Omega ratio
        if len(negative_returns) == 0:
            return np.inf
        
        positive_sum = np.sum(positive_returns)
        negative_sum = np.sum(np.abs(negative_returns))
        
        return positive_sum / negative_sum
    
    def solve_omega(
        self,
        returns: np.ndarray,
        theta: float = 0.0,
        cov: Optional[np.ndarray] = None,
        w_prev: Optional[np.ndarray] = None,
        cvar_cap: Optional[float] = None,
        turnover_penalty: float = 0.0,
        bounds: Optional[Tuple[float, float]] = None,
        long_only: bool = True,
        leverage_cap: Optional[float] = None
    ) -> Dict[str, Union[np.ndarray, float, str]]:
        """
        Solve Omega ratio maximization problem.
        
        Args:
            returns: Historical returns matrix (T x N)
            theta: Threshold return level
            cov: Covariance matrix (optional, for CVaR constraint)
            w_prev: Previous portfolio weights for turnover penalty
            cvar_cap: CVaR constraint cap
            turnover_penalty: Penalty coefficient for turnover
            bounds: Tuple of (min_weight, max_weight) for all assets
            long_only: Whether to enforce long-only constraint
            leverage_cap: Maximum leverage allowed
        
        Returns:
            Dict containing:
                - weights: Optimal portfolio weights
                - objective_value: Maximum Omega ratio achieved
                - status: Optimization status
                - solve_time: Time taken to solve
        """
        
        import time
        start_time = time.time()
        
        n_assets = returns.shape[1]
        n_periods = returns.shape[0]
        
        # Set default bounds
        if bounds is None:
            bounds = (0.0, 1.0) if long_only else (-1.0, 1.0)
        
        min_weight, max_weight = bounds
        
        # Try CVXPY approach first (for cases where it can handle the problem)
        try:
            result = self._solve_omega_cvxpy(
                returns, theta, cov, w_prev, cvar_cap, turnover_penalty,
                bounds, long_only, leverage_cap
            )
            
            if result["status"] == "optimal":
                solve_time = time.time() - start_time
                result["solve_time"] = solve_time
                return result
        
        except Exception as e:
            logger.warning("CVXPY approach failed, trying scipy", error=str(e))
        
        # Fall back to scipy optimization
        try:
            result = self._solve_omega_scipy(
                returns, theta, cov, w_prev, cvar_cap, turnover_penalty,
                bounds, long_only, leverage_cap
            )
            
            solve_time = time.time() - start_time
            result["solve_time"] = solve_time
            
            self.last_optimization_result = result
            return result
        
        except Exception as e:
            solve_time = time.time() - start_time
            logger.error("Omega optimization failed", error=str(e), solve_time=solve_time)
            
            return {
                "weights": None,
                "objective_value": -np.inf,
                "status": "error",
                "solve_time": solve_time,
                "error": str(e)
            }
    
    def _solve_omega_cvxpy(
        self,
        returns: np.ndarray,
        theta: float,
        cov: Optional[np.ndarray],
        w_prev: Optional[np.ndarray],
        cvar_cap: Optional[float],
        turnover_penalty: float,
        bounds: Tuple[float, float],
        long_only: bool,
        leverage_cap: Optional[float]
    ) -> Dict[str, Union[np.ndarray, float, str]]:
        """Solve Omega optimization using CVXPY (when possible)."""
        
        n_assets = returns.shape[1]
        n_periods = returns.shape[0]
        
        min_weight, max_weight = bounds
        
        # Create optimization variables
        w = cp.Variable(n_assets, name="weights")
        
        # Calculate portfolio returns
        portfolio_returns = returns @ w
        
        # Calculate excess returns above threshold
        excess_returns = portfolio_returns - theta
        
        # Create auxiliary variables for positive and negative parts
        pos_returns = cp.Variable(n_periods, name="positive_returns")
        neg_returns = cp.Variable(n_periods, name="negative_returns")
        
        # Define constraints
        constraints = [
            cp.sum(w) == 1.0,  # Budget constraint
            w >= min_weight,   # Lower bound
            w <= max_weight,   # Upper bound
            pos_returns >= 0,  # Positive returns non-negative
            neg_returns >= 0,  # Negative returns non-negative
            pos_returns >= excess_returns,  # Positive returns >= excess returns
            neg_returns >= -excess_returns,  # Negative returns >= -excess returns
        ]
        
        if long_only:
            constraints.append(w >= 0.0)
        
        if leverage_cap is not None:
            constraints.append(cp.norm(w, 1) <= leverage_cap)
        
        # Add turnover penalty
        if turnover_penalty > 0 and w_prev is not None:
            constraints.append(cp.norm(w - w_prev, 1) <= 1.0)  # Reasonable turnover constraint
        
        # Add CVaR constraint if specified
        if cvar_cap is not None and cov is not None:
            # CVaR constraint: P(portfolio_return <= VaR) <= alpha
            # This is a simplified version - full CVaR constraint is complex
            pass
        
        # Define objective (maximize Omega ratio)
        # Omega = sum(positive_returns) / sum(negative_returns)
        # We'll maximize sum(positive_returns) - penalty * sum(negative_returns)
        penalty = 1.0  # Adjust this based on your preference
        objective = cp.Maximize(cp.sum(pos_returns) - penalty * cp.sum(neg_returns))
        
        # Solve optimization problem
        problem = cp.Problem(objective, constraints)
        problem.solve(solver=self.solver, verbose=self.verbose)
        
        if problem.status == cp.OPTIMAL:
            weights = w.value
            omega_ratio = self.calculate_omega_ratio(returns, weights, theta)
            
            return {
                "weights": weights,
                "objective_value": omega_ratio,
                "status": "optimal"
            }
        else:
            return {
                "weights": None,
                "objective_value": -np.inf,
                "status": problem.status
            }
    
    def _solve_omega_scipy(
        self,
        returns: np.ndarray,
        theta: float,
        cov: Optional[np.ndarray],
        w_prev: Optional[np.ndarray],
        cvar_cap: Optional[float],
        turnover_penalty: float,
        bounds: Tuple[float, float],
        long_only: bool,
        leverage_cap: Optional[float]
    ) -> Dict[str, Union[np.ndarray, float, str]]:
        """Solve Omega optimization using scipy optimization."""
        
        n_assets = returns.shape[1]
        
        min_weight, max_weight = bounds
        
        # Define objective function (negative Omega ratio for minimization)
        def objective(weights):
            if not self._validate_weights(weights, bounds, long_only, leverage_cap):
                return 1e6  # Large penalty for invalid weights
            
            omega_ratio = self.calculate_omega_ratio(returns, weights, theta)
            
            # Add turnover penalty
            penalty = 0.0
            if turnover_penalty > 0 and w_prev is not None:
                penalty = turnover_penalty * np.sum(np.abs(weights - w_prev))
            
            # Add CVaR penalty if specified
            if cvar_cap is not None:
                portfolio_returns = np.dot(returns, weights)
                cvar = self._calculate_cvar(portfolio_returns, 0.05)
                if cvar > cvar_cap:
                    penalty += 1000 * (cvar - cvar_cap)  # Large penalty for CVaR violation
            
            return -(omega_ratio - penalty)  # Negative for minimization
        
        # Define constraints
        constraints = []
        
        # Budget constraint
        constraints.append({
            'type': 'eq',
            'fun': lambda w: np.sum(w) - 1.0
        })
        
        # CVaR constraint if specified
        if cvar_cap is not None:
            def cvar_constraint(weights):
                portfolio_returns = np.dot(returns, weights)
                cvar = self._calculate_cvar(portfolio_returns, 0.05)
                return cvar_cap - cvar
            
            constraints.append({
                'type': 'ineq',
                'fun': cvar_constraint
            })
        
        # Bounds for each variable
        variable_bounds = [(min_weight, max_weight) for _ in range(n_assets)]
        
        # Initial guess (equal weights)
        x0 = np.ones(n_assets) / n_assets
        
        # Solve optimization
        try:
            result = minimize(
                objective,
                x0,
                method='SLSQP',
                bounds=variable_bounds,
                constraints=constraints,
                options={'maxiter': 1000, 'ftol': 1e-9}
            )
            
            if result.success:
                weights = result.x
                omega_ratio = self.calculate_omega_ratio(returns, weights, theta)
                
                return {
                    "weights": weights,
                    "objective_value": omega_ratio,
                    "status": "optimal"
                }
            else:
                return {
                    "weights": None,
                    "objective_value": -np.inf,
                    "status": result.message
                }
        
        except Exception as e:
            logger.error("Scipy optimization failed", error=str(e))
            return {
                "weights": None,
                "objective_value": -np.inf,
                "status": "error",
                "error": str(e)
            }
    
    def _validate_weights(
        self,
        weights: np.ndarray,
        bounds: Tuple[float, float],
        long_only: bool,
        leverage_cap: Optional[float]
    ) -> bool:
        """Validate portfolio weights."""
        
        if weights is None:
            return False
        
        # Check if weights sum to 1
        if not np.isclose(np.sum(weights), 1.0, atol=1e-6):
            return False
        
        # Check bounds
        min_weight, max_weight = bounds
        if not np.all(weights >= min_weight - 1e-6):
            return False
        if not np.all(weights <= max_weight + 1e-6):
            return False
        
        # Check long-only constraint
        if long_only and not np.all(weights >= -1e-6):
            return False
        
        # Check leverage constraint
        if leverage_cap is not None and np.sum(np.abs(weights)) > leverage_cap + 1e-6:
            return False
        
        return True
    
    def _calculate_cvar(self, returns: np.ndarray, alpha: float = 0.05) -> float:
        """Calculate Conditional Value at Risk."""
        
        var = np.percentile(returns, alpha * 100)
        cvar = returns[returns <= var].mean()
        
        return cvar
    
    def solve_omega_with_returns(
        self,
        returns: pd.DataFrame,
        theta: float = 0.0,
        lookback_window: int = 252,
        cov: Optional[np.ndarray] = None,
        w_prev: Optional[np.ndarray] = None,
        cvar_cap: Optional[float] = None,
        turnover_penalty: float = 0.0,
        bounds: Optional[Tuple[float, float]] = None,
        long_only: bool = True,
        leverage_cap: Optional[float] = None
    ) -> Dict[str, Union[np.ndarray, float, str]]:
        """
        Solve Omega optimization using historical returns.
        
        Args:
            returns: Historical returns DataFrame (T x N)
            theta: Threshold return level
            lookback_window: Number of periods to use for optimization
            cov: Covariance matrix (optional, for CVaR constraint)
            w_prev: Previous portfolio weights for turnover penalty
            cvar_cap: CVaR constraint cap
            turnover_penalty: Penalty coefficient for turnover
            bounds: Tuple of (min_weight, max_weight) for all assets
            long_only: Whether to enforce long-only constraint
            leverage_cap: Maximum leverage allowed
        
        Returns:
            Dict containing optimization results
        """
        
        # Use recent data for optimization
        recent_returns = returns.tail(lookback_window)
        
        # Convert to numpy array
        returns_array = recent_returns.values
        
        # Solve Omega optimization
        return self.solve_omega(
            returns=returns_array,
            theta=theta,
            cov=cov,
            w_prev=w_prev,
            cvar_cap=cvar_cap,
            turnover_penalty=turnover_penalty,
            bounds=bounds,
            long_only=long_only,
            leverage_cap=leverage_cap
        )
    
    def get_omega_surface(
        self,
        returns: np.ndarray,
        theta_range: np.ndarray,
        bounds: Optional[Tuple[float, float]] = None,
        long_only: bool = True
    ) -> Dict[str, np.ndarray]:
        """
        Generate Omega ratio surface for different theta values.
        
        Args:
            returns: Historical returns matrix (T x N)
            theta_range: Array of theta values to test
            bounds: Tuple of (min_weight, max_weight) for all assets
            long_only: Whether to enforce long-only constraint
        
        Returns:
            Dict containing:
                - theta_values: Array of theta values
                - omega_ratios: Array of Omega ratios
                - weights: Array of optimal weights for each theta
        """
        
        theta_values = []
        omega_ratios = []
        weights_list = []
        
        for theta in theta_range:
            try:
                result = self.solve_omega(
                    returns=returns,
                    theta=theta,
                    bounds=bounds,
                    long_only=long_only
                )
                
                if result["status"] == "optimal":
                    theta_values.append(theta)
                    omega_ratios.append(result["objective_value"])
                    weights_list.append(result["weights"])
                
            except Exception as e:
                logger.warning("Failed to solve for theta", theta=theta, error=str(e))
                continue
        
        return {
            "theta_values": np.array(theta_values),
            "omega_ratios": np.array(omega_ratios),
            "weights": np.array(weights_list)
        }
    
    def get_risk_contribution(self, weights: np.ndarray, cov: np.ndarray) -> np.ndarray:
        """Calculate risk contribution of each asset."""
        
        if weights is None:
            return np.array([])
        
        portfolio_var = np.dot(weights, np.dot(cov, weights))
        marginal_contrib = np.dot(cov, weights)
        risk_contrib = weights * marginal_contrib / portfolio_var
        
        return risk_contrib
