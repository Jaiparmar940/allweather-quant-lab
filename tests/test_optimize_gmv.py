"""
Tests for GMV optimizer.
"""

import pytest
import numpy as np
import pandas as pd
from engine.optimize.gmv import GMVOptimizer


class TestGMVOptimizer:
    """Test GMVOptimizer class."""
    
    def test_gmv_optimizer_init(self):
        """Test GMVOptimizer initialization."""
        optimizer = GMVOptimizer()
        assert optimizer.solver == "ECOS"
        assert optimizer.verbose == False
    
    def test_gmv_optimizer_init_with_params(self):
        """Test GMVOptimizer initialization with parameters."""
        optimizer = GMVOptimizer(solver="SCS", verbose=True)
        assert optimizer.solver == "SCS"
        assert optimizer.verbose == True
    
    def test_solve_gmv_basic(self):
        """Test basic GMV optimization."""
        # Create simple covariance matrix
        cov = np.array([[0.04, 0.02], [0.02, 0.09]])
        
        optimizer = GMVOptimizer()
        result = optimizer.solve_gmv(cov)
        
        assert result["status"] == "optimal"
        assert result["weights"] is not None
        assert len(result["weights"]) == 2
        assert np.isclose(np.sum(result["weights"]), 1.0, atol=1e-6)
        assert np.all(result["weights"] >= 0)  # Long-only constraint
    
    def test_solve_gmv_with_bounds(self):
        """Test GMV optimization with custom bounds."""
        # Create simple covariance matrix
        cov = np.array([[0.04, 0.02], [0.02, 0.09]])
        
        optimizer = GMVOptimizer()
        result = optimizer.solve_gmv(cov, bounds=(0.1, 0.9))
        
        assert result["status"] == "optimal"
        assert result["weights"] is not None
        assert np.all(result["weights"] >= 0.1)
        assert np.all(result["weights"] <= 0.9)
    
    def test_solve_gmv_with_turnover_penalty(self):
        """Test GMV optimization with turnover penalty."""
        # Create simple covariance matrix
        cov = np.array([[0.04, 0.02], [0.02, 0.09]])
        
        optimizer = GMVOptimizer()
        w_prev = np.array([0.5, 0.5])
        result = optimizer.solve_gmv(cov, turnover_penalty=0.1, w_prev=w_prev)
        
        assert result["status"] == "optimal"
        assert result["weights"] is not None
        assert len(result["weights"]) == 2
    
    def test_solve_gmv_with_leverage_cap(self):
        """Test GMV optimization with leverage cap."""
        # Create simple covariance matrix
        cov = np.array([[0.04, 0.02], [0.02, 0.09]])
        
        optimizer = GMVOptimizer()
        result = optimizer.solve_gmv(cov, leverage_cap=1.5)
        
        assert result["status"] == "optimal"
        assert result["weights"] is not None
        assert np.sum(np.abs(result["weights"])) <= 1.5 + 1e-6
    
    def test_solve_gmv_with_returns(self):
        """Test GMV optimization using historical returns."""
        # Create sample returns
        np.random.seed(42)
        returns = pd.DataFrame(
            np.random.normal(0.001, 0.02, (100, 3)),
            columns=['A', 'B', 'C']
        )
        
        optimizer = GMVOptimizer()
        result = optimizer.solve_gmv_with_returns(returns)
        
        assert result["status"] == "optimal"
        assert result["weights"] is not None
        assert len(result["weights"]) == 3
        assert np.isclose(np.sum(result["weights"]), 1.0, atol=1e-6)
    
    def test_solve_gmv_robust(self):
        """Test robust GMV optimization with shrinkage."""
        # Create sample returns
        np.random.seed(42)
        returns = pd.DataFrame(
            np.random.normal(0.001, 0.02, (100, 3)),
            columns=['A', 'B', 'C']
        )
        
        optimizer = GMVOptimizer()
        result = optimizer.solve_gmv_robust(returns, shrinkage_factor=0.1)
        
        assert result["status"] == "optimal"
        assert result["weights"] is not None
        assert len(result["weights"]) == 3
        assert np.isclose(np.sum(result["weights"]), 1.0, atol=1e-6)
    
    def test_get_risk_contribution(self):
        """Test risk contribution calculation."""
        # Create simple covariance matrix
        cov = np.array([[0.04, 0.02], [0.02, 0.09]])
        weights = np.array([0.6, 0.4])
        
        optimizer = GMVOptimizer()
        risk_contrib = optimizer.get_risk_contribution(weights, cov)
        
        assert len(risk_contrib) == 2
        assert np.isclose(np.sum(risk_contrib), 1.0, atol=1e-6)
        assert np.all(risk_contrib >= 0)
    
    def test_get_efficient_frontier(self):
        """Test efficient frontier generation."""
        # Create simple covariance matrix
        cov = np.array([[0.04, 0.02], [0.02, 0.09]])
        
        optimizer = GMVOptimizer()
        frontier = optimizer.get_efficient_frontier(cov, n_points=10)
        
        assert "returns" in frontier
        assert "risks" in frontier
        assert "weights" in frontier
        assert len(frontier["returns"]) == 10
        assert len(frontier["risks"]) == 10
        assert len(frontier["weights"]) == 10
    
    def test_validate_weights(self):
        """Test weight validation."""
        optimizer = GMVOptimizer()
        
        # Valid weights
        valid_weights = np.array([0.6, 0.4])
        assert optimizer.validate_weights(valid_weights)
        
        # Invalid weights (don't sum to 1)
        invalid_weights = np.array([0.6, 0.3])
        assert not optimizer.validate_weights(invalid_weights)
        
        # Invalid weights (negative)
        invalid_weights = np.array([0.6, -0.4])
        assert not optimizer.validate_weights(invalid_weights, bounds=(0.0, 1.0))
    
    def test_solve_gmv_infeasible(self):
        """Test GMV optimization with infeasible constraints."""
        # Create simple covariance matrix
        cov = np.array([[0.04, 0.02], [0.02, 0.09]])
        
        optimizer = GMVOptimizer()
        # Set impossible bounds
        result = optimizer.solve_gmv(cov, bounds=(0.6, 0.7))
        
        # Should fail due to infeasible constraints
        assert result["status"] != "optimal"
        assert result["weights"] is None
    
    def test_solve_gmv_singular_covariance(self):
        """Test GMV optimization with singular covariance matrix."""
        # Create singular covariance matrix
        cov = np.array([[0.04, 0.02], [0.02, 0.01]])
        
        optimizer = GMVOptimizer()
        result = optimizer.solve_gmv(cov)
        
        # Should handle singular matrix gracefully
        assert result["status"] in ["optimal", "optimal_inaccurate", "error"]


if __name__ == "__main__":
    pytest.main([__file__])
