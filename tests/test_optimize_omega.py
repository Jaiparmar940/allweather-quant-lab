"""
Tests for Omega optimizer.
"""

import pytest
import numpy as np
import pandas as pd
from engine.optimize.omega import OmegaOptimizer


class TestOmegaOptimizer:
    """Test OmegaOptimizer class."""
    
    def test_omega_optimizer_init(self):
        """Test OmegaOptimizer initialization."""
        optimizer = OmegaOptimizer()
        assert optimizer.solver == "ECOS"
        assert optimizer.verbose == False
    
    def test_omega_optimizer_init_with_params(self):
        """Test OmegaOptimizer initialization with parameters."""
        optimizer = OmegaOptimizer(solver="SCS", verbose=True)
        assert optimizer.solver == "SCS"
        assert optimizer.verbose == True
    
    def test_calculate_omega_ratio(self):
        """Test Omega ratio calculation."""
        # Create sample returns
        returns = np.array([[0.01, 0.02, -0.01, 0.03, 0.01],
                           [0.02, 0.01, 0.02, -0.01, 0.02]])
        weights = np.array([0.6, 0.4])
        
        optimizer = OmegaOptimizer()
        omega_ratio = optimizer.calculate_omega_ratio(returns, weights, theta=0.0)
        
        assert isinstance(omega_ratio, float)
        assert omega_ratio > 0
    
    def test_calculate_omega_ratio_with_threshold(self):
        """Test Omega ratio calculation with threshold."""
        # Create sample returns
        returns = np.array([[0.01, 0.02, -0.01, 0.03, 0.01],
                           [0.02, 0.01, 0.02, -0.01, 0.02]])
        weights = np.array([0.6, 0.4])
        
        optimizer = OmegaOptimizer()
        omega_ratio = optimizer.calculate_omega_ratio(returns, weights, theta=0.01)
        
        assert isinstance(omega_ratio, float)
        assert omega_ratio > 0
    
    def test_solve_omega_basic(self):
        """Test basic Omega optimization."""
        # Create sample returns
        returns = np.array([[0.01, 0.02, -0.01, 0.03, 0.01],
                           [0.02, 0.01, 0.02, -0.01, 0.02]])
        
        optimizer = OmegaOptimizer()
        result = optimizer.solve_omega(returns, theta=0.0)
        
        assert result["status"] == "optimal"
        assert result["weights"] is not None
        assert len(result["weights"]) == 2
        assert np.isclose(np.sum(result["weights"]), 1.0, atol=1e-6)
        assert np.all(result["weights"] >= 0)  # Long-only constraint
    
    def test_solve_omega_with_bounds(self):
        """Test Omega optimization with custom bounds."""
        # Create sample returns
        returns = np.array([[0.01, 0.02, -0.01, 0.03, 0.01],
                           [0.02, 0.01, 0.02, -0.01, 0.02]])
        
        optimizer = OmegaOptimizer()
        result = optimizer.solve_omega(returns, theta=0.0, bounds=(0.1, 0.9))
        
        assert result["status"] == "optimal"
        assert result["weights"] is not None
        assert np.all(result["weights"] >= 0.1)
        assert np.all(result["weights"] <= 0.9)
    
    def test_solve_omega_with_turnover_penalty(self):
        """Test Omega optimization with turnover penalty."""
        # Create sample returns
        returns = np.array([[0.01, 0.02, -0.01, 0.03, 0.01],
                           [0.02, 0.01, 0.02, -0.01, 0.02]])
        
        optimizer = OmegaOptimizer()
        w_prev = np.array([0.5, 0.5])
        result = optimizer.solve_omega(returns, theta=0.0, turnover_penalty=0.1, w_prev=w_prev)
        
        assert result["status"] == "optimal"
        assert result["weights"] is not None
        assert len(result["weights"]) == 2
    
    def test_solve_omega_with_cvar_constraint(self):
        """Test Omega optimization with CVaR constraint."""
        # Create sample returns
        returns = np.array([[0.01, 0.02, -0.01, 0.03, 0.01],
                           [0.02, 0.01, 0.02, -0.01, 0.02]])
        
        optimizer = OmegaOptimizer()
        result = optimizer.solve_omega(returns, theta=0.0, cvar_cap=0.05)
        
        assert result["status"] == "optimal"
        assert result["weights"] is not None
        assert len(result["weights"]) == 2
    
    def test_solve_omega_with_returns(self):
        """Test Omega optimization using historical returns."""
        # Create sample returns
        np.random.seed(42)
        returns = pd.DataFrame(
            np.random.normal(0.001, 0.02, (100, 3)),
            columns=['A', 'B', 'C']
        )
        
        optimizer = OmegaOptimizer()
        result = optimizer.solve_omega_with_returns(returns, theta=0.01)
        
        assert result["status"] == "optimal"
        assert result["weights"] is not None
        assert len(result["weights"]) == 3
        assert np.isclose(np.sum(result["weights"]), 1.0, atol=1e-6)
    
    def test_get_omega_surface(self):
        """Test Omega ratio surface generation."""
        # Create sample returns
        returns = np.array([[0.01, 0.02, -0.01, 0.03, 0.01],
                           [0.02, 0.01, 0.02, -0.01, 0.02]])
        
        optimizer = OmegaOptimizer()
        theta_range = np.array([0.0, 0.01, 0.02, 0.03])
        surface = optimizer.get_omega_surface(returns, theta_range)
        
        assert "theta_values" in surface
        assert "omega_ratios" in surface
        assert "weights" in surface
        assert len(surface["theta_values"]) > 0
        assert len(surface["omega_ratios"]) > 0
        assert len(surface["weights"]) > 0
    
    def test_get_risk_contribution(self):
        """Test risk contribution calculation."""
        # Create simple covariance matrix
        cov = np.array([[0.04, 0.02], [0.02, 0.09]])
        weights = np.array([0.6, 0.4])
        
        optimizer = OmegaOptimizer()
        risk_contrib = optimizer.get_risk_contribution(weights, cov)
        
        assert len(risk_contrib) == 2
        assert np.isclose(np.sum(risk_contrib), 1.0, atol=1e-6)
        assert np.all(risk_contrib >= 0)
    
    def test_validate_weights(self):
        """Test weight validation."""
        optimizer = OmegaOptimizer()
        
        # Valid weights
        valid_weights = np.array([0.6, 0.4])
        assert optimizer._validate_weights(valid_weights, (0.0, 1.0), True, None)
        
        # Invalid weights (don't sum to 1)
        invalid_weights = np.array([0.6, 0.3])
        assert not optimizer._validate_weights(invalid_weights, (0.0, 1.0), True, None)
        
        # Invalid weights (negative)
        invalid_weights = np.array([0.6, -0.4])
        assert not optimizer._validate_weights(invalid_weights, (0.0, 1.0), True, None)
    
    def test_calculate_cvar(self):
        """Test CVaR calculation."""
        # Create sample returns
        returns = np.array([0.01, 0.02, -0.01, 0.03, 0.01, -0.05, 0.02, 0.01])
        
        optimizer = OmegaOptimizer()
        cvar = optimizer._calculate_cvar(returns, alpha=0.05)
        
        assert isinstance(cvar, float)
        assert cvar <= 0  # CVaR should be negative or zero
    
    def test_solve_omega_infeasible(self):
        """Test Omega optimization with infeasible constraints."""
        # Create sample returns
        returns = np.array([[0.01, 0.02, -0.01, 0.03, 0.01],
                           [0.02, 0.01, 0.02, -0.01, 0.02]])
        
        optimizer = OmegaOptimizer()
        # Set impossible bounds
        result = optimizer.solve_omega(returns, theta=0.0, bounds=(0.6, 0.7))
        
        # Should fail due to infeasible constraints
        assert result["status"] != "optimal"
        assert result["weights"] is None
    
    def test_solve_omega_empty_returns(self):
        """Test Omega optimization with empty returns."""
        # Create empty returns
        returns = np.array([])
        
        optimizer = OmegaOptimizer()
        result = optimizer.solve_omega(returns, theta=0.0)
        
        # Should handle empty returns gracefully
        assert result["status"] == "error"
        assert result["weights"] is None


if __name__ == "__main__":
    pytest.main([__file__])
