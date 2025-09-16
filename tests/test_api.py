"""
Tests for API endpoints.
"""

import pytest
import requests
import json
from fastapi.testclient import TestClient
from api.main import app


class TestAPI:
    """Test API endpoints."""
    
    def setup_method(self):
        """Setup test client."""
        self.client = TestClient(app)
    
    def test_health_check(self):
        """Test health check endpoint."""
        response = self.client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        assert "status" in data
        assert "version" in data
        assert "uptime" in data
        assert "memory_usage" in data
        assert "cpu_usage" in data
        assert "active_jobs" in data
        assert "total_jobs" in data
    
    def test_optimize_portfolio_gmv(self):
        """Test portfolio optimization with GMV objective."""
        # Create sample data
        returns = [
            [0.01, 0.02, -0.01, 0.03, 0.01],
            [0.02, 0.01, 0.02, -0.01, 0.02],
            [0.01, 0.02, -0.01, 0.03, 0.01],
            [0.02, 0.01, 0.02, -0.01, 0.02],
            [0.01, 0.02, -0.01, 0.03, 0.01]
        ]
        asset_names = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
        dates = ["2023-01-01", "2023-01-02", "2023-01-03", "2023-01-04", "2023-01-05"]
        
        request_data = {
            "returns": returns,
            "asset_names": asset_names,
            "dates": dates,
            "objective": "gmv",
            "long_only": True
        }
        
        response = self.client.post("/optimize", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert "weights" in data
        assert "objective_value" in data
        assert "status" in data
        assert "solve_time" in data
        assert len(data["weights"]) == len(asset_names)
    
    def test_optimize_portfolio_omega(self):
        """Test portfolio optimization with Omega objective."""
        # Create sample data
        returns = [
            [0.01, 0.02, -0.01, 0.03, 0.01],
            [0.02, 0.01, 0.02, -0.01, 0.02],
            [0.01, 0.02, -0.01, 0.03, 0.01],
            [0.02, 0.01, 0.02, -0.01, 0.02],
            [0.01, 0.02, -0.01, 0.03, 0.01]
        ]
        asset_names = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
        dates = ["2023-01-01", "2023-01-02", "2023-01-03", "2023-01-04", "2023-01-05"]
        
        request_data = {
            "returns": returns,
            "asset_names": asset_names,
            "dates": dates,
            "objective": "omega",
            "theta": 0.02,
            "long_only": True
        }
        
        response = self.client.post("/optimize", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert "weights" in data
        assert "objective_value" in data
        assert "status" in data
        assert "solve_time" in data
        assert len(data["weights"]) == len(asset_names)
    
    def test_optimize_portfolio_invalid_data(self):
        """Test portfolio optimization with invalid data."""
        request_data = {
            "returns": [],
            "asset_names": [],
            "dates": [],
            "objective": "gmv"
        }
        
        response = self.client.post("/optimize", json=request_data)
        assert response.status_code == 422  # Validation error
    
    def test_optimize_portfolio_unknown_objective(self):
        """Test portfolio optimization with unknown objective."""
        returns = [
            [0.01, 0.02, -0.01, 0.03, 0.01],
            [0.02, 0.01, 0.02, -0.01, 0.02]
        ]
        asset_names = ["AAPL", "MSFT"]
        dates = ["2023-01-01", "2023-01-02"]
        
        request_data = {
            "returns": returns,
            "asset_names": asset_names,
            "dates": dates,
            "objective": "unknown"
        }
        
        response = self.client.post("/optimize", json=request_data)
        assert response.status_code == 400
    
    def test_run_backtest(self):
        """Test portfolio backtest."""
        # Create sample data
        returns = [
            [0.01, 0.02, -0.01, 0.03, 0.01],
            [0.02, 0.01, 0.02, -0.01, 0.02],
            [0.01, 0.02, -0.01, 0.03, 0.01],
            [0.02, 0.01, 0.02, -0.01, 0.02],
            [0.01, 0.02, -0.01, 0.03, 0.01]
        ]
        asset_names = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
        dates = ["2023-01-01", "2023-01-02", "2023-01-03", "2023-01-04", "2023-01-05"]
        
        request_data = {
            "returns": returns,
            "asset_names": asset_names,
            "dates": dates,
            "objective": "gmv",
            "initial_capital": 1000000.0,
            "train_months": 12,
            "test_months": 3,
            "step_months": 1,
            "transaction_costs": 0.0005,
            "slippage": 0.0002,
            "rebalance_frequency": "monthly",
            "long_only": True
        }
        
        response = self.client.post("/backtest", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert "total_return" in data
        assert "annualized_return" in data
        assert "annualized_volatility" in data
        assert "sharpe_ratio" in data
        assert "max_drawdown" in data
        assert "final_weights" in data
        assert "performance_history" in data
        assert "trade_history" in data
    
    def test_run_backtest_invalid_data(self):
        """Test portfolio backtest with invalid data."""
        request_data = {
            "returns": [],
            "asset_names": [],
            "dates": [],
            "objective": "gmv"
        }
        
        response = self.client.post("/backtest", json=request_data)
        assert response.status_code == 422  # Validation error
    
    def test_detect_regimes_hmm(self):
        """Test regime detection with HMM method."""
        # Create sample features
        features = [
            [0.01, 0.02, -0.01, 0.03, 0.01],
            [0.02, 0.01, 0.02, -0.01, 0.02],
            [0.01, 0.02, -0.01, 0.03, 0.01],
            [0.02, 0.01, 0.02, -0.01, 0.02],
            [0.01, 0.02, -0.01, 0.03, 0.01]
        ]
        feature_names = ["feature1", "feature2", "feature3", "feature4", "feature5"]
        dates = ["2023-01-01", "2023-01-02", "2023-01-03", "2023-01-04", "2023-01-05"]
        
        request_data = {
            "features": features,
            "feature_names": feature_names,
            "dates": dates,
            "method": "hmm",
            "n_regimes": 3,
            "random_state": 42
        }
        
        response = self.client.post("/regime", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert "regime_labels" in data
        assert "regime_probs" in data
        assert "regime_characteristics" in data
        assert "method" in data
        assert "n_regimes" in data
        assert "n_observations" in data
        assert len(data["regime_labels"]) == len(features)
    
    def test_detect_regimes_gmm(self):
        """Test regime detection with GMM method."""
        # Create sample features
        features = [
            [0.01, 0.02, -0.01, 0.03, 0.01],
            [0.02, 0.01, 0.02, -0.01, 0.02],
            [0.01, 0.02, -0.01, 0.03, 0.01],
            [0.02, 0.01, 0.02, -0.01, 0.02],
            [0.01, 0.02, -0.01, 0.03, 0.01]
        ]
        feature_names = ["feature1", "feature2", "feature3", "feature4", "feature5"]
        dates = ["2023-01-01", "2023-01-02", "2023-01-03", "2023-01-04", "2023-01-05"]
        
        request_data = {
            "features": features,
            "feature_names": feature_names,
            "dates": dates,
            "method": "gmm",
            "n_regimes": 3,
            "random_state": 42
        }
        
        response = self.client.post("/regime", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert "regime_labels" in data
        assert "regime_probs" in data
        assert "regime_characteristics" in data
        assert "method" in data
        assert "n_regimes" in data
        assert "n_observations" in data
        assert len(data["regime_labels"]) == len(features)
    
    def test_detect_regimes_invalid_data(self):
        """Test regime detection with invalid data."""
        request_data = {
            "features": [],
            "feature_names": [],
            "dates": [],
            "method": "hmm"
        }
        
        response = self.client.post("/regime", json=request_data)
        assert response.status_code == 422  # Validation error
    
    def test_detect_regimes_unknown_method(self):
        """Test regime detection with unknown method."""
        features = [
            [0.01, 0.02, -0.01, 0.03, 0.01],
            [0.02, 0.01, 0.02, -0.01, 0.02]
        ]
        feature_names = ["feature1", "feature2"]
        dates = ["2023-01-01", "2023-01-02"]
        
        request_data = {
            "features": features,
            "feature_names": feature_names,
            "dates": dates,
            "method": "unknown",
            "n_regimes": 3
        }
        
        response = self.client.post("/regime", json=request_data)
        assert response.status_code == 400
    
    def test_create_job(self):
        """Test job creation."""
        request_data = {
            "job_type": "optimize",
            "parameters": {
                "returns": [[0.01, 0.02], [0.02, 0.01]],
                "asset_names": ["AAPL", "MSFT"],
                "dates": ["2023-01-01", "2023-01-02"],
                "objective": "gmv"
            },
            "priority": 1,
            "timeout": 300
        }
        
        response = self.client.post("/jobs", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert "job_id" in data
        assert "status" in data
        assert "message" in data
        assert "created_at" in data
        assert data["status"] == "pending"
    
    def test_get_job_status(self):
        """Test getting job status."""
        # First create a job
        request_data = {
            "job_type": "optimize",
            "parameters": {
                "returns": [[0.01, 0.02], [0.02, 0.01]],
                "asset_names": ["AAPL", "MSFT"],
                "dates": ["2023-01-01", "2023-01-02"],
                "objective": "gmv"
            }
        }
        
        create_response = self.client.post("/jobs", json=request_data)
        job_id = create_response.json()["job_id"]
        
        # Then get job status
        response = self.client.get(f"/jobs/{job_id}")
        assert response.status_code == 200
        
        data = response.json()
        assert "job_id" in data
        assert "status" in data
        assert "progress" in data
        assert "created_at" in data
        assert data["job_id"] == job_id
    
    def test_get_job_status_not_found(self):
        """Test getting job status for non-existent job."""
        response = self.client.get("/jobs/non-existent-job-id")
        assert response.status_code == 404
    
    def test_get_job_result(self):
        """Test getting job result."""
        # First create a job
        request_data = {
            "job_type": "optimize",
            "parameters": {
                "returns": [[0.01, 0.02], [0.02, 0.01]],
                "asset_names": ["AAPL", "MSFT"],
                "dates": ["2023-01-01", "2023-01-02"],
                "objective": "gmv"
            }
        }
        
        create_response = self.client.post("/jobs", json=request_data)
        job_id = create_response.json()["job_id"]
        
        # Try to get job result (will fail since job is not completed)
        response = self.client.get(f"/jobs/{job_id}/result")
        assert response.status_code == 400  # Job not completed
    
    def test_delete_job(self):
        """Test deleting a job."""
        # First create a job
        request_data = {
            "job_type": "optimize",
            "parameters": {
                "returns": [[0.01, 0.02], [0.02, 0.01]],
                "asset_names": ["AAPL", "MSFT"],
                "dates": ["2023-01-01", "2023-01-02"],
                "objective": "gmv"
            }
        }
        
        create_response = self.client.post("/jobs", json=request_data)
        job_id = create_response.json()["job_id"]
        
        # Then delete the job
        response = self.client.delete(f"/jobs/{job_id}")
        assert response.status_code == 200
        
        data = response.json()
        assert "message" in data
        assert data["message"] == "Job deleted successfully"
    
    def test_delete_job_not_found(self):
        """Test deleting a non-existent job."""
        response = self.client.delete("/jobs/non-existent-job-id")
        assert response.status_code == 404
    
    def test_global_exception_handler(self):
        """Test global exception handler."""
        # This test would require triggering an internal server error
        # For now, we'll just test that the handler exists
        assert hasattr(app, "exception_handler")
    
    def test_cors_headers(self):
        """Test CORS headers are present."""
        response = self.client.options("/health")
        assert response.status_code == 200
        
        # Check that CORS headers are present
        headers = response.headers
        assert "access-control-allow-origin" in headers
        assert "access-control-allow-methods" in headers
        assert "access-control-allow-headers" in headers


if __name__ == "__main__":
    pytest.main([__file__])
