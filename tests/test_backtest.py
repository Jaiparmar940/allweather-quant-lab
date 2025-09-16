"""
Tests for backtesting modules.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from engine.backtest.simulator import BacktestSimulator, WalkForwardEngine
from engine.backtest.costs import CostCalculator, TaxCalculator
from engine.backtest.evaluation import BacktestEvaluator, MetricsCalculator


class TestBacktestSimulator:
    """Test BacktestSimulator class."""
    
    def test_backtest_simulator_init(self):
        """Test BacktestSimulator initialization."""
        simulator = BacktestSimulator()
        assert simulator.initial_capital == 1000000.0
        assert simulator.rebalance_frequency == "monthly"
        assert simulator.transaction_costs == 0.0005
        assert simulator.slippage == 0.0002
        assert simulator.tax_aware == False
    
    def test_backtest_simulator_init_with_params(self):
        """Test BacktestSimulator initialization with parameters."""
        simulator = BacktestSimulator(
            initial_capital=500000.0,
            rebalance_frequency="weekly",
            transaction_costs=0.001,
            slippage=0.0005,
            tax_aware=True
        )
        assert simulator.initial_capital == 500000.0
        assert simulator.rebalance_frequency == "weekly"
        assert simulator.transaction_costs == 0.001
        assert simulator.slippage == 0.0005
        assert simulator.tax_aware == True
    
    def test_run_backtest_basic(self):
        """Test basic backtest run."""
        # Create sample data
        np.random.seed(42)
        n_days = 100
        n_assets = 3
        
        returns = pd.DataFrame(
            np.random.normal(0.001, 0.02, (n_days, n_assets)),
            columns=['A', 'B', 'C'],
            index=pd.date_range('2023-01-01', periods=n_days, freq='D')
        )
        
        weights = pd.DataFrame(
            np.ones((n_days, n_assets)) / n_assets,
            columns=['A', 'B', 'C'],
            index=returns.index
        )
        
        prices = pd.DataFrame(
            np.ones((n_days, n_assets)) * 100,
            columns=['A', 'B', 'C'],
            index=returns.index
        )
        
        simulator = BacktestSimulator()
        result = simulator.run_backtest(returns, weights, prices)
        
        assert "final_value" in result
        assert "total_return" in result
        assert "sharpe_ratio" in result
        assert "max_drawdown" in result
        assert result["final_value"] > 0
    
    def test_run_backtest_with_costs(self):
        """Test backtest run with transaction costs."""
        # Create sample data
        np.random.seed(42)
        n_days = 100
        n_assets = 3
        
        returns = pd.DataFrame(
            np.random.normal(0.001, 0.02, (n_days, n_assets)),
            columns=['A', 'B', 'C'],
            index=pd.date_range('2023-01-01', periods=n_days, freq='D')
        )
        
        weights = pd.DataFrame(
            np.ones((n_days, n_assets)) / n_assets,
            columns=['A', 'B', 'C'],
            index=returns.index
        )
        
        prices = pd.DataFrame(
            np.ones((n_days, n_assets)) * 100,
            columns=['A', 'B', 'C'],
            index=returns.index
        )
        
        simulator = BacktestSimulator(transaction_costs=0.01, slippage=0.005)
        result = simulator.run_backtest(returns, weights, prices)
        
        assert "cost_drag" in result
        assert result["cost_drag"] > 0
    
    def test_run_backtest_tax_aware(self):
        """Test backtest run with tax awareness."""
        # Create sample data
        np.random.seed(42)
        n_days = 100
        n_assets = 3
        
        returns = pd.DataFrame(
            np.random.normal(0.001, 0.02, (n_days, n_assets)),
            columns=['A', 'B', 'C'],
            index=pd.date_range('2023-01-01', periods=n_days, freq='D')
        )
        
        weights = pd.DataFrame(
            np.ones((n_days, n_assets)) / n_assets,
            columns=['A', 'B', 'C'],
            index=returns.index
        )
        
        prices = pd.DataFrame(
            np.ones((n_days, n_assets)) * 100,
            columns=['A', 'B', 'C'],
            index=returns.index
        )
        
        simulator = BacktestSimulator(tax_aware=True)
        result = simulator.run_backtest(returns, weights, prices)
        
        assert "tax_drag" in result
        assert result["tax_drag"] >= 0


class TestWalkForwardEngine:
    """Test WalkForwardEngine class."""
    
    def test_walk_forward_engine_init(self):
        """Test WalkForwardEngine initialization."""
        engine = WalkForwardEngine()
        assert engine.train_months == 120
        assert engine.test_months == 12
        assert engine.step_months == 1
        assert engine.min_train_months == 60
    
    def test_walk_forward_engine_init_with_params(self):
        """Test WalkForwardEngine initialization with parameters."""
        engine = WalkForwardEngine(
            train_months=60,
            test_months=6,
            step_months=3,
            min_train_months=30
        )
        assert engine.train_months == 60
        assert engine.test_months == 6
        assert engine.step_months == 3
        assert engine.min_train_months == 30
    
    def test_run_walk_forward(self):
        """Test walk-forward backtest run."""
        # Create sample data
        np.random.seed(42)
        n_days = 500
        n_assets = 3
        
        returns = pd.DataFrame(
            np.random.normal(0.001, 0.02, (n_days, n_assets)),
            columns=['A', 'B', 'C'],
            index=pd.date_range('2020-01-01', periods=n_days, freq='D')
        )
        
        def optimizer_func(returns):
            # Simple equal-weight optimizer
            return pd.Series(1.0 / len(returns.columns), index=returns.columns)
        
        engine = WalkForwardEngine(train_months=60, test_months=12, step_months=6)
        result = engine.run_walk_forward(returns, optimizer_func)
        
        assert "overall_return" in result
        assert "overall_volatility" in result
        assert "overall_sharpe" in result
        assert "overall_max_drawdown" in result
        assert "period_results" in result
        assert len(result["period_results"]) > 0


class TestCostCalculator:
    """Test CostCalculator class."""
    
    def test_cost_calculator_init(self):
        """Test CostCalculator initialization."""
        calculator = CostCalculator()
        assert calculator.cost_bps == 5.0
        assert calculator.slippage_bps == 2.0
        assert calculator.min_trade_size == 0.001
    
    def test_cost_calculator_init_with_params(self):
        """Test CostCalculator initialization with parameters."""
        calculator = CostCalculator(
            cost_bps=10.0,
            slippage_bps=5.0,
            min_trade_size=0.005
        )
        assert calculator.cost_bps == 10.0
        assert calculator.slippage_bps == 5.0
        assert calculator.min_trade_size == 0.005
    
    def test_calculate_trade_cost(self):
        """Test trade cost calculation."""
        calculator = CostCalculator()
        
        # Test buy trade
        cost = calculator.calculate_trade_cost(1000.0, "buy")
        assert cost > 0
        
        # Test sell trade
        cost = calculator.calculate_trade_cost(1000.0, "sell")
        assert cost > 0
        
        # Test small trade (below minimum)
        cost = calculator.calculate_trade_cost(0.0001, "buy")
        assert cost == 0.0
    
    def test_calculate_portfolio_costs(self):
        """Test portfolio cost calculation."""
        calculator = CostCalculator()
        
        weights_old = pd.Series([0.5, 0.3, 0.2], index=['A', 'B', 'C'])
        weights_new = pd.Series([0.4, 0.4, 0.2], index=['A', 'B', 'C'])
        portfolio_value = 1000000.0
        
        cost = calculator.calculate_portfolio_costs(weights_old, weights_new, portfolio_value)
        assert cost > 0
    
    def test_calculate_turnover(self):
        """Test turnover calculation."""
        calculator = CostCalculator()
        
        weights_old = pd.Series([0.5, 0.3, 0.2], index=['A', 'B', 'C'])
        weights_new = pd.Series([0.4, 0.4, 0.2], index=['A', 'B', 'C'])
        
        turnover = calculator.calculate_turnover(weights_old, weights_new)
        assert turnover > 0
        assert turnover <= 1.0
    
    def test_calculate_cost_drag(self):
        """Test cost drag calculation."""
        calculator = CostCalculator()
        
        turnover = 0.5
        cost_drag = calculator.calculate_cost_drag(turnover)
        assert cost_drag > 0
        assert cost_drag < turnover


class TestTaxCalculator:
    """Test TaxCalculator class."""
    
    def test_tax_calculator_init(self):
        """Test TaxCalculator initialization."""
        calculator = TaxCalculator()
        assert calculator.tax_rate_short == 0.37
        assert calculator.tax_rate_long == 0.20
        assert calculator.tax_rate_dividend == 0.20
        assert calculator.lot_method == "HIFO"
    
    def test_tax_calculator_init_with_params(self):
        """Test TaxCalculator initialization with parameters."""
        calculator = TaxCalculator(
            tax_rate_short=0.40,
            tax_rate_long=0.25,
            tax_rate_dividend=0.25,
            lot_method="FIFO"
        )
        assert calculator.tax_rate_short == 0.40
        assert calculator.tax_rate_long == 0.25
        assert calculator.tax_rate_dividend == 0.25
        assert calculator.lot_method == "FIFO"
    
    def test_add_tax_lot(self):
        """Test adding tax lot."""
        calculator = TaxCalculator()
        
        calculator.add_tax_lot("AAPL", "2023-01-01", 100, 150.0)
        
        assert "AAPL" in calculator.tax_lots
        assert len(calculator.tax_lots["AAPL"]) == 1
        assert calculator.tax_lots["AAPL"][0] == ("2023-01-01", 100, 150.0)
    
    def test_process_sale(self):
        """Test processing sale for tax purposes."""
        calculator = TaxCalculator()
        
        # Add tax lot
        calculator.add_tax_lot("AAPL", "2023-01-01", 100, 150.0)
        
        # Process sale
        result = calculator.process_sale("AAPL", "2023-06-01", 50, 200.0)
        
        assert "tax" in result
        assert "gain_loss" in result
        assert "short_term" in result
        assert "long_term" in result
        assert result["gain_loss"] > 0  # Capital gain
    
    def test_calculate_tax_drag(self):
        """Test tax drag calculation."""
        calculator = TaxCalculator()
        
        turnover = 0.5
        tax_drag = calculator.calculate_tax_drag(turnover)
        assert tax_drag > 0
        assert tax_drag < turnover
    
    def test_calculate_dividend_tax(self):
        """Test dividend tax calculation."""
        calculator = TaxCalculator()
        
        dividend_income = 1000.0
        tax = calculator.calculate_dividend_tax(dividend_income)
        assert tax == dividend_income * calculator.tax_rate_dividend
    
    def test_get_tax_summary(self):
        """Test tax summary generation."""
        calculator = TaxCalculator()
        
        # Add some realized gains/losses
        calculator.realized_gains = 1000.0
        calculator.realized_losses = 500.0
        
        summary = calculator.get_tax_summary()
        
        assert "realized_gains" in summary
        assert "realized_losses" in summary
        assert "net_gain_loss" in summary
        assert "tax_on_gains" in summary
        assert summary["realized_gains"] == 1000.0
        assert summary["realized_losses"] == 500.0
        assert summary["net_gain_loss"] == 500.0


class TestMetricsCalculator:
    """Test MetricsCalculator class."""
    
    def test_metrics_calculator_init(self):
        """Test MetricsCalculator initialization."""
        calculator = MetricsCalculator()
        assert calculator.risk_free_rate == 0.02
    
    def test_metrics_calculator_init_with_params(self):
        """Test MetricsCalculator initialization with parameters."""
        calculator = MetricsCalculator(risk_free_rate=0.03)
        assert calculator.risk_free_rate == 0.03
    
    def test_calculate_return_metrics(self):
        """Test return metrics calculation."""
        calculator = MetricsCalculator()
        
        # Create sample returns
        returns = pd.Series([0.01, 0.02, -0.01, 0.03, 0.01])
        
        metrics = calculator.calculate_return_metrics(returns)
        
        assert "total_return" in metrics
        assert "annualized_return" in metrics
        assert "annualized_volatility" in metrics
        assert "sharpe_ratio" in metrics
        assert "sortino_ratio" in metrics
        assert "calmar_ratio" in metrics
        assert all(isinstance(v, float) for v in metrics.values())
    
    def test_calculate_risk_metrics(self):
        """Test risk metrics calculation."""
        calculator = MetricsCalculator()
        
        # Create sample returns
        returns = pd.Series([0.01, 0.02, -0.01, 0.03, 0.01])
        
        metrics = calculator.calculate_risk_metrics(returns)
        
        assert "var_95" in metrics
        assert "cvar_95" in metrics
        assert "max_drawdown" in metrics
        assert "downside_deviation" in metrics
        assert "tail_ratio" in metrics
        assert "skewness" in metrics
        assert "kurtosis" in metrics
        assert all(isinstance(v, float) for v in metrics.values())
    
    def test_calculate_omega_metrics(self):
        """Test Omega metrics calculation."""
        calculator = MetricsCalculator()
        
        # Create sample returns
        returns = pd.Series([0.01, 0.02, -0.01, 0.03, 0.01])
        
        metrics = calculator.calculate_omega_metrics(returns)
        
        assert "omega_0.0%" in metrics
        assert "omega_2.0%" in metrics
        assert "omega_4.0%" in metrics
        assert all(isinstance(v, float) for v in metrics.values())
    
    def test_calculate_turnover_metrics(self):
        """Test turnover metrics calculation."""
        calculator = MetricsCalculator()
        
        # Create sample weights
        weights = pd.DataFrame({
            'A': [0.5, 0.4, 0.6, 0.5],
            'B': [0.3, 0.4, 0.2, 0.3],
            'C': [0.2, 0.2, 0.2, 0.2]
        })
        
        metrics = calculator.calculate_turnover_metrics(weights)
        
        assert "annual_turnover" in metrics
        assert "max_turnover" in metrics
        assert "turnover_volatility" in metrics
        assert all(isinstance(v, float) for v in metrics.values())
    
    def test_calculate_hit_rate(self):
        """Test hit rate calculation."""
        calculator = MetricsCalculator()
        
        # Create sample returns
        returns = pd.Series([0.01, 0.02, -0.01, 0.03, 0.01])
        
        hit_rate = calculator.calculate_hit_rate(returns, threshold=0.0)
        assert isinstance(hit_rate, float)
        assert 0 <= hit_rate <= 1
    
    def test_calculate_max_drawdown(self):
        """Test maximum drawdown calculation."""
        calculator = MetricsCalculator()
        
        # Create sample returns
        returns = pd.Series([0.01, 0.02, -0.01, 0.03, 0.01])
        
        max_dd = calculator.calculate_max_drawdown(returns)
        assert isinstance(max_dd, float)
        assert max_dd <= 0
    
    def test_calculate_deflated_sharpe(self):
        """Test deflated Sharpe ratio calculation."""
        calculator = MetricsCalculator()
        
        sharpe_ratio = 1.5
        n_observations = 252
        n_strategies = 1
        
        deflated_sharpe = calculator.calculate_deflated_sharpe(sharpe_ratio, n_observations, n_strategies)
        assert isinstance(deflated_sharpe, float)
        assert deflated_sharpe > 0
    
    def test_calculate_probability_backtest_overfitting(self):
        """Test PBO calculation."""
        calculator = MetricsCalculator()
        
        in_sample_sharpe = 1.5
        out_of_sample_sharpe = 1.0
        n_observations = 252
        
        pbo = calculator.calculate_probability_backtest_overfitting(
            in_sample_sharpe, out_of_sample_sharpe, n_observations
        )
        assert isinstance(pbo, float)
        assert 0 <= pbo <= 1


class TestBacktestEvaluator:
    """Test BacktestEvaluator class."""
    
    def test_backtest_evaluator_init(self):
        """Test BacktestEvaluator initialization."""
        evaluator = BacktestEvaluator()
        assert evaluator.risk_free_rate == 0.02
        assert evaluator.metrics_calculator is not None
    
    def test_backtest_evaluator_init_with_params(self):
        """Test BacktestEvaluator initialization with parameters."""
        evaluator = BacktestEvaluator(risk_free_rate=0.03)
        assert evaluator.risk_free_rate == 0.03
    
    def test_evaluate_backtest(self):
        """Test backtest evaluation."""
        evaluator = BacktestEvaluator()
        
        # Create sample returns
        returns = pd.Series([0.01, 0.02, -0.01, 0.03, 0.01])
        
        evaluation = evaluator.evaluate_backtest(returns)
        
        assert "return_metrics" in evaluation
        assert "risk_metrics" in evaluation
        assert "omega_metrics" in evaluation
        assert "hit_rate" in evaluation
        assert "deflated_sharpe" in evaluation
        assert "n_observations" in evaluation
    
    def test_evaluate_backtest_with_benchmark(self):
        """Test backtest evaluation with benchmark."""
        evaluator = BacktestEvaluator()
        
        # Create sample returns
        returns = pd.Series([0.01, 0.02, -0.01, 0.03, 0.01])
        benchmark_returns = pd.Series([0.005, 0.015, -0.005, 0.025, 0.005])
        
        evaluation = evaluator.evaluate_backtest(returns, benchmark_returns=benchmark_returns)
        
        assert "benchmark_metrics" in evaluation
        assert "alpha" in evaluation
        assert "beta" in evaluation
    
    def test_evaluate_walk_forward(self):
        """Test walk-forward evaluation."""
        evaluator = BacktestEvaluator()
        
        # Create sample walk-forward results
        walk_forward_results = {
            "period_results": [
                {"total_return": 0.1, "sharpe_ratio": 1.0, "max_drawdown": -0.05},
                {"total_return": 0.15, "sharpe_ratio": 1.2, "max_drawdown": -0.03},
                {"total_return": 0.08, "sharpe_ratio": 0.8, "max_drawdown": -0.08}
            ]
        }
        
        result = evaluator.evaluate_walk_forward(walk_forward_results)
        
        assert "evaluation" in result
        assert "avg_period_return" in result["evaluation"]
        assert "avg_period_sharpe" in result["evaluation"]
        assert "consistency" in result["evaluation"]
        assert "n_periods" in result["evaluation"]
    
    def test_generate_evaluation_report(self):
        """Test evaluation report generation."""
        evaluator = BacktestEvaluator()
        
        # Create sample evaluation
        evaluation = {
            "return_metrics": {"total_return": 0.1, "sharpe_ratio": 1.0},
            "risk_metrics": {"max_drawdown": -0.05, "var_95": -0.02},
            "omega_metrics": {"omega_0.0%": 1.5, "omega_2.0%": 1.2},
            "hit_rate": 0.6,
            "deflated_sharpe": 0.9
        }
        
        report = evaluator.generate_evaluation_report(evaluation)
        
        assert isinstance(report, str)
        assert "Portfolio Backtest Evaluation" in report
        assert "Return Metrics" in report
        assert "Risk Metrics" in report
        assert "Omega Metrics" in report


if __name__ == "__main__":
    pytest.main([__file__])
