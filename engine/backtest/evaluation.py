"""
Portfolio backtest evaluation and metrics calculation.
"""

import logging
from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
import pandas as pd
from scipy import stats
import structlog

logger = structlog.get_logger(__name__)


class MetricsCalculator:
    """Calculate portfolio performance metrics."""
    
    def __init__(self, risk_free_rate: float = 0.02):
        self.risk_free_rate = risk_free_rate
    
    def calculate_return_metrics(
        self,
        returns: pd.Series,
        annualize: bool = True
    ) -> Dict[str, float]:
        """Calculate return-based metrics."""
        
        if len(returns) == 0:
            return {}
        
        # Basic return metrics
        total_return = (1 + returns).prod() - 1
        
        if annualize:
            periods_per_year = 252  # Assuming daily returns
            annualized_return = (1 + total_return) ** (periods_per_year / len(returns)) - 1
        else:
            annualized_return = total_return
        
        # Volatility
        if annualize:
            annualized_vol = returns.std() * np.sqrt(252)
        else:
            annualized_vol = returns.std()
        
        # Sharpe ratio
        excess_returns = returns - self.risk_free_rate / 252
        sharpe_ratio = excess_returns.mean() * 252 / (returns.std() * np.sqrt(252)) if returns.std() > 0 else 0
        
        # Sortino ratio
        downside_returns = excess_returns[excess_returns < 0]
        if len(downside_returns) > 0:
            downside_std = downside_returns.std() * np.sqrt(252)
            sortino_ratio = annualized_return / downside_std if downside_std > 0 else 0
        else:
            sortino_ratio = np.inf
        
        # Calmar ratio
        max_drawdown = self.calculate_max_drawdown(returns)
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else np.inf
        
        return {
            "total_return": total_return,
            "annualized_return": annualized_return,
            "annualized_volatility": annualized_vol,
            "sharpe_ratio": sharpe_ratio,
            "sortino_ratio": sortino_ratio,
            "calmar_ratio": calmar_ratio
        }
    
    def calculate_risk_metrics(
        self,
        returns: pd.Series,
        confidence_level: float = 0.05
    ) -> Dict[str, float]:
        """Calculate risk-based metrics."""
        
        if len(returns) == 0:
            return {}
        
        # Value at Risk (VaR)
        var = returns.quantile(confidence_level)
        
        # Conditional Value at Risk (CVaR)
        cvar = returns[returns <= var].mean()
        
        # Maximum drawdown
        max_drawdown = self.calculate_max_drawdown(returns)
        
        # Downside deviation
        downside_returns = returns[returns < 0]
        downside_deviation = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
        
        # Tail ratio
        p95 = returns.quantile(0.95)
        p5 = returns.quantile(0.05)
        tail_ratio = p95 / abs(p5) if p5 != 0 else np.inf
        
        # Skewness and Kurtosis
        skewness = returns.skew()
        kurtosis = returns.kurtosis()
        
        return {
            "var_95": var,
            "cvar_95": cvar,
            "max_drawdown": max_drawdown,
            "downside_deviation": downside_deviation,
            "tail_ratio": tail_ratio,
            "skewness": skewness,
            "kurtosis": kurtosis
        }
    
    def calculate_omega_metrics(
        self,
        returns: pd.Series,
        theta_values: List[float] = [0.0, 0.02, 0.04]
    ) -> Dict[str, float]:
        """Calculate Omega ratio metrics."""
        
        if len(returns) == 0:
            return {}
        
        omega_metrics = {}
        
        for theta in theta_values:
            # Convert annual theta to daily
            daily_theta = theta / 252
            
            # Calculate excess returns
            excess_returns = returns - daily_theta
            
            # Separate positive and negative excess returns
            positive_returns = excess_returns[excess_returns > 0]
            negative_returns = excess_returns[excess_returns < 0]
            
            # Calculate Omega ratio
            if len(negative_returns) == 0:
                omega_ratio = np.inf
            else:
                positive_sum = positive_returns.sum()
                negative_sum = abs(negative_returns.sum())
                omega_ratio = positive_sum / negative_sum
            
            omega_metrics[f"omega_{theta:.0%}"] = omega_ratio
        
        return omega_metrics
    
    def calculate_turnover_metrics(
        self,
        weights: pd.DataFrame
    ) -> Dict[str, float]:
        """Calculate turnover metrics."""
        
        if len(weights) == 0:
            return {}
        
        # Calculate daily turnover
        daily_turnover = weights.diff().abs().sum(axis=1)
        
        # Annual turnover
        annual_turnover = daily_turnover.mean() * 252
        
        # Maximum turnover
        max_turnover = daily_turnover.max()
        
        # Turnover volatility
        turnover_vol = daily_turnover.std() * np.sqrt(252)
        
        return {
            "annual_turnover": annual_turnover,
            "max_turnover": max_turnover,
            "turnover_volatility": turnover_vol
        }
    
    def calculate_hit_rate(
        self,
        returns: pd.Series,
        threshold: float = 0.0
    ) -> float:
        """Calculate hit rate (percentage of positive returns above threshold)."""
        
        if len(returns) == 0:
            return 0.0
        
        hit_rate = (returns > threshold).mean()
        return hit_rate
    
    def calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown."""
        
        if len(returns) == 0:
            return 0.0
        
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        
        return drawdown.min()
    
    def calculate_deflated_sharpe(
        self,
        sharpe_ratio: float,
        n_observations: int,
        n_strategies: int = 1
    ) -> float:
        """Calculate deflated Sharpe ratio."""
        
        if n_observations <= 1:
            return 0.0
        
        # Calculate deflation factor
        deflation_factor = np.sqrt((n_observations - 1) / (n_observations - 3))
        
        # Calculate deflated Sharpe ratio
        deflated_sharpe = sharpe_ratio * deflation_factor
        
        return deflated_sharpe
    
    def calculate_probability_backtest_overfitting(
        self,
        in_sample_sharpe: float,
        out_of_sample_sharpe: float,
        n_observations: int
    ) -> float:
        """Calculate Probability of Backtest Overfitting (PBO)."""
        
        if n_observations <= 1:
            return 0.0
        
        # Calculate t-statistic
        t_stat = (in_sample_sharpe - out_of_sample_sharpe) / np.sqrt(2 / n_observations)
        
        # Calculate PBO (simplified version)
        pbo = 1 - stats.norm.cdf(t_stat)
        
        return pbo


class BacktestEvaluator:
    """Evaluate portfolio backtest results."""
    
    def __init__(self, risk_free_rate: float = 0.02):
        self.risk_free_rate = risk_free_rate
        self.metrics_calculator = MetricsCalculator(risk_free_rate)
    
    def evaluate_backtest(
        self,
        returns: pd.Series,
        weights: Optional[pd.DataFrame] = None,
        benchmark_returns: Optional[pd.Series] = None
    ) -> Dict[str, Any]:
        """Evaluate portfolio backtest results."""
        
        logger.info("Evaluating backtest results", n_observations=len(returns))
        
        # Calculate return metrics
        return_metrics = self.metrics_calculator.calculate_return_metrics(returns)
        
        # Calculate risk metrics
        risk_metrics = self.metrics_calculator.calculate_risk_metrics(returns)
        
        # Calculate Omega metrics
        omega_metrics = self.metrics_calculator.calculate_omega_metrics(returns)
        
        # Calculate turnover metrics if weights provided
        turnover_metrics = {}
        if weights is not None:
            turnover_metrics = self.metrics_calculator.calculate_turnover_metrics(weights)
        
        # Calculate hit rate
        hit_rate = self.metrics_calculator.calculate_hit_rate(returns)
        
        # Calculate deflated Sharpe ratio
        deflated_sharpe = self.metrics_calculator.calculate_deflated_sharpe(
            return_metrics.get("sharpe_ratio", 0),
            len(returns)
        )
        
        # Combine all metrics
        evaluation = {
            "return_metrics": return_metrics,
            "risk_metrics": risk_metrics,
            "omega_metrics": omega_metrics,
            "turnover_metrics": turnover_metrics,
            "hit_rate": hit_rate,
            "deflated_sharpe": deflated_sharpe,
            "n_observations": len(returns)
        }
        
        # Add benchmark comparison if provided
        if benchmark_returns is not None:
            benchmark_metrics = self.metrics_calculator.calculate_return_metrics(benchmark_returns)
            evaluation["benchmark_metrics"] = benchmark_metrics
            
            # Calculate alpha and beta
            alpha, beta = self._calculate_alpha_beta(returns, benchmark_returns)
            evaluation["alpha"] = alpha
            evaluation["beta"] = beta
        
        logger.info("Backtest evaluation completed", sharpe_ratio=return_metrics.get("sharpe_ratio", 0))
        return evaluation
    
    def evaluate_walk_forward(
        self,
        walk_forward_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Evaluate walk-forward backtest results."""
        
        logger.info("Evaluating walk-forward results")
        
        if not walk_forward_results or "period_results" not in walk_forward_results:
            return {}
        
        period_results = walk_forward_results["period_results"]
        
        # Calculate period-level statistics
        period_returns = [result["total_return"] for result in period_results]
        period_sharpes = [result["sharpe_ratio"] for result in period_results]
        period_drawdowns = [result["max_drawdown"] for result in period_results]
        
        # Calculate statistics
        avg_return = np.mean(period_returns)
        avg_sharpe = np.mean(period_sharpes)
        avg_drawdown = np.mean(period_drawdowns)
        
        # Calculate consistency metrics
        positive_periods = sum(1 for r in period_returns if r > 0)
        consistency = positive_periods / len(period_returns) if period_returns else 0
        
        # Calculate stability metrics
        return_std = np.std(period_returns)
        sharpe_std = np.std(period_sharpes)
        
        # Calculate overall metrics
        overall_metrics = {
            "avg_period_return": avg_return,
            "avg_period_sharpe": avg_sharpe,
            "avg_period_drawdown": avg_drawdown,
            "consistency": consistency,
            "return_volatility": return_std,
            "sharpe_volatility": sharpe_std,
            "n_periods": len(period_results)
        }
        
        # Add to walk-forward results
        walk_forward_results["evaluation"] = overall_metrics
        
        logger.info("Walk-forward evaluation completed", n_periods=len(period_results))
        return walk_forward_results
    
    def _calculate_alpha_beta(
        self,
        returns: pd.Series,
        benchmark_returns: pd.Series
    ) -> Tuple[float, float]:
        """Calculate alpha and beta relative to benchmark."""
        
        # Align returns
        aligned_returns = returns.align(benchmark_returns, join='inner')
        portfolio_returns = aligned_returns[0]
        benchmark_returns = aligned_returns[1]
        
        if len(portfolio_returns) == 0:
            return 0.0, 0.0
        
        # Calculate beta
        covariance = np.cov(portfolio_returns, benchmark_returns)[0, 1]
        benchmark_var = np.var(benchmark_returns)
        beta = covariance / benchmark_var if benchmark_var > 0 else 0
        
        # Calculate alpha
        alpha = portfolio_returns.mean() - beta * benchmark_returns.mean()
        
        return alpha, beta
    
    def generate_evaluation_report(
        self,
        evaluation: Dict[str, Any],
        title: str = "Portfolio Backtest Evaluation"
    ) -> str:
        """Generate a text report of evaluation results."""
        
        report = f"\n{title}\n"
        report += "=" * len(title) + "\n\n"
        
        # Return metrics
        if "return_metrics" in evaluation:
            report += "Return Metrics:\n"
            report += "-" * 20 + "\n"
            for metric, value in evaluation["return_metrics"].items():
                if isinstance(value, float):
                    report += f"{metric}: {value:.4f}\n"
            report += "\n"
        
        # Risk metrics
        if "risk_metrics" in evaluation:
            report += "Risk Metrics:\n"
            report += "-" * 20 + "\n"
            for metric, value in evaluation["risk_metrics"].items():
                if isinstance(value, float):
                    report += f"{metric}: {value:.4f}\n"
            report += "\n"
        
        # Omega metrics
        if "omega_metrics" in evaluation:
            report += "Omega Metrics:\n"
            report += "-" * 20 + "\n"
            for metric, value in evaluation["omega_metrics"].items():
                if isinstance(value, float):
                    report += f"{metric}: {value:.4f}\n"
            report += "\n"
        
        # Turnover metrics
        if "turnover_metrics" in evaluation:
            report += "Turnover Metrics:\n"
            report += "-" * 20 + "\n"
            for metric, value in evaluation["turnover_metrics"].items():
                if isinstance(value, float):
                    report += f"{metric}: {value:.4f}\n"
            report += "\n"
        
        # Additional metrics
        if "hit_rate" in evaluation:
            report += f"Hit Rate: {evaluation['hit_rate']:.4f}\n"
        
        if "deflated_sharpe" in evaluation:
            report += f"Deflated Sharpe: {evaluation['deflated_sharpe']:.4f}\n"
        
        return report
