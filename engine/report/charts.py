"""
Chart generation for portfolio optimization results.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import structlog

logger = structlog.get_logger(__name__)


class ChartGenerator:
    """Generate charts for portfolio optimization results."""
    
    def __init__(self, output_dir: Union[str, Path] = "./results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def generate_performance_charts(
        self,
        backtest_results: Dict[str, Any],
        output_prefix: str = "performance"
    ) -> List[Path]:
        """Generate performance-related charts."""
        
        logger.info("Generating performance charts", prefix=output_prefix)
        
        chart_paths = []
        
        # Cumulative returns chart
        if "performance_history" in backtest_results:
            cum_returns_path = self._plot_cumulative_returns(
                backtest_results["performance_history"],
                output_prefix
            )
            chart_paths.append(cum_returns_path)
        
        # Drawdown chart
        if "performance_history" in backtest_results:
            drawdown_path = self._plot_drawdown(
                backtest_results["performance_history"],
                output_prefix
            )
            chart_paths.append(drawdown_path)
        
        # Rolling Sharpe ratio chart
        if "performance_history" in backtest_results:
            sharpe_path = self._plot_rolling_sharpe(
                backtest_results["performance_history"],
                output_prefix
            )
            chart_paths.append(sharpe_path)
        
        # Returns distribution chart
        if "performance_history" in backtest_results:
            returns_dist_path = self._plot_returns_distribution(
                backtest_results["performance_history"],
                output_prefix
            )
            chart_paths.append(returns_dist_path)
        
        logger.info("Performance charts generated", count=len(chart_paths))
        return chart_paths
    
    def generate_portfolio_charts(
        self,
        backtest_results: Dict[str, Any],
        output_prefix: str = "portfolio"
    ) -> List[Path]:
        """Generate portfolio-related charts."""
        
        logger.info("Generating portfolio charts", prefix=output_prefix)
        
        chart_paths = []
        
        # Portfolio weights over time
        if "weights_history" in backtest_results:
            weights_path = self._plot_portfolio_weights(
                backtest_results["weights_history"],
                output_prefix
            )
            chart_paths.append(weights_path)
        
        # Asset allocation pie chart
        if "final_weights" in backtest_results:
            allocation_path = self._plot_asset_allocation(
                backtest_results["final_weights"],
                output_prefix
            )
            chart_paths.append(allocation_path)
        
        # Risk contribution chart
        if "risk_contribution" in backtest_results:
            risk_contrib_path = self._plot_risk_contribution(
                backtest_results["risk_contribution"],
                output_prefix
            )
            chart_paths.append(risk_contrib_path)
        
        logger.info("Portfolio charts generated", count=len(chart_paths))
        return chart_paths
    
    def generate_regime_charts(
        self,
        regime_results: Dict[str, Any],
        output_prefix: str = "regime"
    ) -> List[Path]:
        """Generate regime-related charts."""
        
        logger.info("Generating regime charts", prefix=output_prefix)
        
        chart_paths = []
        
        # Regime timeline
        if "regime_labels" in regime_results:
            timeline_path = self._plot_regime_timeline(
                regime_results["regime_labels"],
                output_prefix
            )
            chart_paths.append(timeline_path)
        
        # Regime characteristics
        if "regime_characteristics" in regime_results:
            characteristics_path = self._plot_regime_characteristics(
                regime_results["regime_characteristics"],
                output_prefix
            )
            chart_paths.append(characteristics_path)
        
        logger.info("Regime charts generated", count=len(chart_paths))
        return chart_paths
    
    def _plot_cumulative_returns(
        self,
        performance_history: pd.DataFrame,
        output_prefix: str
    ) -> Path:
        """Plot cumulative returns."""
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Calculate cumulative returns
        cumulative_returns = (1 + performance_history["daily_return"]).cumprod()
        
        # Plot cumulative returns
        ax.plot(cumulative_returns.index, cumulative_returns.values, linewidth=2, label="Portfolio")
        
        # Add benchmark if available
        if "benchmark_return" in performance_history.columns:
            benchmark_cumulative = (1 + performance_history["benchmark_return"]).cumprod()
            ax.plot(benchmark_cumulative.index, benchmark_cumulative.values, 
                   linewidth=2, label="Benchmark", alpha=0.7)
        
        ax.set_title("Cumulative Returns", fontsize=16, fontweight='bold')
        ax.set_xlabel("Date", fontsize=12)
        ax.set_ylabel("Cumulative Return", fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Format y-axis as percentage
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1%}'))
        
        # Save chart
        output_path = self.output_dir / f"{output_prefix}_cumulative_returns.png"
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return output_path
    
    def _plot_drawdown(
        self,
        performance_history: pd.DataFrame,
        output_prefix: str
    ) -> Path:
        """Plot drawdown chart."""
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Calculate drawdown
        cumulative_returns = (1 + performance_history["daily_return"]).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        
        # Plot drawdown
        ax.fill_between(drawdown.index, drawdown.values, 0, alpha=0.3, color='red')
        ax.plot(drawdown.index, drawdown.values, color='red', linewidth=1)
        
        ax.set_title("Portfolio Drawdown", fontsize=16, fontweight='bold')
        ax.set_xlabel("Date", fontsize=12)
        ax.set_ylabel("Drawdown", fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # Format y-axis as percentage
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1%}'))
        
        # Save chart
        output_path = self.output_dir / f"{output_prefix}_drawdown.png"
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return output_path
    
    def _plot_rolling_sharpe(
        self,
        performance_history: pd.DataFrame,
        output_prefix: str,
        window: int = 252
    ) -> Path:
        """Plot rolling Sharpe ratio."""
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Calculate rolling Sharpe ratio
        returns = performance_history["daily_return"]
        rolling_sharpe = returns.rolling(window=window).mean() / returns.rolling(window=window).std() * np.sqrt(252)
        
        # Plot rolling Sharpe ratio
        ax.plot(rolling_sharpe.index, rolling_sharpe.values, linewidth=2, color='blue')
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax.axhline(y=1, color='green', linestyle='--', alpha=0.5, label='Sharpe = 1')
        
        ax.set_title(f"Rolling Sharpe Ratio ({window} days)", fontsize=16, fontweight='bold')
        ax.set_xlabel("Date", fontsize=12)
        ax.set_ylabel("Sharpe Ratio", fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Save chart
        output_path = self.output_dir / f"{output_prefix}_rolling_sharpe.png"
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return output_path
    
    def _plot_returns_distribution(
        self,
        performance_history: pd.DataFrame,
        output_prefix: str
    ) -> Path:
        """Plot returns distribution."""
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        returns = performance_history["daily_return"]
        
        # Histogram
        ax1.hist(returns, bins=50, alpha=0.7, color='blue', edgecolor='black')
        ax1.set_title("Returns Distribution", fontsize=14, fontweight='bold')
        ax1.set_xlabel("Daily Return", fontsize=12)
        ax1.set_ylabel("Frequency", fontsize=12)
        ax1.grid(True, alpha=0.3)
        
        # Q-Q plot
        from scipy import stats
        stats.probplot(returns, dist="norm", plot=ax2)
        ax2.set_title("Q-Q Plot (Normal Distribution)", fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # Save chart
        output_path = self.output_dir / f"{output_prefix}_returns_distribution.png"
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return output_path
    
    def _plot_portfolio_weights(
        self,
        weights_history: pd.DataFrame,
        output_prefix: str
    ) -> Path:
        """Plot portfolio weights over time."""
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Plot weights
        for column in weights_history.columns:
            ax.plot(weights_history.index, weights_history[column], 
                   label=column, linewidth=2, alpha=0.8)
        
        ax.set_title("Portfolio Weights Over Time", fontsize=16, fontweight='bold')
        ax.set_xlabel("Date", fontsize=12)
        ax.set_ylabel("Weight", fontsize=12)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        # Format y-axis as percentage
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1%}'))
        
        # Save chart
        output_path = self.output_dir / f"{output_prefix}_weights.png"
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return output_path
    
    def _plot_asset_allocation(
        self,
        final_weights: Dict[str, float],
        output_prefix: str
    ) -> Path:
        """Plot asset allocation pie chart."""
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Prepare data
        assets = list(final_weights.keys())
        weights = list(final_weights.values())
        
        # Create pie chart
        wedges, texts, autotexts = ax.pie(weights, labels=assets, autopct='%1.1f%%', 
                                         startangle=90, colors=plt.cm.Set3.colors)
        
        ax.set_title("Final Asset Allocation", fontsize=16, fontweight='bold')
        
        # Save chart
        output_path = self.output_dir / f"{output_prefix}_allocation.png"
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return output_path
    
    def _plot_risk_contribution(
        self,
        risk_contribution: Dict[str, float],
        output_prefix: str
    ) -> Path:
        """Plot risk contribution chart."""
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Prepare data
        assets = list(risk_contribution.keys())
        contributions = list(risk_contribution.values())
        
        # Create bar chart
        bars = ax.bar(assets, contributions, color=plt.cm.Set3.colors)
        
        ax.set_title("Risk Contribution by Asset", fontsize=16, fontweight='bold')
        ax.set_xlabel("Asset", fontsize=12)
        ax.set_ylabel("Risk Contribution", fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # Format y-axis as percentage
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1%}'))
        
        # Add value labels on bars
        for bar, contribution in zip(bars, contributions):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{contribution:.1%}', ha='center', va='bottom')
        
        # Save chart
        output_path = self.output_dir / f"{output_prefix}_risk_contribution.png"
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return output_path
    
    def _plot_regime_timeline(
        self,
        regime_labels: np.ndarray,
        output_prefix: str
    ) -> Path:
        """Plot regime timeline."""
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Create regime timeline
        ax.plot(regime_labels, linewidth=2, color='blue')
        ax.fill_between(range(len(regime_labels)), regime_labels, alpha=0.3, color='blue')
        
        ax.set_title("Regime Timeline", fontsize=16, fontweight='bold')
        ax.set_xlabel("Time", fontsize=12)
        ax.set_ylabel("Regime", fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # Set y-axis ticks to show regime numbers
        unique_regimes = np.unique(regime_labels)
        ax.set_yticks(unique_regimes)
        ax.set_yticklabels([f"Regime {r}" for r in unique_regimes])
        
        # Save chart
        output_path = self.output_dir / f"{output_prefix}_timeline.png"
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return output_path
    
    def _plot_regime_characteristics(
        self,
        regime_characteristics: Dict[str, Any],
        output_prefix: str
    ) -> Path:
        """Plot regime characteristics."""
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Prepare data
        regimes = list(regime_characteristics.keys())
        counts = [regime_characteristics[regime]["count"] for regime in regimes]
        percentages = [regime_characteristics[regime]["percentage"] for regime in regimes]
        
        # Create bar chart
        bars = ax.bar(regimes, percentages, color=plt.cm.Set3.colors)
        
        ax.set_title("Regime Characteristics", fontsize=16, fontweight='bold')
        ax.set_xlabel("Regime", fontsize=12)
        ax.set_ylabel("Percentage of Time", fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # Format y-axis as percentage
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1f}%'))
        
        # Add value labels on bars
        for bar, percentage in zip(bars, percentages):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{percentage:.1f}%', ha='center', va='bottom')
        
        # Save chart
        output_path = self.output_dir / f"{output_prefix}_characteristics.png"
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return output_path
    
    def generate_interactive_charts(
        self,
        backtest_results: Dict[str, Any],
        output_prefix: str = "interactive"
    ) -> List[Path]:
        """Generate interactive charts using Plotly."""
        
        logger.info("Generating interactive charts", prefix=output_prefix)
        
        chart_paths = []
        
        # Interactive cumulative returns
        if "performance_history" in backtest_results:
            cum_returns_path = self._plot_interactive_cumulative_returns(
                backtest_results["performance_history"],
                output_prefix
            )
            chart_paths.append(cum_returns_path)
        
        # Interactive portfolio weights
        if "weights_history" in backtest_results:
            weights_path = self._plot_interactive_weights(
                backtest_results["weights_history"],
                output_prefix
            )
            chart_paths.append(weights_path)
        
        logger.info("Interactive charts generated", count=len(chart_paths))
        return chart_paths
    
    def _plot_interactive_cumulative_returns(
        self,
        performance_history: pd.DataFrame,
        output_prefix: str
    ) -> Path:
        """Plot interactive cumulative returns using Plotly."""
        
        # Calculate cumulative returns
        cumulative_returns = (1 + performance_history["daily_return"]).cumprod()
        
        # Create figure
        fig = go.Figure()
        
        # Add portfolio line
        fig.add_trace(go.Scatter(
            x=cumulative_returns.index,
            y=cumulative_returns.values,
            mode='lines',
            name='Portfolio',
            line=dict(width=2, color='blue')
        ))
        
        # Add benchmark if available
        if "benchmark_return" in performance_history.columns:
            benchmark_cumulative = (1 + performance_history["benchmark_return"]).cumprod()
            fig.add_trace(go.Scatter(
                x=benchmark_cumulative.index,
                y=benchmark_cumulative.values,
                mode='lines',
                name='Benchmark',
                line=dict(width=2, color='red', dash='dash')
            ))
        
        # Update layout
        fig.update_layout(
            title="Cumulative Returns",
            xaxis_title="Date",
            yaxis_title="Cumulative Return",
            hovermode='x unified',
            template='plotly_white'
        )
        
        # Format y-axis as percentage
        fig.update_yaxis(tickformat='.1%')
        
        # Save chart
        output_path = self.output_dir / f"{output_prefix}_cumulative_returns.html"
        fig.write_html(str(output_path))
        
        return output_path
    
    def _plot_interactive_weights(
        self,
        weights_history: pd.DataFrame,
        output_prefix: str
    ) -> Path:
        """Plot interactive portfolio weights using Plotly."""
        
        # Create figure
        fig = go.Figure()
        
        # Add weight lines for each asset
        for column in weights_history.columns:
            fig.add_trace(go.Scatter(
                x=weights_history.index,
                y=weights_history[column],
                mode='lines',
                name=column,
                line=dict(width=2)
            ))
        
        # Update layout
        fig.update_layout(
            title="Portfolio Weights Over Time",
            xaxis_title="Date",
            yaxis_title="Weight",
            hovermode='x unified',
            template='plotly_white'
        )
        
        # Format y-axis as percentage
        fig.update_yaxis(tickformat='.1%')
        
        # Save chart
        output_path = self.output_dir / f"{output_prefix}_weights.html"
        fig.write_html(str(output_path))
        
        return output_path
