"""
Portfolio backtesting simulator.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
import pandas as pd
import structlog

logger = structlog.get_logger(__name__)


class BacktestSimulator:
    """Portfolio backtesting simulator."""
    
    def __init__(
        self,
        initial_capital: float = 1000000.0,
        rebalance_frequency: str = "monthly",
        transaction_costs: float = 0.0005,
        slippage: float = 0.0002,
        tax_aware: bool = False,
        tax_rate_short: float = 0.37,
        tax_rate_long: float = 0.20,
        lot_method: str = "HIFO"
    ):
        self.initial_capital = initial_capital
        self.rebalance_frequency = rebalance_frequency
        self.transaction_costs = transaction_costs
        self.slippage = slippage
        self.tax_aware = tax_aware
        self.tax_rate_short = tax_rate_short
        self.tax_rate_long = tax_rate_long
        self.lot_method = lot_method
        
        # Portfolio state
        self.current_capital = initial_capital
        self.current_weights = None
        self.current_positions = None
        self.cash = initial_capital
        self.positions = {}
        self.trade_history = []
        self.performance_history = []
        
        # Tax tracking
        self.tax_lots = {}  # {asset: [(date, quantity, cost_basis), ...]}
        self.realized_gains = 0.0
        self.realized_losses = 0.0
        
        logger.info("Backtest simulator initialized", initial_capital=initial_capital)
    
    def run_backtest(
        self,
        returns: pd.DataFrame,
        weights: pd.DataFrame,
        prices: pd.DataFrame,
        start_date: Optional[Union[str, datetime]] = None,
        end_date: Optional[Union[str, datetime]] = None
    ) -> Dict[str, Any]:
        """
        Run portfolio backtest.
        
        Args:
            returns: Asset returns DataFrame (T x N)
            weights: Portfolio weights DataFrame (T x N)
            prices: Asset prices DataFrame (T x N)
            start_date: Start date for backtest
            end_date: End date for backtest
        
        Returns:
            Dict containing backtest results
        """
        
        logger.info("Starting portfolio backtest", start_date=start_date, end_date=end_date)
        
        # Set date range
        if start_date is None:
            start_date = returns.index[0]
        if end_date is None:
            end_date = returns.index[-1]
        
        # Convert to datetime if needed
        if isinstance(start_date, str):
            start_date = pd.to_datetime(start_date)
        if isinstance(end_date, str):
            end_date = pd.to_datetime(end_date)
        
        # Filter data to date range
        returns = returns[(returns.index >= start_date) & (returns.index <= end_date)]
        weights = weights[(weights.index >= start_date) & (weights.index <= end_date)]
        prices = prices[(prices.index >= start_date) & (prices.index <= end_date)]
        
        # Initialize portfolio
        self._initialize_portfolio(weights.iloc[0], prices.iloc[0])
        
        # Run backtest
        for i, (date, row_returns) in enumerate(returns.iterrows()):
            if i == 0:
                continue  # Skip first row (no previous weights)
            
            # Get current weights and prices
            current_weights = weights.iloc[i]
            current_prices = prices.iloc[i]
            
            # Check if rebalancing is needed
            if self._should_rebalance(date):
                self._rebalance_portfolio(current_weights, current_prices, date)
            
            # Update portfolio value
            self._update_portfolio_value(current_prices, date)
            
            # Record performance
            self._record_performance(date)
        
        # Calculate final results
        results = self._calculate_results(returns, weights, prices)
        
        logger.info("Portfolio backtest completed", final_value=results["final_value"])
        return results
    
    def _initialize_portfolio(self, initial_weights: pd.Series, initial_prices: pd.Series) -> None:
        """Initialize portfolio with initial weights and prices."""
        
        # Calculate initial positions
        total_value = self.initial_capital
        self.positions = {}
        
        for asset, weight in initial_weights.items():
            if weight > 0:
                target_value = total_value * weight
                shares = target_value / initial_prices[asset]
                self.positions[asset] = shares
                
                # Initialize tax lots
                if self.tax_aware:
                    self.tax_lots[asset] = [(initial_prices.index, shares, initial_prices[asset])]
        
        # Calculate cash
        self.cash = total_value - sum(
            shares * initial_prices[asset] 
            for asset, shares in self.positions.items()
        )
        
        self.current_capital = total_value
        self.current_weights = initial_weights
        
        logger.debug("Portfolio initialized", positions=len(self.positions), cash=self.cash)
    
    def _should_rebalance(self, date: Union[str, datetime]) -> bool:
        """Check if portfolio should be rebalanced."""
        
        if self.rebalance_frequency == "daily":
            return True
        elif self.rebalance_frequency == "weekly":
            return date.weekday() == 0  # Monday
        elif self.rebalance_frequency == "monthly":
            return date.day == 1
        elif self.rebalance_frequency == "quarterly":
            return date.month in [1, 4, 7, 10] and date.day == 1
        elif self.rebalance_frequency == "annually":
            return date.month == 1 and date.day == 1
        else:
            return False
    
    def _rebalance_portfolio(
        self,
        target_weights: pd.Series,
        current_prices: pd.Series,
        date: Union[str, datetime]
    ) -> None:
        """Rebalance portfolio to target weights."""
        
        # Calculate current portfolio value
        current_value = self._calculate_portfolio_value(current_prices)
        
        # Calculate target positions
        target_positions = {}
        for asset, weight in target_weights.items():
            if weight > 0:
                target_value = current_value * weight
                target_shares = target_value / current_prices[asset]
                target_positions[asset] = target_shares
        
        # Calculate trades needed
        trades = {}
        for asset in set(list(self.positions.keys()) + list(target_positions.keys())):
            current_shares = self.positions.get(asset, 0)
            target_shares = target_positions.get(asset, 0)
            trade_shares = target_shares - current_shares
            
            if abs(trade_shares) > 1e-6:  # Minimum trade size
                trades[asset] = trade_shares
        
        # Execute trades
        if trades:
            self._execute_trades(trades, current_prices, date)
        
        # Update positions
        self.positions = target_positions
        self.current_weights = target_weights
        
        logger.debug("Portfolio rebalanced", trades=len(trades), date=date)
    
    def _execute_trades(
        self,
        trades: Dict[str, float],
        prices: pd.Series,
        date: Union[str, datetime]
    ) -> None:
        """Execute trades with costs and slippage."""
        
        total_cost = 0.0
        
        for asset, trade_shares in trades.items():
            if trade_shares > 0:  # Buy
                # Apply slippage
                effective_price = prices[asset] * (1 + self.slippage)
                trade_value = trade_shares * effective_price
                
                # Calculate transaction cost
                transaction_cost = trade_value * self.transaction_costs
                total_cost += transaction_cost
                
                # Update cash
                self.cash -= (trade_value + transaction_cost)
                
                # Update tax lots
                if self.tax_aware:
                    if asset not in self.tax_lots:
                        self.tax_lots[asset] = []
                    self.tax_lots[asset].append((date, trade_shares, effective_price))
                
                # Record trade
                self.trade_history.append({
                    "date": date,
                    "asset": asset,
                    "action": "buy",
                    "shares": trade_shares,
                    "price": effective_price,
                    "value": trade_value,
                    "cost": transaction_cost
                })
            
            elif trade_shares < 0:  # Sell
                # Apply slippage
                effective_price = prices[asset] * (1 - self.slippage)
                trade_value = abs(trade_shares) * effective_price
                
                # Calculate transaction cost
                transaction_cost = trade_value * self.transaction_costs
                total_cost += transaction_cost
                
                # Update cash
                self.cash += (trade_value - transaction_cost)
                
                # Handle tax lots
                if self.tax_aware:
                    self._process_sale(asset, abs(trade_shares), effective_price, date)
                
                # Record trade
                self.trade_history.append({
                    "date": date,
                    "asset": asset,
                    "action": "sell",
                    "shares": abs(trade_shares),
                    "price": effective_price,
                    "value": trade_value,
                    "cost": transaction_cost
                })
        
        # Update capital
        self.current_capital -= total_cost
        
        logger.debug("Trades executed", total_cost=total_cost, trades=len(trades))
    
    def _process_sale(
        self,
        asset: str,
        shares: float,
        price: float,
        date: Union[str, datetime]
    ) -> None:
        """Process sale for tax purposes."""
        
        if asset not in self.tax_lots or not self.tax_lots[asset]:
            return
        
        remaining_shares = shares
        total_proceeds = shares * price
        
        # Sort lots by method
        if self.lot_method == "FIFO":
            lots = self.tax_lots[asset]
        elif self.lot_method == "LIFO":
            lots = list(reversed(self.tax_lots[asset]))
        elif self.lot_method == "HIFO":
            lots = sorted(self.tax_lots[asset], key=lambda x: x[2], reverse=True)
        else:  # Average
            lots = self.tax_lots[asset]
        
        # Process lots
        for i, (lot_date, lot_shares, lot_cost) in enumerate(lots):
            if remaining_shares <= 0:
                break
            
            shares_to_sell = min(remaining_shares, lot_shares)
            cost_basis = shares_to_sell * lot_cost
            proceeds = shares_to_sell * price
            gain_loss = proceeds - cost_basis
            
            # Determine tax rate
            holding_period = (date - lot_date).days
            if holding_period >= 365:
                tax_rate = self.tax_rate_long
            else:
                tax_rate = self.tax_rate_short
            
            # Calculate tax
            if gain_loss > 0:
                tax = gain_loss * tax_rate
                self.realized_gains += gain_loss
            else:
                tax = 0
                self.realized_losses += abs(gain_loss)
            
            # Update lot
            self.tax_lots[asset][i] = (lot_date, lot_shares - shares_to_sell, lot_cost)
            remaining_shares -= shares_to_sell
        
        # Remove empty lots
        self.tax_lots[asset] = [lot for lot in self.tax_lots[asset] if lot[1] > 0]
    
    def _update_portfolio_value(self, prices: pd.Series, date: Union[str, datetime]) -> None:
        """Update portfolio value based on current prices."""
        
        # Calculate position values
        position_values = {}
        for asset, shares in self.positions.items():
            position_values[asset] = shares * prices[asset]
        
        # Calculate total portfolio value
        self.current_capital = self.cash + sum(position_values.values())
        
        # Update current weights
        if self.current_capital > 0:
            self.current_weights = pd.Series({
                asset: value / self.current_capital
                for asset, value in position_values.items()
            })
    
    def _record_performance(self, date: Union[str, datetime]) -> None:
        """Record portfolio performance for the date."""
        
        # Calculate daily return
        if len(self.performance_history) > 0:
            prev_value = self.performance_history[-1]["portfolio_value"]
            daily_return = (self.current_capital - prev_value) / prev_value
        else:
            daily_return = 0.0
        
        # Record performance
        self.performance_history.append({
            "date": date,
            "portfolio_value": self.current_capital,
            "daily_return": daily_return,
            "cash": self.cash,
            "positions": self.positions.copy(),
            "weights": self.current_weights.copy() if self.current_weights is not None else {}
        })
    
    def _calculate_portfolio_value(self, prices: pd.Series) -> float:
        """Calculate current portfolio value."""
        
        position_value = sum(
            shares * prices[asset]
            for asset, shares in self.positions.items()
        )
        
        return self.cash + position_value
    
    def _calculate_results(
        self,
        returns: pd.DataFrame,
        weights: pd.DataFrame,
        prices: pd.DataFrame
    ) -> Dict[str, Any]:
        """Calculate backtest results."""
        
        if not self.performance_history:
            return {}
        
        # Convert performance history to DataFrame
        perf_df = pd.DataFrame(self.performance_history)
        perf_df.set_index("date", inplace=True)
        
        # Calculate metrics
        total_return = (perf_df["portfolio_value"].iloc[-1] / perf_df["portfolio_value"].iloc[0]) - 1
        annualized_return = (1 + total_return) ** (252 / len(perf_df)) - 1
        annualized_vol = perf_df["daily_return"].std() * np.sqrt(252)
        sharpe_ratio = annualized_return / annualized_vol if annualized_vol > 0 else 0
        
        # Calculate maximum drawdown
        cumulative = (1 + perf_df["daily_return"]).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Calculate turnover
        turnover = perf_df["daily_return"].abs().mean() * 252
        
        # Calculate costs
        total_costs = sum(trade["cost"] for trade in self.trade_history)
        cost_drag = total_costs / self.initial_capital
        
        # Calculate taxes
        total_tax = (self.realized_gains - self.realized_losses) * self.tax_rate_long
        tax_drag = total_tax / self.initial_capital
        
        results = {
            "initial_capital": self.initial_capital,
            "final_value": perf_df["portfolio_value"].iloc[-1],
            "total_return": total_return,
            "annualized_return": annualized_return,
            "annualized_volatility": annualized_vol,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "turnover": turnover,
            "cost_drag": cost_drag,
            "tax_drag": tax_drag,
            "total_costs": total_costs,
            "total_tax": total_tax,
            "performance_history": perf_df,
            "trade_history": self.trade_history
        }
        
        return results


class WalkForwardEngine:
    """Walk-forward backtesting engine."""
    
    def __init__(
        self,
        train_months: int = 120,
        test_months: int = 12,
        step_months: int = 1,
        min_train_months: int = 60
    ):
        self.train_months = train_months
        self.test_months = test_months
        self.step_months = step_months
        self.min_train_months = min_train_months
        
        logger.info("Walk-forward engine initialized", train_months=train_months, test_months=test_months)
    
    def run_walk_forward(
        self,
        returns: pd.DataFrame,
        optimizer_func: callable,
        start_date: Optional[Union[str, datetime]] = None,
        end_date: Optional[Union[str, datetime]] = None
    ) -> Dict[str, Any]:
        """
        Run walk-forward backtest.
        
        Args:
            returns: Asset returns DataFrame (T x N)
            optimizer_func: Function that takes returns and returns weights
            start_date: Start date for backtest
            end_date: End date for backtest
        
        Returns:
            Dict containing walk-forward results
        """
        
        logger.info("Starting walk-forward backtest", start_date=start_date, end_date=end_date)
        
        # Set date range
        if start_date is None:
            start_date = returns.index[0]
        if end_date is None:
            end_date = returns.index[-1]
        
        # Convert to datetime if needed
        if isinstance(start_date, str):
            start_date = pd.to_datetime(start_date)
        if isinstance(end_date, str):
            end_date = pd.to_datetime(end_date)
        
        # Generate walk-forward periods
        periods = self._generate_periods(returns.index, start_date, end_date)
        
        # Run backtest for each period
        results = []
        for i, (train_start, train_end, test_start, test_end) in enumerate(periods):
            logger.info(f"Running period {i+1}/{len(periods)}", train_start=train_start, test_start=test_start)
            
            # Get training data
            train_returns = returns[(returns.index >= train_start) & (returns.index < train_end)]
            
            if len(train_returns) < self.min_train_months * 21:  # Approximate trading days
                logger.warning("Insufficient training data", period=i+1, train_days=len(train_returns))
                continue
            
            # Optimize portfolio
            try:
                weights = optimizer_func(train_returns)
                if weights is None:
                    logger.warning("Optimization failed", period=i+1)
                    continue
                
                # Get test data
                test_returns = returns[(returns.index >= test_start) & (returns.index < test_end)]
                
                # Calculate test performance
                test_performance = self._calculate_test_performance(weights, test_returns)
                
                results.append({
                    "period": i + 1,
                    "train_start": train_start,
                    "train_end": train_end,
                    "test_start": test_start,
                    "test_end": test_end,
                    "weights": weights,
                    "test_performance": test_performance
                })
                
            except Exception as e:
                logger.error("Error in period", period=i+1, error=str(e))
                continue
        
        # Combine results
        combined_results = self._combine_results(results)
        
        logger.info("Walk-forward backtest completed", periods=len(results))
        return combined_results
    
    def _generate_periods(
        self,
        dates: pd.DatetimeIndex,
        start_date: datetime,
        end_date: datetime
    ) -> List[Tuple[datetime, datetime, datetime, datetime]]:
        """Generate walk-forward periods."""
        
        periods = []
        current_date = start_date
        
        while current_date < end_date:
            # Training period
            train_start = current_date
            train_end = train_start + pd.DateOffset(months=self.train_months)
            
            # Test period
            test_start = train_end
            test_end = test_start + pd.DateOffset(months=self.test_months)
            
            # Check if we have enough data
            if test_end <= end_date:
                periods.append((train_start, train_end, test_start, test_end))
            
            # Move to next period
            current_date += pd.DateOffset(months=self.step_months)
        
        return periods
    
    def _calculate_test_performance(
        self,
        weights: pd.Series,
        test_returns: pd.DataFrame
    ) -> Dict[str, float]:
        """Calculate test period performance."""
        
        # Calculate portfolio returns
        portfolio_returns = (test_returns * weights).sum(axis=1)
        
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
        
        return {
            "total_return": total_return,
            "annualized_return": annualized_return,
            "annualized_volatility": annualized_vol,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "portfolio_returns": portfolio_returns
        }
    
    def _combine_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Combine walk-forward results."""
        
        if not results:
            return {}
        
        # Combine all test performances
        all_returns = []
        all_metrics = []
        
        for result in results:
            test_perf = result["test_performance"]
            all_returns.extend(test_perf["portfolio_returns"].tolist())
            all_metrics.append({
                "period": result["period"],
                "total_return": test_perf["total_return"],
                "annualized_return": test_perf["annualized_return"],
                "annualized_volatility": test_perf["annualized_volatility"],
                "sharpe_ratio": test_perf["sharpe_ratio"],
                "max_drawdown": test_perf["max_drawdown"]
            })
        
        # Calculate overall metrics
        all_returns = pd.Series(all_returns)
        overall_return = (1 + all_returns).prod() - 1
        overall_vol = all_returns.std() * np.sqrt(252)
        overall_sharpe = overall_return / overall_vol if overall_vol > 0 else 0
        
        # Calculate maximum drawdown
        cumulative = (1 + all_returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        overall_max_drawdown = drawdown.min()
        
        return {
            "overall_return": overall_return,
            "overall_volatility": overall_vol,
            "overall_sharpe": overall_sharpe,
            "overall_max_drawdown": overall_max_drawdown,
            "period_results": all_metrics,
            "all_returns": all_returns
        }
