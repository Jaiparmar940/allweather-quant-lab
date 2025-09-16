"""
Cost calculation modules for portfolio backtesting.
"""

import logging
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import structlog

logger = structlog.get_logger(__name__)


class CostCalculator:
    """Calculate transaction costs and slippage."""
    
    def __init__(
        self,
        cost_bps: float = 5.0,
        slippage_bps: float = 2.0,
        min_trade_size: float = 0.001
    ):
        self.cost_bps = cost_bps
        self.slippage_bps = slippage_bps
        self.min_trade_size = min_trade_size
        
        logger.info("Cost calculator initialized", cost_bps=cost_bps, slippage_bps=slippage_bps)
    
    def calculate_trade_cost(
        self,
        trade_value: float,
        trade_type: str = "both"
    ) -> float:
        """
        Calculate transaction cost for a trade.
        
        Args:
            trade_value: Value of the trade
            trade_type: Type of trade ("buy", "sell", "both")
        
        Returns:
            Transaction cost
        """
        
        if trade_value < self.min_trade_size:
            return 0.0
        
        # Base transaction cost
        base_cost = trade_value * (self.cost_bps / 10000)
        
        # Add slippage cost
        slippage_cost = trade_value * (self.slippage_bps / 10000)
        
        total_cost = base_cost + slippage_cost
        
        logger.debug("Trade cost calculated", trade_value=trade_value, cost=total_cost)
        return total_cost
    
    def calculate_portfolio_costs(
        self,
        weights_old: pd.Series,
        weights_new: pd.Series,
        portfolio_value: float
    ) -> float:
        """
        Calculate total transaction costs for portfolio rebalancing.
        
        Args:
            weights_old: Previous portfolio weights
            weights_new: New portfolio weights
            portfolio_value: Total portfolio value
        
        Returns:
            Total transaction costs
        """
        
        # Calculate weight changes
        weight_changes = (weights_new - weights_old).abs()
        
        # Calculate trade values
        trade_values = weight_changes * portfolio_value
        
        # Calculate total costs
        total_cost = 0.0
        for asset, trade_value in trade_values.items():
            if trade_value > 0:
                cost = self.calculate_trade_cost(trade_value)
                total_cost += cost
        
        logger.debug("Portfolio costs calculated", total_cost=total_cost, portfolio_value=portfolio_value)
        return total_cost
    
    def calculate_turnover(
        self,
        weights_old: pd.Series,
        weights_new: pd.Series
    ) -> float:
        """
        Calculate portfolio turnover.
        
        Args:
            weights_old: Previous portfolio weights
            weights_new: New portfolio weights
        
        Returns:
            Portfolio turnover
        """
        
        turnover = (weights_new - weights_old).abs().sum() / 2
        
        logger.debug("Turnover calculated", turnover=turnover)
        return turnover
    
    def calculate_cost_drag(
        self,
        turnover: float,
        cost_bps: Optional[float] = None
    ) -> float:
        """
        Calculate cost drag from turnover.
        
        Args:
            turnover: Portfolio turnover
            cost_bps: Cost in basis points (uses default if None)
        
        Returns:
            Cost drag
        """
        
        if cost_bps is None:
            cost_bps = self.cost_bps
        
        cost_drag = turnover * (cost_bps / 10000)
        
        logger.debug("Cost drag calculated", turnover=turnover, cost_drag=cost_drag)
        return cost_drag


class TaxCalculator:
    """Calculate tax implications of portfolio trades."""
    
    def __init__(
        self,
        tax_rate_short: float = 0.37,
        tax_rate_long: float = 0.20,
        tax_rate_dividend: float = 0.20,
        lot_method: str = "HIFO"
    ):
        self.tax_rate_short = tax_rate_short
        self.tax_rate_long = tax_rate_long
        self.tax_rate_dividend = tax_rate_dividend
        self.lot_method = lot_method
        
        # Tax lot tracking
        self.tax_lots = {}  # {asset: [(date, quantity, cost_basis), ...]}
        self.realized_gains = 0.0
        self.realized_losses = 0.0
        
        logger.info("Tax calculator initialized", lot_method=lot_method)
    
    def add_tax_lot(
        self,
        asset: str,
        date: Union[str, pd.Timestamp],
        quantity: float,
        cost_basis: float
    ) -> None:
        """Add a tax lot for an asset."""
        
        if asset not in self.tax_lots:
            self.tax_lots[asset] = []
        
        self.tax_lots[asset].append((date, quantity, cost_basis))
        
        logger.debug("Tax lot added", asset=asset, quantity=quantity, cost_basis=cost_basis)
    
    def process_sale(
        self,
        asset: str,
        date: Union[str, pd.Timestamp],
        quantity: float,
        sale_price: float
    ) -> Dict[str, float]:
        """
        Process a sale for tax purposes.
        
        Args:
            asset: Asset being sold
            date: Sale date
            quantity: Quantity sold
            sale_price: Sale price per share
        
        Returns:
            Dict containing tax information
        """
        
        if asset not in self.tax_lots or not self.tax_lots[asset]:
            logger.warning("No tax lots found for asset", asset=asset)
            return {"tax": 0.0, "gain_loss": 0.0, "short_term": 0.0, "long_term": 0.0}
        
        remaining_quantity = quantity
        total_tax = 0.0
        total_gain_loss = 0.0
        short_term_gain_loss = 0.0
        long_term_gain_loss = 0.0
        
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
        for i, (lot_date, lot_quantity, lot_cost_basis) in enumerate(lots):
            if remaining_quantity <= 0:
                break
            
            # Calculate quantity to sell from this lot
            sell_quantity = min(remaining_quantity, lot_quantity)
            
            # Calculate gain/loss
            cost_basis = sell_quantity * lot_cost_basis
            proceeds = sell_quantity * sale_price
            gain_loss = proceeds - cost_basis
            
            # Determine holding period
            holding_period = (pd.to_datetime(date) - pd.to_datetime(lot_date)).days
            
            # Calculate tax
            if gain_loss > 0:
                if holding_period >= 365:
                    tax = gain_loss * self.tax_rate_long
                    long_term_gain_loss += gain_loss
                else:
                    tax = gain_loss * self.tax_rate_short
                    short_term_gain_loss += gain_loss
            else:
                tax = 0
                if holding_period >= 365:
                    long_term_gain_loss += gain_loss
                else:
                    short_term_gain_loss += gain_loss
            
            total_tax += tax
            total_gain_loss += gain_loss
            
            # Update lot
            self.tax_lots[asset][i] = (lot_date, lot_quantity - sell_quantity, lot_cost_basis)
            remaining_quantity -= sell_quantity
        
        # Remove empty lots
        self.tax_lots[asset] = [lot for lot in self.tax_lots[asset] if lot[1] > 0]
        
        # Update realized gains/losses
        if total_gain_loss > 0:
            self.realized_gains += total_gain_loss
        else:
            self.realized_losses += abs(total_gain_loss)
        
        logger.debug("Sale processed", asset=asset, quantity=quantity, tax=total_tax)
        
        return {
            "tax": total_tax,
            "gain_loss": total_gain_loss,
            "short_term": short_term_gain_loss,
            "long_term": long_term_gain_loss
        }
    
    def calculate_tax_drag(
        self,
        turnover: float,
        tax_rate: Optional[float] = None
    ) -> float:
        """
        Calculate tax drag from turnover.
        
        Args:
            turnover: Portfolio turnover
            tax_rate: Tax rate (uses average if None)
        
        Returns:
            Tax drag
        """
        
        if tax_rate is None:
            tax_rate = (self.tax_rate_short + self.tax_rate_long) / 2
        
        tax_drag = turnover * tax_rate
        
        logger.debug("Tax drag calculated", turnover=turnover, tax_drag=tax_drag)
        return tax_drag
    
    def calculate_dividend_tax(
        self,
        dividend_income: float
    ) -> float:
        """
        Calculate tax on dividend income.
        
        Args:
            dividend_income: Dividend income received
        
        Returns:
            Dividend tax
        """
        
        dividend_tax = dividend_income * self.tax_rate_dividend
        
        logger.debug("Dividend tax calculated", dividend_income=dividend_income, tax=dividend_tax)
        return dividend_tax
    
    def get_tax_summary(self) -> Dict[str, float]:
        """Get summary of tax implications."""
        
        total_gain_loss = self.realized_gains - self.realized_losses
        
        # Calculate tax on gains
        if total_gain_loss > 0:
            # Assume long-term rate for simplicity
            tax_on_gains = total_gain_loss * self.tax_rate_long
        else:
            tax_on_gains = 0.0
        
        return {
            "realized_gains": self.realized_gains,
            "realized_losses": self.realized_losses,
            "net_gain_loss": total_gain_loss,
            "tax_on_gains": tax_on_gains,
            "total_tax_lots": sum(len(lots) for lots in self.tax_lots.values())
        }
    
    def reset_tax_lots(self) -> None:
        """Reset all tax lots."""
        
        self.tax_lots.clear()
        self.realized_gains = 0.0
        self.realized_losses = 0.0
        
        logger.debug("Tax lots reset")
    
    def get_asset_tax_lots(self, asset: str) -> List[Tuple[Union[str, pd.Timestamp], float, float]]:
        """Get tax lots for a specific asset."""
        
        return self.tax_lots.get(asset, [])
    
    def get_unrealized_gain_loss(
        self,
        asset: str,
        current_price: float
    ) -> float:
        """
        Calculate unrealized gain/loss for an asset.
        
        Args:
            asset: Asset symbol
            current_price: Current market price
        
        Returns:
            Unrealized gain/loss
        """
        
        if asset not in self.tax_lots:
            return 0.0
        
        total_quantity = sum(lot[1] for lot in self.tax_lots[asset])
        total_cost_basis = sum(lot[1] * lot[2] for lot in self.tax_lots[asset])
        
        current_value = total_quantity * current_price
        unrealized_gain_loss = current_value - total_cost_basis
        
        logger.debug("Unrealized gain/loss calculated", asset=asset, gain_loss=unrealized_gain_loss)
        return unrealized_gain_loss
