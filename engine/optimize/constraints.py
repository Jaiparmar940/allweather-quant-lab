"""
Portfolio optimization constraints management.
"""

import logging
from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
import pandas as pd
import cvxpy as cp
import structlog

logger = structlog.get_logger(__name__)


class ConstraintManager:
    """Manages portfolio optimization constraints."""
    
    def __init__(self):
        self.constraints = []
        self.constraint_names = []
    
    def add_budget_constraint(self, n_assets: int) -> None:
        """Add budget constraint (weights sum to 1)."""
        
        w = cp.Variable(n_assets, name="weights")
        constraint = cp.sum(w) == 1.0
        
        self.constraints.append(constraint)
        self.constraint_names.append("budget")
        
        logger.debug("Budget constraint added", n_assets=n_assets)
    
    def add_long_only_constraint(self, n_assets: int) -> None:
        """Add long-only constraint (weights >= 0)."""
        
        w = cp.Variable(n_assets, name="weights")
        constraint = w >= 0.0
        
        self.constraints.append(constraint)
        self.constraint_names.append("long_only")
        
        logger.debug("Long-only constraint added", n_assets=n_assets)
    
    def add_weight_bounds_constraint(
        self,
        n_assets: int,
        min_weight: float = 0.0,
        max_weight: float = 1.0
    ) -> None:
        """Add weight bounds constraint."""
        
        w = cp.Variable(n_assets, name="weights")
        constraint = w >= min_weight
        
        self.constraints.append(constraint)
        self.constraint_names.append("weight_lower_bound")
        
        constraint = w <= max_weight
        self.constraints.append(constraint)
        self.constraint_names.append("weight_upper_bound")
        
        logger.debug("Weight bounds constraint added", min_weight=min_weight, max_weight=max_weight)
    
    def add_leverage_constraint(self, n_assets: int, max_leverage: float = 1.0) -> None:
        """Add leverage constraint."""
        
        w = cp.Variable(n_assets, name="weights")
        constraint = cp.norm(w, 1) <= max_leverage
        
        self.constraints.append(constraint)
        self.constraint_names.append("leverage")
        
        logger.debug("Leverage constraint added", max_leverage=max_leverage)
    
    def add_sector_constraint(
        self,
        n_assets: int,
        sector_mapping: Dict[str, List[int]],
        sector_bounds: Dict[str, Tuple[float, float]]
    ) -> None:
        """Add sector weight constraints."""
        
        w = cp.Variable(n_assets, name="weights")
        
        for sector, asset_indices in sector_mapping.items():
            if sector in sector_bounds:
                min_weight, max_weight = sector_bounds[sector]
                
                # Calculate sector weight
                sector_weight = cp.sum([w[i] for i in asset_indices])
                
                # Add constraints
                if min_weight > 0:
                    constraint = sector_weight >= min_weight
                    self.constraints.append(constraint)
                    self.constraint_names.append(f"sector_{sector}_min")
                
                if max_weight < 1.0:
                    constraint = sector_weight <= max_weight
                    self.constraints.append(constraint)
                    self.constraint_names.append(f"sector_{sector}_max")
        
        logger.debug("Sector constraints added", sectors=list(sector_bounds.keys()))
    
    def add_asset_constraint(
        self,
        n_assets: int,
        asset_bounds: Dict[int, Tuple[float, float]]
    ) -> None:
        """Add individual asset weight constraints."""
        
        w = cp.Variable(n_assets, name="weights")
        
        for asset_idx, (min_weight, max_weight) in asset_bounds.items():
            if min_weight > 0:
                constraint = w[asset_idx] >= min_weight
                self.constraints.append(constraint)
                self.constraint_names.append(f"asset_{asset_idx}_min")
            
            if max_weight < 1.0:
                constraint = w[asset_idx] <= max_weight
                self.constraints.append(constraint)
                self.constraint_names.append(f"asset_{asset_idx}_max")
        
        logger.debug("Asset constraints added", count=len(asset_bounds))
    
    def add_turnover_constraint(
        self,
        n_assets: int,
        w_prev: np.ndarray,
        max_turnover: float = 0.5
    ) -> None:
        """Add turnover constraint."""
        
        w = cp.Variable(n_assets, name="weights")
        constraint = cp.norm(w - w_prev, 1) <= max_turnover
        
        self.constraints.append(constraint)
        self.constraint_names.append("turnover")
        
        logger.debug("Turnover constraint added", max_turnover=max_turnover)
    
    def add_cvar_constraint(
        self,
        n_assets: int,
        returns: np.ndarray,
        alpha: float = 0.05,
        cvar_cap: float = 0.05
    ) -> None:
        """Add CVaR constraint."""
        
        w = cp.Variable(n_assets, name="weights")
        
        # Calculate portfolio returns
        portfolio_returns = returns @ w
        
        # CVaR constraint (simplified version)
        # This is a complex constraint that may not be directly expressible in CVXPY
        # We'll implement a simplified version using VaR approximation
        
        # Calculate VaR (simplified)
        var = cp.quantile(portfolio_returns, alpha)
        
        # CVaR constraint (approximate)
        constraint = var >= -cvar_cap
        
        self.constraints.append(constraint)
        self.constraint_names.append("cvar")
        
        logger.debug("CVaR constraint added", alpha=alpha, cvar_cap=cvar_cap)
    
    def add_esg_constraint(
        self,
        n_assets: int,
        esg_scores: np.ndarray,
        min_esg_score: float = 50.0
    ) -> None:
        """Add ESG constraint."""
        
        w = cp.Variable(n_assets, name="weights")
        
        # Calculate weighted average ESG score
        weighted_esg = cp.sum(w * esg_scores)
        
        # ESG constraint
        constraint = weighted_esg >= min_esg_score
        
        self.constraints.append(constraint)
        self.constraint_names.append("esg")
        
        logger.debug("ESG constraint added", min_esg_score=min_esg_score)
    
    def add_liquidity_constraint(
        self,
        n_assets: int,
        liquidity_scores: np.ndarray,
        min_liquidity: float = 0.5
    ) -> None:
        """Add liquidity constraint."""
        
        w = cp.Variable(n_assets, name="weights")
        
        # Calculate weighted average liquidity score
        weighted_liquidity = cp.sum(w * liquidity_scores)
        
        # Liquidity constraint
        constraint = weighted_liquidity >= min_liquidity
        
        self.constraints.append(constraint)
        self.constraint_names.append("liquidity")
        
        logger.debug("Liquidity constraint added", min_liquidity=min_liquidity)
    
    def add_geographic_constraint(
        self,
        n_assets: int,
        geographic_mapping: Dict[str, List[int]],
        geographic_bounds: Dict[str, Tuple[float, float]]
    ) -> None:
        """Add geographic weight constraints."""
        
        w = cp.Variable(n_assets, name="weights")
        
        for region, asset_indices in geographic_mapping.items():
            if region in geographic_bounds:
                min_weight, max_weight = geographic_bounds[region]
                
                # Calculate regional weight
                regional_weight = cp.sum([w[i] for i in asset_indices])
                
                # Add constraints
                if min_weight > 0:
                    constraint = regional_weight >= min_weight
                    self.constraints.append(constraint)
                    self.constraint_names.append(f"region_{region}_min")
                
                if max_weight < 1.0:
                    constraint = regional_weight <= max_weight
                    self.constraints.append(constraint)
                    self.constraint_names.append(f"region_{region}_max")
        
        logger.debug("Geographic constraints added", regions=list(geographic_bounds.keys()))
    
    def add_risk_parity_constraint(
        self,
        n_assets: int,
        cov: np.ndarray,
        target_risk_contrib: Optional[np.ndarray] = None
    ) -> None:
        """Add risk parity constraint."""
        
        w = cp.Variable(n_assets, name="weights")
        
        # Calculate risk contributions
        portfolio_var = cp.quad_form(w, cov)
        marginal_contrib = cov @ w
        risk_contrib = w * marginal_contrib / portfolio_var
        
        if target_risk_contrib is None:
            # Equal risk contribution
            target_risk_contrib = np.ones(n_assets) / n_assets
        
        # Risk parity constraint
        for i in range(n_assets):
            constraint = risk_contrib[i] == target_risk_contrib[i]
            self.constraints.append(constraint)
            self.constraint_names.append(f"risk_parity_{i}")
        
        logger.debug("Risk parity constraint added", n_assets=n_assets)
    
    def add_max_weight_constraint(
        self,
        n_assets: int,
        max_weight: float = 0.1
    ) -> None:
        """Add maximum weight per asset constraint."""
        
        w = cp.Variable(n_assets, name="weights")
        constraint = w <= max_weight
        
        self.constraints.append(constraint)
        self.constraint_names.append("max_weight")
        
        logger.debug("Max weight constraint added", max_weight=max_weight)
    
    def add_min_weight_constraint(
        self,
        n_assets: int,
        min_weight: float = 0.01
    ) -> None:
        """Add minimum weight per asset constraint."""
        
        w = cp.Variable(n_assets, name="weights")
        constraint = w >= min_weight
        
        self.constraints.append(constraint)
        self.constraint_names.append("min_weight")
        
        logger.debug("Min weight constraint added", min_weight=min_weight)
    
    def add_concentration_constraint(
        self,
        n_assets: int,
        max_concentration: float = 0.3
    ) -> None:
        """Add concentration constraint (max weight of any single asset)."""
        
        w = cp.Variable(n_assets, name="weights")
        constraint = w <= max_concentration
        
        self.constraints.append(constraint)
        self.constraint_names.append("concentration")
        
        logger.debug("Concentration constraint added", max_concentration=max_concentration)
    
    def add_herfindahl_constraint(
        self,
        n_assets: int,
        max_herfindahl: float = 0.25
    ) -> None:
        """Add Herfindahl index constraint."""
        
        w = cp.Variable(n_assets, name="weights")
        herfindahl = cp.sum_squares(w)
        constraint = herfindahl <= max_herfindahl
        
        self.constraints.append(constraint)
        self.constraint_names.append("herfindahl")
        
        logger.debug("Herfindahl constraint added", max_herfindahl=max_herfindahl)
    
    def get_constraints(self) -> List[cp.Constraint]:
        """Get all constraints."""
        
        return self.constraints
    
    def get_constraint_names(self) -> List[str]:
        """Get constraint names."""
        
        return self.constraint_names
    
    def clear_constraints(self) -> None:
        """Clear all constraints."""
        
        self.constraints.clear()
        self.constraint_names.clear()
        
        logger.debug("All constraints cleared")
    
    def validate_constraints(self, n_assets: int) -> bool:
        """Validate that constraints are compatible."""
        
        # Check if we have at least a budget constraint
        if "budget" not in self.constraint_names:
            logger.warning("No budget constraint found")
            return False
        
        # Check for conflicting constraints
        if "long_only" in self.constraint_names and "leverage" in self.constraint_names:
            # This might be fine, but worth checking
            pass
        
        # Check if constraints are feasible
        try:
            w = cp.Variable(n_assets, name="weights")
            problem = cp.Problem(cp.Minimize(0), self.constraints)
            problem.solve()
            
            if problem.status == cp.INFEASIBLE:
                logger.error("Constraints are infeasible")
                return False
            
        except Exception as e:
            logger.error("Error validating constraints", error=str(e))
            return False
        
        logger.debug("Constraints validated successfully")
        return True
    
    def get_constraint_summary(self) -> Dict[str, Any]:
        """Get summary of all constraints."""
        
        summary = {
            "total_constraints": len(self.constraints),
            "constraint_types": list(set(self.constraint_names)),
            "constraints": self.constraint_names
        }
        
        return summary
