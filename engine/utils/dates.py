"""
Date and time utilities for portfolio optimization.
"""

import logging
from datetime import datetime, timedelta
from typing import List, Optional, Union, Tuple
import pandas as pd
import numpy as np
import structlog

logger = structlog.get_logger(__name__)


class DateUtils:
    """Utility class for date and time operations."""
    
    @staticmethod
    def get_trading_days(
        start_date: Union[str, datetime],
        end_date: Union[str, datetime],
        frequency: str = "daily"
    ) -> pd.DatetimeIndex:
        """Get trading days between start and end dates."""
        
        # Convert to datetime if needed
        if isinstance(start_date, str):
            start_date = pd.to_datetime(start_date)
        if isinstance(end_date, str):
            end_date = pd.to_datetime(end_date)
        
        # Generate date range
        if frequency == "daily":
            date_range = pd.bdate_range(start=start_date, end=end_date, freq="B")
        elif frequency == "weekly":
            date_range = pd.bdate_range(start=start_date, end=end_date, freq="W")
        elif frequency == "monthly":
            date_range = pd.bdate_range(start=start_date, end=end_date, freq="M")
        elif frequency == "quarterly":
            date_range = pd.bdate_range(start=start_date, end=end_date, freq="Q")
        elif frequency == "annually":
            date_range = pd.bdate_range(start=start_date, end=end_date, freq="Y")
        else:
            raise ValueError(f"Unknown frequency: {frequency}")
        
        logger.debug("Trading days generated", count=len(date_range), frequency=frequency)
        return date_range
    
    @staticmethod
    def get_rebalancing_dates(
        start_date: Union[str, datetime],
        end_date: Union[str, datetime],
        frequency: str = "monthly",
        day_of_month: int = 1
    ) -> pd.DatetimeIndex:
        """Get rebalancing dates based on frequency."""
        
        # Convert to datetime if needed
        if isinstance(start_date, str):
            start_date = pd.to_datetime(start_date)
        if isinstance(end_date, str):
            end_date = pd.to_datetime(end_date)
        
        # Generate rebalancing dates
        if frequency == "daily":
            dates = pd.bdate_range(start=start_date, end=end_date, freq="B")
        elif frequency == "weekly":
            dates = pd.bdate_range(start=start_date, end=end_date, freq="W")
        elif frequency == "monthly":
            dates = pd.bdate_range(start=start_date, end=end_date, freq="M")
        elif frequency == "quarterly":
            dates = pd.bdate_range(start=start_date, end=end_date, freq="Q")
        elif frequency == "annually":
            dates = pd.bdate_range(start=start_date, end=end_date, freq="Y")
        else:
            raise ValueError(f"Unknown frequency: {frequency}")
        
        # Adjust to specific day of month if needed
        if frequency in ["monthly", "quarterly", "annually"] and day_of_month != 1:
            adjusted_dates = []
            for date in dates:
                # Try to set the day of month
                try:
                    adjusted_date = date.replace(day=day_of_month)
                    if adjusted_date >= start_date and adjusted_date <= end_date:
                        adjusted_dates.append(adjusted_date)
                except ValueError:
                    # If day doesn't exist in month, use last day
                    last_day = (date + pd.DateOffset(months=1) - pd.DateOffset(days=1)).day
                    adjusted_date = date.replace(day=min(day_of_month, last_day))
                    if adjusted_date >= start_date and adjusted_date <= end_date:
                        adjusted_dates.append(adjusted_date)
            
            dates = pd.DatetimeIndex(adjusted_dates)
        
        logger.debug("Rebalancing dates generated", count=len(dates), frequency=frequency)
        return dates
    
    @staticmethod
    def get_walk_forward_periods(
        start_date: Union[str, datetime],
        end_date: Union[str, datetime],
        train_months: int = 120,
        test_months: int = 12,
        step_months: int = 1
    ) -> List[Tuple[datetime, datetime, datetime, datetime]]:
        """Get walk-forward periods for backtesting."""
        
        # Convert to datetime if needed
        if isinstance(start_date, str):
            start_date = pd.to_datetime(start_date)
        if isinstance(end_date, str):
            end_date = pd.to_datetime(end_date)
        
        periods = []
        current_date = start_date
        
        while current_date < end_date:
            # Training period
            train_start = current_date
            train_end = train_start + pd.DateOffset(months=train_months)
            
            # Test period
            test_start = train_end
            test_end = test_start + pd.DateOffset(months=test_months)
            
            # Check if we have enough data
            if test_end <= end_date:
                periods.append((train_start, train_end, test_start, test_end))
            
            # Move to next period
            current_date += pd.DateOffset(months=step_months)
        
        logger.debug("Walk-forward periods generated", count=len(periods))
        return periods
    
    @staticmethod
    def get_quarterly_dates(
        start_date: Union[str, datetime],
        end_date: Union[str, datetime]
    ) -> pd.DatetimeIndex:
        """Get quarterly dates between start and end dates."""
        
        # Convert to datetime if needed
        if isinstance(start_date, str):
            start_date = pd.to_datetime(start_date)
        if isinstance(end_date, str):
            end_date = pd.to_datetime(end_date)
        
        # Generate quarterly dates
        dates = pd.bdate_range(start=start_date, end=end_date, freq="Q")
        
        logger.debug("Quarterly dates generated", count=len(dates))
        return dates
    
    @staticmethod
    def get_monthly_dates(
        start_date: Union[str, datetime],
        end_date: Union[str, datetime]
    ) -> pd.DatetimeIndex:
        """Get monthly dates between start and end dates."""
        
        # Convert to datetime if needed
        if isinstance(start_date, str):
            start_date = pd.to_datetime(start_date)
        if isinstance(end_date, str):
            end_date = pd.to_datetime(end_date)
        
        # Generate monthly dates
        dates = pd.bdate_range(start=start_date, end=end_date, freq="M")
        
        logger.debug("Monthly dates generated", count=len(dates))
        return dates
    
    @staticmethod
    def get_annual_dates(
        start_date: Union[str, datetime],
        end_date: Union[str, datetime]
    ) -> pd.DatetimeIndex:
        """Get annual dates between start and end dates."""
        
        # Convert to datetime if needed
        if isinstance(start_date, str):
            start_date = pd.to_datetime(start_date)
        if isinstance(end_date, str):
            end_date = pd.to_datetime(end_date)
        
        # Generate annual dates
        dates = pd.bdate_range(start=start_date, end=end_date, freq="Y")
        
        logger.debug("Annual dates generated", count=len(dates))
        return dates
    
    @staticmethod
    def get_business_days_between(
        start_date: Union[str, datetime],
        end_date: Union[str, datetime]
    ) -> int:
        """Get number of business days between two dates."""
        
        # Convert to datetime if needed
        if isinstance(start_date, str):
            start_date = pd.to_datetime(start_date)
        if isinstance(end_date, str):
            end_date = pd.to_datetime(end_date)
        
        # Calculate business days
        business_days = len(pd.bdate_range(start=start_date, end=end_date))
        
        logger.debug("Business days calculated", count=business_days)
        return business_days
    
    @staticmethod
    def get_calendar_days_between(
        start_date: Union[str, datetime],
        end_date: Union[str, datetime]
    ) -> int:
        """Get number of calendar days between two dates."""
        
        # Convert to datetime if needed
        if isinstance(start_date, str):
            start_date = pd.to_datetime(start_date)
        if isinstance(end_date, str):
            end_date = pd.to_datetime(end_date)
        
        # Calculate calendar days
        calendar_days = (end_date - start_date).days
        
        logger.debug("Calendar days calculated", count=calendar_days)
        return calendar_days
    
    @staticmethod
    def get_quarter_end_dates(
        start_date: Union[str, datetime],
        end_date: Union[str, datetime]
    ) -> pd.DatetimeIndex:
        """Get quarter end dates between start and end dates."""
        
        # Convert to datetime if needed
        if isinstance(start_date, str):
            start_date = pd.to_datetime(start_date)
        if isinstance(end_date, str):
            end_date = pd.to_datetime(end_date)
        
        # Generate quarter end dates
        dates = pd.bdate_range(start=start_date, end=end_date, freq="Q")
        
        logger.debug("Quarter end dates generated", count=len(dates))
        return dates
    
    @staticmethod
    def get_year_end_dates(
        start_date: Union[str, datetime],
        end_date: Union[str, datetime]
    ) -> pd.DatetimeIndex:
        """Get year end dates between start and end dates."""
        
        # Convert to datetime if needed
        if isinstance(start_date, str):
            start_date = pd.to_datetime(start_date)
        if isinstance(end_date, str):
            end_date = pd.to_datetime(end_date)
        
        # Generate year end dates
        dates = pd.bdate_range(start=start_date, end=end_date, freq="Y")
        
        logger.debug("Year end dates generated", count=len(dates))
        return dates
    
    @staticmethod
    def is_trading_day(date: Union[str, datetime]) -> bool:
        """Check if a date is a trading day."""
        
        # Convert to datetime if needed
        if isinstance(date, str):
            date = pd.to_datetime(date)
        
        # Check if it's a weekday
        is_weekday = date.weekday() < 5
        
        # Check if it's not a holiday (simplified)
        # In practice, you would use a proper holiday calendar
        is_holiday = False
        
        is_trading = is_weekday and not is_holiday
        
        logger.debug("Trading day check", date=date, is_trading=is_trading)
        return is_trading
    
    @staticmethod
    def get_next_trading_day(
        date: Union[str, datetime],
        n_days: int = 1
    ) -> datetime:
        """Get the next trading day after a given date."""
        
        # Convert to datetime if needed
        if isinstance(date, str):
            date = pd.to_datetime(date)
        
        # Get next trading day
        next_date = date + pd.Timedelta(days=1)
        while not DateUtils.is_trading_day(next_date):
            next_date += pd.Timedelta(days=1)
        
        logger.debug("Next trading day calculated", date=date, next_date=next_date)
        return next_date
    
    @staticmethod
    def get_previous_trading_day(
        date: Union[str, datetime],
        n_days: int = 1
    ) -> datetime:
        """Get the previous trading day before a given date."""
        
        # Convert to datetime if needed
        if isinstance(date, str):
            date = pd.to_datetime(date)
        
        # Get previous trading day
        prev_date = date - pd.Timedelta(days=1)
        while not DateUtils.is_trading_day(prev_date):
            prev_date -= pd.Timedelta(days=1)
        
        logger.debug("Previous trading day calculated", date=date, prev_date=prev_date)
        return prev_date
    
    @staticmethod
    def get_date_range(
        start_date: Union[str, datetime],
        end_date: Union[str, datetime],
        frequency: str = "daily"
    ) -> pd.DatetimeIndex:
        """Get date range with specified frequency."""
        
        # Convert to datetime if needed
        if isinstance(start_date, str):
            start_date = pd.to_datetime(start_date)
        if isinstance(end_date, str):
            end_date = pd.to_datetime(end_date)
        
        # Generate date range
        if frequency == "daily":
            dates = pd.bdate_range(start=start_date, end=end_date, freq="B")
        elif frequency == "weekly":
            dates = pd.bdate_range(start=start_date, end=end_date, freq="W")
        elif frequency == "monthly":
            dates = pd.bdate_range(start=start_date, end=end_date, freq="M")
        elif frequency == "quarterly":
            dates = pd.bdate_range(start=start_date, end=end_date, freq="Q")
        elif frequency == "annually":
            dates = pd.bdate_range(start=start_date, end=end_date, freq="Y")
        else:
            raise ValueError(f"Unknown frequency: {frequency}")
        
        logger.debug("Date range generated", count=len(dates), frequency=frequency)
        return dates
