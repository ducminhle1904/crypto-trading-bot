"""
Base strategy class for implementing trading strategies.
"""
import pandas as pd
from abc import ABC, abstractmethod
from typing import Tuple, List, Dict, Any, Optional

from trading_bot.models import Position
from trading_bot.config import logger, RISK_PER_TRADE, TRADING_FEE


class BaseStrategy(ABC):
    """Base class for all trading strategies."""
    
    def __init__(self, name: str = "base_strategy", timeframe: str = "3m"):
        """Initialize the strategy."""
        self.name = name
        self.timeframe = timeframe
        self.timeframe_minutes = self._parse_timeframe_minutes(timeframe)
        logger.info(f"Initialized {self.name} strategy for {self.timeframe} timeframe")
        
    def _parse_timeframe_minutes(self, timeframe: str) -> int:
        """Convert timeframe string to minutes."""
        if timeframe.endswith('m'):
            return int(timeframe[:-1])
        elif timeframe.endswith('h'):
            return int(timeframe[:-1]) * 60
        elif timeframe.endswith('d'):
            return int(timeframe[:-1]) * 1440  # 24 * 60
        elif timeframe.endswith('w'):
            return int(timeframe[:-1]) * 10080  # 7 * 24 * 60
        else:
            # Default to the provided value or 3m
            return 3
    
    def update_timeframe(self, timeframe: str):
        """Update the strategy timeframe."""
        self.timeframe = timeframe
        self.timeframe_minutes = self._parse_timeframe_minutes(timeframe)
        logger.info(f"Updated {self.name} timeframe to {timeframe}")
        
    @abstractmethod
    async def calculate_indicators(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Calculate indicators for the strategy."""
        pass
        
    @abstractmethod
    async def check_signals(self, df: pd.DataFrame, position: Optional[Position] = None) -> Tuple[
        bool, bool, bool, List[str], List[str], List[str], List[str]
    ]:
        """
        Check for trade signals.
        
        Returns:
            Tuple containing:
            - long_signal: True if should enter long position
            - short_signal: True if should enter short position
            - close_signal: True if should close position
            - long_signals: List of reasons for long signal
            - short_signals: List of reasons for short signal
            - close_signals: List of reasons for close signal
            - fail_reasons: List of reasons why signals failed
        """
        pass
        
    @abstractmethod
    async def manage_position(self, df: pd.DataFrame, position: Position, balance: float) -> Tuple[
        Position, bool, List[str]
    ]:
        """
        Manage existing position with trailing stops and dynamic exits.
        
        Returns:
            Tuple containing:
            - updated position object
            - close_signal: True if position should be closed
            - close_signals: List of reasons for closing
        """
        pass
        
    async def calculate_position_size(self, price: float, balance: float) -> float:
        """Calculate position size based on risk management."""
        stop_loss_percent = 0.0075  # Default, can be overridden
        amount_to_risk = balance * RISK_PER_TRADE
        position_size = amount_to_risk / (price * stop_loss_percent)
        return position_size
        
    async def calculate_profit(self, position: Position, exit_price: float) -> Tuple[float, float]:
        """Calculate profit amount and percentage for a position."""
        if position.side == 'long':
            profit = (exit_price * (1 - TRADING_FEE) - position.entry * (1 + TRADING_FEE)) * position.size
            profit_pct = (exit_price / position.entry - 1) * 100
        else:  # short
            profit = (position.entry * (1 + TRADING_FEE) - exit_price * (1 - TRADING_FEE)) * position.size
            profit_pct = (position.entry / exit_price - 1) * 100
            
        return profit, profit_pct 