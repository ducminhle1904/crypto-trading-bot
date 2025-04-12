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
        if not timeframe:
            logger.warning(f"No timeframe provided to {self.name}, defaulting to 3m")
            return 3
            
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
        
    async def initialize_stop_loss(self, position: Position, last_price: float, 
                                 atr_value: float, atr_multiple: float = 1.5) -> Position:
        """Initialize stop loss for a position based on ATR."""
        if not position.trailing_stop:
            if position.side == 'long':
                position.trailing_stop = position.entry - (atr_value * atr_multiple)
            else:  # short
                position.trailing_stop = position.entry + (atr_value * atr_multiple)
                
            position.open_candles = 0
            logger.info(f"Initial stop set at {position.trailing_stop:.2f} ({atr_multiple}x ATR)")
        
        return position
        
    async def move_to_breakeven(self, position: Position, last_price: float, 
                               min_profit_pct: float = 0.002) -> Tuple[Position, List[str]]:
        """Move stop loss to breakeven if price has moved in favor by min_profit_pct."""
        signals = []
        
        if position.side == 'long':
            breakeven_price = position.entry * (1 + min_profit_pct)
            if last_price > breakeven_price and position.trailing_stop < position.entry:
                position.trailing_stop = position.entry
                signals.append(f"Moved stop-loss to breakeven ({position.entry:.2f})")
                
        elif position.side == 'short':
            breakeven_price = position.entry * (1 - min_profit_pct)
            if last_price < breakeven_price and position.trailing_stop > position.entry:
                position.trailing_stop = position.entry
                signals.append(f"Moved stop-loss to breakeven ({position.entry:.2f})")
                
        return position, signals
        
    async def update_trailing_stop(self, position: Position, last_price: float, 
                                 atr_value: float, trailing_atr_mult: float = 1.0) -> Tuple[Position, List[str]]:
        """Update trailing stop based on ATR and current price."""
        signals = []
        
        if position.side == 'long':
            potential_stop = last_price - (atr_value * trailing_atr_mult)
            if potential_stop > position.trailing_stop:
                position.trailing_stop = potential_stop
                signals.append(f"Raised stop to {potential_stop:.2f}")
                
        elif position.side == 'short':
            potential_stop = last_price + (atr_value * trailing_atr_mult)
            if potential_stop < position.trailing_stop:
                position.trailing_stop = potential_stop
                signals.append(f"Lowered stop to {potential_stop:.2f}")
                
        return position, signals
    
    async def check_max_holding_time(self, position: Position, max_candles: int) -> Tuple[bool, List[str]]:
        """Check if position has reached maximum holding time."""
        signals = []
        close_condition = False
        
        if not hasattr(position, 'open_candles'):
            position.open_candles = 0
            
        if position.open_candles >= max_candles:
            signals.append(f"Position time limit reached ({position.open_candles} candles)")
            close_condition = True
            
        return close_condition, signals
        
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