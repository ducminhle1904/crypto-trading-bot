"""
SSL Basic Strategy implementation.
"""
import pandas as pd
from typing import Tuple, List, Optional, Dict, Any

from trading_bot.strategies.base_strategy import BaseStrategy
from trading_bot.models import Position
from trading_bot.utils.indicators import calculate_ssl_channel, apply_standard_indicators
from trading_bot.config import logger


class SslBasicStrategy(BaseStrategy):
    """
    SSL Basic Strategy uses the SSL Channel (Buy Stop Line and Sell Stop Line) for trend detection.
    Entry signals are generated when price crosses the SSL lines.
    """
    
    def __init__(self, timeframe: str = "15m", use_trailing_profit: bool = True):
        """Initialize the strategy with parameters."""
        super().__init__(name="ssl_basic_strategy", timeframe=timeframe)
        
        # Whether to use trailing profit or fixed take profit
        self.use_trailing_profit = use_trailing_profit
        
        # Set parameters based on timeframe
        self._set_parameters_for_timeframe()
        
        logger.info(f"Initialized SSL Basic Strategy with parameters: SSL Period={self.ssl_period}, "
                   f"ATR Period={self.atr_period}, Trailing Profit: {'Enabled' if use_trailing_profit else 'Disabled'}")
    
    def _set_parameters_for_timeframe(self):
        """Set strategy parameters based on the timeframe."""
        minutes = self.timeframe_minutes
        
        # Base parameters - these will be adjusted based on timeframe
        if minutes <= 15:  # For short timeframes (1m-15m)
            # Faster settings for scalping
            self.ssl_period = 7
            self.atr_period = 10
            self.position_max_candles = 10
            self.profit_target_pct = 0.015  # 1.5%
            self.stop_loss_pct = 0.01  # 1%
        elif minutes <= 60:  # For medium timeframes (30m-1h)
            # Balanced parameters
            self.ssl_period = 10
            self.atr_period = 14
            self.position_max_candles = 8
            self.profit_target_pct = 0.02  # 2%
            self.stop_loss_pct = 0.012  # 1.2%
        else:  # For higher timeframes (4h+)
            # Slower settings for trend following
            self.ssl_period = 14
            self.atr_period = 14
            self.position_max_candles = 6
            self.profit_target_pct = 0.025  # 2.5%
            self.stop_loss_pct = 0.015  # 1.5%
    
    def update_timeframe(self, timeframe: str):
        """Update the strategy timeframe and adjust parameters."""
        super().update_timeframe(timeframe)
        self._set_parameters_for_timeframe()
        logger.info(f"Updated SSL Basic Strategy parameters for {timeframe} timeframe")
    
    async def calculate_indicators(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Calculate indicators for the strategy."""
        try:
            # Apply standard indicators (including ATR)
            df = apply_standard_indicators(
                df,
                atr_period=self.atr_period
            )
            
            # Calculate SSL Channel
            df = calculate_ssl_channel(df, period=self.ssl_period)
            
            return df
        except Exception as e:
            logger.error(f"Error calculating indicators for SSL Basic Strategy: {str(e)}")
            return None
    
    async def check_signals(self, df: pd.DataFrame, position: Optional[Position] = None) -> Tuple[
        bool, bool, bool, List[str], List[str], List[str], List[str]
    ]:
        """Check for trading signals."""
        if df is None or len(df) < self.ssl_period + 2:
            logger.warning("Insufficient data for signal calculation")
            return False, False, False, [], [], [], ["Insufficient data"]
        
        last = df.iloc[-1]
        prev = df.iloc[-2]
        
        long_signals = []
        short_signals = []
        close_signals = []
        fail_reasons = []
        
        # Check for trend changes
        ssl_crossover_bull = prev['ssl_trend'] <= 0 and last['ssl_trend'] > 0
        ssl_crossover_bear = prev['ssl_trend'] >= 0 and last['ssl_trend'] < 0
        
        # Price crossing above ssl_down is bullish
        price_cross_up = prev['close'] <= prev['ssl_down'] and last['close'] > last['ssl_down']
        
        # Price crossing below ssl_up is bearish
        price_cross_down = prev['close'] >= prev['ssl_up'] and last['close'] < last['ssl_up']
        
        # Combined signals
        long_condition = (ssl_crossover_bull or price_cross_up) and last['ssl_trend'] > 0
        short_condition = (ssl_crossover_bear or price_cross_down) and last['ssl_trend'] < 0
        
        if long_condition:
            long_signals.append("SSL bullish trend change detected")
            long_signals.append(f"SSL Up: {last['ssl_up']:.2f}, SSL Down: {last['ssl_down']:.2f}")
            if ssl_crossover_bull:
                long_signals.append("SSL Crossover: Trend changed from bearish/neutral to bullish")
            if price_cross_up:
                long_signals.append("Price crossed above SSL Down line")
        else:
            if not (ssl_crossover_bull or price_cross_up):
                fail_reasons.append("No SSL bullish crossover or price cross up")
            if last['ssl_trend'] <= 0:
                fail_reasons.append(f"SSL trend not bullish: {last['ssl_trend']}")
        
        if short_condition:
            short_signals.append("SSL bearish trend change detected")
            short_signals.append(f"SSL Up: {last['ssl_up']:.2f}, SSL Down: {last['ssl_down']:.2f}")
            if ssl_crossover_bear:
                short_signals.append("SSL Crossover: Trend changed from bullish/neutral to bearish")
            if price_cross_down:
                short_signals.append("Price crossed below SSL Up line")
        
        close_condition = False
        if position:
            if position.side == 'long' and last['ssl_trend'] < 0:
                close_signals.append("SSL trend changed to bearish")
                close_condition = True
            elif position.side == 'short' and last['ssl_trend'] > 0:
                close_signals.append("SSL trend changed to bullish")
                close_condition = True
        
        long_signal = long_condition and not position
        short_signal = short_condition and not position
        close_signal = close_condition and position
        
        return long_signal, short_signal, close_signal, long_signals, short_signals, close_signals, fail_reasons
    
    async def manage_position(self, df: pd.DataFrame, position: Position, balance: float) -> Tuple[
        Position, bool, List[str]
    ]:
        """Manages existing positions with trailing stops and dynamic exits."""
        if not position:
            return position, False, []
        
        last = df.iloc[-1]
        close_signals = []
        close_condition = False
        
        # Initialize if first check of this position
        if not position.trailing_stop:
            position = await self.initialize_stop_loss(position, last['close'], last['atr'], 1.5)
            
            # Initialize trailing profit tracking
            if self.use_trailing_profit:
                position.highest_profit_pct = 0.0
                position.trailing_profit_activated = False
        
        # Increment candle counter
        if not hasattr(position, 'open_candles'):
            position.open_candles = 0
        position.open_candles += 1
        
        # Move to breakeven when the position has a profit of 1%
        position, breakeven_signals = await self.move_to_breakeven(position, last['close'], 0.01)
        close_signals.extend(breakeven_signals)
        
        # Handle trailing profit with dynamic settings
        if self.use_trailing_profit:
            # Calculate current profit percentage
            if position.side == 'long':
                current_profit_pct = (last['close'] - position.entry) / position.entry * 100
            else:  # short
                current_profit_pct = (position.entry - last['close']) / position.entry * 100
                
            # Update highest profit seen
            if not hasattr(position, 'highest_profit_pct'):
                position.highest_profit_pct = current_profit_pct
            elif current_profit_pct > position.highest_profit_pct:
                position.highest_profit_pct = current_profit_pct
                
            # Trailing profit logic based on SSL trend
            if not hasattr(position, 'trailing_profit_activated'):
                position.trailing_profit_activated = False
                
            # Activate trailing profit when profit reaches target
            if current_profit_pct > self.profit_target_pct * 100:
                position.trailing_profit_activated = True
                
                # Allow for 30% pullback from peak profit
                max_pullback = position.highest_profit_pct * 0.3
                
                # Close position if price pulls back too much from the peak
                if position.trailing_profit_activated and current_profit_pct < (position.highest_profit_pct - max_pullback):
                    close_signals.append(f"Trailing profit: Locked in {current_profit_pct:.2f}% (max: {position.highest_profit_pct:.2f}%)")
                    close_condition = True
                
        # Check if the SSL trend has reversed
        if (position.side == 'long' and last['ssl_trend'] < 0) or (position.side == 'short' and last['ssl_trend'] > 0):
            close_signals.append(f"SSL trend reversed: {last['ssl_trend']}")
            close_condition = True
        
        # Check for stop loss hit
        if position.trailing_stop:
            if (position.side == 'long' and last['close'] < position.trailing_stop) or \
               (position.side == 'short' and last['close'] > position.trailing_stop):
                close_signals.append(f"Stop loss hit at {position.trailing_stop:.2f}")
                close_condition = True
        
        # Update trailing stop based on SSL lines
        if position.side == 'long' and last['ssl_down'] > position.trailing_stop:
            position.trailing_stop = last['ssl_down']
            close_signals.append(f"Raised stop to SSL Down: {position.trailing_stop:.2f}")
        elif position.side == 'short' and last['ssl_up'] < position.trailing_stop:
            position.trailing_stop = last['ssl_up']
            close_signals.append(f"Lowered stop to SSL Up: {position.trailing_stop:.2f}")
        
        # Check for time-based exit
        time_exit, time_signals = await self.check_max_holding_time(position, self.position_max_candles)
        close_signals.extend(time_signals)
        if time_exit:
            close_condition = True
        
        return position, close_condition, close_signals 