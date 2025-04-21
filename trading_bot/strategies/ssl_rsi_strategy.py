"""
SSL RSI Strategy implementation.
"""
import pandas as pd
from typing import Tuple, List, Optional, Dict, Any

from trading_bot.strategies.base_strategy import BaseStrategy
from trading_bot.models import Position
from trading_bot.utils.indicators import calculate_ssl_channel, apply_standard_indicators
from trading_bot.config import logger


class SslRsiStrategy(BaseStrategy):
    """
    SSL RSI Strategy combines SSL Channel with RSI for confirmation.
    Entry signals require both SSL trend change and RSI confirmation.
    """
    
    def __init__(self, timeframe: str = "15m", use_trailing_profit: bool = True):
        """Initialize the strategy with parameters."""
        super().__init__(name="ssl_rsi_strategy", timeframe=timeframe)
        
        # Whether to use trailing profit or fixed take profit
        self.use_trailing_profit = use_trailing_profit
        
        # Set parameters based on timeframe
        self._set_parameters_for_timeframe()
        
        logger.info(f"Initialized SSL RSI Strategy with parameters: SSL Period={self.ssl_period}, "
                   f"RSI Period={self.rsi_period}, RSI Overbought={self.rsi_overbought}, "
                   f"RSI Oversold={self.rsi_oversold}, Trailing Profit: {'Enabled' if use_trailing_profit else 'Disabled'}")
    
    def _set_parameters_for_timeframe(self):
        """Set strategy parameters based on the timeframe."""
        minutes = self.timeframe_minutes
        
        # Base parameters - these will be adjusted based on timeframe
        if minutes <= 15:  # For short timeframes (1m-15m)
            # Faster settings for scalping
            self.ssl_period = 7
            self.rsi_period = 7
            self.rsi_overbought = 75
            self.rsi_oversold = 25
            self.atr_period = 10
            self.position_max_candles = 10
            self.profit_target_pct = 0.015  # 1.5%
            self.stop_loss_pct = 0.01  # 1%
        elif minutes <= 60:  # For medium timeframes (30m-1h)
            # Balanced parameters
            self.ssl_period = 10
            self.rsi_period = 10
            self.rsi_overbought = 70
            self.rsi_oversold = 30
            self.atr_period = 14
            self.position_max_candles = 8
            self.profit_target_pct = 0.02  # 2%
            self.stop_loss_pct = 0.012  # 1.2%
        else:  # For higher timeframes (4h+)
            # Slower settings for trend following
            self.ssl_period = 14
            self.rsi_period = 14
            self.rsi_overbought = 70
            self.rsi_oversold = 30
            self.atr_period = 14
            self.position_max_candles = 6
            self.profit_target_pct = 0.025  # 2.5%
            self.stop_loss_pct = 0.015  # 1.5%
    
    def update_timeframe(self, timeframe: str):
        """Update the strategy timeframe and adjust parameters."""
        super().update_timeframe(timeframe)
        self._set_parameters_for_timeframe()
        logger.info(f"Updated SSL RSI Strategy parameters for {timeframe} timeframe")
    
    async def calculate_indicators(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Calculate indicators for the strategy."""
        try:
            # Apply standard indicators (including RSI and ATR)
            df = apply_standard_indicators(
                df,
                rsi_period=self.rsi_period,
                atr_period=self.atr_period
            )
            
            # Calculate SSL Channel
            df = calculate_ssl_channel(df, period=self.ssl_period)
            
            return df
        except Exception as e:
            logger.error(f"Error calculating indicators for SSL RSI Strategy: {str(e)}")
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
        
        # SSL signals
        ssl_crossover_bull = prev['ssl_trend'] <= 0 and last['ssl_trend'] > 0
        ssl_crossover_bear = prev['ssl_trend'] >= 0 and last['ssl_trend'] < 0
        
        # Price crossing above ssl_down is bullish
        price_cross_up = prev['close'] <= prev['ssl_down'] and last['close'] > last['ssl_down']
        
        # Price crossing below ssl_up is bearish
        price_cross_down = prev['close'] >= prev['ssl_up'] and last['close'] < last['ssl_up']
        
        # RSI conditions
        rsi_oversold = last['rsi'] < self.rsi_oversold
        rsi_overbought = last['rsi'] > self.rsi_overbought
        rsi_rising = last['rsi'] > prev['rsi']
        rsi_falling = last['rsi'] < prev['rsi']
        
        # Combined signals
        long_condition = ((ssl_crossover_bull or price_cross_up) and last['ssl_trend'] > 0) and \
                        (rsi_oversold or (rsi_rising and last['rsi'] < 60))
        
        short_condition = ((ssl_crossover_bear or price_cross_down) and last['ssl_trend'] < 0) and \
                        (rsi_overbought or (rsi_falling and last['rsi'] > 40))
        
        if long_condition:
            long_signals.append("SSL bullish trend with RSI confirmation")
            long_signals.append(f"SSL Up: {last['ssl_up']:.2f}, SSL Down: {last['ssl_down']:.2f}")
            long_signals.append(f"RSI: {last['rsi']:.2f}")
            if ssl_crossover_bull:
                long_signals.append("SSL Crossover: Trend changed from bearish/neutral to bullish")
            if price_cross_up:
                long_signals.append("Price crossed above SSL Down line")
            if rsi_oversold:
                long_signals.append(f"RSI oversold: {last['rsi']:.2f} < {self.rsi_oversold}")
            elif rsi_rising:
                long_signals.append(f"RSI rising: {prev['rsi']:.2f} -> {last['rsi']:.2f}")
        else:
            if not (ssl_crossover_bull or price_cross_up) or last['ssl_trend'] <= 0:
                fail_reasons.append("No SSL bullish signal")
            if not (rsi_oversold or (rsi_rising and last['rsi'] < 60)):
                fail_reasons.append(f"No RSI confirmation: {last['rsi']:.2f}")
        
        if short_condition:
            short_signals.append("SSL bearish trend with RSI confirmation")
            short_signals.append(f"SSL Up: {last['ssl_up']:.2f}, SSL Down: {last['ssl_down']:.2f}")
            short_signals.append(f"RSI: {last['rsi']:.2f}")
            if ssl_crossover_bear:
                short_signals.append("SSL Crossover: Trend changed from bullish/neutral to bearish")
            if price_cross_down:
                short_signals.append("Price crossed below SSL Up line")
            if rsi_overbought:
                short_signals.append(f"RSI overbought: {last['rsi']:.2f} > {self.rsi_overbought}")
            elif rsi_falling:
                short_signals.append(f"RSI falling: {prev['rsi']:.2f} -> {last['rsi']:.2f}")
        
        close_condition = False
        if position:
            # Exit long when SSL turns bearish or RSI gets overbought
            if position.side == 'long' and (last['ssl_trend'] < 0 or last['rsi'] > self.rsi_overbought):
                if last['ssl_trend'] < 0:
                    close_signals.append("SSL trend changed to bearish")
                if last['rsi'] > self.rsi_overbought:
                    close_signals.append(f"RSI overbought: {last['rsi']:.2f} > {self.rsi_overbought}")
                close_condition = True
            
            # Exit short when SSL turns bullish or RSI gets oversold
            elif position.side == 'short' and (last['ssl_trend'] > 0 or last['rsi'] < self.rsi_oversold):
                if last['ssl_trend'] > 0:
                    close_signals.append("SSL trend changed to bullish")
                if last['rsi'] < self.rsi_oversold:
                    close_signals.append(f"RSI oversold: {last['rsi']:.2f} < {self.rsi_oversold}")
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
        prev = df.iloc[-2] if len(df) > 1 else None
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
                
            # Advanced trailing profit logic based on both SSL and RSI
            if not hasattr(position, 'trailing_profit_activated'):
                position.trailing_profit_activated = False
                
            # Activate trailing profit when profit reaches target
            if current_profit_pct > self.profit_target_pct * 100:
                position.trailing_profit_activated = True
                
                # Adjust pullback allowance based on RSI
                if position.side == 'long':
                    # For longs, allow smaller pullbacks when RSI is falling
                    rsi_factor = 0.3 if (prev and last['rsi'] < prev['rsi']) else 0.4
                else:
                    # For shorts, allow smaller pullbacks when RSI is rising
                    rsi_factor = 0.3 if (prev and last['rsi'] > prev['rsi']) else 0.4
                
                max_pullback = position.highest_profit_pct * rsi_factor
                
                # Close position if price pulls back too much from the peak
                if position.trailing_profit_activated and current_profit_pct < (position.highest_profit_pct - max_pullback):
                    close_signals.append(f"Trailing profit: Locked in {current_profit_pct:.2f}% (max: {position.highest_profit_pct:.2f}%)")
                    close_condition = True
        
        # Check for SSL and RSI exit signals
        if position.side == 'long':
            # Exit long when SSL turns bearish or RSI gets overbought and starts falling
            if last['ssl_trend'] < 0:
                close_signals.append("SSL trend changed to bearish")
                close_condition = True
            elif last['rsi'] > self.rsi_overbought and prev and last['rsi'] < prev['rsi']:
                close_signals.append(f"RSI overbought and falling: {last['rsi']:.2f}")
                close_condition = True
        else:  # short
            # Exit short when SSL turns bullish or RSI gets oversold and starts rising
            if last['ssl_trend'] > 0:
                close_signals.append("SSL trend changed to bullish")
                close_condition = True
            elif last['rsi'] < self.rsi_oversold and prev and last['rsi'] > prev['rsi']:
                close_signals.append(f"RSI oversold and rising: {last['rsi']:.2f}")
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