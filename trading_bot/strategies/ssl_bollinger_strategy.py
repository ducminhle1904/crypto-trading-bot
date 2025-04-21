"""
SSL Bollinger Strategy implementation.
"""
import pandas as pd
import numpy as np
from typing import Tuple, List, Optional, Dict, Any

from trading_bot.strategies.base_strategy import BaseStrategy
from trading_bot.models import Position
from trading_bot.utils.indicators import calculate_ssl_channel, apply_standard_indicators
from trading_bot.config import logger


class SslBollingerStrategy(BaseStrategy):
    """
    SSL Bollinger Strategy combines SSL Channel with Bollinger Bands.
    Entry signals occur when SSL trend changes and price is near Bollinger Band extremes.
    """
    
    def __init__(self, timeframe: str = "15m", use_trailing_profit: bool = True):
        """Initialize the strategy with parameters."""
        super().__init__(name="ssl_bollinger_strategy", timeframe=timeframe)
        
        # Whether to use trailing profit or fixed take profit
        self.use_trailing_profit = use_trailing_profit
        
        # Set parameters based on timeframe
        self._set_parameters_for_timeframe()
        
        logger.info(f"Initialized SSL Bollinger Strategy with parameters: SSL Period={self.ssl_period}, "
                   f"BB Period={self.bb_period}, BB Std={self.bb_std}, "
                   f"Trailing Profit: {'Enabled' if use_trailing_profit else 'Disabled'}")
    
    def _set_parameters_for_timeframe(self):
        """Set strategy parameters based on the timeframe."""
        minutes = self.timeframe_minutes
        
        # Base parameters - these will be adjusted based on timeframe
        if minutes <= 15:  # For short timeframes (1m-15m)
            # Faster settings for scalping
            self.ssl_period = 7
            self.bb_period = 14
            self.bb_std = 2.2  # Wider bands for short timeframes (more noise)
            self.atr_period = 10
            self.position_max_candles = 10
            self.profit_target_pct = 0.015  # 1.5%
            self.stop_loss_pct = 0.01  # 1%
        elif minutes <= 60:  # For medium timeframes (30m-1h)
            # Balanced parameters
            self.ssl_period = 10
            self.bb_period = 20
            self.bb_std = 2.0
            self.atr_period = 14
            self.position_max_candles = 8
            self.profit_target_pct = 0.02  # 2%
            self.stop_loss_pct = 0.012  # 1.2%
        else:  # For higher timeframes (4h+)
            # Slower settings for trend following
            self.ssl_period = 14
            self.bb_period = 20
            self.bb_std = 2.0
            self.atr_period = 14
            self.position_max_candles = 6
            self.profit_target_pct = 0.025  # 2.5%
            self.stop_loss_pct = 0.015  # 1.5%
    
    def update_timeframe(self, timeframe: str):
        """Update the strategy timeframe and adjust parameters."""
        super().update_timeframe(timeframe)
        self._set_parameters_for_timeframe()
        logger.info(f"Updated SSL Bollinger Strategy parameters for {timeframe} timeframe")
    
    async def calculate_indicators(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Calculate indicators for the strategy."""
        try:
            # Calculate Bollinger Bands
            df['bb_middle'] = df['close'].rolling(window=self.bb_period).mean()
            std_dev = df['close'].rolling(window=self.bb_period).std()
            df['bb_upper'] = df['bb_middle'] + (std_dev * self.bb_std)
            df['bb_lower'] = df['bb_middle'] - (std_dev * self.bb_std)
            
            # BB Width and %B indicators
            df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
            df['bb_pct_b'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
            
            # Apply standard indicators (including ATR)
            df = apply_standard_indicators(
                df,
                atr_period=self.atr_period
            )
            
            # Calculate SSL Channel
            df = calculate_ssl_channel(df, period=self.ssl_period)
            
            return df
        except Exception as e:
            logger.error(f"Error calculating indicators for SSL Bollinger Strategy: {str(e)}")
            return None
    
    async def check_signals(self, df: pd.DataFrame, position: Optional[Position] = None) -> Tuple[
        bool, bool, bool, List[str], List[str], List[str], List[str]
    ]:
        """Check for trading signals."""
        if df is None or len(df) < max(self.ssl_period, self.bb_period) + 2:
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
        
        # Bollinger Band conditions
        bb_lower_touch = last['close'] <= last['bb_lower'] * 1.01  # Within 1% of lower band
        bb_upper_touch = last['close'] >= last['bb_upper'] * 0.99  # Within 1% of upper band
        bb_squeeze = last['bb_width'] < np.percentile(df['bb_width'].tail(20), 20)  # Lower 20% of recent width
        
        # Calculate if price is bouncing off band
        bounce_off_lower = prev['close'] < prev['bb_lower'] and last['close'] > last['bb_lower']
        bounce_off_upper = prev['close'] > prev['bb_upper'] and last['close'] < last['bb_upper']
        
        # Combined signals
        # Long signal: SSL bullish + price near/bouncing off lower BB
        long_condition = (ssl_crossover_bull or (price_cross_up and last['ssl_trend'] > 0)) and \
                        (bb_lower_touch or bounce_off_lower or (bb_squeeze and last['ssl_trend'] > 0))
        
        # Short signal: SSL bearish + price near/bouncing off upper BB
        short_condition = (ssl_crossover_bear or (price_cross_down and last['ssl_trend'] < 0)) and \
                         (bb_upper_touch or bounce_off_upper or (bb_squeeze and last['ssl_trend'] < 0))
        
        if long_condition:
            long_signals.append("SSL bullish trend with Bollinger Band confirmation")
            long_signals.append(f"SSL Up: {last['ssl_up']:.2f}, SSL Down: {last['ssl_down']:.2f}")
            if ssl_crossover_bull:
                long_signals.append("SSL Crossover: Trend changed from bearish/neutral to bullish")
            if price_cross_up:
                long_signals.append("Price crossed above SSL Down line")
            if bb_lower_touch:
                long_signals.append(f"Price near lower Bollinger Band: {last['close']:.2f} ~ {last['bb_lower']:.2f}")
            if bounce_off_lower:
                long_signals.append(f"Price bouncing off lower Bollinger Band")
            if bb_squeeze:
                long_signals.append(f"Bollinger Band squeeze detected: {last['bb_width']:.4f}")
        else:
            if not (ssl_crossover_bull or (price_cross_up and last['ssl_trend'] > 0)):
                fail_reasons.append("No SSL bullish signal")
            if not (bb_lower_touch or bounce_off_lower or (bb_squeeze and last['ssl_trend'] > 0)):
                fail_reasons.append("No Bollinger Band confirmation")
        
        if short_condition:
            short_signals.append("SSL bearish trend with Bollinger Band confirmation")
            short_signals.append(f"SSL Up: {last['ssl_up']:.2f}, SSL Down: {last['ssl_down']:.2f}")
            if ssl_crossover_bear:
                short_signals.append("SSL Crossover: Trend changed from bullish/neutral to bearish")
            if price_cross_down:
                short_signals.append("Price crossed below SSL Up line")
            if bb_upper_touch:
                short_signals.append(f"Price near upper Bollinger Band: {last['close']:.2f} ~ {last['bb_upper']:.2f}")
            if bounce_off_upper:
                short_signals.append(f"Price bouncing off upper Bollinger Band")
            if bb_squeeze:
                short_signals.append(f"Bollinger Band squeeze detected: {last['bb_width']:.4f}")
        
        close_condition = False
        if position:
            # Exit long when SSL turns bearish or price hits upper BB
            if position.side == 'long':
                if last['ssl_trend'] < 0:
                    close_signals.append("SSL trend changed to bearish")
                    close_condition = True
                elif last['close'] >= last['bb_upper'] * 0.98:  # Within 2% of upper band
                    close_signals.append(f"Price reached upper Bollinger Band: {last['close']:.2f} ~ {last['bb_upper']:.2f}")
                    close_condition = True
            
            # Exit short when SSL turns bullish or price hits lower BB
            elif position.side == 'short':
                if last['ssl_trend'] > 0:
                    close_signals.append("SSL trend changed to bullish")
                    close_condition = True
                elif last['close'] <= last['bb_lower'] * 1.02:  # Within 2% of lower band
                    close_signals.append(f"Price reached lower Bollinger Band: {last['close']:.2f} ~ {last['bb_lower']:.2f}")
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
                
            # Advanced trailing profit logic based on both SSL and Bollinger Bands
            if not hasattr(position, 'trailing_profit_activated'):
                position.trailing_profit_activated = False
                
            # Activate trailing profit when profit reaches target
            if current_profit_pct > self.profit_target_pct * 100:
                position.trailing_profit_activated = True
                
                # Adjust trailing based on price position within Bollinger Bands
                pct_b = last['bb_pct_b']
                
                # For longs, tighten trailing stop when price is high in the bands
                # For shorts, tighten trailing stop when price is low in the bands
                if position.side == 'long':
                    trailing_factor = 0.5 - (pct_b * 0.2)  # 0.5 to 0.3 as pct_b goes from 0 to 1
                else:
                    trailing_factor = 0.3 + (pct_b * 0.2)  # 0.3 to 0.5 as pct_b goes from 0 to 1
                
                # Ensure trailing factor is in reasonable range
                trailing_factor = max(0.2, min(0.5, trailing_factor))
                
                max_pullback = position.highest_profit_pct * trailing_factor
                
                # Close position if price pulls back too much from the peak
                if position.trailing_profit_activated and current_profit_pct < (position.highest_profit_pct - max_pullback):
                    close_signals.append(f"Trailing profit: Locked in {current_profit_pct:.2f}% (max: {position.highest_profit_pct:.2f}%)")
                    close_condition = True
        
        # Check for SSL and Bollinger Band exit signals
        if position.side == 'long':
            # Exit long when SSL turns bearish or price hits upper BB
            if last['ssl_trend'] < 0:
                close_signals.append("SSL trend changed to bearish")
                close_condition = True
            elif last['close'] >= last['bb_upper'] * 0.98:  # Within 2% of upper band
                close_signals.append(f"Price reached upper Bollinger Band: {last['close']:.2f} ~ {last['bb_upper']:.2f}")
                close_condition = True
        else:  # short
            # Exit short when SSL turns bullish or price hits lower BB
            if last['ssl_trend'] > 0:
                close_signals.append("SSL trend changed to bullish")
                close_condition = True
            elif last['close'] <= last['bb_lower'] * 1.02:  # Within 2% of lower band
                close_signals.append(f"Price reached lower Bollinger Band: {last['close']:.2f} ~ {last['bb_lower']:.2f}")
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