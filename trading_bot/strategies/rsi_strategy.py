"""
RSI-based trading strategy.
"""
import pandas as pd
from typing import Tuple, List, Optional, Dict, Any

from trading_bot.strategies.base_strategy import BaseStrategy
from trading_bot.models import Position
from trading_bot.utils.indicators import calculate_rsi, calculate_ema, calculate_atr
from trading_bot.config import logger


class RsiStrategy(BaseStrategy):
    """
    RSI Strategy uses RSI oversold/overbought conditions and EMA filter.
    """
    
    def __init__(self, timeframe: str = "1h", use_trailing_profit: bool = True):
        """Initialize the strategy with parameters."""
        super().__init__(name="rsi_strategy", timeframe=timeframe)
        
        # Whether to use trailing profit or fixed take profit
        self.use_trailing_profit = use_trailing_profit
        
        # Set parameters based on timeframe
        self._set_parameters_for_timeframe()
        
        logger.info(f"Initialized RSI Strategy with parameters: RSI Period={self.rsi_period}, "
                   f"RSI Levels: {self.rsi_oversold}/{self.rsi_overbought}, EMA={self.ema_period}, "
                   f"Trailing Profit: {'Enabled' if use_trailing_profit else 'Disabled'}")
    
    def _set_parameters_for_timeframe(self):
        """Set strategy parameters based on the timeframe."""
        self.rsi_period = 12
        self.rsi_overbought = 75
        self.rsi_oversold = 25
        self.ema_period = 50
        self.atr_period = 14
        self.atr_multiple = 1.5  # Still tighter stops for intraday
        self.position_max_candles = 8
        self.use_close_price_filter = True
        self.profit_target_pct = 0.01  # 1% target
    
    def update_timeframe(self, timeframe: str):
        """Update the strategy timeframe and adjust parameters."""
        super().update_timeframe(timeframe)
        self._set_parameters_for_timeframe()
        logger.info(f"Updated RSI Strategy parameters for {timeframe} timeframe")
    
    async def calculate_indicators(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Calculate indicators for the strategy."""
        try:
            if df is None or len(df) < self.rsi_period:
                return None
                
            df = df.copy()
            
            # Calculate RSI
            df['rsi'] = calculate_rsi(df['close'], self.rsi_period)
            
            # Calculate EMA for trend filter
            df['ema'] = calculate_ema(df['close'], self.ema_period)
            
            # Calculate ATR for volatility-based stops
            df['atr'] = calculate_atr(df, self.atr_period)
            
            # Add previous RSI for momentum checks
            df['prev_rsi'] = df['rsi'].shift(1)
            
            # Calculate price rate of change for additional filter
            df['price_rate_of_change'] = (df['close'] - df['close'].shift(3)) / df['close'].shift(3) * 100
            
            return df
        except Exception as e:
            logger.error(f"Error calculating RSI strategy indicators: {e}")
            return None
    
    async def check_signals(self, df: pd.DataFrame, position: Optional[Position] = None) -> Tuple[
        bool, bool, bool, List[str], List[str], List[str], List[str]
    ]:
        """Check for trading signals."""
        if df is None or len(df) < self.rsi_period + 5:
            logger.warning("Insufficient data for RSI signal calculation")
            return False, False, False, [], [], [], ["Insufficient data"]
        
        last = df.iloc[-1]
        prev = df.iloc[-2] if len(df) > 2 else last
        
        long_signals = []
        short_signals = []
        close_signals = []
        fail_reasons = []
        
        # Price above/below EMA for trend direction
        price_above_ema = last['close'] > last['ema']
        price_below_ema = last['close'] < last['ema']
        
        # RSI conditions
        rsi_oversold = last['rsi'] < self.rsi_oversold
        rsi_overbought = last['rsi'] > self.rsi_overbought
        rsi_rising = last['rsi'] > last['prev_rsi']
        rsi_falling = last['rsi'] < last['prev_rsi']
        
        # RSI reversal conditions
        oversold_reversal = rsi_oversold and rsi_rising
        overbought_reversal = rsi_overbought and rsi_falling
        
        # Price momentum for day trading
        price_momentum_up = last['price_rate_of_change'] > 0
        price_momentum_down = last['price_rate_of_change'] < 0
        
        # Long signal: RSI oversold and rising, price above EMA (bullish trend)
        # For day trading, add momentum as a condition
        long_condition = (oversold_reversal and price_above_ema) or (price_above_ema and rsi_rising and price_momentum_up and last['rsi'] < 40)
        
        # Short signal: RSI overbought and falling, price below EMA (bearish trend)
        # For day trading, add momentum as a condition
        short_condition = (overbought_reversal and price_below_ema) or (price_below_ema and rsi_falling and price_momentum_down and last['rsi'] > 60)
        
        if long_condition:
            if oversold_reversal:
                long_signals.append(f"RSI({self.rsi_period}) oversold and rising ({last['rsi']:.2f} < {self.rsi_oversold})")
            else:
                long_signals.append(f"RSI({self.rsi_period}) rising and below 40 ({last['rsi']:.2f})")
                
            long_signals.append(f"Price above EMA({self.ema_period}): {last['close']:.2f} > {last['ema']:.2f}")
            
            if price_momentum_up:
                long_signals.append(f"Price momentum up: {last['price_rate_of_change']:.2f}%")
        else:
            if not oversold_reversal and not (rsi_rising and last['rsi'] < 40):
                fail_reasons.append(f"RSI({self.rsi_period}) not oversold or not rising: {last['rsi']:.2f}")
            if not price_above_ema:
                fail_reasons.append(f"Price not above EMA({self.ema_period}): {last['close']:.2f} < {last['ema']:.2f}")
            if not price_momentum_up and not oversold_reversal:
                fail_reasons.append(f"No upward price momentum: {last['price_rate_of_change']:.2f}%")
        
        if short_condition:
            if overbought_reversal:
                short_signals.append(f"RSI({self.rsi_period}) overbought and falling ({last['rsi']:.2f} > {self.rsi_overbought})")
            else:
                short_signals.append(f"RSI({self.rsi_period}) falling and above 60 ({last['rsi']:.2f})")
                
            short_signals.append(f"Price below EMA({self.ema_period}): {last['close']:.2f} < {last['ema']:.2f}")
            
            if price_momentum_down:
                short_signals.append(f"Price momentum down: {last['price_rate_of_change']:.2f}%")
        else:
            if not overbought_reversal and not (rsi_falling and last['rsi'] > 60):
                fail_reasons.append(f"RSI({self.rsi_period}) not overbought or not falling: {last['rsi']:.2f}")
            if not price_below_ema:
                fail_reasons.append(f"Price not below EMA({self.ema_period}): {last['close']:.2f} > {last['ema']:.2f}")
            if not price_momentum_down and not overbought_reversal:
                fail_reasons.append(f"No downward price momentum: {last['price_rate_of_change']:.2f}%")
        
        # Exit conditions
        close_condition = False
        if position:
            # Take profit level for day trading
            take_profit_pct = self.profit_target_pct
            
            if position.side == 'long':
                take_profit = position.entry * (1 + take_profit_pct)
                
                # Exit long if RSI becomes overbought or price falls below EMA
                if position.trailing_stop and last['close'] <= position.trailing_stop:
                    close_signals.append(f"Stop loss hit at {position.trailing_stop:.2f}")
                    close_condition = True
                elif last['close'] >= take_profit:
                    close_signals.append(f"Take profit reached: {last['close']:.2f} >= {take_profit:.2f}")
                    close_condition = True
                elif overbought_reversal:
                    close_signals.append(f"RSI overbought and falling ({last['rsi']:.2f} > {self.rsi_overbought})")
                    close_condition = True
                elif self.use_close_price_filter and price_below_ema and not rsi_oversold:
                    close_signals.append(f"Price fell below EMA({self.ema_period}): {last['close']:.2f} < {last['ema']:.2f}")
                    close_condition = True
                # Additional day trading exit - RSI momentum shift
                elif self.use_close_price_filter and price_momentum_down and rsi_falling and last['rsi'] > 50:
                    close_signals.append(f"RSI momentum shifting down from above 50: {last['rsi']:.2f}")
                    close_condition = True
            
            elif position.side == 'short':
                take_profit = position.entry * (1 - take_profit_pct)
                
                # Exit short if RSI becomes oversold or price rises above EMA
                if position.trailing_stop and last['close'] >= position.trailing_stop:
                    close_signals.append(f"Stop loss hit at {position.trailing_stop:.2f}")
                    close_condition = True
                elif last['close'] <= take_profit:
                    close_signals.append(f"Take profit reached: {last['close']:.2f} <= {take_profit:.2f}")
                    close_condition = True
                elif oversold_reversal:
                    close_signals.append(f"RSI oversold and rising ({last['rsi']:.2f} < {self.rsi_oversold})")
                    close_condition = True
                elif self.use_close_price_filter and price_above_ema and not rsi_overbought:
                    close_signals.append(f"Price rose above EMA({self.ema_period}): {last['close']:.2f} > {last['ema']:.2f}")
                    close_condition = True
                # Additional day trading exit - RSI momentum shift
                elif self.use_close_price_filter and price_momentum_up and rsi_rising and last['rsi'] < 50:
                    close_signals.append(f"RSI momentum shifting up from below 50: {last['rsi']:.2f}")
                    close_condition = True
        
        long_signal = long_condition and not position
        short_signal = short_condition and not position
        close_signal = close_condition and position
        
        return long_signal, short_signal, close_signal, long_signals, short_signals, close_signals, fail_reasons
    
    async def manage_position(self, df: pd.DataFrame, position: Position, balance: float) -> Tuple[
        Position, bool, List[str]
    ]:
        """Manages existing positions with ATR-based trailing stops."""
        if not position:
            return position, False, []
        
        last = df.iloc[-1]
        close_signals = []
        close_condition = False
        
        # Initialize if first check of this position
        if not position.trailing_stop:
            # Initialize ATR-based stop loss
            position = await self.initialize_stop_loss(position, last['close'], last['atr'], self.atr_multiple)
            
            # Initialize trailing profit tracking
            if self.use_trailing_profit:
                position.highest_profit_pct = 0.0
                position.trailing_profit_activated = False
            
        # Increment candle counter
        if not hasattr(position, 'open_candles'):
            position.open_candles = 0
        position.open_candles += 1
        
        # Move to breakeven quickly for day trading
        min_profit_pct = 0.003 if self.timeframe_minutes <= 60 else 0.005
        position, breakeven_signals = await self.move_to_breakeven(position, last['close'], min_profit_pct)
        close_signals.extend(breakeven_signals)
        
        # Handle trailing profit if enabled
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
                
            # Lock in profit with trailing take-profit once we reach 1.5x the target
            trailing_profit_threshold = self.profit_target_pct * 100 * 1.5
            if not hasattr(position, 'trailing_profit_activated'):
                position.trailing_profit_activated = False
                
            # RSI can help with exit timing for trailing profit
            rsi_extreme = (position.side == 'long' and last['rsi'] > 70) or \
                          (position.side == 'short' and last['rsi'] < 30)
                
            if current_profit_pct > trailing_profit_threshold:
                position.trailing_profit_activated = True
                
                # Allow only 25% pullback from highest profit (tighter for RSI-based strategy)
                max_pullback = position.highest_profit_pct * 0.25
                
                # Close position if price pulls back too much from the peak
                # Also consider RSI extremes for exits
                if position.trailing_profit_activated and (current_profit_pct < (position.highest_profit_pct - max_pullback) or rsi_extreme):
                    close_signals.append(f"Trailing profit: Locked in {current_profit_pct:.2f}% (max: {position.highest_profit_pct:.2f}%)")
                    if rsi_extreme:
                        close_signals.append(f"RSI extreme value: {last['rsi']:.2f}")
                    close_condition = True
        
        # Update trailing stop - tighter for day trading
        trailing_atr_mult = self.atr_multiple * 0.6  # Use 60% of the initial ATR multiple for tighter trailing
        position, trailing_signals = await self.update_trailing_stop(
            position, last['close'], last['atr'], trailing_atr_mult
        )
        close_signals.extend(trailing_signals)
        
        # Check for time-based exit
        time_exit, time_signals = await self.check_max_holding_time(position, self.position_max_candles)
        close_signals.extend(time_signals)
        if time_exit:
            close_condition = True
        
        return position, close_condition, close_signals 