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
    
    def __init__(self, timeframe: str = "3m"):
        """Initialize the strategy with parameters."""
        super().__init__(name="rsi_strategy", timeframe=timeframe)
        
        # Set parameters based on timeframe
        self._set_parameters_for_timeframe()
        
        logger.info(f"Initialized RSI Strategy with parameters: RSI Period={self.rsi_period}, "
                   f"Overbought={self.rsi_overbought}, Oversold={self.rsi_oversold}, EMA={self.ema_period}")
    
    def _set_parameters_for_timeframe(self):
        """Set strategy parameters based on the timeframe."""
        minutes = self.timeframe_minutes
        
        # Scale parameters based on timeframe
        if minutes <= 5:  # 1m to 5m - faster settings
            self.rsi_period = 14
            self.rsi_overbought = 70
            self.rsi_oversold = 30
            self.ema_period = 50
            self.atr_period = 14
            self.atr_multiple = 2.0
            self.position_max_candles = 15
            
        elif minutes <= 60:  # 15m to 1h - medium settings
            self.rsi_period = 14
            self.rsi_overbought = 75
            self.rsi_oversold = 25
            self.ema_period = 100
            self.atr_period = 20
            self.atr_multiple = 2.5
            self.position_max_candles = 10
            
        else:  # 4h, daily - slower settings
            self.rsi_period = 14
            self.rsi_overbought = 80
            self.rsi_oversold = 20
            self.ema_period = 200
            self.atr_period = 30
            self.atr_multiple = 3.0
            self.position_max_candles = 5
    
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
        rsi_rising = last['rsi'] > prev['rsi']
        rsi_falling = last['rsi'] < prev['rsi']
        
        # RSI reversal conditions
        oversold_reversal = rsi_oversold and rsi_rising
        overbought_reversal = rsi_overbought and rsi_falling
        
        # Long signal: RSI oversold and rising, price above EMA (bullish trend)
        long_condition = oversold_reversal and price_above_ema
        
        # Short signal: RSI overbought and falling, price below EMA (bearish trend)
        short_condition = overbought_reversal and price_below_ema
        
        if long_condition:
            long_signals.append(f"RSI({self.rsi_period}) oversold and rising ({last['rsi']:.2f} < {self.rsi_oversold})")
            long_signals.append(f"Price above EMA({self.ema_period}): {last['close']:.2f} > {last['ema']:.2f}")
        else:
            if not oversold_reversal:
                fail_reasons.append(f"RSI({self.rsi_period}) not oversold or not rising: {last['rsi']:.2f}")
            if not price_above_ema:
                fail_reasons.append(f"Price not above EMA({self.ema_period}): {last['close']:.2f} < {last['ema']:.2f}")
        
        if short_condition:
            short_signals.append(f"RSI({self.rsi_period}) overbought and falling ({last['rsi']:.2f} > {self.rsi_overbought})")
            short_signals.append(f"Price below EMA({self.ema_period}): {last['close']:.2f} < {last['ema']:.2f}")
        else:
            if not overbought_reversal:
                fail_reasons.append(f"RSI({self.rsi_period}) not overbought or not falling: {last['rsi']:.2f}")
            if not price_below_ema:
                fail_reasons.append(f"Price not below EMA({self.ema_period}): {last['close']:.2f} > {last['ema']:.2f}")
        
        # Exit conditions
        close_condition = False
        if position:
            # Calculate stop and target based on ATR
            atr_value = last['atr']
            
            if position.side == 'long':
                # Exit long if RSI becomes overbought or price falls below EMA
                if position.trailing_stop and last['close'] <= position.trailing_stop:
                    close_signals.append(f"Stop loss hit at {position.trailing_stop:.2f}")
                    close_condition = True
                elif overbought_reversal:
                    close_signals.append(f"RSI overbought and falling ({last['rsi']:.2f} > {self.rsi_overbought})")
                    close_condition = True
                elif price_below_ema and not rsi_oversold:
                    close_signals.append(f"Price fell below EMA({self.ema_period}): {last['close']:.2f} < {last['ema']:.2f}")
                    close_condition = True
            
            elif position.side == 'short':
                # Exit short if RSI becomes oversold or price rises above EMA
                if position.trailing_stop and last['close'] >= position.trailing_stop:
                    close_signals.append(f"Stop loss hit at {position.trailing_stop:.2f}")
                    close_condition = True
                elif oversold_reversal:
                    close_signals.append(f"RSI oversold and rising ({last['rsi']:.2f} < {self.rsi_oversold})")
                    close_condition = True
                elif price_above_ema and not rsi_overbought:
                    close_signals.append(f"Price rose above EMA({self.ema_period}): {last['close']:.2f} > {last['ema']:.2f}")
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
            # Calculate initial stop-loss based on ATR
            atr_value = last['atr']
            
            if position.side == 'long':
                position.trailing_stop = position.entry - (atr_value * self.atr_multiple)
            else:
                position.trailing_stop = position.entry + (atr_value * self.atr_multiple)
            
            position.open_candles = 0
            close_signals.append(f"Initial stop set at {position.trailing_stop:.2f} ({self.atr_multiple}x ATR)")
        else:
            # Increment candle counter
            position.open_candles += 1
        
        # Update trailing stop based on price movement
        atr_value = last['atr']
        trailing_atr = self.atr_multiple * 0.75  # Use a tighter trailing factor
        
        if position.side == 'long':
            # Calculate potential new stop level based on ATR
            potential_stop = last['close'] - (atr_value * trailing_atr)
            
            # Move stop up if price has moved favorably
            if position.trailing_stop < potential_stop:
                position.trailing_stop = potential_stop
                close_signals.append(f"Raised stop to {potential_stop:.2f}")
        
        elif position.side == 'short':
            # Calculate potential new stop level based on ATR
            potential_stop = last['close'] + (atr_value * trailing_atr)
            
            # Move stop down if price has moved favorably
            if position.trailing_stop > potential_stop:
                position.trailing_stop = potential_stop
                close_signals.append(f"Lowered stop to {potential_stop:.2f}")
        
        # Time-based exit
        if position.open_candles > self.position_max_candles:
            close_signals.append(f"Position time limit reached ({position.open_candles} candles)")
            close_condition = True
        
        return position, close_condition, close_signals 