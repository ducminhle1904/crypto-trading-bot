"""
EMA Dow Volume Strategy implementation.

This strategy combines:
1. EMA crossovers for trend identification
2. Dow Theory principles for trend confirmation
3. Volume analysis for entry/exit signals
"""
import pandas as pd
import numpy as np
from typing import Tuple, List, Optional, Dict, Any

from trading_bot.strategies.base_strategy import BaseStrategy
from trading_bot.models import Position
from trading_bot.utils.indicators import apply_standard_indicators
from trading_bot.config import logger


class EmaDowVolumeStrategy(BaseStrategy):
    """
    EMA Dow Volume Strategy combines EMA crossovers with Dow Theory principles and volume analysis.
    
    Key components:
    - EMA crossovers to identify primary trends
    - Higher timeframe trend alignment (Dow Theory)
    - Price structure analysis (higher highs/higher lows or lower highs/lower lows)
    - Volume gaps and surges for entry confirmation
    - Volume divergence for exit signals
    """
    
    def __init__(self, timeframe: str = "1h", use_trailing_profit: bool = True):
        """Initialize the strategy with parameters."""
        super().__init__(name="ema_dow_volume_strategy", timeframe=timeframe)
        
        # Whether to use trailing profit or fixed take profit
        self.use_trailing_profit = use_trailing_profit
        
        # Set parameters based on timeframe
        self._set_parameters_for_timeframe()
        
        logger.info(f"Initialized EMA Dow Volume Strategy with parameters: "
                   f"Fast EMA={self.fast_ema}, Slow EMA={self.slow_ema}, "
                   f"Trend EMA={self.trend_ema}, Volume MA={self.volume_ma_period}, "
                   f"Swing Period={self.swing_period}, "
                   f"Trailing Profit: {'Enabled' if use_trailing_profit else 'Disabled'}")
    
    def _set_parameters_for_timeframe(self):
        """Set strategy parameters based on the timeframe."""
        minutes = self.timeframe_minutes
        
        # Base parameters - these will be adjusted based on timeframe
        if minutes <= 15:  # For short timeframes (1m-15m)
            # Faster settings for intraday trading
            self.fast_ema = 9
            self.slow_ema = 21
            self.trend_ema = 50
            self.volume_ma_period = 10
            self.swing_period = 5
            self.volume_threshold = 1.5
            self.atr_period = 10
            self.position_max_candles = 15
            self.profit_target_pct = 0.015  # 1.5%
            self.stop_loss_pct = 0.01  # 1%
        elif minutes <= 60:  # For medium timeframes (30m-1h)
            # Balanced parameters for swing trading
            self.fast_ema = 12
            self.slow_ema = 26
            self.trend_ema = 100
            self.volume_ma_period = 20
            self.swing_period = 10
            self.volume_threshold = 1.7
            self.atr_period = 14
            self.position_max_candles = 12
            self.profit_target_pct = 0.025  # 2.5%
            self.stop_loss_pct = 0.015  # 1.5%
        else:  # For higher timeframes (4h+)
            # Slower settings for position trading
            self.fast_ema = 21
            self.slow_ema = 50
            self.trend_ema = 200
            self.volume_ma_period = 30
            self.swing_period = 15
            self.volume_threshold = 2.0
            self.atr_period = 14
            self.position_max_candles = 10
            self.profit_target_pct = 0.035  # 3.5%
            self.stop_loss_pct = 0.02  # 2%
    
    def update_timeframe(self, timeframe: str):
        """Update the strategy timeframe and adjust parameters."""
        super().update_timeframe(timeframe)
        self._set_parameters_for_timeframe()
        logger.info(f"Updated EMA Dow Volume Strategy parameters for {timeframe} timeframe")
    
    async def calculate_indicators(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Calculate indicators for the strategy."""
        try:
            # Apply standard indicators (including ATR)
            df = apply_standard_indicators(
                df,
                atr_period=self.atr_period,
                ema_short=self.fast_ema,
                ema_long=self.slow_ema
            )
            
            # Add trend EMA
            df['trend_ema'] = df['close'].ewm(span=self.trend_ema, adjust=False).mean()
            
            # Volume indicators
            df['volume_ma'] = df['volume'].rolling(window=self.volume_ma_period).mean()
            df['volume_ratio'] = df['volume'] / df['volume_ma']
            
            # Identify volume surges
            df['volume_surge'] = (df['volume_ratio'] > self.volume_threshold).astype(int)
            
            # Calculate price swings for Dow Theory
            self._calculate_swings(df)
            
            # Detect volume gaps
            df['volume_gap'] = 0
            for i in range(1, len(df)):
                if df.iloc[i]['volume'] > df.iloc[i-1]['volume'] * self.volume_threshold:
                    df.iloc[i, df.columns.get_loc('volume_gap')] = 1
                elif df.iloc[i]['volume'] < df.iloc[i-1]['volume'] / self.volume_threshold:
                    df.iloc[i, df.columns.get_loc('volume_gap')] = -1
            
            # Add Dow Theory trend identification
            self._apply_dow_trend(df)
            
            return df
        except Exception as e:
            logger.error(f"Error calculating indicators for EMA Dow Volume Strategy: {str(e)}")
            return None
    
    def _calculate_swings(self, df: pd.DataFrame) -> None:
        """Calculate price swings for Dow Theory analysis."""
        # Initialize swing high/low columns
        df['swing_high'] = 0
        df['swing_low'] = 0
        df['higher_high'] = False
        df['higher_low'] = False
        df['lower_high'] = False
        df['lower_low'] = False
        
        # Identify swing highs and lows
        for i in range(self.swing_period, len(df) - self.swing_period):
            # Check if current point is a swing high
            if all(df.iloc[i]['high'] > df.iloc[i-j]['high'] for j in range(1, self.swing_period+1)) and \
               all(df.iloc[i]['high'] > df.iloc[i+j]['high'] for j in range(1, self.swing_period+1)):
                df.iloc[i, df.columns.get_loc('swing_high')] = df.iloc[i]['high']
            
            # Check if current point is a swing low
            if all(df.iloc[i]['low'] < df.iloc[i-j]['low'] for j in range(1, self.swing_period+1)) and \
               all(df.iloc[i]['low'] < df.iloc[i+j]['low'] for j in range(1, self.swing_period+1)):
                df.iloc[i, df.columns.get_loc('swing_low')] = df.iloc[i]['low']
        
        # Find the progression of swing highs and lows (Dow Theory)
        prev_swing_high = None
        prev_swing_low = None
        
        for i in range(self.swing_period, len(df)):
            if df.iloc[i]['swing_high'] > 0:
                if prev_swing_high is not None:
                    if df.iloc[i]['swing_high'] > prev_swing_high:
                        df.iloc[i, df.columns.get_loc('higher_high')] = True
                    else:
                        df.iloc[i, df.columns.get_loc('lower_high')] = True
                prev_swing_high = df.iloc[i]['swing_high']
            
            if df.iloc[i]['swing_low'] > 0:
                if prev_swing_low is not None:
                    if df.iloc[i]['swing_low'] > prev_swing_low:
                        df.iloc[i, df.columns.get_loc('higher_low')] = True
                    else:
                        df.iloc[i, df.columns.get_loc('lower_low')] = True
                prev_swing_low = df.iloc[i]['swing_low']
    
    def _apply_dow_trend(self, df: pd.DataFrame) -> None:
        """Apply Dow Theory trend identification."""
        df['dow_trend'] = 0  # 0: neutral, 1: uptrend, -1: downtrend
        
        # Calculate rolling windows for trend analysis
        window_size = min(20, len(df) // 2)
        if window_size < 4:
            return  # Not enough data
        
        for i in range(window_size, len(df)):
            window = df.iloc[i-window_size:i]
            
            # Count higher highs/lows and lower highs/lows in the window
            higher_highs = window['higher_high'].sum()
            higher_lows = window['higher_low'].sum()
            lower_highs = window['lower_high'].sum()
            lower_lows = window['lower_low'].sum()
            
            # Determine trend based on Dow Theory principles
            if higher_highs > 0 and higher_lows > 0 and higher_highs + higher_lows > lower_highs + lower_lows:
                df.iloc[i, df.columns.get_loc('dow_trend')] = 1  # Uptrend
            elif lower_highs > 0 and lower_lows > 0 and lower_highs + lower_lows > higher_highs + higher_lows:
                df.iloc[i, df.columns.get_loc('dow_trend')] = -1  # Downtrend
    
    async def check_signals(self, df: pd.DataFrame, position: Optional[Position] = None) -> Tuple[
        bool, bool, bool, List[str], List[str], List[str], List[str]
    ]:
        """Check for trading signals."""
        if df is None or len(df) < max(self.trend_ema, self.swing_period * 2) + 2:
            logger.warning("Insufficient data for signal calculation")
            return False, False, False, [], [], [], ["Insufficient data"]
        
        last = df.iloc[-1]
        prev = df.iloc[-2]
        
        long_signals = []
        short_signals = []
        close_signals = []
        fail_reasons = []
        
        # EMA signals
        ema_crossover_bull = prev['ema_short'] <= prev['ema_long'] and last['ema_short'] > last['ema_long']
        ema_crossover_bear = prev['ema_short'] >= prev['ema_long'] and last['ema_short'] < last['ema_long']
        
        trend_filter_bull = last['close'] > last['trend_ema']
        trend_filter_bear = last['close'] < last['trend_ema']
        
        # Volume signals
        volume_confirmation_bull = last['volume_ratio'] > self.volume_threshold or last['volume_gap'] > 0
        volume_confirmation_bear = last['volume_ratio'] > self.volume_threshold or last['volume_gap'] < 0
        
        # Dow Theory signals
        dow_uptrend = last['dow_trend'] > 0
        dow_downtrend = last['dow_trend'] < 0
        
        # Combined signals for entry
        # Long: EMA crossover + above trend EMA + Dow uptrend + volume confirmation
        long_condition = (ema_crossover_bull or (last['ema_short'] > last['ema_long'] and trend_filter_bull)) and \
                         dow_uptrend and volume_confirmation_bull
        
        # Short: EMA crossover + below trend EMA + Dow downtrend + volume confirmation
        short_condition = (ema_crossover_bear or (last['ema_short'] < last['ema_long'] and trend_filter_bear)) and \
                          dow_downtrend and volume_confirmation_bear
        
        if long_condition:
            long_signals.append("EMA crossover with Dow uptrend and volume confirmation")
            long_signals.append(f"Fast EMA ({self.fast_ema}): {last['ema_short']:.2f} > Slow EMA ({self.slow_ema}): {last['ema_long']:.2f}")
            long_signals.append(f"Price {last['close']:.2f} > Trend EMA ({self.trend_ema}): {last['trend_ema']:.2f}")
            
            if dow_uptrend:
                long_signals.append("Dow Theory uptrend confirmed (higher highs & higher lows)")
            
            if last['volume_ratio'] > self.volume_threshold:
                long_signals.append(f"Volume surge: {last['volume_ratio']:.2f}x average volume")
            if last['volume_gap'] > 0:
                long_signals.append("Bullish volume gap detected")
        else:
            if not (ema_crossover_bull or (last['ema_short'] > last['ema_long'] and trend_filter_bull)):
                fail_reasons.append("No bullish EMA signal")
            if not dow_uptrend:
                fail_reasons.append("No Dow Theory uptrend confirmation")
            if not volume_confirmation_bull:
                fail_reasons.append("Insufficient volume confirmation")
        
        if short_condition:
            short_signals.append("EMA crossover with Dow downtrend and volume confirmation")
            short_signals.append(f"Fast EMA ({self.fast_ema}): {last['ema_short']:.2f} < Slow EMA ({self.slow_ema}): {last['ema_long']:.2f}")
            short_signals.append(f"Price {last['close']:.2f} < Trend EMA ({self.trend_ema}): {last['trend_ema']:.2f}")
            
            if dow_downtrend:
                short_signals.append("Dow Theory downtrend confirmed (lower highs & lower lows)")
            
            if last['volume_ratio'] > self.volume_threshold:
                short_signals.append(f"Volume surge: {last['volume_ratio']:.2f}x average volume")
            if last['volume_gap'] < 0:
                short_signals.append("Bearish volume gap detected")
        
        # Exit signals
        close_condition = False
        if position:
            # Exit long position
            if position.side == 'long':
                # 1. EMA crossover bearish
                if ema_crossover_bear:
                    close_signals.append("Bearish EMA crossover")
                    close_condition = True
                # 2. Price below trend EMA
                elif last['close'] < last['trend_ema'] and prev['close'] >= prev['trend_ema']:
                    close_signals.append(f"Price broke below Trend EMA ({self.trend_ema})")
                    close_condition = True
                # 3. Dow Theory trend reversal
                elif last['dow_trend'] < 0 and prev['dow_trend'] >= 0:
                    close_signals.append("Dow Theory trend reversal to downtrend")
                    close_condition = True
                # 4. Lower high formed with high volume
                elif last['lower_high'] and last['volume_ratio'] > self.volume_threshold:
                    close_signals.append("Lower high formed with high volume (trend weakening)")
                    close_condition = True
            
            # Exit short position
            elif position.side == 'short':
                # 1. EMA crossover bullish
                if ema_crossover_bull:
                    close_signals.append("Bullish EMA crossover")
                    close_condition = True
                # 2. Price above trend EMA
                elif last['close'] > last['trend_ema'] and prev['close'] <= prev['trend_ema']:
                    close_signals.append(f"Price broke above Trend EMA ({self.trend_ema})")
                    close_condition = True
                # 3. Dow Theory trend reversal
                elif last['dow_trend'] > 0 and prev['dow_trend'] <= 0:
                    close_signals.append("Dow Theory trend reversal to uptrend")
                    close_condition = True
                # 4. Higher low formed with high volume
                elif last['higher_low'] and last['volume_ratio'] > self.volume_threshold:
                    close_signals.append("Higher low formed with high volume (trend weakening)")
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
                position.volume_at_entry = last['volume_ratio']
        
        # Increment candle counter
        if not hasattr(position, 'open_candles'):
            position.open_candles = 0
        position.open_candles += 1
        
        # Move to breakeven after 3 candles or when profit exceeds 1%
        if position.open_candles >= 3:
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
                
            # Advanced trailing profit logic based on price and volume
            if not hasattr(position, 'trailing_profit_activated'):
                position.trailing_profit_activated = False
                
            # Activate trailing profit when profit reaches target
            if current_profit_pct > self.profit_target_pct * 100:
                position.trailing_profit_activated = True
                
                # Adjust trailing factor based on:
                # 1. Dow trend strength
                # 2. Volume expansion/contraction
                # 3. Profit magnitude
                
                # Base trailing factor
                trailing_factor = 0.4  # Default 40% retracement allowed
                
                # Adjust for trend strength
                if position.side == 'long' and last['dow_trend'] > 0:
                    trailing_factor -= 0.1  # Tighter trailing in strong uptrend (30%)
                elif position.side == 'short' and last['dow_trend'] < 0:
                    trailing_factor -= 0.1  # Tighter trailing in strong downtrend (30%)
                    
                # Adjust for volume expansion
                if last['volume_ratio'] > self.volume_threshold:
                    trailing_factor -= 0.1  # Even tighter with high volume (20%)
                
                # Adjust for profit magnitude - tighter trailing as profit grows
                if current_profit_pct > self.profit_target_pct * 200:
                    trailing_factor -= 0.05  # Very tight trailing with big profits (15%)
                
                # Ensure trailing factor is in reasonable range
                trailing_factor = max(0.15, min(0.5, trailing_factor))
                
                max_pullback = position.highest_profit_pct * trailing_factor
                
                # Close position if price pulls back too much from the peak
                if position.trailing_profit_activated and current_profit_pct < (position.highest_profit_pct - max_pullback):
                    close_signals.append(f"Trailing profit: Locked in {current_profit_pct:.2f}% (max: {position.highest_profit_pct:.2f}%)")
                    close_condition = True
        
        # Update trailing stop based on Dow trend structure
        if position.side == 'long':
            # In uptrend, use recent swing lows for trailing stop
            recent_swing_lows = [row['swing_low'] for _, row in df.tail(10).iterrows() if row['swing_low'] > 0]
            if recent_swing_lows and recent_swing_lows[-1] > position.trailing_stop:
                position.trailing_stop = recent_swing_lows[-1]
                close_signals.append(f"Raised stop to swing low: {position.trailing_stop:.2f}")
        else:  # short
            # In downtrend, use recent swing highs for trailing stop
            recent_swing_highs = [row['swing_high'] for _, row in df.tail(10).iterrows() if row['swing_high'] > 0]
            if recent_swing_highs and recent_swing_highs[-1] < position.trailing_stop:
                position.trailing_stop = recent_swing_highs[-1]
                close_signals.append(f"Lowered stop to swing high: {position.trailing_stop:.2f}")
        
        # Check for significant volume expansion against position
        if (position.side == 'long' and 
            last['volume_ratio'] > self.volume_threshold * 1.5 and 
            last['close'] < last['open']):
            # Heavy selling volume
            close_signals.append(f"Heavy selling volume detected: {last['volume_ratio']:.2f}x average")
            close_condition = True
        elif (position.side == 'short' and 
              last['volume_ratio'] > self.volume_threshold * 1.5 and 
              last['close'] > last['open']):
            # Heavy buying volume
            close_signals.append(f"Heavy buying volume detected: {last['volume_ratio']:.2f}x average")
            close_condition = True
        
        # Check for trend reversal using EMAs and Dow trend
        if position.side == 'long':
            if last['ema_short'] < last['ema_long'] and last['dow_trend'] < 0:
                close_signals.append("EMA crossover bearish with Dow downtrend")
                close_condition = True
        else:  # short
            if last['ema_short'] > last['ema_long'] and last['dow_trend'] > 0:
                close_signals.append("EMA crossover bullish with Dow uptrend")
                close_condition = True
        
        # Check for time-based exit
        time_exit, time_signals = await self.check_max_holding_time(position, self.position_max_candles)
        close_signals.extend(time_signals)
        if time_exit:
            close_condition = True
        
        return position, close_condition, close_signals 