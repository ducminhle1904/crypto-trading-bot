"""
Volume Profile combined with Bollinger Bands and RSI Strategy.
"""
import pandas as pd
import numpy as np
from typing import Tuple, List, Optional, Dict, Any

from trading_bot.strategies.base_strategy import BaseStrategy
from trading_bot.models import Position
from trading_bot.utils.indicators import (
    calculate_rsi, calculate_bollinger_bands, calculate_atr,
    apply_volume_profile_indicators, calculate_ema
)
from trading_bot.config import logger


class VolumeProfileBollingerRsiStrategy(BaseStrategy):
    """
    Volume Profile + Bollinger Bands + RSI Strategy combines volume distribution analysis 
    with volatility and momentum indicators. 
    
    This strategy identifies high-probability trades where:
    1. Volume profile shows accumulation/distribution near key levels
    2. Bollinger Bands indicate volatility expansion or contraction
    3. RSI confirms momentum and potential reversals
    """
    
    def __init__(self, timeframe: str = "30m", use_trailing_profit: bool = True):
        """Initialize the strategy with parameters."""
        super().__init__(name="volume_profile_bollinger_rsi_strategy", timeframe=timeframe)
        
        # Whether to use trailing profit or fixed take profit
        self.use_trailing_profit = use_trailing_profit
        
        # Set parameters based on timeframe
        self._set_parameters_for_timeframe()
        
        logger.info(f"Initialized Volume Profile + Bollinger + RSI Strategy for {timeframe}, "
                   f"RSI Period={self.rsi_period}, BB Period={self.bb_period}, "
                   f"VP Lookback={self.vp_lookback_periods}, "
                   f"Trailing Profit: {'Enabled' if use_trailing_profit else 'Disabled'}")
    
    def _set_parameters_for_timeframe(self):
        """Set strategy parameters based on the timeframe."""
        minutes = self.timeframe_minutes
        
        # Base parameters for all timeframes
        self.rsi_period = 14
        self.rsi_overbought = 70
        self.rsi_oversold = 30
        self.bb_period = 20
        self.bb_std = 2.0
        self.ema_period = 34  # Mid-term trend filter
        self.atr_period = 14
        self.position_max_candles = 12
        
        # Volume Profile parameters
        self.vp_bins = 20  # Number of price bins
        self.vp_lookback_periods = 100  # How far back to look for volume profile
        
        # Adjust parameters based on timeframe
        if minutes <= 15:  # 1m-15m (more aggressive)
            self.atr_multiple = 1.0  # Tighter stops
            self.profit_target_pct = 0.007  # 0.7% target
            self.position_max_candles = 16  # Allow slightly more time for setup to develop
            self.rsi_period = 10  # More responsive RSI
        elif minutes <= 60:  # 30m-1h (balanced)
            self.atr_multiple = 1.2
            self.profit_target_pct = 0.01  # 1% target
            self.position_max_candles = 12
        else:  # 4h+ (conservative)
            self.atr_multiple = 1.5
            self.profit_target_pct = 0.02  # 2% target
            self.position_max_candles = 8
            self.rsi_period = 16  # Smoother RSI for higher timeframes
    
    def update_timeframe(self, timeframe: str):
        """Update the strategy timeframe and adjust parameters."""
        super().update_timeframe(timeframe)
        self._set_parameters_for_timeframe()
        logger.info(f"Updated Volume Profile + Bollinger + RSI Strategy parameters for {timeframe} timeframe")
    
    async def calculate_indicators(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Calculate indicators for the strategy."""
        try:
            if df is None or len(df) < max(self.rsi_period, self.bb_period) + 5:
                return None
                
            df = df.copy()
            
            # Calculate RSI
            df['rsi'] = calculate_rsi(df['close'], self.rsi_period)
            df['prev_rsi'] = df['rsi'].shift(1)
            
            # Calculate Bollinger Bands
            bb = calculate_bollinger_bands(df['close'], self.bb_period, self.bb_std)
            df['bb_upper'] = bb['upper']
            df['bb_middle'] = bb['middle']
            df['bb_lower'] = bb['lower']
            df['bb_width'] = bb['bandwidth']
            
            # Calculate EMA for trend filter
            df['ema'] = calculate_ema(df['close'], self.ema_period)
            
            # Calculate ATR for volatility-based stops
            df['atr'] = calculate_atr(df, self.atr_period)
            
            # Calculate Bollinger Band conditions
            df['bb_squeeze'] = df['bb_width'] < df['bb_width'].rolling(window=20).mean() * 0.8
            df['bb_expansion'] = df['bb_width'] > df['bb_width'].shift(1) * 1.05
            
            # Calculate RSI conditions
            df['rsi_overbought'] = df['rsi'] > self.rsi_overbought
            df['rsi_oversold'] = df['rsi'] < self.rsi_oversold
            df['rsi_rising'] = df['rsi'] > df['prev_rsi']
            df['rsi_falling'] = df['rsi'] < df['prev_rsi']
            
            # Apply Volume Profile indicators
            df = apply_volume_profile_indicators(
                df,
                num_bins=self.vp_bins,
                lookback_periods=self.vp_lookback_periods
            )
            
            return df
            
        except Exception as e:
            logger.error(f"Error calculating Volume Profile + Bollinger + RSI indicators: {e}")
            return None
    
    async def check_signals(self, df: pd.DataFrame, position: Optional[Position] = None) -> Tuple[
        bool, bool, bool, List[str], List[str], List[str], List[str]
    ]:
        """Check for trading signals."""
        if df is None or len(df) < max(self.rsi_period, self.bb_period) + 5:
            logger.warning("Insufficient data for Volume Profile + Bollinger + RSI signal calculation")
            return False, False, False, [], [], [], ["Insufficient data"]
        
        last = df.iloc[-1]
        prev = df.iloc[-2] if len(df) > 2 else last
        
        long_signals = []
        short_signals = []
        close_signals = []
        fail_reasons = []
        
        # Extract key indicators
        price = last['close']
        bb_upper = last['bb_upper']
        bb_middle = last['bb_middle']
        bb_lower = last['bb_lower']
        bb_squeeze = last['bb_squeeze']
        bb_expansion = last['bb_expansion']
        rsi = last['rsi']
        ema = last['ema']
        
        # Volume Profile conditions
        price_near_poc = 'vp_poc' in last and last['vp_poc'] and abs(price - last['vp_poc']) / last['vp_poc'] < 0.005
        price_above_poc = 'vp_poc' in last and last['vp_poc'] and price > last['vp_poc']
        price_below_poc = 'vp_poc' in last and last['vp_poc'] and price < last['vp_poc']
        price_in_value_area = last.get('vp_in_value_area', True)
        vp_potential_long = last.get('vp_potential_long', False)
        vp_potential_short = last.get('vp_potential_short', False)
        
        # Trend conditions
        above_ema = price > ema
        below_ema = price < ema
        
        # RSI conditions
        rsi_oversold = last['rsi_oversold']
        rsi_overbought = last['rsi_overbought']
        rsi_rising = last['rsi_rising']
        rsi_falling = last['rsi_falling']
        
        # Bollinger Band conditions
        price_near_upper = abs(price - bb_upper) / bb_upper < 0.01
        price_near_lower = abs(price - bb_lower) / bb_lower < 0.01
        
        # Long signal conditions:
        # 1. Volume Profile shows accumulation (potential_long) OR price is near POC
        # 2. Price is near lower Bollinger Band OR squeeze is forming
        # 3. RSI is oversold OR rising from low levels
        # 4. (Optional) Price is above EMA for trend confirmation in higher timeframes
        long_condition = (
            (vp_potential_long or (price_near_poc and price_above_poc)) and
            (price_near_lower or (price < bb_middle and bb_squeeze)) and
            (rsi_oversold or (rsi_rising and rsi < 45)) and
            (above_ema or self.timeframe_minutes <= 30)  # Less strict trend filter on lower timeframes
        )
        
        # Short signal conditions:
        # 1. Volume Profile shows distribution (potential_short) OR price is near POC
        # 2. Price is near upper Bollinger Band OR squeeze is forming
        # 3. RSI is overbought OR falling from high levels
        # 4. (Optional) Price is below EMA for trend confirmation in higher timeframes
        short_condition = (
            (vp_potential_short or (price_near_poc and price_below_poc)) and
            (price_near_upper or (price > bb_middle and bb_squeeze)) and
            (rsi_overbought or (rsi_falling and rsi > 55)) and
            (below_ema or self.timeframe_minutes <= 30)  # Less strict trend filter on lower timeframes
        )
        
        # Generate long signal reasons
        if long_condition:
            # Volume Profile reasons
            if vp_potential_long:
                long_signals.append(f"Volume Profile showing accumulation")
            if price_near_poc:
                long_signals.append(f"Price near Point of Control: {last['vp_poc']:.2f}")
            if 'vp_val' in last and last['vp_val'] and abs(price - last['vp_val']) / last['vp_val'] < 0.01:
                long_signals.append(f"Price near Value Area Low: {last['vp_val']:.2f}")
                
            # Bollinger Band reasons
            if price_near_lower:
                long_signals.append(f"Price testing lower Bollinger Band: {bb_lower:.2f}")
            if bb_squeeze:
                long_signals.append(f"Bollinger Band squeeze forming (volatility contraction)")
            if bb_expansion:
                long_signals.append(f"Bollinger Bands expanding (potential breakout)")
                
            # RSI reasons
            if rsi_oversold:
                long_signals.append(f"RSI oversold: {rsi:.2f}")
            if rsi_rising:
                long_signals.append(f"RSI rising: {rsi:.2f}")
                
            # Trend reasons
            if above_ema:
                long_signals.append(f"Price above EMA({self.ema_period}): {price:.2f} > {ema:.2f}")
        else:
            # Add fail reasons for debugging
            if not (vp_potential_long or (price_near_poc and price_above_poc)):
                fail_reasons.append("Volume Profile not showing accumulation")
            if not (price_near_lower or (price < bb_middle and bb_squeeze)):
                fail_reasons.append(f"Price not near lower BB or below middle during squeeze")
            if not (rsi_oversold or (rsi_rising and rsi < 45)):
                fail_reasons.append(f"RSI not signaling buy: {rsi:.2f}")
            if not (above_ema or self.timeframe_minutes <= 30):
                fail_reasons.append(f"Price below EMA in higher timeframe: {price:.2f} < {ema:.2f}")
        
        # Generate short signal reasons
        if short_condition:
            # Volume Profile reasons
            if vp_potential_short:
                short_signals.append(f"Volume Profile showing distribution")
            if price_near_poc:
                short_signals.append(f"Price near Point of Control: {last['vp_poc']:.2f}")
            if 'vp_vah' in last and last['vp_vah'] and abs(price - last['vp_vah']) / last['vp_vah'] < 0.01:
                short_signals.append(f"Price near Value Area High: {last['vp_vah']:.2f}")
                
            # Bollinger Band reasons
            if price_near_upper:
                short_signals.append(f"Price testing upper Bollinger Band: {bb_upper:.2f}")
            if bb_squeeze:
                short_signals.append(f"Bollinger Band squeeze forming (volatility contraction)")
            if bb_expansion:
                short_signals.append(f"Bollinger Bands expanding (potential breakdown)")
                
            # RSI reasons
            if rsi_overbought:
                short_signals.append(f"RSI overbought: {rsi:.2f}")
            if rsi_falling:
                short_signals.append(f"RSI falling: {rsi:.2f}")
                
            # Trend reasons
            if below_ema:
                short_signals.append(f"Price below EMA({self.ema_period}): {price:.2f} < {ema:.2f}")
        else:
            # Add fail reasons for debugging
            if not (vp_potential_short or (price_near_poc and price_below_poc)):
                fail_reasons.append("Volume Profile not showing distribution")
            if not (price_near_upper or (price > bb_middle and bb_squeeze)):
                fail_reasons.append(f"Price not near upper BB or above middle during squeeze")
            if not (rsi_overbought or (rsi_falling and rsi > 55)):
                fail_reasons.append(f"RSI not signaling sell: {rsi:.2f}")
            if not (below_ema or self.timeframe_minutes <= 30):
                fail_reasons.append(f"Price above EMA in higher timeframe: {price:.2f} > {ema:.2f}")
        
        # Exit conditions
        close_condition = False
        if position:
            take_profit_pct = self.profit_target_pct
            
            if position.side == 'long':
                take_profit = position.entry * (1 + take_profit_pct)
                
                # Exit long position conditions
                if position.trailing_stop and price <= position.trailing_stop:
                    close_signals.append(f"Stop loss hit at {position.trailing_stop:.2f}")
                    close_condition = True
                elif price >= take_profit:
                    close_signals.append(f"Take profit hit: {price:.2f} >= {take_profit:.2f}")
                    close_condition = True
                elif rsi_overbought:
                    close_signals.append(f"RSI overbought: {rsi:.2f}")
                    close_condition = True
                elif price_near_upper and rsi_falling:
                    close_signals.append(f"Price at upper Bollinger Band ({bb_upper:.2f}) with falling RSI")
                    close_condition = True
                elif price > bb_upper and vp_potential_short:
                    close_signals.append(f"Price above upper Bollinger Band with distribution pattern")
                    close_condition = True
                
            elif position.side == 'short':
                take_profit = position.entry * (1 - take_profit_pct)
                
                # Exit short position conditions
                if position.trailing_stop and price >= position.trailing_stop:
                    close_signals.append(f"Stop loss hit at {position.trailing_stop:.2f}")
                    close_condition = True
                elif price <= take_profit:
                    close_signals.append(f"Take profit hit: {price:.2f} <= {take_profit:.2f}")
                    close_condition = True
                elif rsi_oversold:
                    close_signals.append(f"RSI oversold: {rsi:.2f}")
                    close_condition = True
                elif price_near_lower and rsi_rising:
                    close_signals.append(f"Price at lower Bollinger Band ({bb_lower:.2f}) with rising RSI")
                    close_condition = True
                elif price < bb_lower and vp_potential_long:
                    close_signals.append(f"Price below lower Bollinger Band with accumulation pattern")
                    close_condition = True
        
        long_signal = long_condition and not position
        short_signal = short_condition and not position
        close_signal = close_condition and position
        
        return long_signal, short_signal, close_signal, long_signals, short_signals, close_signals, fail_reasons
    
    async def manage_position(self, df: pd.DataFrame, position: Position, balance: float) -> Tuple[
        Position, bool, List[str]
    ]:
        """Manages existing positions with dynamic stop loss and trailing profit."""
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
                position.profit_lockout_level = 0
            
        # Increment candle counter
        if not hasattr(position, 'open_candles'):
            position.open_candles = 0
        position.open_candles += 1
        
        # Move to breakeven after minimal profit
        min_profit_pct = 0.004  # 0.4% profit to move to breakeven
        position, breakeven_signals = await self.move_to_breakeven(position, last['close'], min_profit_pct)
        close_signals.extend(breakeven_signals)
        
        # Volume Profile aware trailing stop
        # Use tighter stops near important volume levels
        price_near_poc = 'vp_poc' in last and last['vp_poc'] and abs(last['close'] - last['vp_poc']) / last['vp_poc'] < 0.005
        
        if price_near_poc:
            # Use tighter trailing stops near Point of Control (high volume area)
            trailing_atr_mult = 0.5  # Tighter at POC
            position, trailing_signals = await self.update_trailing_stop(
                position, last['close'], last['atr'], trailing_atr_mult
            )
            if trailing_signals:
                trailing_signals = [s + " (near POC - tighter stop)" for s in trailing_signals]
                close_signals.extend(trailing_signals)
        else:
            # Standard trailing stop updates
            trailing_atr_mult = 0.7
            position, trailing_signals = await self.update_trailing_stop(
                position, last['close'], last['atr'], trailing_atr_mult
            )
            close_signals.extend(trailing_signals)
        
        # Enhanced trailing profit with multiple lockout levels
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
                
            # Multi-level profit locking
            profit_target_pct = self.profit_target_pct * 100
            
            # Define profit lockout levels (30%, 50%, 70%, 90% of target)
            lockout_levels = [
                profit_target_pct * 0.3,  # First level - 30% of target - allow 50% pullback
                profit_target_pct * 0.5,  # Second level - 50% of target - allow 40% pullback
                profit_target_pct * 0.7,  # Third level - 70% of target - allow 30% pullback
                profit_target_pct * 0.9   # Fourth level - 90% of target - allow 20% pullback
            ]
            
            lockout_pullbacks = [0.5, 0.4, 0.3, 0.2]  # Allowed pullback at each level
            
            # Initialize profit lockout level if not present
            if not hasattr(position, 'profit_lockout_level'):
                position.profit_lockout_level = 0
                
            # Check if we've reached a new lockout level
            current_level = position.profit_lockout_level
            for level in range(current_level, len(lockout_levels)):
                if current_profit_pct >= lockout_levels[level]:
                    position.profit_lockout_level = level + 1  # Set to next level (1-indexed)
                    close_signals.append(f"Reached profit lockout level {position.profit_lockout_level}: {current_profit_pct:.2f}%")
                    
            # Check for pullback beyond allowed level
            if position.profit_lockout_level > 0:
                level_idx = position.profit_lockout_level - 1  # Convert to 0-indexed
                max_pullback = position.highest_profit_pct * lockout_pullbacks[min(level_idx, len(lockout_pullbacks)-1)]
                
                if current_profit_pct < (position.highest_profit_pct - max_pullback):
                    close_signals.append(
                        f"Trailing profit: Locked in {current_profit_pct:.2f}% at level {position.profit_lockout_level} "
                        f"(max: {position.highest_profit_pct:.2f}%, allowed pullback: {max_pullback:.2f}%)"
                    )
                    close_condition = True
        
        # Check for time-based exit
        time_exit, time_signals = await self.check_max_holding_time(position, self.position_max_candles)
        close_signals.extend(time_signals)
        if time_exit:
            close_condition = True
        
        return position, close_condition, close_signals 