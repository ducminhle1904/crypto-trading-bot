"""
Bollinger Band Squeeze Strategy for scalping.
"""
import pandas as pd
from typing import Tuple, List, Optional, Dict, Any

from trading_bot.strategies.base_strategy import BaseStrategy
from trading_bot.models import Position
from trading_bot.utils.indicators import apply_bollinger_squeeze_indicators
from trading_bot.utils.position_manager import TrailingProfitManager
from trading_bot.config import logger


class BollingerSqueezeStrategy(BaseStrategy):
    """
    Bollinger Squeeze Strategy identifies consolidation patterns followed by volatility expansions.
    Perfect for scalping as it catches explosive moves after periods of low volatility.
    Combined with RSI Divergence for confirmation and improved entry timing.
    Now with trailing profit to let winning trades run longer.
    """
    
    def __init__(self, timeframe: str = "3m", use_trailing_profit: bool = True):
        """Initialize the strategy with parameters."""
        super().__init__(name="bollinger_squeeze_strategy", timeframe=timeframe)
        
        # Set parameters based on timeframe
        self._set_parameters_for_timeframe()
        
        # Initialize trailing profit manager
        self.trailing_profit_manager = TrailingProfitManager(enable_trailing_profit=use_trailing_profit)
        self.use_trailing_profit = use_trailing_profit
        
        logger.info(f"Initialized Bollinger Squeeze Strategy with parameters: BB Length={self.bb_length}, "
                   f"KC Length={self.keltner_length}, MACD Fast={self.macd_fast}, MACD Slow={self.macd_slow}, "
                   f"RSI Period={self.rsi_period}, Trailing Profit={use_trailing_profit}")
    
    def _set_parameters_for_timeframe(self):
        """Set strategy parameters based on the timeframe."""
        minutes = self.timeframe_minutes
        
        # Scale parameters based on timeframe
        if minutes <= 5:  # 1m to 5m - optimal for scalping
            self.bb_length = 20
            self.bb_std = 2.0
            self.keltner_length = 20
            self.keltner_atr_mult = 1.5
            self.macd_fast = 12
            self.macd_slow = 26
            self.macd_signal = 9
            self.rsi_period = 14
            self.rsi_divergence_lookback = 5
            self.position_max_candles = 10  # Quick exits for scalping
            self.atr_multiple = 1.0  # Tight stops for scalping
            self.profit_target_pct = 0.005  # 0.5% target (small but frequent wins)
            
        elif minutes <= 60:  # 15m to 1h - medium settings
            self.bb_length = 20
            self.bb_std = 2.0
            self.keltner_length = 20
            self.keltner_atr_mult = 1.5
            self.macd_fast = 12
            self.macd_slow = 26
            self.macd_signal = 9
            self.rsi_period = 14
            self.rsi_divergence_lookback = 8
            self.position_max_candles = 8
            self.atr_multiple = 1.5
            self.profit_target_pct = 0.01  # 1% target
            
        else:  # 4h, daily - slower settings
            self.bb_length = 20
            self.bb_std = 2.0
            self.keltner_length = 20
            self.keltner_atr_mult = 1.5
            self.macd_fast = 12
            self.macd_slow = 26
            self.macd_signal = 9
            self.rsi_period = 14
            self.rsi_divergence_lookback = 10
            self.position_max_candles = 5
            self.atr_multiple = 2.0
            self.profit_target_pct = 0.02  # 2% target
    
    def update_timeframe(self, timeframe: str):
        """Update the strategy timeframe and adjust parameters."""
        super().update_timeframe(timeframe)
        self._set_parameters_for_timeframe()
        logger.info(f"Updated Bollinger Squeeze Strategy parameters for {timeframe} timeframe")
    
    async def calculate_indicators(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Calculate indicators for the strategy."""
        return apply_bollinger_squeeze_indicators(
            df,
            bb_length=self.bb_length,
            bb_std=self.bb_std,
            kc_length=self.keltner_length,
            kc_atr_mult=self.keltner_atr_mult,
            macd_fast=self.macd_fast,
            macd_slow=self.macd_slow,
            macd_signal=self.macd_signal,
            rsi_period=self.rsi_period,
            rsi_divergence_lookback=self.rsi_divergence_lookback
        )
    
    async def check_signals(self, df: pd.DataFrame, position: Optional[Position] = None) -> Tuple[
        bool, bool, bool, List[str], List[str], List[str], List[str]
    ]:
        """Check for trading signals."""
        if df is None or len(df) < self.macd_slow + 5:
            logger.warning("Insufficient data for Bollinger Squeeze signal calculation")
            return False, False, False, [], [], [], ["Insufficient data"]
        
        last = df.iloc[-1]
        prev = df.iloc[-2] if len(df) > 2 else last
        
        long_signals = []
        short_signals = []
        close_signals = []
        fail_reasons = []
        
        # Check for squeeze conditions
        is_squeeze = last['squeeze']
        was_in_squeeze = prev['squeeze']
        squeeze_release = last['squeeze_release']
        
        # MACD conditions for direction
        macd_above_signal = last['macd'] > last['macd_signal']
        macd_below_signal = last['macd'] < last['macd_signal']
        macd_cross_up = last['macd'] > last['macd_signal'] and prev['macd'] <= prev['macd_signal']
        macd_cross_down = last['macd'] < last['macd_signal'] and prev['macd'] >= prev['macd_signal']
        
        # Price breakout conditions
        breaking_bb_upper = last['close'] > last['bb_upper'] and prev['close'] <= prev['bb_upper']
        breaking_bb_lower = last['close'] < last['bb_lower'] and prev['close'] >= prev['bb_lower']
        
        # Volatility expansion conditions - useful for determining when a squeeze is ending
        expanding_bands = last['bb_bandwidth'] > prev['bb_bandwidth']
        
        # RSI Divergence conditions
        bullish_divergence = last['bullish_divergence']
        bearish_divergence = last['bearish_divergence']
        
        # RSI levels
        rsi_value = last['rsi']
        rsi_oversold = rsi_value < 30
        rsi_overbought = rsi_value > 70
        
        # Long conditions - Squeeze has formed and now breaking out to upside
        # Add RSI divergence confirmation for higher quality signals
        long_condition = ((squeeze_release or was_in_squeeze) and 
                          (breaking_bb_upper or macd_cross_up) and 
                          macd_above_signal and
                          expanding_bands and
                          (bullish_divergence or rsi_oversold))  # Add divergence confirmation
        
        # Short conditions - Squeeze has formed and now breaking out to downside
        # Add RSI divergence confirmation for higher quality signals
        short_condition = ((squeeze_release or was_in_squeeze) and 
                           (breaking_bb_lower or macd_cross_down) and 
                           macd_below_signal and
                           expanding_bands and
                           (bearish_divergence or rsi_overbought))  # Add divergence confirmation
        
        if long_condition:
            if squeeze_release:
                long_signals.append("Bollinger Bands Squeeze just released")
            elif was_in_squeeze:
                long_signals.append("Market in squeeze state - low volatility")
                
            if breaking_bb_upper:
                long_signals.append(f"Price broke above upper Bollinger Band: {last['close']:.2f} > {last['bb_upper']:.2f}")
            if macd_cross_up:
                long_signals.append(f"MACD crossed above signal line: {last['macd']:.4f} > {last['macd_signal']:.4f}")
            else:
                long_signals.append(f"MACD above signal line: {last['macd']:.4f} > {last['macd_signal']:.4f}")
                
            long_signals.append(f"Volatility expansion: Bandwidth {last['bb_bandwidth']:.2f}% increasing")
            
            # Add RSI divergence signal
            if bullish_divergence:
                long_signals.append(f"Bullish RSI divergence detected: Price making lower lows, RSI making higher lows")
            if rsi_oversold:
                long_signals.append(f"RSI oversold: {rsi_value:.2f} < 30")
        else:
            if not (squeeze_release or was_in_squeeze):
                fail_reasons.append("No squeeze condition detected")
            if not (breaking_bb_upper or macd_cross_up):
                fail_reasons.append("No upside breakout signal")
            if not macd_above_signal:
                fail_reasons.append(f"MACD not above signal: {last['macd']:.4f} < {last['macd_signal']:.4f}")
            if not expanding_bands:
                fail_reasons.append("No volatility expansion")
            if not (bullish_divergence or rsi_oversold):
                fail_reasons.append(f"No RSI confirmation: No bullish divergence and RSI not oversold ({rsi_value:.2f})")
        
        if short_condition:
            if squeeze_release:
                short_signals.append("Bollinger Bands Squeeze just released")
            elif was_in_squeeze:
                short_signals.append("Market in squeeze state - low volatility")
                
            if breaking_bb_lower:
                short_signals.append(f"Price broke below lower Bollinger Band: {last['close']:.2f} < {last['bb_lower']:.2f}")
            if macd_cross_down:
                short_signals.append(f"MACD crossed below signal line: {last['macd']:.4f} < {last['macd_signal']:.4f}")
            else:
                short_signals.append(f"MACD below signal line: {last['macd']:.4f} < {last['macd_signal']:.4f}")
                
            short_signals.append(f"Volatility expansion: Bandwidth {last['bb_bandwidth']:.2f}% increasing")
            
            # Add RSI divergence signal
            if bearish_divergence:
                short_signals.append(f"Bearish RSI divergence detected: Price making higher highs, RSI making lower highs")
            if rsi_overbought:
                short_signals.append(f"RSI overbought: {rsi_value:.2f} > 70")
        else:
            if not (squeeze_release or was_in_squeeze):
                fail_reasons.append("No squeeze condition detected")
            if not (breaking_bb_lower or macd_cross_down):
                fail_reasons.append("No downside breakout signal")
            if not macd_below_signal:
                fail_reasons.append(f"MACD not below signal: {last['macd']:.4f} > {last['macd_signal']:.4f}")
            if not expanding_bands:
                fail_reasons.append("No volatility expansion")
            if not (bearish_divergence or rsi_overbought):
                fail_reasons.append(f"No RSI confirmation: No bearish divergence and RSI not overbought ({rsi_value:.2f})")
        
        # Exit conditions - now with trailing profit
        close_condition = False
        if position:
            # For standard strategy (fixed take profit)
            if not self.use_trailing_profit:
                take_profit_pct = self.profit_target_pct
                
                if position.side == 'long':
                    take_profit = position.entry * (1 + take_profit_pct)
                    
                    # Exit long if we hit stops or targets
                    if position.trailing_stop and last['close'] <= position.trailing_stop:
                        close_signals.append(f"Stop loss hit at {position.trailing_stop:.2f}")
                        close_condition = True
                    # Take profit for scalping
                    elif last['close'] >= take_profit:
                        close_signals.append(f"Take profit hit: {last['close']:.2f} >= {take_profit:.2f}")
                        close_condition = True
                    # Exit if we get MACD cross or close below middle band
                    elif macd_cross_down:
                        close_signals.append(f"MACD crossed below signal line: {last['macd']:.4f} < {last['macd_signal']:.4f}")
                        close_condition = True
                    elif last['close'] < last['bb_middle']:
                        close_signals.append(f"Price fell below BB middle band: {last['close']:.2f} < {last['bb_middle']:.2f}")
                        close_condition = True
                    # Add exit on bearish divergence
                    elif bearish_divergence:
                        close_signals.append(f"Bearish RSI divergence detected while in long position")
                        close_condition = True
                
                elif position.side == 'short':
                    take_profit = position.entry * (1 - take_profit_pct)
                    
                    # Exit short if we hit stops or targets
                    if position.trailing_stop and last['close'] >= position.trailing_stop:
                        close_signals.append(f"Stop loss hit at {position.trailing_stop:.2f}")
                        close_condition = True
                    # Take profit for scalping
                    elif last['close'] <= take_profit:
                        close_signals.append(f"Take profit hit: {last['close']:.2f} <= {take_profit:.2f}")
                        close_condition = True
                    # Exit if we get MACD cross or close above middle band
                    elif macd_cross_up:
                        close_signals.append(f"MACD crossed above signal line: {last['macd']:.4f} > {last['macd_signal']:.4f}")
                        close_condition = True
                    elif last['close'] > last['bb_middle']:
                        close_signals.append(f"Price rose above BB middle band: {last['close']:.2f} > {last['bb_middle']:.2f}")
                        close_condition = True
                    # Add exit on bullish divergence
                    elif bullish_divergence:
                        close_signals.append(f"Bullish RSI divergence detected while in short position")
                        close_condition = True
            
            # When using trailing profit manager, let trailing profit logic handle exits
            # Traditional exit criteria are not used, replaced by trend-based trailing
        
        long_signal = long_condition and not position
        short_signal = short_condition and not position
        close_signal = close_condition and position
        
        return long_signal, short_signal, close_signal, long_signals, short_signals, close_signals, fail_reasons
    
    async def manage_position(self, df: pd.DataFrame, position: Position, balance: float) -> Tuple[
        Position, bool, List[str]
    ]:
        """Manages existing positions with ATR-based trailing stops optimized for scalping."""
        if not position:
            return position, False, []
        
        last = df.iloc[-1]
        close_signals = []
        close_condition = False
        
        # When using trailing profit, delegate to trailing profit manager first
        if self.use_trailing_profit:
            position, trail_close, trail_signals = self.trailing_profit_manager.manage_trailing_profit(df, position)
            
            # If trailing profit manager says to close, respect that
            if trail_close:
                return position, trail_close, trail_signals
                
            # Otherwise, add any signals to our list
            close_signals.extend(trail_signals)
        
        # Otherwise, use standard ATR-based stop management
        # Initialize if first check of this position and trailing_stop not already set
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
            if not hasattr(position, 'open_candles'):
                position.open_candles = 0
            position.open_candles += 1
        
        # Standard stop management logic only if not using trailing profit
        # or if trailing profit not yet activated
        if not self.use_trailing_profit or not getattr(position, 'trailing_profit_activated', False):
            # For scalping, move to breakeven very quickly (after just 1-2 candles)
            if position.open_candles >= 2:
                if position.side == 'long' and last['close'] > position.entry and position.trailing_stop < position.entry:
                    position.trailing_stop = position.entry
                    close_signals.append("Moved stop-loss to breakeven after 2 candles")
                elif position.side == 'short' and last['close'] < position.entry and position.trailing_stop > position.entry:
                    position.trailing_stop = position.entry
                    close_signals.append("Moved stop-loss to breakeven after 2 candles")
            
            # Very aggressive trailing for scalping - much tighter than usual
            atr_value = last['atr']
            trailing_atr = self.atr_multiple * 0.5  # Use half the initial ATR multiple for trailing
            
            if position.side == 'long':
                # Calculate potential new stop level based on ATR
                potential_stop = last['close'] - (atr_value * trailing_atr)
                
                # Move stop up if price has moved favorably
                if potential_stop > position.trailing_stop:
                    position.trailing_stop = potential_stop
                    close_signals.append(f"Raised stop to {potential_stop:.2f}")
                    
                # For scalping, also consider closing when returning to the middle band
                if last['close'] < last['bb_middle'] and last['close'] > position.entry:
                    close_signals.append(f"Price returning to middle band ({last['bb_middle']:.2f}), consider closing")
            
            elif position.side == 'short':
                # Calculate potential new stop level based on ATR
                potential_stop = last['close'] + (atr_value * trailing_atr)
                
                # Move stop down if price has moved favorably
                if potential_stop < position.trailing_stop:
                    position.trailing_stop = potential_stop
                    close_signals.append(f"Lowered stop to {potential_stop:.2f}")
                    
                # For scalping, also consider closing when returning to the middle band  
                if last['close'] > last['bb_middle'] and last['close'] < position.entry:
                    close_signals.append(f"Price returning to middle band ({last['bb_middle']:.2f}), consider closing")
        
        # Quicker time-based exit for scalping - still applicable for trailing profit
        # to avoid holding positions too long
        max_candles = self.position_max_candles
        if self.use_trailing_profit:
            # Extend time limit if we're in significant profit using trailing profit
            if hasattr(position, 'highest_profit_pct') and position.highest_profit_pct > 2.0:
                # Double the allowed time if we're over 2% profit
                max_candles = self.position_max_candles * 2
                
        if position.open_candles > max_candles:
            close_signals.append(f"Position time limit reached ({position.open_candles} candles)")
            close_condition = True
        
        return position, close_condition, close_signals 