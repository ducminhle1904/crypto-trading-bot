"""
Multi-Timeframe Strategy implementation.
"""
import pandas as pd
import numpy as np
from typing import Tuple, List, Optional, Dict, Any
import asyncio

from trading_bot.strategies.base_strategy import BaseStrategy
from trading_bot.models import Position
from trading_bot.utils.indicators import (
    calculate_ema, calculate_rsi, calculate_atr,
    calculate_bollinger_bands, apply_volume_profile_indicators
)
from trading_bot.config import logger
from trading_bot.exchange_client import ExchangeClient


class MultiTimeframeStrategy(BaseStrategy):
    """
    Multi-Timeframe Strategy uses data from multiple timeframes to confirm trends and generate signals.
    
    This strategy analyzes:
    1. Primary timeframe (default 15m): For main signal generation
    2. Higher timeframe (4x primary): For trend confirmation
    3. Lower timeframe (1/3 primary): For entry timing optimization
    """
    
    def __init__(self, timeframe: str = "15m", use_trailing_profit: bool = True):
        """Initialize the strategy with parameters."""
        super().__init__(name="multi_timeframe_strategy", timeframe=timeframe)
        
        # Whether to use trailing profit or fixed take profit
        self.use_trailing_profit = use_trailing_profit
        
        # Set parameters based on timeframe
        self._set_parameters_for_timeframe()
        
        # Store data frames for different timeframes
        self.primary_tf_df = None
        self.higher_tf_df = None 
        self.lower_tf_df = None
        
        # Set related timeframes
        self._set_related_timeframes()
        
        logger.info(f"Initialized Multi-Timeframe Strategy with primary TF={timeframe}, "
                   f"higher TF={self.higher_timeframe}, lower TF={self.lower_timeframe}, "
                   f"Trailing Profit: {'Enabled' if use_trailing_profit else 'Disabled'}")
    
    def _set_related_timeframes(self):
        """Set higher and lower timeframes based on primary timeframe."""
        minutes = self.timeframe_minutes
        
        # Higher timeframe is 4x the primary (or closest standard timeframe)
        if minutes < 15:
            self.higher_timeframe = "15m"
        elif minutes < 60:
            self.higher_timeframe = "1h"
        elif minutes < 240:
            self.higher_timeframe = "4h"
        else:
            self.higher_timeframe = "1d"
            
        # Lower timeframe is 1/3 the primary (or closest standard timeframe)
        if minutes <= 3:
            self.lower_timeframe = "1m"
        elif minutes <= 15:
            self.lower_timeframe = "5m"
        elif minutes <= 60:
            self.lower_timeframe = "15m"
        elif minutes <= 240:
            self.lower_timeframe = "1h"
        else:
            self.lower_timeframe = "4h"
    
    def _set_parameters_for_timeframe(self):
        """Set strategy parameters based on the timeframe."""
        minutes = self.timeframe_minutes
        
        # Base parameters for all timeframes
        self.ema_short = 13
        self.ema_long = 34
        self.ema_slow = 89  # Extra slow EMA for filtering
        self.rsi_period = 14
        self.rsi_overbought = 70
        self.rsi_oversold = 30
        self.bb_period = 20
        self.bb_std = 2.0
        self.atr_period = 14
        
        # Volume Profile parameters
        self.vp_bins = 20
        self.vp_lookback_periods = 100
        
        # Timeframe-specific adjustments
        if minutes <= 5:  # 1m-5m
            self.ema_short = 8
            self.ema_long = 21
            self.ema_slow = 55
            self.rsi_period = 7
            self.atr_period = 10
            self.position_max_candles = 12
            self.profit_target_pct = 0.01  # 1.0%
            self.stop_loss_pct = 0.006  # 0.6%
        elif minutes <= 60:  # 15m-1h
            self.ema_short = 13
            self.ema_long = 34
            self.ema_slow = 89
            self.position_max_candles = 8
            self.profit_target_pct = 0.015  # 1.5%
            self.stop_loss_pct = 0.01  # 1.0%
        else:  # 4h+
            self.ema_short = 18
            self.ema_long = 42
            self.ema_slow = 144
            self.position_max_candles = 5
            self.profit_target_pct = 0.025  # 2.5%
            self.stop_loss_pct = 0.015  # 1.5%
    
    def update_timeframe(self, timeframe: str):
        """Update the strategy timeframe and adjust parameters."""
        super().update_timeframe(timeframe)
        self._set_parameters_for_timeframe()
        self._set_related_timeframes()
        logger.info(f"Updated Multi-Timeframe Strategy parameters for {timeframe} timeframe")
    
    async def _fetch_multi_timeframe_data(self, symbol: str, limit: int = 100) -> Dict[str, pd.DataFrame]:
        """Fetch data for all required timeframes."""
        exchange_client = await ExchangeClient().initialize()
        
        primary_data_task = exchange_client.fetch_ohlcv(symbol, self.timeframe, limit)
        higher_data_task = exchange_client.fetch_ohlcv(symbol, self.higher_timeframe, limit // 2)
        lower_data_task = exchange_client.fetch_ohlcv(symbol, self.lower_timeframe, limit * 2)
        
        # Gather all data
        primary_df, higher_df, lower_df = await asyncio.gather(
            primary_data_task, 
            higher_data_task, 
            lower_data_task
        )
        
        # Close the exchange client
        await exchange_client.close()
        
        return {
            "primary": primary_df,
            "higher": higher_df,
            "lower": lower_df
        }
    
    async def _calculate_single_timeframe_indicators(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Calculate indicators for a single timeframe dataframe."""
        if df is None or len(df) < max(self.rsi_period, self.bb_period) + 5:
            return None
            
        df = df.copy()
        
        # EMA indicators
        df['ema_short'] = calculate_ema(df['close'], self.ema_short)
        df['ema_long'] = calculate_ema(df['close'], self.ema_long)
        df['ema_slow'] = calculate_ema(df['close'], self.ema_slow)
        
        # RSI
        df['rsi'] = calculate_rsi(df['close'], self.rsi_period)
        df['rsi_prev'] = df['rsi'].shift(1)
        df['rsi_rising'] = df['rsi'] > df['rsi_prev']
        df['rsi_falling'] = df['rsi'] < df['rsi_prev']
        df['rsi_overbought'] = df['rsi'] > self.rsi_overbought
        df['rsi_oversold'] = df['rsi'] < self.rsi_oversold
        
        # Bollinger Bands
        bb = calculate_bollinger_bands(df['close'], self.bb_period, self.bb_std)
        df['bb_upper'] = bb['upper']
        df['bb_middle'] = bb['middle']
        df['bb_lower'] = bb['lower']
        df['bb_width'] = bb['bandwidth']
        df['price_above_upper'] = df['close'] > df['bb_upper']
        df['price_below_lower'] = df['close'] < df['bb_lower']
        df['price_above_middle'] = df['close'] > df['bb_middle']
        df['price_below_middle'] = df['close'] < df['bb_middle']
        
        # ATR for volatility
        df['atr'] = calculate_atr(df, self.atr_period)
        
        # Trend calculations
        df['trend_up'] = (df['ema_short'] > df['ema_long']) & (df['ema_long'] > df['ema_slow'])
        df['trend_down'] = (df['ema_short'] < df['ema_long']) & (df['ema_long'] < df['ema_slow'])
        
        # EMA crossovers
        df['ema_cross_up'] = (df['ema_short'] > df['ema_long']) & (df['ema_short'].shift(1) <= df['ema_long'].shift(1))
        df['ema_cross_down'] = (df['ema_short'] < df['ema_long']) & (df['ema_short'].shift(1) >= df['ema_long'].shift(1))
        
        # Volume Profile (optional for some timeframes)
        if len(df) >= self.vp_lookback_periods:
            df = apply_volume_profile_indicators(df, num_bins=self.vp_bins, lookback_periods=self.vp_lookback_periods)
        
        return df
    
    async def calculate_indicators(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """
        Calculate indicators for the primary timeframe, and store higher/lower timeframe data.
        
        Note: This will be called automatically with the primary timeframe data,
        but we'll also fetch and process data from other timeframes.
        """
        try:
            symbol = None
            
            # Extract symbol from df if possible
            if hasattr(df, 'symbol'):
                symbol = df.symbol
            elif '_symbol' in df.attrs:
                symbol = df._symbol
            
            if not symbol:
                # Default symbol from config if not available
                from trading_bot.config import DEFAULT_SYMBOL
                symbol = DEFAULT_SYMBOL
            
            # Calculate indicators for primary timeframe
            self.primary_tf_df = await self._calculate_single_timeframe_indicators(df)
            
            # Fetch and calculate indicators for other timeframes
            if self.primary_tf_df is not None:
                timeframe_data = await self._fetch_multi_timeframe_data(symbol)
                
                self.higher_tf_df = await self._calculate_single_timeframe_indicators(timeframe_data['higher'])
                self.lower_tf_df = await self._calculate_single_timeframe_indicators(timeframe_data['lower'])
                
                # Log successful multi-timeframe data fetch
                logger.info(f"Multi-timeframe data ready: Primary ({self.timeframe}), "
                           f"Higher ({self.higher_timeframe}), Lower ({self.lower_timeframe})")
            
            return self.primary_tf_df
            
        except Exception as e:
            logger.error(f"Error calculating multi-timeframe indicators: {e}")
            return None
    
    def _get_timeframe_alignment(self) -> Dict[str, Any]:
        """
        Determine trend alignment across timeframes.
        
        Returns dictionary with alignment status for different aspects:
        - trend_aligned: Overall trend alignment
        - bullish_aligned: Bullish alignment across timeframes
        - bearish_aligned: Bearish alignment across timeframes
        - etc.
        """
        alignment = {
            'trend_aligned': False,
            'bullish_aligned': False,
            'bearish_aligned': False,
            'trend_strength': 0,  # -3 to +3 scale
            'overbought_alignment': False,
            'oversold_alignment': False,
            'volume_confirms': False
        }
        
        # Check if we have data for all timeframes
        if not all([self.primary_tf_df is not None, 
                   self.higher_tf_df is not None, 
                   self.lower_tf_df is not None]):
            return alignment
        
        # Get last candle from each timeframe
        p_last = self.primary_tf_df.iloc[-1]
        h_last = self.higher_tf_df.iloc[-1]
        l_last = self.lower_tf_df.iloc[-1]
        
        # Check trend alignment - add trend strength for each aligned timeframe
        # For bullish alignment
        trend_strength = 0
        
        # Higher timeframe (most weight)
        if h_last['trend_up']:
            trend_strength += 1.5
        elif h_last['trend_down']:
            trend_strength -= 1.5
            
        # Primary timeframe (medium weight)
        if p_last['trend_up']:
            trend_strength += 1.0
        elif p_last['trend_down']:
            trend_strength -= 1.0
            
        # Lower timeframe (least weight)
        if l_last['trend_up']:
            trend_strength += 0.5
        elif l_last['trend_down']:
            trend_strength -= 0.5
        
        alignment['trend_strength'] = trend_strength
        alignment['trend_aligned'] = abs(trend_strength) >= 2.0  # At least 2 out of 3 scores
        alignment['bullish_aligned'] = trend_strength >= 2.0
        alignment['bearish_aligned'] = trend_strength <= -2.0
        
        # Check RSI alignment
        alignment['overbought_alignment'] = (
            h_last['rsi_overbought'] and 
            p_last['rsi'] > 60 and 
            l_last['rsi'] > 55
        )
        
        alignment['oversold_alignment'] = (
            h_last['rsi_oversold'] and 
            p_last['rsi'] < 40 and 
            l_last['rsi'] < 45
        )
        
        # Check volume profile alignment if available
        if all(['vp_potential_long' in p_last, 'vp_potential_long' in h_last]):
            alignment['volume_long_confirms'] = p_last['vp_potential_long'] and h_last['vp_potential_long']
            alignment['volume_short_confirms'] = p_last['vp_potential_short'] and h_last['vp_potential_short']
        
        return alignment
    
    async def check_signals(self, df: pd.DataFrame, position: Optional[Position] = None) -> Tuple[
        bool, bool, bool, List[str], List[str], List[str], List[str]
    ]:
        """
        Check for signals using multi-timeframe analysis.
        """
        if self.primary_tf_df is None or len(self.primary_tf_df) < 10:
            logger.warning("Insufficient primary timeframe data for signal calculation")
            return False, False, False, [], [], [], ["Insufficient data"]
        
        if self.higher_tf_df is None or self.lower_tf_df is None:
            logger.warning("Missing higher or lower timeframe data")
            return False, False, False, [], [], [], ["Missing multi-timeframe data"]
        
        # Get last candle from each timeframe
        p_last = self.primary_tf_df.iloc[-1]  # Primary
        h_last = self.higher_tf_df.iloc[-1]   # Higher
        l_last = self.lower_tf_df.iloc[-1]    # Lower
        
        long_signals = []
        short_signals = []
        close_signals = []
        fail_reasons = []
        
        # Get multi-timeframe alignment
        alignment = self._get_timeframe_alignment()
        
        # LONG SIGNAL CONDITIONS
        # 1. Higher timeframe confirms uptrend (ema alignment)
        # 2. Primary timeframe shows entry opportunity (oversold, EMA cross, etc.)
        # 3. Lower timeframe confirms momentum (RSI direction, etc.)
        
        higher_tf_bullish = h_last['trend_up'] or (h_last['ema_short'] > h_last['ema_long'])
        primary_tf_entry = (p_last['rsi_oversold'] or p_last['ema_cross_up'] or 
                            (p_last['price_below_lower'] and p_last['rsi_rising']))
        lower_tf_momentum = l_last['rsi_rising'] or l_last['ema_cross_up']
        
        # Volume profile confirmation (if available)
        volume_confirms_long = alignment.get('volume_long_confirms', False)
        
        # Enhanced with BB conditions
        bb_buy_setup = p_last['price_below_lower'] or (
            p_last['price_below_middle'] and 
            p_last['bb_width'] < p_last['bb_width'].rolling(window=14).mean() * 0.8  # BB squeeze
        )
        
        # LONG CONDITION
        long_condition = (
            higher_tf_bullish and 
            primary_tf_entry and 
            lower_tf_momentum and
            (not position or position.side != 'long')
        )
        
        # Add volume confirmation if available
        if 'vp_potential_long' in p_last and volume_confirms_long:
            long_condition = long_condition and volume_confirms_long
            
        # SHORT SIGNAL CONDITIONS (inverse of long)
        higher_tf_bearish = h_last['trend_down'] or (h_last['ema_short'] < h_last['ema_long'])
        primary_tf_entry_short = (p_last['rsi_overbought'] or p_last['ema_cross_down'] or 
                                 (p_last['price_above_upper'] and p_last['rsi_falling']))
        lower_tf_momentum_short = l_last['rsi_falling'] or l_last['ema_cross_down']
        
        # Volume profile confirmation (if available)
        volume_confirms_short = alignment.get('volume_short_confirms', False)
        
        # Enhanced with BB conditions
        bb_sell_setup = p_last['price_above_upper'] or (
            p_last['price_above_middle'] and 
            p_last['bb_width'] < p_last['bb_width'].rolling(window=14).mean() * 0.8  # BB squeeze
        )
        
        # SHORT CONDITION
        short_condition = (
            higher_tf_bearish and 
            primary_tf_entry_short and 
            lower_tf_momentum_short and
            (not position or position.side != 'short')
        )
        
        # Add volume confirmation if available
        if 'vp_potential_short' in p_last and volume_confirms_short:
            short_condition = short_condition and volume_confirms_short
        
        # Generate detailed signal explanation
        if long_condition:
            # Higher timeframe reasons
            if h_last['trend_up']:
                long_signals.append(f"Higher timeframe ({self.higher_timeframe}) shows strong uptrend")
            else:
                long_signals.append(f"Higher timeframe ({self.higher_timeframe}) EMAs bullish: {h_last['ema_short']:.2f} > {h_last['ema_long']:.2f}")
            
            # Primary timeframe reasons
            if p_last['rsi_oversold']:
                long_signals.append(f"Primary timeframe ({self.timeframe}) RSI oversold: {p_last['rsi']:.2f}")
            if p_last['ema_cross_up']:
                long_signals.append(f"Primary timeframe ({self.timeframe}) bullish EMA cross")
            if p_last['price_below_lower']:
                long_signals.append(f"Primary timeframe ({self.timeframe}) price below lower BB: {p_last['close']:.2f} < {p_last['bb_lower']:.2f}")
            
            # Lower timeframe reasons
            if l_last['rsi_rising']:
                long_signals.append(f"Lower timeframe ({self.lower_timeframe}) RSI rising: {l_last['rsi']:.2f}")
            if l_last['ema_cross_up']:
                long_signals.append(f"Lower timeframe ({self.lower_timeframe}) shows momentum with EMA cross")
            
            # Volume profile reasons
            if volume_confirms_long:
                long_signals.append(f"Volume profile confirms long setup across timeframes")
            
            # Alignment score
            long_signals.append(f"Multi-timeframe alignment score: {alignment['trend_strength']:.1f}/3.0 (bullish)")
        else:
            # Add failure reasons
            if not higher_tf_bullish:
                fail_reasons.append(f"Higher timeframe ({self.higher_timeframe}) not bullish")
            if not primary_tf_entry:
                fail_reasons.append(f"No entry opportunity on primary timeframe ({self.timeframe})")
            if not lower_tf_momentum:
                fail_reasons.append(f"No bullish momentum on lower timeframe ({self.lower_timeframe})")
        
        if short_condition:
            # Higher timeframe reasons
            if h_last['trend_down']:
                short_signals.append(f"Higher timeframe ({self.higher_timeframe}) shows strong downtrend")
            else:
                short_signals.append(f"Higher timeframe ({self.higher_timeframe}) EMAs bearish: {h_last['ema_short']:.2f} < {h_last['ema_long']:.2f}")
            
            # Primary timeframe reasons
            if p_last['rsi_overbought']:
                short_signals.append(f"Primary timeframe ({self.timeframe}) RSI overbought: {p_last['rsi']:.2f}")
            if p_last['ema_cross_down']:
                short_signals.append(f"Primary timeframe ({self.timeframe}) bearish EMA cross")
            if p_last['price_above_upper']:
                short_signals.append(f"Primary timeframe ({self.timeframe}) price above upper BB: {p_last['close']:.2f} > {p_last['bb_upper']:.2f}")
            
            # Lower timeframe reasons
            if l_last['rsi_falling']:
                short_signals.append(f"Lower timeframe ({self.lower_timeframe}) RSI falling: {l_last['rsi']:.2f}")
            if l_last['ema_cross_down']:
                short_signals.append(f"Lower timeframe ({self.lower_timeframe}) shows momentum with EMA cross")
            
            # Volume profile reasons
            if volume_confirms_short:
                short_signals.append(f"Volume profile confirms short setup across timeframes")
            
            # Alignment score
            short_signals.append(f"Multi-timeframe alignment score: {-alignment['trend_strength']:.1f}/3.0 (bearish)")
        else:
            # Add failure reasons
            if not higher_tf_bearish:
                fail_reasons.append(f"Higher timeframe ({self.higher_timeframe}) not bearish")
            if not primary_tf_entry_short:
                fail_reasons.append(f"No short entry opportunity on primary timeframe ({self.timeframe})")
            if not lower_tf_momentum_short:
                fail_reasons.append(f"No bearish momentum on lower timeframe ({self.lower_timeframe})")
        
        # EXIT CONDITIONS - Also use multi-timeframe confirmation
        close_condition = False
        if position:
            price = p_last['close']
            
            # Common exit conditions
            if position.trailing_stop:
                if (position.side == 'long' and price <= position.trailing_stop) or \
                   (position.side == 'short' and price >= position.trailing_stop):
                    close_signals.append(f"Stop loss hit at {position.trailing_stop:.2f}")
                    close_condition = True
            
            # Trend reversal exit 
            if position.side == 'long':
                # Take profit level
                take_profit = position.entry * (1 + self.profit_target_pct)
                
                # Exit long if:
                if price >= take_profit:
                    close_signals.append(f"Take profit reached: {price:.2f} >= {take_profit:.2f}")
                    close_condition = True
                # Exit on multi-timeframe trend reversal
                elif alignment['bearish_aligned']:
                    close_signals.append(f"Multi-timeframe trend reversal (bearish alignment)")
                    close_condition = True
                # Exit on primary timeframe reversal with higher timeframe confirmation
                elif p_last['ema_cross_down'] and h_last['ema_short'] < h_last['ema_long']:
                    close_signals.append(f"EMA cross down with higher timeframe confirmation")
                    close_condition = True
                # Exit on RSI overbought on primary and higher timeframe
                elif p_last['rsi_overbought'] and h_last['rsi'] > 65:
                    close_signals.append(f"Overbought on multiple timeframes: Primary RSI={p_last['rsi']:.2f}, Higher RSI={h_last['rsi']:.2f}")
                    close_condition = True
            
            elif position.side == 'short':
                # Take profit level
                take_profit = position.entry * (1 - self.profit_target_pct)
                
                # Exit short if:
                if price <= take_profit:
                    close_signals.append(f"Take profit reached: {price:.2f} <= {take_profit:.2f}")
                    close_condition = True
                # Exit on multi-timeframe trend reversal
                elif alignment['bullish_aligned']:
                    close_signals.append(f"Multi-timeframe trend reversal (bullish alignment)")
                    close_condition = True
                # Exit on primary timeframe reversal with higher timeframe confirmation
                elif p_last['ema_cross_up'] and h_last['ema_short'] > h_last['ema_long']:
                    close_signals.append(f"EMA cross up with higher timeframe confirmation")
                    close_condition = True
                # Exit on RSI oversold on primary and higher timeframe
                elif p_last['rsi_oversold'] and h_last['rsi'] < 35:
                    close_signals.append(f"Oversold on multiple timeframes: Primary RSI={p_last['rsi']:.2f}, Higher RSI={h_last['rsi']:.2f}")
                    close_condition = True
        
        # Final signal Boolean values
        long_signal = long_condition
        short_signal = short_condition
        close_signal = close_condition and position is not None
        
        return long_signal, short_signal, close_signal, long_signals, short_signals, close_signals, fail_reasons
    
    async def manage_position(self, df: pd.DataFrame, position: Position, balance: float) -> Tuple[
        Position, bool, List[str]
    ]:
        """Manages existing positions with multi-timeframe awareness."""
        if not position:
            return position, False, []
        
        # Check if we have all required dataframes
        if any(df is None for df in [self.primary_tf_df, self.higher_tf_df, self.lower_tf_df]):
            logger.warning("Missing timeframe data for position management")
            return position, False, []
        
        # Get the current price from primary timeframe
        last = self.primary_tf_df.iloc[-1]
        price = last['close']
        
        close_signals = []
        close_condition = False
        
        # Initialize if first check of this position
        if not position.trailing_stop:
            # Calculate initial stop-loss based on ATR from the primary timeframe
            atr_value = last['atr']
            atr_multiple = 1.2  # Default multiple
            
            # Use different ATR multiples based on position side and trend alignment
            alignment = self._get_timeframe_alignment()
            
            if position.side == 'long' and alignment['bullish_aligned']:
                # Use tighter stop for strong trend alignment
                atr_multiple = 1.0
            elif position.side == 'short' and alignment['bearish_aligned']:
                # Use tighter stop for strong trend alignment
                atr_multiple = 1.0
                
            position = await self.initialize_stop_loss(position, price, atr_value, atr_multiple)
            
            # Initialize trailing profit tracking
            if self.use_trailing_profit:
                position.highest_profit_pct = 0.0
                position.trailing_profit_activated = False
            
        # Increment candle counter
        if not hasattr(position, 'open_candles'):
            position.open_candles = 0
        position.open_candles += 1
        
        # Move to breakeven after minimal profit
        # - Use tighter threshold on higher volatility (measured by BB width)
        bb_width_ratio = last['bb_width'] / last['bb_width'].rolling(window=20).mean()
        
        # Adjust breakeven threshold - tighter in higher volatility
        if bb_width_ratio > 1.5:  # High volatility
            breakeven_threshold = 0.002  # 0.2%
        elif bb_width_ratio > 1.0:  # Normal volatility
            breakeven_threshold = 0.003  # 0.3%
        else:  # Low volatility
            breakeven_threshold = 0.004  # 0.4%
            
        position, breakeven_signals = await self.move_to_breakeven(position, price, breakeven_threshold)
        close_signals.extend(breakeven_signals)
        
        # Enhanced trailing stop with multi-timeframe awareness
        # Check lower timeframe for more responsive trailing
        l_last = self.lower_tf_df.iloc[-1]
        
        # Use different trailing stop factors based on:
        # 1. Position side
        # 2. Trend strength across timeframes
        # 3. Current price relative to Bollinger Bands
        alignment = self._get_timeframe_alignment()
        trailing_atr_mult = 0.6  # Default
        
        # Adjust trailing stop based on context
        if position.side == 'long':
            if alignment['trend_strength'] > 2:  # Strong bullish alignment
                trailing_atr_mult = 0.8  # Looser stops to let profits run
            elif last['price_above_upper']:
                trailing_atr_mult = 0.4  # Tighter stops when extended
        else:  # short
            if alignment['trend_strength'] < -2:  # Strong bearish alignment
                trailing_atr_mult = 0.8  # Looser stops to let profits run
            elif last['price_below_lower']:
                trailing_atr_mult = 0.4  # Tighter stops when extended
        
        # Update trailing stop based on context
        position, trailing_signals = await self.update_trailing_stop(
            position, price, last['atr'], trailing_atr_mult
        )
        close_signals.extend(trailing_signals)
        
        # Handle complex trailing profit logic with multi-timeframe awareness
        if self.use_trailing_profit:
            # Calculate current profit percentage
            if position.side == 'long':
                current_profit_pct = (price - position.entry) / position.entry * 100
            else:  # short
                current_profit_pct = (position.entry - price) / position.entry * 100
                
            # Update highest profit seen
            if not hasattr(position, 'highest_profit_pct'):
                position.highest_profit_pct = current_profit_pct
            elif current_profit_pct > position.highest_profit_pct:
                position.highest_profit_pct = current_profit_pct
                
            # Adjust trailing profit threshold based on trend strength
            alignment = self._get_timeframe_alignment()
            if (position.side == 'long' and alignment['trend_strength'] > 2) or \
               (position.side == 'short' and alignment['trend_strength'] < -2):
                # In strong trend, wait for more profit before trailing
                trailing_profit_threshold = self.profit_target_pct * 100 * 1.3
                max_pullback_pct = 0.25  # Allow smaller pullback in strong trend
            else:
                # In weaker trend, start trailing earlier and allow larger pullback
                trailing_profit_threshold = self.profit_target_pct * 100 * 1.0
                max_pullback_pct = 0.4
                
            if not hasattr(position, 'trailing_profit_activated'):
                position.trailing_profit_activated = False
                
            if current_profit_pct > trailing_profit_threshold:
                position.trailing_profit_activated = True
                
                # Apply calculated pullback percentage
                max_pullback = position.highest_profit_pct * max_pullback_pct
                
                # Close position if price pulls back too much from the peak
                if position.trailing_profit_activated and current_profit_pct < (position.highest_profit_pct - max_pullback):
                    close_signals.append(f"Trailing profit: Locked in {current_profit_pct:.2f}% (max: {position.highest_profit_pct:.2f}%, allowed pullback: {max_pullback:.2f}%)")
                    close_condition = True
        
        # Check for time-based exit - consider trend alignment
        alignment = self._get_timeframe_alignment()
        
        # Adjust max holding time based on trend alignment
        # In strong aligned trend, allow position to run longer
        adjusted_max_candles = self.position_max_candles
        
        if (position.side == 'long' and alignment['trend_strength'] > 2) or \
           (position.side == 'short' and alignment['trend_strength'] < -2):
            # Extend max holding time in strong trend
            adjusted_max_candles = int(self.position_max_candles * 1.5)
        
        time_exit, time_signals = await self.check_max_holding_time(position, adjusted_max_candles)
        close_signals.extend(time_signals)
        if time_exit:
            close_condition = True
            
        # Multi-timeframe exit signals - look for reversal patterns
        # Check if at least 2 timeframes show reversal patterns
        reversals_count = 0
        
        # For long positions, check for bearish signals
        if position.side == 'long':
            if self.higher_tf_df.iloc[-1]['ema_cross_down']:
                reversals_count += 1
                close_signals.append(f"Higher timeframe bearish EMA cross")
                
            if self.primary_tf_df.iloc[-1]['ema_cross_down']:
                reversals_count += 1
                close_signals.append(f"Primary timeframe bearish EMA cross")
                
            if self.lower_tf_df.iloc[-1]['rsi_overbought'] and self.lower_tf_df.iloc[-1]['rsi_falling']:
                reversals_count += 0.5
                close_signals.append(f"Lower timeframe RSI overbought and falling")
                
        # For short positions, check for bullish signals
        elif position.side == 'short':
            if self.higher_tf_df.iloc[-1]['ema_cross_up']:
                reversals_count += 1
                close_signals.append(f"Higher timeframe bullish EMA cross")
                
            if self.primary_tf_df.iloc[-1]['ema_cross_up']:
                reversals_count += 1
                close_signals.append(f"Primary timeframe bullish EMA cross")
                
            if self.lower_tf_df.iloc[-1]['rsi_oversold'] and self.lower_tf_df.iloc[-1]['rsi_rising']:
                reversals_count += 0.5
                close_signals.append(f"Lower timeframe RSI oversold and rising")
        
        # Exit on multi-timeframe reversal
        if reversals_count >= 2:
            close_signals.append(f"Multi-timeframe reversal detected ({reversals_count} signals)")
            close_condition = True
        
        return position, close_condition, close_signals 