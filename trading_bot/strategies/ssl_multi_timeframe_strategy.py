"""
SSL Multi-Timeframe Strategy implementation.
"""
import pandas as pd
from typing import Tuple, List, Optional, Dict, Any, Set

from trading_bot.strategies.base_strategy import BaseStrategy
from trading_bot.models import Position
from trading_bot.utils.indicators import calculate_ssl_channel, apply_standard_indicators
from trading_bot.config import logger
from trading_bot.exchange_client import ExchangeClient


class SslMultiTimeframeStrategy(BaseStrategy):
    """
    SSL Multi-Timeframe Strategy uses SSL Channel across multiple timeframes.
    Entry signals require SSL alignment across timeframes for high-probability entries.
    """
    
    def __init__(self, timeframe: str = "15m", use_trailing_profit: bool = True):
        """Initialize the strategy with parameters."""
        super().__init__(name="ssl_mtf_strategy", timeframe=timeframe)
        
        # Whether to use trailing profit or fixed take profit
        self.use_trailing_profit = use_trailing_profit
        
        # Define timeframes - primary (current), higher, and lower
        self.primary_timeframe = timeframe
        
        # Set parameters based on timeframe
        self._set_parameters_for_timeframe()
        
        # Cache for higher and lower timeframe data
        self.higher_tf_data = None
        self.lower_tf_data = None
        self.last_update_time = None
        
        logger.info(f"Initialized SSL MTF Strategy with parameters: SSL Period={self.ssl_period}, "
                   f"Higher TF={self.higher_timeframe}, Lower TF={self.lower_timeframe}, "
                   f"Alignment Weight Higher={self.higher_tf_weight}, Lower={self.lower_tf_weight}, "
                   f"Trailing Profit: {'Enabled' if use_trailing_profit else 'Disabled'}")
    
    def _set_parameters_for_timeframe(self):
        """Set strategy parameters based on the timeframe."""
        minutes = self.timeframe_minutes
        
        # Set higher and lower timeframes based on primary timeframe
        if minutes <= 5:  # Very short timeframes (1m-5m)
            self.higher_timeframe = "15m"
            self.lower_timeframe = "1m"
        elif minutes <= 15:  # Short timeframes (15m)
            self.higher_timeframe = "1h"
            self.lower_timeframe = "5m"
        elif minutes <= 60:  # Medium timeframes (30m-1h)
            self.higher_timeframe = "4h"
            self.lower_timeframe = "15m"
        else:  # Higher timeframes (4h+)
            self.higher_timeframe = "1d"
            self.lower_timeframe = "1h"
        
        # Base parameters - these will be adjusted based on timeframe
        if minutes <= 15:  # For short timeframes (1m-15m)
            # Faster settings for scalping
            self.ssl_period = 7
            self.higher_tf_ssl_period = 10
            self.lower_tf_ssl_period = 5
            self.atr_period = 10
            self.position_max_candles = 12
            self.profit_target_pct = 0.015  # 1.5%
            self.stop_loss_pct = 0.01  # 1%
            # Weights for alignment score (total of 3)
            self.higher_tf_weight = 1.5  # Higher timeframe has more importance
            self.primary_tf_weight = 1.0  # Primary timeframe has standard weight
            self.lower_tf_weight = 0.5   # Lower timeframe has less importance
        elif minutes <= 60:  # For medium timeframes (30m-1h)
            # Balanced parameters
            self.ssl_period = 10
            self.higher_tf_ssl_period = 14
            self.lower_tf_ssl_period = 7
            self.atr_period = 14
            self.position_max_candles = 10
            self.profit_target_pct = 0.02  # 2%
            self.stop_loss_pct = 0.012  # 1.2%
            # Weights for alignment score (total of 3)
            self.higher_tf_weight = 1.5
            self.primary_tf_weight = 1.0
            self.lower_tf_weight = 0.5
        else:  # For higher timeframes (4h+)
            # Slower settings for trend following
            self.ssl_period = 14
            self.higher_tf_ssl_period = 20
            self.lower_tf_ssl_period = 10
            self.atr_period = 14
            self.position_max_candles = 8
            self.profit_target_pct = 0.025  # 2.5%
            self.stop_loss_pct = 0.015  # 1.5%
            # Weights for alignment score (total of 3)
            self.higher_tf_weight = 1.5
            self.primary_tf_weight = 1.0
            self.lower_tf_weight = 0.5
    
    def update_timeframe(self, timeframe: str):
        """Update the strategy timeframe and adjust parameters."""
        super().update_timeframe(timeframe)
        self.primary_timeframe = timeframe
        self._set_parameters_for_timeframe()
        logger.info(f"Updated SSL MTF Strategy parameters for {timeframe} timeframe")
    
    async def _fetch_multi_timeframe_data(self, df: pd.DataFrame, symbol: str) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """Fetch data for higher and lower timeframes."""
        exchange_client = ExchangeClient()
        await exchange_client.initialize()
        
        try:
            # Fetch higher timeframe data
            higher_tf_df = await exchange_client.fetch_ohlcv(symbol, self.higher_timeframe, 100)
            if higher_tf_df is not None:
                higher_tf_df = calculate_ssl_channel(higher_tf_df, period=self.higher_tf_ssl_period)
            
            # Fetch lower timeframe data
            lower_tf_df = await exchange_client.fetch_ohlcv(symbol, self.lower_timeframe, 100)
            if lower_tf_df is not None:
                lower_tf_df = calculate_ssl_channel(lower_tf_df, period=self.lower_tf_ssl_period)
            
            # Close the client
            await exchange_client.close()
            
            return higher_tf_df, lower_tf_df
        except Exception as e:
            logger.error(f"Error fetching multi-timeframe data: {str(e)}")
            await exchange_client.close()
            return None, None
    
    async def calculate_indicators(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Calculate indicators for the strategy."""
        try:
            # Apply standard indicators (including ATR)
            df = apply_standard_indicators(
                df,
                atr_period=self.atr_period
            )
            
            # Calculate SSL Channel for primary timeframe
            df = calculate_ssl_channel(df, period=self.ssl_period)
            
            # Symbol extraction for multi-timeframe data
            symbol = None
            if df.attrs.get('symbol'):
                symbol = df.attrs.get('symbol')
            
            if symbol:
                # Fetch or update multi-timeframe data
                current_timestamp = df.iloc[-1]['timestamp']
                if self.last_update_time is None or (current_timestamp - self.last_update_time).total_seconds() > 60:
                    self.higher_tf_data, self.lower_tf_data = await self._fetch_multi_timeframe_data(df, symbol)
                    self.last_update_time = current_timestamp
            
            return df
        except Exception as e:
            logger.error(f"Error calculating indicators for SSL MTF Strategy: {str(e)}")
            return None
    
    async def _calculate_alignment_score(self, primary_df: pd.DataFrame) -> Tuple[float, Dict[str, Any]]:
        """Calculate alignment score across timeframes."""
        last_primary = primary_df.iloc[-1]
        primary_trend = last_primary['ssl_trend']
        
        alignment_details = {}
        alignment_score = 0.0
        
        # Add primary timeframe contribution
        alignment_score += self.primary_tf_weight if primary_trend != 0 else 0
        alignment_details['primary'] = {
            'trend': primary_trend,
            'weight': self.primary_tf_weight,
            'score': self.primary_tf_weight if primary_trend != 0 else 0
        }
        
        # Add higher timeframe contribution if available
        if self.higher_tf_data is not None and len(self.higher_tf_data) > 0:
            higher_trend = self.higher_tf_data.iloc[-1]['ssl_trend']
            # Only add score if trends align
            if higher_trend != 0 and higher_trend == primary_trend:
                alignment_score += self.higher_tf_weight
            alignment_details['higher'] = {
                'trend': higher_trend,
                'weight': self.higher_tf_weight,
                'score': self.higher_tf_weight if higher_trend != 0 and higher_trend == primary_trend else 0
            }
        
        # Add lower timeframe contribution if available
        if self.lower_tf_data is not None and len(self.lower_tf_data) > 0:
            lower_trend = self.lower_tf_data.iloc[-1]['ssl_trend']
            # Only add score if trends align
            if lower_trend != 0 and lower_trend == primary_trend:
                alignment_score += self.lower_tf_weight
            alignment_details['lower'] = {
                'trend': lower_trend,
                'weight': self.lower_tf_weight,
                'score': self.lower_tf_weight if lower_trend != 0 and lower_trend == primary_trend else 0
            }
        
        return alignment_score, alignment_details
    
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
        
        # Calculate multi-timeframe alignment score
        alignment_score, alignment_details = await self._calculate_alignment_score(df)
        
        # SSL signals on primary timeframe
        ssl_crossover_bull = prev['ssl_trend'] <= 0 and last['ssl_trend'] > 0
        ssl_crossover_bear = prev['ssl_trend'] >= 0 and last['ssl_trend'] < 0
        
        # Price crossing above ssl_down is bullish
        price_cross_up = prev['close'] <= prev['ssl_down'] and last['close'] > last['ssl_down']
        
        # Price crossing below ssl_up is bearish
        price_cross_down = prev['close'] >= prev['ssl_up'] and last['close'] < last['ssl_up']
        
        # Require significant alignment score for signals
        strong_alignment = alignment_score >= 2.0  # At least 2 out of 3 possible points
        
        # Combined signals with multi-timeframe alignment
        # Long signal: Primary SSL bullish + strong multi-timeframe alignment
        long_condition = (ssl_crossover_bull or price_cross_up) and last['ssl_trend'] > 0 and strong_alignment
        
        # Short signal: Primary SSL bearish + strong multi-timeframe alignment  
        short_condition = (ssl_crossover_bear or price_cross_down) and last['ssl_trend'] < 0 and strong_alignment
        
        if long_condition:
            long_signals.append(f"Multi-timeframe bullish alignment (Score: {alignment_score:.1f}/3.0)")
            long_signals.append(f"SSL Up: {last['ssl_up']:.2f}, SSL Down: {last['ssl_down']:.2f}")
            
            # Add individual timeframe details
            for tf_name, details in alignment_details.items():
                trend_str = "Bullish" if details['trend'] > 0 else "Bearish" if details['trend'] < 0 else "Neutral"
                long_signals.append(f"{tf_name.capitalize()} timeframe: {trend_str} (Score: {details['score']:.1f})")
            
            if ssl_crossover_bull:
                long_signals.append("SSL Crossover: Trend changed from bearish/neutral to bullish")
            if price_cross_up:
                long_signals.append("Price crossed above SSL Down line")
        else:
            if not (ssl_crossover_bull or price_cross_up):
                fail_reasons.append("No SSL bullish signal on primary timeframe")
            if not strong_alignment:
                fail_reasons.append(f"Insufficient trend alignment across timeframes: {alignment_score:.1f}/3.0")
            for tf_name, details in alignment_details.items():
                if details['score'] == 0:
                    trend_str = "Neutral" if details['trend'] == 0 else "Bearish"
                    fail_reasons.append(f"{tf_name.capitalize()} timeframe not aligned: {trend_str}")
        
        if short_condition:
            short_signals.append(f"Multi-timeframe bearish alignment (Score: {alignment_score:.1f}/3.0)")
            short_signals.append(f"SSL Up: {last['ssl_up']:.2f}, SSL Down: {last['ssl_down']:.2f}")
            
            # Add individual timeframe details
            for tf_name, details in alignment_details.items():
                trend_str = "Bullish" if details['trend'] > 0 else "Bearish" if details['trend'] < 0 else "Neutral"
                short_signals.append(f"{tf_name.capitalize()} timeframe: {trend_str} (Score: {details['score']:.1f})")
            
            if ssl_crossover_bear:
                short_signals.append("SSL Crossover: Trend changed from bullish/neutral to bearish")
            if price_cross_down:
                short_signals.append("Price crossed below SSL Up line")
        
        close_condition = False
        if position:
            # Calculate alignment score for exit (looser criteria)
            trend_reversal = (position.side == 'long' and last['ssl_trend'] < 0) or \
                             (position.side == 'short' and last['ssl_trend'] > 0)
            
            # Calculate how many timeframes are showing reversal
            reversed_timeframes = []
            if position.side == 'long':
                if last['ssl_trend'] < 0:
                    reversed_timeframes.append("Primary")
                if self.higher_tf_data is not None and len(self.higher_tf_data) > 0 and self.higher_tf_data.iloc[-1]['ssl_trend'] < 0:
                    reversed_timeframes.append("Higher")
                if self.lower_tf_data is not None and len(self.lower_tf_data) > 0 and self.lower_tf_data.iloc[-1]['ssl_trend'] < 0:
                    reversed_timeframes.append("Lower")
            else:  # short
                if last['ssl_trend'] > 0:
                    reversed_timeframes.append("Primary")
                if self.higher_tf_data is not None and len(self.higher_tf_data) > 0 and self.higher_tf_data.iloc[-1]['ssl_trend'] > 0:
                    reversed_timeframes.append("Higher")
                if self.lower_tf_data is not None and len(self.lower_tf_data) > 0 and self.lower_tf_data.iloc[-1]['ssl_trend'] > 0:
                    reversed_timeframes.append("Lower")
            
            # Exit position if primary timeframe has reversed and at least one other timeframe confirms
            if trend_reversal and len(reversed_timeframes) >= 2:
                close_signals.append(f"Multi-timeframe trend reversal: {', '.join(reversed_timeframes)}")
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
                position.alignment_score_at_entry = 0.0
        
        # Calculate current multi-timeframe alignment
        alignment_score, alignment_details = await self._calculate_alignment_score(df)
        
        # Store alignment score at entry for reference
        if not hasattr(position, 'alignment_score_at_entry') or position.alignment_score_at_entry == 0:
            position.alignment_score_at_entry = alignment_score
        
        # Increment candle counter
        if not hasattr(position, 'open_candles'):
            position.open_candles = 0
        position.open_candles += 1
        
        # Move to breakeven based on alignment score - tighter with strong alignment
        breakeven_threshold = 0.01 - (alignment_score * 0.002)  # 0.01 to 0.004 as score goes from 0 to 3
        position, breakeven_signals = await self.move_to_breakeven(position, last['close'], breakeven_threshold)
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
                
            # Advanced trailing profit logic based on multi-timeframe alignment
            if not hasattr(position, 'trailing_profit_activated'):
                position.trailing_profit_activated = False
                
            # Activate trailing profit when profit reaches target
            profit_target_factor = max(0.7, min(1.3, alignment_score / 2))  # Scale based on alignment
            adaptive_profit_target = self.profit_target_pct * profit_target_factor * 100
            
            if current_profit_pct > adaptive_profit_target:
                position.trailing_profit_activated = True
                
                # Calculate how many timeframes are still aligned with the position
                aligned_count = 0
                if position.side == 'long':
                    if last['ssl_trend'] > 0:
                        aligned_count += 1
                    if self.higher_tf_data is not None and len(self.higher_tf_data) > 0 and self.higher_tf_data.iloc[-1]['ssl_trend'] > 0:
                        aligned_count += 1
                    if self.lower_tf_data is not None and len(self.lower_tf_data) > 0 and self.lower_tf_data.iloc[-1]['ssl_trend'] > 0:
                        aligned_count += 1
                else:  # short
                    if last['ssl_trend'] < 0:
                        aligned_count += 1
                    if self.higher_tf_data is not None and len(self.higher_tf_data) > 0 and self.higher_tf_data.iloc[-1]['ssl_trend'] < 0:
                        aligned_count += 1
                    if self.lower_tf_data is not None and len(self.lower_tf_data) > 0 and self.lower_tf_data.iloc[-1]['ssl_trend'] < 0:
                        aligned_count += 1
                
                # Adjust trailing factor based on alignment (tighter with less alignment)
                trailing_factor = 0.5 - (aligned_count * 0.1)  # 0.5 to 0.2 as alignment decreases
                
                # Ensure trailing factor is in reasonable range
                trailing_factor = max(0.2, min(0.5, trailing_factor))
                
                max_pullback = position.highest_profit_pct * trailing_factor
                
                # Close position if price pulls back too much from the peak
                if position.trailing_profit_activated and current_profit_pct < (position.highest_profit_pct - max_pullback):
                    close_signals.append(f"Trailing profit: Locked in {current_profit_pct:.2f}% (max: {position.highest_profit_pct:.2f}%)")
                    close_condition = True
        
        # Check for multi-timeframe trend reversal
        reversed_timeframes = []
        if position.side == 'long':
            if last['ssl_trend'] < 0:
                reversed_timeframes.append("Primary")
            if self.higher_tf_data is not None and len(self.higher_tf_data) > 0 and self.higher_tf_data.iloc[-1]['ssl_trend'] < 0:
                reversed_timeframes.append("Higher")
            if self.lower_tf_data is not None and len(self.lower_tf_data) > 0 and self.lower_tf_data.iloc[-1]['ssl_trend'] < 0:
                reversed_timeframes.append("Lower")
        else:  # short
            if last['ssl_trend'] > 0:
                reversed_timeframes.append("Primary")
            if self.higher_tf_data is not None and len(self.higher_tf_data) > 0 and self.higher_tf_data.iloc[-1]['ssl_trend'] > 0:
                reversed_timeframes.append("Higher")
            if self.lower_tf_data is not None and len(self.lower_tf_data) > 0 and self.lower_tf_data.iloc[-1]['ssl_trend'] > 0:
                reversed_timeframes.append("Lower")
        
        # Exit position if primary timeframe has reversed and at least one other timeframe confirms
        if len(reversed_timeframes) >= 2 and "Primary" in reversed_timeframes:
            close_signals.append(f"Multi-timeframe trend reversal: {', '.join(reversed_timeframes)}")
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
        
        # Check for time-based exit - extended if alignment is strong
        max_candles_adjustment = min(4, int(alignment_score))  # Add up to 4 candles for strong alignment
        adjusted_max_candles = self.position_max_candles + max_candles_adjustment
        
        time_exit, time_signals = await self.check_max_holding_time(position, adjusted_max_candles)
        if time_exit:
            time_signals[0] = time_signals[0].replace(f"({position.open_candles} candles)", 
                                                     f"({position.open_candles}/{adjusted_max_candles} candles)")
        close_signals.extend(time_signals)
        if time_exit:
            close_condition = True
        
        return position, close_condition, close_signals 