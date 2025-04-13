"""
Volume Profile enhanced VWAP Strategy for day trading.
"""
import pandas as pd
from typing import Tuple, List, Optional, Dict, Any

from trading_bot.strategies.base_strategy import BaseStrategy
from trading_bot.models import Position
from trading_bot.utils.indicators import apply_vwap_stoch_indicators, apply_volume_profile_indicators
from trading_bot.config import logger


class VolumeProfileVwapStrategy(BaseStrategy):
    """
    Volume Profile + VWAP Strategy combines VWAP with Volume Profile analysis to identify
    high-probability zones for entries and exits. Perfect for day trading with increased signal accuracy.
    """
    
    def __init__(self, timeframe: str = "5m", use_trailing_profit: bool = True):
        """Initialize the strategy with parameters."""
        super().__init__(name="volume_profile_vwap_strategy", timeframe=timeframe)
        
        # Whether to use trailing profit or fixed take profit
        self.use_trailing_profit = use_trailing_profit
        
        # Set parameters based on timeframe
        self._set_parameters_for_timeframe()
        
        logger.info(f"Initialized Volume Profile + VWAP Strategy for {timeframe}, "
                   f"Stoch K={self.stoch_k}, Stoch D={self.stoch_d}, "
                   f"VP Lookback={self.vp_lookback_periods}, "
                   f"Trailing Profit: {'Enabled' if use_trailing_profit else 'Disabled'}")
    
    def _set_parameters_for_timeframe(self):
        """Set strategy parameters based on the timeframe."""
        minutes = self.timeframe_minutes
        
        # Base parameters (derived from VWAP strategy)
        self.stoch_k = 14
        self.stoch_d = 3
        self.stoch_smooth = 3
        self.ema_period = 8  # Very fast EMA
        self.position_max_candles = 8  # Quick exits for scalping
        self.atr_multiple = 1.0  # Tight stops for scalping
        
        # Volume Profile parameters
        self.vp_bins = 20  # Number of price bins
        self.vp_lookback_periods = 100  # How far back to look for volume profile
        
        # Adjust profit target based on timeframe
        if minutes <= 5:  # 1m-5m (aggressive scalping)
            self.profit_target_pct = 0.005  # 0.5% target
            self.atr_multiple = 0.8  # Even tighter stops for scalping
            self.position_max_candles = 10
        elif minutes <= 60:  # 15m-1h (standard day trading)
            self.profit_target_pct = 0.008  # 0.8% target
            self.atr_multiple = 1.0
        else:  # 4h+ (swing trading)
            self.profit_target_pct = 0.015  # 1.5% target
            self.atr_multiple = 1.5
            self.position_max_candles = 5
        
        # Stochastic thresholds
        self.oversold_threshold = 20
        self.overbought_threshold = 80
    
    def update_timeframe(self, timeframe: str):
        """Update the strategy timeframe and adjust parameters."""
        super().update_timeframe(timeframe)
        self._set_parameters_for_timeframe()
        logger.info(f"Updated Volume Profile + VWAP Strategy parameters for {timeframe} timeframe")
    
    async def calculate_indicators(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Calculate indicators for the strategy."""
        if df is None or len(df) < self.stoch_k + 5:
            return None
            
        # Apply VWAP + Stochastic indicators first
        df = apply_vwap_stoch_indicators(
            df,
            stoch_k=self.stoch_k,
            stoch_d=self.stoch_d,
            stoch_smooth=self.stoch_smooth,
            ema_period=self.ema_period
        )
        
        if df is None:
            return None
            
        # Then apply Volume Profile indicators
        df = apply_volume_profile_indicators(
            df,
            num_bins=self.vp_bins,
            lookback_periods=self.vp_lookback_periods
        )
        
        return df
    
    async def check_signals(self, df: pd.DataFrame, position: Optional[Position] = None) -> Tuple[
        bool, bool, bool, List[str], List[str], List[str], List[str]
    ]:
        """Check for trading signals."""
        if df is None or len(df) < self.stoch_k + 5:
            logger.warning("Insufficient data for Volume Profile + VWAP signal calculation")
            return False, False, False, [], [], [], ["Insufficient data"]
        
        last = df.iloc[-1]
        prev = df.iloc[-2] if len(df) > 2 else last
        
        long_signals = []
        short_signals = []
        close_signals = []
        fail_reasons = []
        
        # VWAP conditions
        above_vwap = last['above_vwap']
        below_vwap = last['below_vwap']
        vwap_cross_up = last['vwap_cross_up']
        vwap_cross_down = last['vwap_cross_down']
        
        # Stochastic conditions
        stoch_oversold = last['stoch_oversold']
        stoch_overbought = last['stoch_overbought']
        stoch_cross_up = last['stoch_cross_up']
        stoch_cross_down = last['stoch_cross_down']
        
        # EMA conditions for overall trend
        price_above_ema = last['close'] > last['ema']
        price_below_ema = last['close'] < last['ema']
        
        # Volume Profile conditions
        vp_potential_long = last.get('vp_potential_long', False)
        vp_potential_short = last.get('vp_potential_short', False)
        price_near_poc = abs(last['close'] - last['vp_poc']) / last['vp_poc'] < 0.005 if 'vp_poc' in last and last['vp_poc'] else False
        price_in_value_area = last.get('vp_in_value_area', True)
        
        # Enhanced Long signal conditions:
        # 1. Price crosses above VWAP (value zone) AND
        # 2. Stochastic is oversold OR Stochastic K crosses above D
        # 3. Price is above fast EMA for trend confirmation
        # 4. Volume Profile agrees (near POC or at value area low)
        long_condition = ((vwap_cross_up or above_vwap) and 
                         (stoch_oversold or stoch_cross_up) and 
                         price_above_ema and 
                         vp_potential_long)
        
        # Enhanced Short signal conditions:
        # 1. Price crosses below VWAP (value zone) AND
        # 2. Stochastic is overbought OR Stochastic K crosses below D
        # 3. Price is below fast EMA for trend confirmation
        # 4. Volume Profile agrees (near POC or at value area high)
        short_condition = ((vwap_cross_down or below_vwap) and 
                          (stoch_overbought or stoch_cross_down) and 
                          price_below_ema and 
                          vp_potential_short)
        
        # Generate reasons for long signals
        if long_condition:
            if vwap_cross_up:
                long_signals.append(f"Price crossed above VWAP: {last['close']:.2f} > {last['vwap']:.2f}")
            else:
                long_signals.append(f"Price above VWAP: {last['close']:.2f} > {last['vwap']:.2f}")
                
            if stoch_oversold:
                long_signals.append(f"Stochastic oversold: K={last['stoch_k']:.2f}, D={last['stoch_d']:.2f}")
            if stoch_cross_up:
                long_signals.append(f"Stochastic bullish cross: K={last['stoch_k']:.2f} crossed above D={last['stoch_d']:.2f}")
                
            long_signals.append(f"Price above EMA({self.ema_period}): {last['close']:.2f} > {last['ema']:.2f}")
            
            # Add Volume Profile specific reasons
            if price_near_poc:
                long_signals.append(f"Price near high-volume Point of Control: {last['vp_poc']:.2f}")
            
            if 'vp_val' in last and last['vp_val']:
                long_signals.append(f"Price near value area low: {last['vp_val']:.2f}")
            
            if 'vp_val_dist' in last and last['vp_val_dist'] > 0:
                long_signals.append(f"Price bouncing up from value area: {abs(last['vp_val_dist']):.2f}% from VAL")
        else:
            if not (vwap_cross_up or above_vwap):
                fail_reasons.append(f"Price not above VWAP: {last['close']:.2f} ≤ {last['vwap']:.2f}")
            if not (stoch_oversold or stoch_cross_up):
                fail_reasons.append(f"Stochastic not giving buy signal: K={last['stoch_k']:.2f}, D={last['stoch_d']:.2f}")
            if not price_above_ema:
                fail_reasons.append(f"Price below EMA({self.ema_period}): {last['close']:.2f} < {last['ema']:.2f}")
            if not vp_potential_long:
                fail_reasons.append("Volume Profile not showing long potential")
        
        # Generate reasons for short signals
        if short_condition:
            if vwap_cross_down:
                short_signals.append(f"Price crossed below VWAP: {last['close']:.2f} < {last['vwap']:.2f}")
            else:
                short_signals.append(f"Price below VWAP: {last['close']:.2f} < {last['vwap']:.2f}")
                
            if stoch_overbought:
                short_signals.append(f"Stochastic overbought: K={last['stoch_k']:.2f}, D={last['stoch_d']:.2f}")
            if stoch_cross_down:
                short_signals.append(f"Stochastic bearish cross: K={last['stoch_k']:.2f} crossed below D={last['stoch_d']:.2f}")
                
            short_signals.append(f"Price below EMA({self.ema_period}): {last['close']:.2f} < {last['ema']:.2f}")
            
            # Add Volume Profile specific reasons
            if price_near_poc:
                short_signals.append(f"Price near high-volume Point of Control: {last['vp_poc']:.2f}")
            
            if 'vp_vah' in last and last['vp_vah']:
                short_signals.append(f"Price near value area high: {last['vp_vah']:.2f}")
                
            if 'vp_vah_dist' in last and last['vp_vah_dist'] < 0:
                short_signals.append(f"Price dropping from value area: {abs(last['vp_vah_dist']):.2f}% from VAH")
        else:
            if not (vwap_cross_down or below_vwap):
                fail_reasons.append(f"Price not below VWAP: {last['close']:.2f} ≥ {last['vwap']:.2f}")
            if not (stoch_overbought or stoch_cross_down):
                fail_reasons.append(f"Stochastic not giving sell signal: K={last['stoch_k']:.2f}, D={last['stoch_d']:.2f}")
            if not price_below_ema:
                fail_reasons.append(f"Price above EMA({self.ema_period}): {last['close']:.2f} > {last['ema']:.2f}")
            if not vp_potential_short:
                fail_reasons.append("Volume Profile not showing short potential")
        
        # Exit conditions - enhanced with Volume Profile
        close_condition = False
        if position:
            take_profit_pct = self.profit_target_pct
            
            if position.side == 'long':
                take_profit = position.entry * (1 + take_profit_pct)
                
                # Exit long positions when:
                if position.trailing_stop and last['close'] <= position.trailing_stop:
                    close_signals.append(f"Stop loss hit at {position.trailing_stop:.2f}")
                    close_condition = True
                # Take profit target hit
                elif last['close'] >= take_profit:
                    close_signals.append(f"Take profit hit: {last['close']:.2f} >= {take_profit:.2f}")
                    close_condition = True
                # Stochastic overbought OR bearish cross
                elif stoch_overbought:
                    close_signals.append(f"Stochastic overbought: K={last['stoch_k']:.2f}, D={last['stoch_d']:.2f}")
                    close_condition = True
                elif stoch_cross_down:
                    close_signals.append(f"Stochastic bearish cross: K={last['stoch_k']:.2f} crossed below D={last['stoch_d']:.2f}")
                    close_condition = True
                # Price moves below VWAP
                elif vwap_cross_down:
                    close_signals.append(f"Price crossed below VWAP: {last['close']:.2f} < {last['vwap']:.2f}")
                    close_condition = True
                # Volume Profile signals potential short (reversal)
                elif vp_potential_short and price_near_poc:
                    close_signals.append(f"Volume Profile showing potential reversal at {last['vp_poc']:.2f}")
                    close_condition = True
                # Price above value area high
                elif 'vp_vah' in last and last['vp_vah'] and last['close'] > last['vp_vah']:
                    close_signals.append(f"Price above value area high: {last['close']:.2f} > {last['vp_vah']:.2f}")
                    close_condition = True
            
            elif position.side == 'short':
                take_profit = position.entry * (1 - take_profit_pct)
                
                # Exit short positions when:
                if position.trailing_stop and last['close'] >= position.trailing_stop:
                    close_signals.append(f"Stop loss hit at {position.trailing_stop:.2f}")
                    close_condition = True
                # Take profit target hit
                elif last['close'] <= take_profit:
                    close_signals.append(f"Take profit hit: {last['close']:.2f} <= {take_profit:.2f}")
                    close_condition = True
                # Stochastic oversold OR bullish cross
                elif stoch_oversold:
                    close_signals.append(f"Stochastic oversold: K={last['stoch_k']:.2f}, D={last['stoch_d']:.2f}")
                    close_condition = True
                elif stoch_cross_up:
                    close_signals.append(f"Stochastic bullish cross: K={last['stoch_k']:.2f} crossed above D={last['stoch_d']:.2f}")
                    close_condition = True
                # Price moves above VWAP
                elif vwap_cross_up:
                    close_signals.append(f"Price crossed above VWAP: {last['close']:.2f} > {last['vwap']:.2f}")
                    close_condition = True
                # Volume Profile signals potential long (reversal)
                elif vp_potential_long and price_near_poc:
                    close_signals.append(f"Volume Profile showing potential reversal at {last['vp_poc']:.2f}")
                    close_condition = True
                # Price below value area low
                elif 'vp_val' in last and last['vp_val'] and last['close'] < last['vp_val']:
                    close_signals.append(f"Price below value area low: {last['close']:.2f} < {last['vp_val']:.2f}")
                    close_condition = True
        
        long_signal = long_condition and not position
        short_signal = short_condition and not position
        close_signal = close_condition and position
        
        return long_signal, short_signal, close_signal, long_signals, short_signals, close_signals, fail_reasons
    
    async def manage_position(self, df: pd.DataFrame, position: Position, balance: float) -> Tuple[
        Position, bool, List[str]
    ]:
        """Manages existing positions with Volume Profile + VWAP-based stops."""
        if not position:
            return position, False, []
        
        last = df.iloc[-1]
        close_signals = []
        close_condition = False
        
        # Initialize if first check of this position
        if not position.trailing_stop:
            # Calculate initial stop-loss based on ATR
            position = await self.initialize_stop_loss(position, last['close'], last['atr'], self.atr_multiple)
            
            # Initialize trailing profit tracking
            if self.use_trailing_profit:
                position.highest_profit_pct = 0.0
                position.trailing_profit_activated = False
            
        # Increment candle counter
        if not hasattr(position, 'open_candles'):
            position.open_candles = 0
        position.open_candles += 1
        
        # Quick breakeven for scalping - just 0.2% profit
        min_profit_pct = 0.002
        position, breakeven_signals = await self.move_to_breakeven(position, last['close'], min_profit_pct)
        close_signals.extend(breakeven_signals)
        
        # Volume Profile enhanced trailing stops
        # If price is near POC, tighten the stop
        price_near_poc = 'vp_poc' in last and last['vp_poc'] and abs(last['close'] - last['vp_poc']) / last['vp_poc'] < 0.005
        
        if price_near_poc:
            # Use tighter trailing stops near POC (high volume areas have higher reversal potential)
            trailing_atr_mult = 0.5  # Tighter at POC
            position, trailing_signals = await self.update_trailing_stop(
                position, last['close'], last['atr'], trailing_atr_mult
            )
            if trailing_signals:
                trailing_signals = [s + " (near POC - tighter stop)" for s in trailing_signals]
                close_signals.extend(trailing_signals)
        else:
            # Standard trailing stop updates
            trailing_atr_mult = 0.8
            position, trailing_signals = await self.update_trailing_stop(
                position, last['close'], last['atr'], trailing_atr_mult
            )
            close_signals.extend(trailing_signals)
        
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
                
            # Lock in profit with trailing take-profit once we reach the target
            # For Volume Profile strategy, we can be more aggressive with locking in profits
            trailing_profit_threshold = self.profit_target_pct * 100 * 0.7  # 70% of target activates trailing
            
            if not hasattr(position, 'trailing_profit_activated'):
                position.trailing_profit_activated = False
                
            if current_profit_pct > trailing_profit_threshold:
                position.trailing_profit_activated = True
                
            # Allow only 30% pullback from highest profit once trailing activated
            if position.trailing_profit_activated:
                max_pullback = min(position.highest_profit_pct * 0.3, 0.5)  # Cap at 0.5%
                
                if current_profit_pct < (position.highest_profit_pct - max_pullback):
                    close_signals.append(f"Trailing profit: Locked in {current_profit_pct:.2f}% (max: {position.highest_profit_pct:.2f}%)")
                    close_condition = True
        
        # Check for time-based exit
        time_exit, time_signals = await self.check_max_holding_time(position, self.position_max_candles)
        close_signals.extend(time_signals)
        if time_exit:
            close_condition = True
        
        return position, close_condition, close_signals 