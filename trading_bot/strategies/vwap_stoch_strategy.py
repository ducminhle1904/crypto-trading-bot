"""
VWAP + Stochastic Strategy for scalping.
"""
import pandas as pd
from typing import Tuple, List, Optional, Dict, Any

from trading_bot.strategies.base_strategy import BaseStrategy
from trading_bot.models import Position
from trading_bot.utils.indicators import apply_vwap_stoch_indicators
from trading_bot.config import logger


class VwapStochStrategy(BaseStrategy):
    """
    VWAP + Stochastic Strategy uses VWAP for intraday value area and Stochastic 
    for overbought/oversold levels. Perfect for intraday scalping.
    """
    
    def __init__(self, timeframe: str = "5m", use_trailing_profit: bool = True):
        """Initialize the strategy with parameters."""
        super().__init__(name="vwap_stoch_strategy", timeframe=timeframe)
        
        # Whether to use trailing profit or fixed take profit
        self.use_trailing_profit = use_trailing_profit
        
        # Set parameters based on timeframe
        self._set_parameters_for_timeframe()
        
        logger.info(f"Initialized VWAP+Stochastic Strategy for {timeframe}, "
                   f"Stoch K={self.stoch_k}, Stoch D={self.stoch_d}, "
                   f"Trailing Profit: {'Enabled' if use_trailing_profit else 'Disabled'}")
    
    def _set_parameters_for_timeframe(self):
        """Set strategy parameters based on the timeframe."""
        minutes = self.timeframe_minutes
        
        # Scale parameters based on timeframe
        if minutes <= 5:  # 1m to 5m - optimal for scalping
            self.stoch_k = 14
            self.stoch_d = 3
            self.stoch_smooth = 3
            self.ema_period = 8  # Very fast EMA
            self.position_max_candles = 8  # Quick exits for scalping
            self.atr_multiple = 1.0  # Tight stops for scalping
            self.profit_target_pct = 0.005  # 0.5% target (small but frequent wins)
            # Adjust overbought/oversold thresholds
            self.oversold_threshold = 20
            self.overbought_threshold = 80
            
        elif minutes <= 60:  # 15m to 1h - medium settings
            self.stoch_k = 14
            self.stoch_d = 3
            self.stoch_smooth = 3
            self.ema_period = 13
            self.position_max_candles = 6
            self.atr_multiple = 1.5
            self.profit_target_pct = 0.01  # 1% target
            self.oversold_threshold = 20
            self.overbought_threshold = 80
            
        else:  # 4h, daily - slower settings
            self.stoch_k = 14
            self.stoch_d = 3
            self.stoch_smooth = 3
            self.ema_period = 21
            self.position_max_candles = 5
            self.atr_multiple = 2.0
            self.profit_target_pct = 0.02  # 2% target
            self.oversold_threshold = 20
            self.overbought_threshold = 80
    
    def update_timeframe(self, timeframe: str):
        """Update the strategy timeframe and adjust parameters."""
        super().update_timeframe(timeframe)
        self._set_parameters_for_timeframe()
        logger.info(f"Updated VWAP + Stochastic Strategy parameters for {timeframe} timeframe")
    
    async def calculate_indicators(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Calculate indicators for the strategy."""
        return apply_vwap_stoch_indicators(
            df,
            stoch_k=self.stoch_k,
            stoch_d=self.stoch_d,
            stoch_smooth=self.stoch_smooth,
            ema_period=self.ema_period
        )
    
    async def check_signals(self, df: pd.DataFrame, position: Optional[Position] = None) -> Tuple[
        bool, bool, bool, List[str], List[str], List[str], List[str]
    ]:
        """Check for trading signals."""
        if df is None or len(df) < self.stoch_k + 5:
            logger.warning("Insufficient data for VWAP + Stochastic signal calculation")
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
        
        # Long signal conditions (ideal for scalping):
        # 1. Price crosses above VWAP (value zone) AND
        # 2. Stochastic is oversold OR Stochastic K crosses above D
        # 3. Price is above fast EMA for trend confirmation
        long_condition = (vwap_cross_up or above_vwap) and (stoch_oversold or stoch_cross_up) and price_above_ema
        
        # Short signal conditions:
        # 1. Price crosses below VWAP (value zone) AND
        # 2. Stochastic is overbought OR Stochastic K crosses below D
        # 3. Price is below fast EMA for trend confirmation
        short_condition = (vwap_cross_down or below_vwap) and (stoch_overbought or stoch_cross_down) and price_below_ema
        
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
        else:
            if not (vwap_cross_up or above_vwap):
                fail_reasons.append(f"Price not above VWAP: {last['close']:.2f} ≤ {last['vwap']:.2f}")
            if not (stoch_oversold or stoch_cross_up):
                fail_reasons.append(f"Stochastic not giving buy signal: K={last['stoch_k']:.2f}, D={last['stoch_d']:.2f}")
            if not price_above_ema:
                fail_reasons.append(f"Price below EMA({self.ema_period}): {last['close']:.2f} < {last['ema']:.2f}")
        
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
        else:
            if not (vwap_cross_down or below_vwap):
                fail_reasons.append(f"Price not below VWAP: {last['close']:.2f} ≥ {last['vwap']:.2f}")
            if not (stoch_overbought or stoch_cross_down):
                fail_reasons.append(f"Stochastic not giving sell signal: K={last['stoch_k']:.2f}, D={last['stoch_d']:.2f}")
            if not price_below_ema:
                fail_reasons.append(f"Price above EMA({self.ema_period}): {last['close']:.2f} > {last['ema']:.2f}")
        
        # Exit conditions - quick exits for scalping
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
        
        long_signal = long_condition and not position
        short_signal = short_condition and not position
        close_signal = close_condition and position
        
        return long_signal, short_signal, close_signal, long_signals, short_signals, close_signals, fail_reasons
    
    async def manage_position(self, df: pd.DataFrame, position: Position, balance: float) -> Tuple[
        Position, bool, List[str]
    ]:
        """Manages existing positions with VWAP-based trailing stops optimized for scalping."""
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
                
            # Consider stochastic signals for trailing profit activation
            stoch_extreme = (position.side == 'long' and last['stoch_k'] > 80) or \
                           (position.side == 'short' and last['stoch_k'] < 20)
                
            if current_profit_pct > trailing_profit_threshold:
                position.trailing_profit_activated = True
                
                # Allow only 20% pullback from highest profit (tighter for intraday VWAP strategy)
                max_pullback = position.highest_profit_pct * 0.2
                
                # Close position if price pulls back too much from the peak
                # Also consider stochastic extremes for earlier exits
                if position.trailing_profit_activated and (current_profit_pct < (position.highest_profit_pct - max_pullback) or stoch_extreme):
                    close_signals.append(f"Trailing profit: Locked in {current_profit_pct:.2f}% (max: {position.highest_profit_pct:.2f}%)")
                    if stoch_extreme:
                        close_signals.append(f"Stochastic extreme: K={last['stoch_k']:.2f}, D={last['stoch_d']:.2f}")
                    close_condition = True
        
        # VWAP-based trailing logic
        if position.side == 'long':
            # If we're above VWAP and making good profit, use VWAP as a stop
            if last['close'] > position.entry*1.003 and last['close'] > last['vwap'] and last['vwap'] > position.trailing_stop:
                position.trailing_stop = last['vwap']
                close_signals.append(f"Using VWAP as trailing stop: {last['vwap']:.2f}")
        
        elif position.side == 'short':
            # If we're below VWAP and making good profit, use VWAP as a stop
            if last['close'] < position.entry*0.997 and last['close'] < last['vwap'] and last['vwap'] < position.trailing_stop:
                position.trailing_stop = last['vwap']
                close_signals.append(f"Using VWAP as trailing stop: {last['vwap']:.2f}")
        
        # Also use ATR for tight trailing (0.5x ATR)
        trailing_atr_mult = 0.5
        position, trailing_signals = await self.update_trailing_stop(
            position, last['close'], last['atr'], trailing_atr_mult
        )
        close_signals.extend(trailing_signals)
        
        # Check for time-based exit
        time_exit, time_signals = await self.check_max_holding_time(position, self.position_max_candles)
        close_signals.extend(time_signals)
        if time_exit:
            close_condition = True
        
        # Consider stochastic signals for exit
        if ((position.side == 'long' and last['stoch_k'] > 80 and last['stoch_d'] > 80) or
            (position.side == 'short' and last['stoch_k'] < 20 and last['stoch_d'] < 20)):
            
            close_signals.append(f"Stochastic extreme: K={last['stoch_k']:.2f}, D={last['stoch_d']:.2f}")
            if position.open_candles > 3:  # Only force exit if we've been in position for 3+ candles
                close_condition = True
        
        return position, close_condition, close_signals 