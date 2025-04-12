"""
Dow Theory with EMA Strategy implementation.

This strategy combines principles of Dow Theory with EMA34 and EMA89 indicators
to identify trend confirmations and reversals.
"""
import pandas as pd
from typing import Tuple, List, Optional, Dict, Any

from trading_bot.strategies.base_strategy import BaseStrategy
from trading_bot.models import Position
from trading_bot.utils.indicators import calculate_ema, calculate_atr, calculate_rsi, calculate_momentum
from trading_bot.config import logger


class DowEmaStrategy(BaseStrategy):
    """
    Dow Theory with EMA Strategy combines classical Dow Theory principles 
    with EMA34 and EMA89 to generate signals.
    
    Dow Theory principles used:
    1. The market discounts everything
    2. The market has three trends (primary, secondary, minor)
    3. Primary trends have three phases (accumulation, public participation, distribution)
    4. Indices/averages must confirm each other
    5. Volume must confirm the trend
    6. A trend is assumed to be in effect until definite signals of reversal
    """
    
    def __init__(self, timeframe: str = "4h", use_trailing_profit: bool = True):
        """Initialize the strategy with parameters."""
        super().__init__(name="dow_ema_strategy", timeframe=timeframe)
        
        # Whether to use trailing profit or fixed take profit
        self.use_trailing_profit = use_trailing_profit
        
        # Set parameters based on timeframe
        self._set_parameters_for_timeframe()
        
        logger.info(f"Initialized Dow EMA Strategy with parameters: EMA1={self.ema1}, EMA2={self.ema2}, "
                   f"RSI Period={self.rsi_period}, Trailing Profit: {'Enabled' if use_trailing_profit else 'Disabled'}")
    
    def _set_parameters_for_timeframe(self):
        """Set strategy parameters based on the timeframe."""
        minutes = self.timeframe_minutes
        
        # Fixed EMA periods according to request
        self.ema1 = 34  # First EMA (faster)
        self.ema2 = 89  # Second EMA (slower)
        
        # Scale other parameters based on timeframe
        if minutes <= 15:  # 1m to 15m - short-term trading
            self.trend_window = 14
            self.rsi_period = 7
            self.atr_period = 10
            self.momentum_period = 3
            self.volume_ma_period = 20
            self.position_max_candles = 10
            self.profit_target_pct = 0.01  # 1% target for shorter timeframes
            self.stop_loss_pct = 0.005     # 0.5% stop loss
            self.rsi_overbought = 75
            self.rsi_oversold = 25
            
        elif minutes <= 60:  # 15m to 1h - medium settings (swing trading)
            self.trend_window = 20
            self.rsi_period = 10
            self.atr_period = 14
            self.momentum_period = 5
            self.volume_ma_period = 30
            self.position_max_candles = 12
            self.profit_target_pct = 0.02  # 2% profit target
            self.stop_loss_pct = 0.01      # 1% stop loss
            self.rsi_overbought = 70
            self.rsi_oversold = 30
            
        else:  # 4h, daily - longer-term settings
            self.trend_window = 30
            self.rsi_period = 14
            self.atr_period = 14
            self.momentum_period = 8
            self.volume_ma_period = 50
            self.position_max_candles = 15
            self.profit_target_pct = 0.03  # 3% target
            self.stop_loss_pct = 0.02      # 2% stop loss
            self.rsi_overbought = 70
            self.rsi_oversold = 30
    
    def update_timeframe(self, timeframe: str):
        """Update the strategy timeframe and adjust parameters."""
        super().update_timeframe(timeframe)
        self._set_parameters_for_timeframe()
        logger.info(f"Updated Dow EMA Strategy parameters for {timeframe} timeframe")
    
    async def calculate_indicators(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Calculate indicators for the strategy."""
        if df is None or len(df) < self.ema2 + 10:
            return None
            
        df = df.copy()
        
        # Calculate EMAs
        df['ema34'] = calculate_ema(df['close'], self.ema1)
        df['ema89'] = calculate_ema(df['close'], self.ema2)
        
        # Volume indicators for Dow Theory volume confirmation
        if 'volume' in df.columns:
            df['volume_ma'] = df['volume'].rolling(window=self.volume_ma_period).mean()
            df['volume_ratio'] = df['volume'] / df['volume_ma']
        
        # Calculate RSI
        df['rsi'] = calculate_rsi(df['close'], self.rsi_period)
        
        # Calculate Momentum
        df['momentum'] = calculate_momentum(df['close'], self.momentum_period)
        
        # Calculate ATR for volatility
        df['atr'] = calculate_atr(df, self.atr_period)
        
        # Calculate higher highs, lower lows for Dow Theory trend analysis
        # Look back 5 bars
        lookback = 5
        
        # Initialize columns
        df['higher_high'] = False
        df['higher_low'] = False
        df['lower_high'] = False
        df['lower_low'] = False
        
        # Calculate higher highs/lows and lower highs/lows
        for i in range(lookback, len(df)):
            prev_high = df['high'].iloc[i-lookback:i].max()
            prev_low = df['low'].iloc[i-lookback:i].min()
            current_high = df['high'].iloc[i]
            current_low = df['low'].iloc[i]
            
            df.loc[df.index[i], 'higher_high'] = current_high > prev_high
            df.loc[df.index[i], 'lower_low'] = current_low < prev_low
            df.loc[df.index[i], 'higher_low'] = current_low > prev_low
            df.loc[df.index[i], 'lower_high'] = current_high < prev_high
        
        # Additional Dow Theory indicators - moving average of highs and lows
        df['highs_ma'] = df['high'].rolling(window=self.trend_window).mean()
        df['lows_ma'] = df['low'].rolling(window=self.trend_window).mean()
        
        return df
    
    async def check_signals(self, df: pd.DataFrame, position: Optional[Position] = None) -> Tuple[
        bool, bool, bool, List[str], List[str], List[str], List[str]
    ]:
        """Check for trading signals using Dow Theory principles and EMAs."""
        if df is None or len(df) < self.ema2 + self.trend_window:
            logger.warning("Insufficient data for signal calculation")
            return False, False, False, [], [], [], ["Insufficient data"]
        
        last = df.iloc[-1]
        prev = df.iloc[-2:] if len(df) > 2 else df.iloc[-len(df):]
        
        long_signals = []
        short_signals = []
        close_signals = []
        long_fail_reasons = []
        short_fail_reasons = []
        
        # --- Dow Theory with EMA signals ---
        
        # 1. Trend identification using EMAs
        uptrend = last['ema34'] > last['ema89']
        downtrend = last['ema34'] < last['ema89']
        
        # 2. EMA cross signals
        ema_cross_up = last['ema34'] > last['ema89'] and prev['ema34'].iloc[-1] <= prev['ema34'].iloc[-1]
        ema_cross_down = last['ema34'] < last['ema89'] and prev['ema34'].iloc[-1] >= prev['ema34'].iloc[-1]
        
        # 3. Price position relative to EMAs
        price_above_emas = last['close'] > last['ema34'] and last['close'] > last['ema89']
        price_below_emas = last['close'] < last['ema34'] and last['close'] < last['ema89']
        
        # 4. Dow Theory higher highs/lows pattern for uptrend
        higher_high_pattern = prev['higher_high'].any() and prev['higher_low'].any()
        
        # 5. Dow Theory lower highs/lows pattern for downtrend
        lower_low_pattern = prev['lower_low'].any() and prev['lower_high'].any()
        
        # 6. Volume confirmation (if volume data is available)
        volume_confirms_up = df['volume_ratio'].iloc[-1] > 1.0 if 'volume_ratio' in df.columns else True
        volume_confirms_down = df['volume_ratio'].iloc[-1] > 1.0 if 'volume_ratio' in df.columns else True
        
        # 7. RSI conditions
        rsi_oversold = last['rsi'] < self.rsi_oversold
        rsi_overbought = last['rsi'] > self.rsi_overbought
        
        # --- Combine signals for entry decisions ---
        
        # Long signal conditions based on Dow Theory
        long_condition = (
            (uptrend or ema_cross_up) and
            price_above_emas and
            (higher_high_pattern or rsi_oversold) and
            volume_confirms_up
        )
        
        # Short signal conditions based on Dow Theory
        short_condition = (
            (downtrend or ema_cross_down) and
            price_below_emas and
            (lower_low_pattern or rsi_overbought) and
            volume_confirms_down
        )
        
        # Record signal reasons for long entries
        if long_condition:
            if uptrend:
                long_signals.append("Uptrend: EMA34 above EMA89")
                long_signals.append(f"EMA34: {last['ema34']:.2f}, EMA89: {last['ema89']:.2f}")
            if ema_cross_up:
                long_signals.append("Bullish EMA crossover: EMA34 crossed above EMA89")
            if price_above_emas:
                long_signals.append(f"Price ({last['close']:.2f}) above both EMAs")
            if higher_high_pattern:
                long_signals.append("Dow Theory confirmed: Higher highs and higher lows")
            if rsi_oversold:
                long_signals.append(f"Oversold RSI: {last['rsi']:.2f}")
            if volume_confirms_up and 'volume_ratio' in df.columns:
                long_signals.append(f"Volume confirms trend: {last['volume_ratio']:.2f}x average")
        else:
            if not (uptrend or ema_cross_up):
                long_fail_reasons.append("No uptrend or bullish crossover")
            if not price_above_emas:
                long_fail_reasons.append(f"Price not above EMAs: Close {last['close']:.2f}, EMA34 {last['ema34']:.2f}, EMA89 {last['ema89']:.2f}")
            if not (higher_high_pattern or rsi_oversold):
                long_fail_reasons.append("No higher highs/lows pattern or oversold condition")
            if not volume_confirms_up and 'volume_ratio' in df.columns:
                long_fail_reasons.append(f"Volume doesn't confirm: {last['volume_ratio']:.2f}x average")
        
        # Record signal reasons for short entries
        if short_condition:
            if downtrend:
                short_signals.append("Downtrend: EMA34 below EMA89")
                short_signals.append(f"EMA34: {last['ema34']:.2f}, EMA89: {last['ema89']:.2f}")
            if ema_cross_down:
                short_signals.append("Bearish EMA crossover: EMA34 crossed below EMA89")
            if price_below_emas:
                short_signals.append(f"Price ({last['close']:.2f}) below both EMAs")
            if lower_low_pattern:
                short_signals.append("Dow Theory confirmed: Lower lows and lower highs")
            if rsi_overbought:
                short_signals.append(f"Overbought RSI: {last['rsi']:.2f}")
            if volume_confirms_down and 'volume_ratio' in df.columns:
                short_signals.append(f"Volume confirms trend: {last['volume_ratio']:.2f}x average")
        else:
            if not (downtrend or ema_cross_down):
                short_fail_reasons.append("No downtrend or bearish crossover")
            if not price_below_emas:
                short_fail_reasons.append(f"Price not below EMAs: Close {last['close']:.2f}, EMA34 {last['ema34']:.2f}, EMA89 {last['ema89']:.2f}")
            if not (lower_low_pattern or rsi_overbought):
                short_fail_reasons.append("No lower lows/highs pattern or overbought condition")
            if not volume_confirms_down and 'volume_ratio' in df.columns:
                short_fail_reasons.append(f"Volume doesn't confirm: {last['volume_ratio']:.2f}x average")
        
        # Close position conditions
        close_condition = False
        if position:
            volatility_factor = last['atr'] / last['close'] * 100
            stop_loss_pct = max(self.stop_loss_pct * 0.9, min(self.stop_loss_pct * 1.1, volatility_factor * 0.75))
            take_profit_pct = max(self.profit_target_pct * 0.9, min(self.profit_target_pct * 1.3, volatility_factor * 1.5))
            
            if position.side == 'long':
                # Check for exit signals
                if position.trailing_stop:
                    stop_loss = position.trailing_stop
                else:
                    stop_loss = position.entry * (1 - stop_loss_pct)
                    
                take_profit = position.entry * (1 + take_profit_pct)
                
                # Exit signals based on Dow Theory reversals
                reversal = (downtrend or ema_cross_down or lower_low_pattern or 
                           (last['close'] < last['ema34'] and prev['close'].iloc[-1] >= prev['ema34'].iloc[-1]))
                
                if last['close'] <= stop_loss:
                    close_signals.append(f"Stop Loss Hit: {last['close']:.2f} <= {stop_loss:.2f}")
                    close_condition = True
                elif last['close'] >= take_profit:
                    close_signals.append(f"Take Profit Hit: {last['close']:.2f} >= {take_profit:.2f}")
                    close_condition = True
                elif reversal:
                    close_signals.append("Dow Theory trend reversal detected")
                    if downtrend:
                        close_signals.append("EMA34 crossed below EMA89")
                    if lower_low_pattern:
                        close_signals.append("Lower lows and lower highs pattern")
                    if last['close'] < last['ema34'] and prev['close'].iloc[-1] >= prev['ema34'].iloc[-1]:
                        close_signals.append("Price crossed below EMA34")
                    close_condition = True
                    
            elif position.side == 'short':
                # Check for exit signals
                if position.trailing_stop:
                    stop_loss = position.trailing_stop
                else:
                    stop_loss = position.entry * (1 + stop_loss_pct)
                    
                take_profit = position.entry * (1 - take_profit_pct)
                
                # Exit signals based on Dow Theory reversals
                reversal = (uptrend or ema_cross_up or higher_high_pattern or 
                           (last['close'] > last['ema34'] and prev['close'].iloc[-1] <= prev['ema34'].iloc[-1]))
                
                if last['close'] >= stop_loss:
                    close_signals.append(f"Stop Loss Hit: {last['close']:.2f} >= {stop_loss:.2f}")
                    close_condition = True
                elif last['close'] <= take_profit:
                    close_signals.append(f"Take Profit Hit: {last['close']:.2f} <= {take_profit:.2f}")
                    close_condition = True
                elif reversal:
                    close_signals.append("Dow Theory trend reversal detected")
                    if uptrend:
                        close_signals.append("EMA34 crossed above EMA89")
                    if higher_high_pattern:
                        close_signals.append("Higher highs and higher lows pattern")
                    if last['close'] > last['ema34'] and prev['close'].iloc[-1] <= prev['ema34'].iloc[-1]:
                        close_signals.append("Price crossed above EMA34")
                    close_condition = True
        
        long_signal = long_condition and not position
        short_signal = short_condition and not position
        close_signal = close_condition and position
        
        if not long_signal and not position:
            logger.info(f"No long signal: " + "; ".join(long_fail_reasons))
        if not short_signal and not position:
            logger.info(f"No short signal: " + "; ".join(short_fail_reasons))
        if not close_signal and position:
            logger.info(f"No close signal for {position.side}")
        
        return long_signal, short_signal, close_signal, long_signals, short_signals, close_signals, long_fail_reasons
    
    async def manage_position(self, df: pd.DataFrame, position: Position, balance: float) -> Tuple[
        Position, bool, List[str]
    ]:
        """Manages existing positions with trailing stops based on Dow Theory principles."""
        if not position:
            return position, False, []
        
        last = df.iloc[-1]
        close_signals = []
        close_condition = False
        
        # Initialize if first check of this position
        if not position.trailing_stop:
            # Calculate initial stop-loss based on volatility
            volatility_pct = last['atr'] / last['close'] * 100 if 'atr' in last else 1.0
            stop_loss_pct = max(self.stop_loss_pct * 0.9, min(self.stop_loss_pct * 1.1, volatility_pct * 0.75))
            atr_multiple = stop_loss_pct * 100 / volatility_pct if volatility_pct > 0 else self.stop_loss_pct / 0.01
            
            position = await self.initialize_stop_loss(position, last['close'], last['atr'], atr_multiple)
            
            # Initialize trailing profit tracking
            if self.use_trailing_profit:
                position.highest_profit_pct = 0.0
                position.trailing_profit_activated = False
            
        # Increment the candles held counter
        if not hasattr(position, 'open_candles'):
            position.open_candles = 0
        position.open_candles += 1
            
        # Use common method to move to breakeven
        min_profit_pct = 0.005  # 0.5% minimum profit before moving to breakeven
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
                
            if current_profit_pct > trailing_profit_threshold:
                position.trailing_profit_activated = True
                # Allow only 30% pullback from highest profit
                max_pullback = position.highest_profit_pct * 0.3
                
                # Close position if price pulls back too much from the peak
                if position.trailing_profit_activated and current_profit_pct < (position.highest_profit_pct - max_pullback):
                    close_signals.append(f"Trailing profit: Locked in {current_profit_pct:.2f}% (max: {position.highest_profit_pct:.2f}%)")
                    close_condition = True
        
        # Update trailing stop with custom logic for Dow Theory strategy
        if position.side == 'long':
            # For longs, trail the stop higher when price moves up
            # Using Dow Theory principles, we use higher lows as potential stop levels
            trailing_atr_mult = 0.5  # Tighter trailing factor
            position, trailing_signals = await self.update_trailing_stop(
                position, last['close'], last['atr'], trailing_atr_mult
            )
            close_signals.extend(trailing_signals)
            
            # Use EMA34 as trailing stop if price has moved significantly above it
            if last['close'] > last['ema34'] * 1.01:  # 1% above EMA34
                ema_based_stop = last['ema34'] * 0.995  # Just below EMA34
                if ema_based_stop > position.trailing_stop:
                    position.trailing_stop = ema_based_stop
                    close_signals.append(f"Updated stop using EMA34: {position.trailing_stop:.2f}")
                
        else:  # short position
            # For shorts, trail the stop lower when price moves down
            trailing_atr_mult = 0.5  # Tighter trailing factor
            position, trailing_signals = await self.update_trailing_stop(
                position, last['close'], last['atr'], trailing_atr_mult
            )
            close_signals.extend(trailing_signals)
            
            # Use EMA34 as trailing stop if price has moved significantly below it
            if last['close'] < last['ema34'] * 0.99:  # 1% below EMA34
                ema_based_stop = last['ema34'] * 1.005  # Just above EMA34
                if ema_based_stop < position.trailing_stop:
                    position.trailing_stop = ema_based_stop
                    close_signals.append(f"Updated stop using EMA34: {position.trailing_stop:.2f}")
        
        # Check for time-based exit (max holding period)
        time_exit, time_signals = await self.check_max_holding_time(position, self.position_max_candles)
        close_signals.extend(time_signals)
        if time_exit:
            close_condition = True
        
        return position, close_condition, close_signals 