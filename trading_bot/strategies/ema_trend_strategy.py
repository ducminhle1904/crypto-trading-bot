"""
EMA Trend Strategy implementation.
"""
import pandas as pd
from typing import Tuple, List, Optional, Dict, Any

from trading_bot.strategies.base_strategy import BaseStrategy
from trading_bot.models import Position
from trading_bot.utils.indicators import apply_standard_indicators
from trading_bot.config import logger


class EmaTrendStrategy(BaseStrategy):
    """
    EMA Trend Strategy uses EMA crossovers, trend slope, and RSI to generate signals.
    """
    
    def __init__(self, timeframe: str = "3m"):
        """Initialize the strategy with parameters."""
        super().__init__(name="ema_trend_strategy", timeframe=timeframe)
        
        # Set parameters based on timeframe
        self._set_parameters_for_timeframe()
        
        logger.info(f"Initialized EMA Trend Strategy with parameters: EMA Short={self.ema_short}, EMA Long={self.ema_long}, "
                   f"Trend Window={self.trend_window}, RSI Period={self.rsi_period}")
    
    def _set_parameters_for_timeframe(self):
        """Set strategy parameters based on the timeframe."""
        minutes = self.timeframe_minutes
        
        # Scale parameters based on timeframe
        if minutes <= 5:  # 1m to 5m - faster settings for scalping/day trading
            self.ema_short = 9  # Faster EMA for short-term moves
            self.ema_long = 21  # Shorter long EMA for day trading
            self.trend_window = 20  # Shorter trend window for faster signals
            self.rsi_period = 7   # More responsive RSI
            self.atr_period = 10
            self.momentum_period = 3
            self.position_max_candles = 8  # Quicker exits for scalping
            self.profit_target_pct = 0.005  # 0.5% target for scalping
            self.stop_loss_pct = 0.004  # Tighter 0.4% stop loss
            self.rsi_overbought = 75  # Tighter RSI bounds
            self.rsi_oversold = 25
            
        elif minutes <= 60:  # 15m to 1h - medium settings
            self.ema_short = 13
            self.ema_long = 34
            self.trend_window = 24
            self.rsi_period = 10
            self.atr_period = 14
            self.momentum_period = 4
            self.position_max_candles = 6  # Still fast exits for day trading
            self.profit_target_pct = 0.008  # 0.8% - still quick profit target
            self.stop_loss_pct = 0.006  # Tighter stop than original
            self.rsi_overbought = 75
            self.rsi_oversold = 25
            
        else:  # 4h, daily - slower settings
            self.ema_short = 18
            self.ema_long = 42
            self.trend_window = 20
            self.rsi_period = 14
            self.atr_period = 14
            self.momentum_period = 5
            self.position_max_candles = 5
            self.profit_target_pct = 0.025  # 2.5%
            self.stop_loss_pct = 0.015  # 1.5%
            self.rsi_overbought = 70
            self.rsi_oversold = 30
    
    def update_timeframe(self, timeframe: str):
        """Update the strategy timeframe and adjust parameters."""
        super().update_timeframe(timeframe)
        self._set_parameters_for_timeframe()
        logger.info(f"Updated EMA Trend Strategy parameters for {timeframe} timeframe")
    
    async def calculate_indicators(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Calculate indicators for the strategy."""
        return apply_standard_indicators(
            df, 
            ema_short=self.ema_short,
            ema_long=self.ema_long,
            trend_window=self.trend_window,
            rsi_period=self.rsi_period,
            atr_period=self.atr_period,
            momentum_period=self.momentum_period
        )
    
    async def check_signals(self, df: pd.DataFrame, position: Optional[Position] = None) -> Tuple[
        bool, bool, bool, List[str], List[str], List[str], List[str]
    ]:
        """Check for trading signals."""
        if df is None or len(df) < self.ema_long + self.trend_window:
            logger.warning("Insufficient data for signal calculation")
            return False, False, False, [], [], [], ["Insufficient data"]
        
        last = df.iloc[-1]
        prev = df.iloc[-2] if len(df) > 2 else last
        
        long_signals = []
        short_signals = []
        close_signals = []
        long_fail_reasons = []
        short_fail_reasons = []
        
        # Original trend and EMA conditions
        trend_up = last['trend_slope'] > 0
        trend_down = last['trend_slope'] < 0
        ema_bullish = last['ema_short'] > last['ema_long'] and last['close'] > last['ema_short']
        ema_bearish = last['ema_short'] < last['ema_long'] and last['close'] < last['ema_short']
        
        # Confirmation indicators
        momentum_bullish = last['momentum'] > 0
        momentum_bearish = last['momentum'] < 0
        rsi_oversold = last['fast_rsi'] < self.rsi_oversold  # Use adaptive RSI levels
        rsi_overbought = last['fast_rsi'] > self.rsi_overbought
        
        # Scalping-specific conditions
        ema_crossover_bull = last['ema_short'] > last['ema_long'] and prev['ema_short'] <= prev['ema_long']
        ema_crossover_bear = last['ema_short'] < last['ema_long'] and prev['ema_short'] >= prev['ema_long']
        
        # Dynamic risk based on volatility
        volatility_factor = last['atr'] / last['close'] * 100  # ATR as percentage of price
        
        # Enhanced signal conditions for day trading
        # For day trading, make the EMA crossover more important for faster entries
        long_confirmation = momentum_bullish or rsi_oversold or ema_crossover_bull
        long_condition = (trend_up and ema_bullish and long_confirmation)
        
        short_confirmation = momentum_bearish or rsi_overbought or ema_crossover_bear
        short_condition = (trend_down and ema_bearish and short_confirmation)
        
        if long_condition:
            long_signals.append("Trendline slope positive")
            long_signals.append(f"Slope: {last['trend_slope']:.2f}")
            long_signals.append(f"EMA({self.ema_short}) > EMA({self.ema_long}) and Price > EMA({self.ema_short})")
            long_signals.append(f"EMA({self.ema_short}): {last['ema_short']:.2f}, EMA({self.ema_long}): {last['ema_long']:.2f}")
            if momentum_bullish:
                long_signals.append(f"Bullish momentum: {last['momentum']:.2f}%")
            if rsi_oversold:
                long_signals.append(f"Oversold RSI: {last['fast_rsi']:.2f}")
            if ema_crossover_bull:
                long_signals.append(f"EMA crossover: Short EMA crossed above Long EMA")
        else:
            if not trend_up:
                long_fail_reasons.append(f"No trend up: Slope ({last['trend_slope']:.2f})")
            if not ema_bullish:
                long_fail_reasons.append(f"Not bullish: Price ({last['close']:.2f}), EMA({self.ema_short}) ({last['ema_short']:.2f}), EMA({self.ema_long}) ({last['ema_long']:.2f})")
            if not long_confirmation:
                long_fail_reasons.append(f"No confirmation: Momentum ({last['momentum']:.2f}%), RSI ({last['fast_rsi']:.2f})")
        
        if short_condition:
            short_signals.append("Trendline slope negative")
            short_signals.append(f"Slope: {last['trend_slope']:.2f}")
            short_signals.append(f"EMA({self.ema_short}) < EMA({self.ema_long}) and Price < EMA({self.ema_short})")
            short_signals.append(f"EMA({self.ema_short}): {last['ema_short']:.2f}, EMA({self.ema_long}): {last['ema_long']:.2f}")
            if momentum_bearish:
                short_signals.append(f"Bearish momentum: {last['momentum']:.2f}%")
            if rsi_overbought:
                short_signals.append(f"Overbought RSI: {last['fast_rsi']:.2f}")
            if ema_crossover_bear:
                short_signals.append(f"EMA crossover: Short EMA crossed below Long EMA")
        else:
            if not trend_down:
                short_fail_reasons.append(f"No trend down: Slope ({last['trend_slope']:.2f})")
            if not ema_bearish:
                short_fail_reasons.append(f"Not bearish: Price ({last['close']:.2f}), EMA({self.ema_short}) ({last['ema_short']:.2f}), EMA({self.ema_long}) ({last['ema_long']:.2f})")
            if not short_confirmation:
                short_fail_reasons.append(f"No confirmation: Momentum ({last['momentum']:.2f}%), RSI ({last['fast_rsi']:.2f})")
        
        close_condition = False
        if position:
            # Dynamic stop-loss and take-profit based on volatility - tighter for scalping
            stop_loss_pct = max(self.stop_loss_pct * 0.67, min(self.stop_loss_pct, volatility_factor * 0.5))  # Even tighter stops for day trading
            take_profit_pct = max(self.profit_target_pct * 0.67, min(self.profit_target_pct * 1.2, volatility_factor * 1.0))  # Quicker profit taking
            
            if position.side == 'long':
                # Check if trailing stop exists, otherwise use initial stop loss
                if position.trailing_stop:
                    stop_loss = position.trailing_stop
                else:
                    stop_loss = position.entry * (1 - stop_loss_pct)
                    
                take_profit = position.entry * (1 + take_profit_pct)
                # Quicker exits for day trading - include EMA crossovers
                reversal = trend_down or ema_bearish or ema_crossover_bear
                
                if last['close'] <= stop_loss:
                    close_signals.append("Stop Loss Hit")
                    close_condition = True
                elif last['close'] >= take_profit:
                    close_signals.append("Take Profit Hit")
                    close_condition = True
                elif reversal:
                    close_signals.append("Bearish reversal detected")
                    close_signals.append(f"Slope: {last['trend_slope']:.2f}, EMA({self.ema_short}): {last['ema_short']:.2f}, EMA({self.ema_long}): {last['ema_long']:.2f}")
                    close_condition = True
                    
            elif position.side == 'short':
                # Check if trailing stop exists, otherwise use initial stop loss
                if position.trailing_stop:
                    stop_loss = position.trailing_stop
                else:
                    stop_loss = position.entry * (1 + stop_loss_pct)
                    
                take_profit = position.entry * (1 - take_profit_pct)
                # Quicker exits for day trading - include EMA crossovers
                reversal = trend_up or ema_bullish or ema_crossover_bull
                
                if last['close'] >= stop_loss:
                    close_signals.append("Stop Loss Hit")
                    close_condition = True
                elif last['close'] <= take_profit:
                    close_signals.append("Take Profit Hit")
                    close_condition = True
                elif reversal:
                    close_signals.append("Bullish reversal detected")
                    close_signals.append(f"Slope: {last['trend_slope']:.2f}, EMA({self.ema_short}): {last['ema_short']:.2f}, EMA({self.ema_long}): {last['ema_long']:.2f}")
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
        """Manages existing positions with trailing stops and dynamic exits."""
        if not position:
            return position, False, []
        
        last = df.iloc[-1]
        close_signals = []
        close_condition = False
        
        # Initialize if first check of this position
        if not position.trailing_stop:
            # Calculate initial stop-loss based on volatility - tighter for day trading
            volatility_pct = last['atr'] / last['close'] * 100 if 'atr' in last else 0.5
            stop_loss_pct = max(self.stop_loss_pct * 0.67, min(self.stop_loss_pct, volatility_pct * 0.5))
            
            if position.side == 'long':
                position.trailing_stop = position.entry * (1 - stop_loss_pct)
            else:
                position.trailing_stop = position.entry * (1 + stop_loss_pct)
            
            position.open_candles = 0
        else:
            # Increment candle counter
            position.open_candles += 1
        
        # Implement quick trailing stop activation - even more aggressive for day trading
        breakeven_threshold = 0.002  # For day trading, move to breakeven at just 0.2%
        if self.timeframe_minutes > 60:
            breakeven_threshold = 0.004  # 0.4% for 4h+
        elif self.timeframe_minutes > 5:
            breakeven_threshold = 0.003  # 0.3% for 15m-1h
            
        trailing_threshold = breakeven_threshold * 1.5  # 1.5x the breakeven for trailing (more aggressive)
            
        if position.side == 'long':
            breakeven_level = position.entry * (1 + breakeven_threshold)
            
            # Move to breakeven after small profit
            if last['close'] > breakeven_level and position.trailing_stop < position.entry:
                position.trailing_stop = position.entry
                close_signals.append("Moved stop-loss to breakeven")
            # More aggressive trailing as profit increases
            elif last['close'] > position.entry * (1 + trailing_threshold):
                potential_stop = max(position.entry, last['close'] * (1 - breakeven_threshold * 0.75))  # Even tighter trailing
                if potential_stop > position.trailing_stop:
                    position.trailing_stop = potential_stop
                    close_signals.append(f"Updated trailing stop to {potential_stop:.2f}")
        
        elif position.side == 'short':
            breakeven_level = position.entry * (1 - breakeven_threshold)
            
            # Move to breakeven after small profit
            if last['close'] < breakeven_level and position.trailing_stop > position.entry:
                position.trailing_stop = position.entry
                close_signals.append("Moved stop-loss to breakeven")
            # More aggressive trailing as profit increases
            elif last['close'] < position.entry * (1 - trailing_threshold):
                potential_stop = min(position.entry, last['close'] * (1 + breakeven_threshold * 0.75))  # Even tighter trailing
                if potential_stop < position.trailing_stop:
                    position.trailing_stop = potential_stop
                    close_signals.append(f"Updated trailing stop to {potential_stop:.2f}")
        
        # Quicker time-based exit for scalping
        if position.open_candles > self.position_max_candles:
            close_signals.append(f"Position time limit reached ({position.open_candles} candles)")
            close_condition = True
        
        return position, close_condition, close_signals 