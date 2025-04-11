"""
Position management utilities with enhanced trailing profit features.
"""
import pandas as pd
from typing import Tuple, List, Dict, Any, Optional
from datetime import datetime

from trading_bot.models import Position
from trading_bot.config import logger


class TrailingProfitManager:
    """
    Enhanced position management with trailing profit features.
    This allows profitable positions to stay open longer when the trend is still favorable.
    """
    
    def __init__(self, enable_trailing_profit: bool = True):
        """Initialize the trailing profit manager."""
        self.enable_trailing_profit = enable_trailing_profit
        
        # Configurable parameters
        self.initial_target_reached = False  # Flag to indicate if initial profit target was reached
        self.breakeven_pct = 0.003  # 0.3% profit to move stop to breakeven
        self.initial_target_pct = 0.005  # 0.5% initial profit target
        self.tighten_stop_increment = 0.002  # 0.2% increment to tighten stop as profit grows
        self.profit_lockout_levels = [0.005, 0.01, 0.02, 0.03, 0.05]  # Profit levels where we lock in gains (0.5%, 1%, 2%, 3%, 5%)
    
    def check_trend(self, df: pd.DataFrame, position: Position) -> Tuple[bool, str]:
        """
        Check if the trend is still favorable for the position.
        Returns (trend_favorable, reason)
        """
        last = df.iloc[-1]
        
        # For long positions
        if position.side == 'long':
            # Check various trend indicators if available
            if 'trend_slope' in last:
                if last['trend_slope'] <= 0:
                    return False, f"Trend slope turned negative: {last['trend_slope']:.4f}"
            
            if 'ema_short' in last and 'ema_long' in last:
                if last['ema_short'] < last['ema_long']:
                    return False, f"EMA bearish cross: EMA short {last['ema_short']:.2f} < EMA long {last['ema_long']:.2f}"
            
            if 'macd' in last and 'macd_signal' in last:
                if last['macd'] < last['macd_signal']:
                    return False, f"MACD bearish cross: MACD {last['macd']:.4f} < Signal {last['macd_signal']:.4f}"
                    
            if 'rsi' in last:
                if last['rsi'] > 70:
                    return False, f"RSI overbought: {last['rsi']:.2f}"
                    
            # If we have Bollinger Bands and price is above upper band
            if 'bb_upper' in last and last['close'] > last['bb_upper']:
                # This is not automatically negative - price can ride the upper band in strong trends
                # But we'll flag it if price is far above the upper band (extended)
                extension = (last['close'] - last['bb_upper']) / last['bb_upper'] * 100
                if extension > 1.0:  # More than 1% above the upper band
                    return False, f"Price extended above upper Bollinger Band: {extension:.2f}%"
            
            # Default - trend is still good for longs
            return True, "Trend still bullish"
            
        # For short positions    
        else:  
            # Check various trend indicators if available
            if 'trend_slope' in last:
                if last['trend_slope'] >= 0:
                    return False, f"Trend slope turned positive: {last['trend_slope']:.4f}"
            
            if 'ema_short' in last and 'ema_long' in last:
                if last['ema_short'] > last['ema_long']:
                    return False, f"EMA bullish cross: EMA short {last['ema_short']:.2f} > EMA long {last['ema_long']:.2f}"
            
            if 'macd' in last and 'macd_signal' in last:
                if last['macd'] > last['macd_signal']:
                    return False, f"MACD bullish cross: MACD {last['macd']:.4f} > Signal {last['macd_signal']:.4f}"
                    
            if 'rsi' in last:
                if last['rsi'] < 30:
                    return False, f"RSI oversold: {last['rsi']:.2f}"
                    
            # If we have Bollinger Bands and price is below lower band
            if 'bb_lower' in last and last['close'] < last['bb_lower']:
                # This is not automatically negative - price can ride the lower band in strong trends
                # But we'll flag it if price is far below the lower band (extended)
                extension = (last['bb_lower'] - last['close']) / last['bb_lower'] * 100
                if extension > 1.0:  # More than 1% below the lower band
                    return False, f"Price extended below lower Bollinger Band: {extension:.2f}%"
            
            # Default - trend is still good for shorts
            return True, "Trend still bearish"
    
    def manage_trailing_profit(self, df: pd.DataFrame, position: Position) -> Tuple[Position, bool, List[str]]:
        """
        Manage position with trailing profit logic.
        
        Returns:
            Tuple containing:
            - updated position object
            - close_signal: True if position should be closed
            - close_signals: List of reasons for closing
        """
        if not position or not self.enable_trailing_profit:
            return position, False, []
        
        if not hasattr(position, 'trailing_profit_activated'):
            position.trailing_profit_activated = False
            position.highest_profit_pct = 0.0
            position.profit_lockout_level = 0
        
        last = df.iloc[-1]
        current_price = last['close']
        close_signals = []
        close_condition = False
        
        # Calculate current profit percentage
        if position.side == 'long':
            current_profit_pct = (current_price / position.entry - 1) * 100
        else:  # short
            current_profit_pct = (position.entry / current_price - 1) * 100
        
        # Update highest profit seen
        if current_profit_pct > position.highest_profit_pct:
            position.highest_profit_pct = current_profit_pct
        
        # Check if we've reached initial profit target
        if not position.trailing_profit_activated and current_profit_pct >= self.initial_target_pct:
            position.trailing_profit_activated = True
            close_signals.append(f"Trailing profit activated at {current_profit_pct:.2f}% profit")
        
        # If trailing profit is activated, check if trend is still favorable
        if position.trailing_profit_activated:
            trend_favorable, trend_reason = self.check_trend(df, position)
            
            # If trend is no longer favorable, close the position
            if not trend_favorable:
                close_signals.append(f"Closing position - trend no longer favorable: {trend_reason}")
                close_condition = True
                return position, close_condition, close_signals
            
            # Update trailing stop based on profit levels
            self.update_trailing_stop(position, current_price, current_profit_pct, close_signals)
            
            # Check if we need to lock in more profit
            for i, level in enumerate(self.profit_lockout_levels):
                if current_profit_pct >= level and position.profit_lockout_level < i + 1:
                    position.profit_lockout_level = i + 1
                    
                    # Tighten stop more aggressively at higher profit levels
                    if position.side == 'long':
                        new_stop = current_price * (1 - level * 0.3)  # Lock in 70% of this profit level
                        if new_stop > position.trailing_stop:
                            position.trailing_stop = new_stop
                            close_signals.append(f"Locked in profit at {level:.1f}% level - trailing stop raised to {new_stop:.2f}")
                    else:  # short
                        new_stop = current_price * (1 + level * 0.3)  # Lock in 70% of this profit level
                        if new_stop < position.trailing_stop:
                            position.trailing_stop = new_stop
                            close_signals.append(f"Locked in profit at {level:.1f}% level - trailing stop lowered to {new_stop:.2f}")
        
        # Check if trailing stop has been hit
        if position.trailing_stop:
            if (position.side == 'long' and current_price <= position.trailing_stop) or \
               (position.side == 'short' and current_price >= position.trailing_stop):
                close_signals.append(f"Trailing stop hit at {position.trailing_stop:.2f}")
                close_condition = True
        
        return position, close_condition, close_signals
    
    def update_trailing_stop(self, position: Position, current_price: float, current_profit_pct: float, close_signals: List[str]):
        """Update trailing stop based on profit percentage."""
        # First, make sure we have a trailing stop
        if not position.trailing_stop:
            # Set initial trailing stop at a conservative level
            if position.side == 'long':
                position.trailing_stop = position.entry * 0.99  # 1% below entry
            else:
                position.trailing_stop = position.entry * 1.01  # 1% above entry
            close_signals.append(f"Set initial trailing stop at {position.trailing_stop:.2f}")
            return
        
        # For long positions - only move stop up
        if position.side == 'long':
            # Move to breakeven once we have a small profit
            if current_profit_pct >= self.breakeven_pct and position.trailing_stop < position.entry:
                position.trailing_stop = position.entry
                close_signals.append(f"Moved stop to breakeven at {position.entry:.2f}")
                
            # In profit with trailing stop activated - keep tightening
            elif position.trailing_profit_activated:
                # Calculate new potential stop level
                # As profit increases, trailing stop gets tighter
                tightness_factor = min(0.5, 0.2 + (current_profit_pct * 0.05))  # More profit = tighter trailing
                distance_pct = self.tighten_stop_increment * (1 - tightness_factor)
                potential_stop = current_price * (1 - distance_pct)
                
                # Only move stop up, never down
                if potential_stop > position.trailing_stop:
                    position.trailing_stop = potential_stop
                    close_signals.append(f"Raised trailing stop to {potential_stop:.2f} ({distance_pct*100:.2f}% below price)")
        
        # For short positions - only move stop down
        else:
            # Move to breakeven once we have a small profit
            if current_profit_pct >= self.breakeven_pct and position.trailing_stop > position.entry:
                position.trailing_stop = position.entry
                close_signals.append(f"Moved stop to breakeven at {position.entry:.2f}")
                
            # In profit with trailing stop activated - keep tightening
            elif position.trailing_profit_activated:
                # Calculate new potential stop level
                # As profit increases, trailing stop gets tighter
                tightness_factor = min(0.5, 0.2 + (current_profit_pct * 0.05))  # More profit = tighter trailing
                distance_pct = self.tighten_stop_increment * (1 - tightness_factor)
                potential_stop = current_price * (1 + distance_pct)
                
                # Only move stop down, never up
                if potential_stop < position.trailing_stop:
                    position.trailing_stop = potential_stop
                    close_signals.append(f"Lowered trailing stop to {potential_stop:.2f} ({distance_pct*100:.2f}% above price)") 