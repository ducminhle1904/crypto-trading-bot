"""
Backtester for evaluating trading strategies.
"""
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, List

from trading_bot.config import logger, INITIAL_BALANCE, TRADING_FEE
from trading_bot.strategies.base_strategy import BaseStrategy
from trading_bot.models import Position


async def backtest_strategy(df: pd.DataFrame, 
                           strategy: BaseStrategy, 
                           initial_balance: float = INITIAL_BALANCE,
                           trading_fee: float = TRADING_FEE) -> Dict[str, Any]:
    """
    Backtest a trading strategy on historical data.
    
    Args:
        df: DataFrame with OHLCV data
        strategy: Strategy instance to test
        initial_balance: Initial account balance
        trading_fee: Trading fee percentage
        
    Returns:
        Dictionary with backtest results
    """
    if df is None or len(df) < 50:  # Minimum data required
        logger.warning("Insufficient data for backtesting")
        return {
            "trades": 0,
            "win_rate": 0,
            "avg_profit": 0,
            "final_balance": initial_balance,
            "max_drawdown": 0,
            "profit_factor": 0
        }
    
    # Apply strategy indicators
    df = await strategy.calculate_indicators(df.copy())
    if df is None:
        logger.warning("Failed to calculate indicators for backtest")
        return {
            "trades": 0,
            "win_rate": 0,
            "avg_profit": 0, 
            "final_balance": initial_balance,
            "max_drawdown": 0,
            "profit_factor": 0
        }
    
    balance = initial_balance
    position = None
    trades = []
    equity = [initial_balance]
    trade_id = 1
    
    winning_trades = 0
    losing_trades = 0
    total_profit = 0
    total_loss = 0
    
    # Metrics to track
    biggest_winner = 0
    biggest_loser = 0
    max_drawdown = 0
    peak_balance = initial_balance
    
    for i in range(1, len(df)):
        temp_df = df.iloc[:i+1].copy()
        current_price = df.iloc[i]['close']
        current_time = df.iloc[i]['timestamp']
        
        # Manage existing position
        if position:
            position, close_signal_from_mgmt, close_signals_mgmt = await strategy.manage_position(temp_df, position, balance)
        else:
            close_signal_from_mgmt = False
            close_signals_mgmt = []
        
        # Check for signals
        long_signal, short_signal, close_signal, long_signals, short_signals, close_signals, _ = await strategy.check_signals(temp_df, position)
        
        # Combine close signals
        if close_signal_from_mgmt:
            close_signal = True
            close_signals.extend(close_signals_mgmt)
        
        # Process signals
        if not position and long_signal:
            size = await strategy.calculate_position_size(current_price, balance)
            position = Position(
                side='long', 
                entry=current_price, 
                size=size, 
                open_time=current_time,
                trade_id=trade_id,
                strategy_name=strategy.name
            )
            trade_id += 1
            logger.debug(f"Backtest Long at {current_price:.2f} - Reasons: {'; '.join(long_signals)}")
        
        elif not position and short_signal:
            size = await strategy.calculate_position_size(current_price, balance)
            position = Position(
                side='short', 
                entry=current_price, 
                size=size,
                open_time=current_time,
                trade_id=trade_id,
                strategy_name=strategy.name
            )
            trade_id += 1
            logger.debug(f"Backtest Short at {current_price:.2f} - Reasons: {'; '.join(short_signals)}")
        
        elif position and close_signal:
            exit_price = current_price
            profit, profit_pct = await strategy.calculate_profit(position, exit_price)
            
            balance += profit
            trades.append(profit_pct)
            equity.append(balance)
            
            # Update metrics
            if profit > 0:
                winning_trades += 1
                total_profit += profit
                biggest_winner = max(biggest_winner, profit_pct)
            else:
                losing_trades += 1
                total_loss += abs(profit)
                biggest_loser = min(biggest_loser, profit_pct)
                
            # Track drawdown
            if balance > peak_balance:
                peak_balance = balance
            else:
                drawdown = (peak_balance - balance) / peak_balance * 100
                max_drawdown = max(max_drawdown, drawdown)
            
            logger.debug(f"Backtest Close {position.side} at {exit_price:.2f}, Profit: {profit_pct:.2f}% - Reasons: {'; '.join(close_signals)}")
            position = None
        
        # Update equity curve at each step
        if len(equity) < i + 1:
            equity.append(balance)
    
    # Calculate results
    total_trades = len(trades)
    win_rate = winning_trades / total_trades if total_trades > 0 else 0
    avg_profit = sum(trades) / total_trades if total_trades > 0 else 0
    profit_factor = total_profit / total_loss if total_loss > 0 else float('inf') if total_profit > 0 else 0
    
    results = {
        "trades": total_trades,
        "win_rate": win_rate,
        "avg_profit": avg_profit,
        "final_balance": balance,
        "max_drawdown": max_drawdown,
        "profit_factor": profit_factor,
        "biggest_winner": biggest_winner,
        "biggest_loser": biggest_loser,
        "equity_curve": equity
    }
    
    logger.info(f"Backtest {strategy.name}: {total_trades} trades, Win rate: {win_rate:.2%}, "
                f"Avg profit: {avg_profit:.2f}%, Final balance: {balance:.2f}, "
                f"Max drawdown: {max_drawdown:.2f}%, Profit factor: {profit_factor:.2f}")
    
    return results 