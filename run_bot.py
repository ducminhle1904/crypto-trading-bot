#!/usr/bin/env python
"""
Script to run the multi-strategy trading bot.
"""
import os
import asyncio
import argparse

from trading_bot.main import TradingBot
from trading_bot.strategies.ema_trend_strategy import EmaTrendStrategy
from trading_bot.strategies.rsi_strategy import RsiStrategy
from trading_bot.strategies.bollinger_squeeze_strategy import BollingerSqueezeStrategy
from trading_bot.strategies.vwap_stoch_strategy import VwapStochStrategy
from trading_bot.config import DEFAULT_SYMBOL, DEFAULT_TIMEFRAME, DEFAULT_LIMIT


async def run_bot(symbol, timeframe, limit, strategies, trailing_profit=True):
    """Run the trading bot with specified strategies."""
    # Initialize bot with the specified timeframe
    bot = await TradingBot(timeframe=timeframe).initialize()
    
    # Save any restored positions, balances and trades before clearing
    restored_positions = dict(bot.positions)
    restored_balances = dict(bot.balances)
    restored_trades = dict(bot.trades)
    trade_id = bot.trade_id
    
    # Clear default strategies
    bot.strategies.clear()
    bot.positions.clear()
    bot.balances.clear()
    bot.trades.clear()
    
    # Add selected strategies with the proper timeframe
    if 'ema' in strategies or 'all' in strategies:
        bot.add_strategy(EmaTrendStrategy(timeframe=timeframe))
    
    if 'rsi' in strategies or 'all' in strategies:
        bot.add_strategy(RsiStrategy(timeframe=timeframe))
    
    # Add new scalping strategies
    if 'squeeze' in strategies or 'all' in strategies:
        bot.add_strategy(BollingerSqueezeStrategy(timeframe=timeframe, use_trailing_profit=trailing_profit))
    
    if 'vwap' in strategies or 'all' in strategies:
        bot.add_strategy(VwapStochStrategy(timeframe=timeframe))
    
    # Restore positions, balances and trades for strategies that were re-added
    for strategy_name, strategy in bot.strategies.items():
        # Restore trades history if available
        if strategy_name in restored_trades:
            bot.trades[strategy_name] = restored_trades[strategy_name]
            
        # Restore balance if available
        if strategy_name in restored_balances:
            bot.balances[strategy_name] = restored_balances[strategy_name]
            
        # Restore position if available
        if strategy_name in restored_positions and restored_positions[strategy_name] is not None:
            bot.positions[strategy_name] = restored_positions[strategy_name]
    
    # Restore trade ID counter
    bot.trade_id = trade_id
    
    # Run the bot
    await bot.run(symbol, timeframe, limit)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run multi-strategy trading bot')
    parser.add_argument('--symbol', type=str, default=DEFAULT_SYMBOL, 
                        help=f'Trading symbol (default: {DEFAULT_SYMBOL})')
    parser.add_argument('--timeframe', type=str, default=DEFAULT_TIMEFRAME,
                        help=f'Candle timeframe (default: {DEFAULT_TIMEFRAME})')
    parser.add_argument('--limit', type=int, default=DEFAULT_LIMIT,
                        help=f'Number of candles to fetch (default: {DEFAULT_LIMIT})')
    parser.add_argument('--strategies', type=str, default='all',
                        help='Strategies to run (comma-separated): ema,rsi,squeeze,vwap,all (default: all)')
    parser.add_argument('--trailing-profit', action='store_true', default=True,
                        help='Enable trailing profit feature to let profitable trades run longer (default: enabled)')
    parser.add_argument('--no-trailing-profit', action='store_false', dest='trailing_profit',
                        help='Disable trailing profit feature and use fixed take profit targets')
    
    args = parser.parse_args()
    strategies = args.strategies.lower().split(',')
    
    # Create .env file if it doesn't exist
    if not os.path.exists(".env"):
        with open(".env", "w") as f:
            f.write("# API Keys\nOKX_API_KEY=\nOKX_API_SECRET=\nOKX_PASSWORD=\nTELEGRAM_BOT_TOKEN=\nTELEGRAM_CHAT_ID=\n")
        print("Created .env file template. Please fill in your API credentials.")
        exit(0)
    
    # Display startup information
    print(f"Starting trading bot with {args.timeframe} timeframe")
    print(f"Symbol: {args.symbol}")
    print(f"Strategies: {', '.join(strategies)}")
    print(f"Trailing Profit: {'Enabled' if args.trailing_profit else 'Disabled'}")
    print(f"Note: The bot will automatically restore previous trading data from CSV if available")
    
    try:
        asyncio.run(run_bot(args.symbol, args.timeframe, args.limit, strategies, args.trailing_profit))
    except KeyboardInterrupt:
        print("Bot stopped by user")
    except Exception as e:
        print(f"Error: {e}") 