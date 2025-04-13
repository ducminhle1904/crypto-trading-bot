#!/usr/bin/env python
"""
Script to run the single-strategy trading bot.
"""
import os
import asyncio
import argparse

from trading_bot.main import TradingBot
from trading_bot.strategies.ema_trend_strategy import EmaTrendStrategy
from trading_bot.strategies.rsi_strategy import RsiStrategy
from trading_bot.strategies.bollinger_squeeze_strategy import BollingerSqueezeStrategy
from trading_bot.strategies.vwap_stoch_strategy import VwapStochStrategy
from trading_bot.strategies.dow_ema_strategy import DowEmaStrategy
from trading_bot.config import DEFAULT_SYMBOL, DEFAULT_TIMEFRAME, DEFAULT_LIMIT


async def run_bot(symbol, timeframe, limit, strategy_name, trailing_profit=True, frequent_updates=True):
    """Run the trading bot with a specific strategy."""
    # Create the strategy first to use its optimal timeframe
    strategy = None
    
    # Select strategy based on name - using each strategy's optimal timeframe
    if strategy_name == 'ema':
        strategy = EmaTrendStrategy(use_trailing_profit=trailing_profit)  # Default 4h
    elif strategy_name == 'rsi':
        strategy = RsiStrategy(use_trailing_profit=trailing_profit)  # Default 1h
    elif strategy_name == 'squeeze':
        strategy = BollingerSqueezeStrategy(use_trailing_profit=trailing_profit)  # Default 15m
    elif strategy_name == 'vwap':
        strategy = VwapStochStrategy(use_trailing_profit=trailing_profit)  # Default 5m
    elif strategy_name == 'dow':
        strategy = DowEmaStrategy(use_trailing_profit=trailing_profit)  # Default 4h
    else:
        print(f"Strategy '{strategy_name}' not recognized, using default EMA strategy.")
        strategy = EmaTrendStrategy(use_trailing_profit=trailing_profit)
    
    # Override strategy timeframe only if explicitly requested
    if timeframe and hasattr(strategy, 'update_timeframe'):
        strategy.update_timeframe(timeframe)
        print(f"Overriding optimal timeframe with user-specified: {timeframe}")
    
    # Get the strategy's timeframe
    strategy_timeframe = getattr(strategy, 'timeframe', DEFAULT_TIMEFRAME)
    
    # Initialize bot with the strategy's timeframe
    bot = await TradingBot(timeframe=strategy_timeframe, frequent_updates=frequent_updates).initialize()
    
    # Set the strategy
    bot.set_strategy(strategy)
    
    # Run the bot
    await bot.run(symbol, None, limit)  # Pass None for timeframe to use strategy's


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run single-strategy trading bot')
    parser.add_argument('--symbol', type=str, default=DEFAULT_SYMBOL, 
                        help=f'Trading symbol (default: {DEFAULT_SYMBOL})')
    parser.add_argument('--timeframe', type=str, default=None,
                        help=f'Override strategy\'s optimal timeframe (optional)')
    parser.add_argument('--limit', type=int, default=DEFAULT_LIMIT,
                        help=f'Number of candles to fetch (default: {DEFAULT_LIMIT})')
    parser.add_argument('--strategy', type=str, default='ema',
                        help='Strategy to run: ema (4h), rsi (1h), squeeze (15m), vwap (5m), dow (4h)')
    parser.add_argument('--trailing-profit', action='store_true', default=True,
                        help='Enable trailing profit feature to let profitable trades run longer (default: enabled)')
    parser.add_argument('--no-trailing-profit', action='store_false', dest='trailing_profit',
                        help='Disable trailing profit feature and use fixed take profit targets')
    parser.add_argument('--frequent-updates', action='store_true', default=True,
                        help='Enable frequent updates mode (fetches data every 5m but uses strategy timeframe for analysis)')
    parser.add_argument('--no-frequent-updates', action='store_false', dest='frequent_updates',
                        help='Disable frequent updates mode and only fetch data at strategy timeframe intervals')
    
    args = parser.parse_args()
    
    # Create .env file if it doesn't exist
    if not os.path.exists(".env"):
        with open(".env", "w") as f:
            f.write("# API Keys\nOKX_API_KEY=\nOKX_API_SECRET=\nOKX_PASSWORD=\nTELEGRAM_BOT_TOKEN=\nTELEGRAM_CHAT_ID=\n")
        print("Created .env file template. Please fill in your API credentials.")
        exit(0)
    
    # Get optimal timeframes for help message
    timeframe_map = {
        'ema': '4h',
        'rsi': '1h',
        'squeeze': '15m',
        'vwap': '5m',
        'dow': '4h'
    }
    optimal_timeframe = timeframe_map.get(args.strategy, DEFAULT_TIMEFRAME)
    
    # Display startup information
    print(f"Starting trading bot with strategy: {args.strategy}")
    timeframe_info = args.timeframe if args.timeframe else f"optimal timeframe for {args.strategy} ({optimal_timeframe})"
    print(f"Timeframe: {timeframe_info}")
    print(f"Symbol: {args.symbol}")
    print(f"Trailing Profit: {'Enabled' if args.trailing_profit else 'Disabled'}")
    if args.frequent_updates:
        print(f"Frequent Updates: Enabled (data fetched every 5m, analyzed using {timeframe_info} parameters)")
    print(f"Note: The bot will automatically restore previous trading data from CSV if available")
    
    # Run the bot
    asyncio.run(run_bot(args.symbol, args.timeframe, args.limit, args.strategy, 
                         args.trailing_profit, args.frequent_updates)) 