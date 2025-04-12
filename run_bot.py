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


async def run_bot(symbol, timeframe, limit, strategy_name, trailing_profit=True):
    """Run the trading bot with a specific strategy."""
    # Initialize bot with the specified timeframe
    bot = await TradingBot(timeframe=timeframe).initialize()
    
    # Select strategy based on name
    if strategy_name == 'ema':
        bot.set_strategy(EmaTrendStrategy(timeframe=timeframe, use_trailing_profit=trailing_profit))
    elif strategy_name == 'rsi':
        bot.set_strategy(RsiStrategy(timeframe=timeframe, use_trailing_profit=trailing_profit))
    elif strategy_name == 'squeeze':
        bot.set_strategy(BollingerSqueezeStrategy(timeframe=timeframe, use_trailing_profit=trailing_profit))
    elif strategy_name == 'vwap':
        bot.set_strategy(VwapStochStrategy(timeframe=timeframe, use_trailing_profit=trailing_profit))
    elif strategy_name == 'dow':
        bot.set_strategy(DowEmaStrategy(timeframe=timeframe, use_trailing_profit=trailing_profit))
    else:
        # Default to EMA trend strategy if unrecognized
        print(f"Strategy '{strategy_name}' not recognized. Using EMA Trend Strategy.")
        bot.set_strategy(EmaTrendStrategy(timeframe=timeframe, use_trailing_profit=trailing_profit))
    
    # Run the bot
    await bot.run(symbol, timeframe, limit)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run single-strategy trading bot')
    parser.add_argument('--symbol', type=str, default=DEFAULT_SYMBOL, 
                        help=f'Trading symbol (default: {DEFAULT_SYMBOL})')
    parser.add_argument('--timeframe', type=str, default=None,
                        help=f'Candle timeframe (default: Uses strategy\'s optimal timeframe)')
    parser.add_argument('--limit', type=int, default=DEFAULT_LIMIT,
                        help=f'Number of candles to fetch (default: {DEFAULT_LIMIT})')
    parser.add_argument('--strategy', type=str, default='ema',
                        help='Strategy to run: ema, rsi, squeeze, vwap, dow (default: ema)')
    parser.add_argument('--trailing-profit', action='store_true', default=True,
                        help='Enable trailing profit feature to let profitable trades run longer (default: enabled)')
    parser.add_argument('--no-trailing-profit', action='store_false', dest='trailing_profit',
                        help='Disable trailing profit feature and use fixed take profit targets')
    
    args = parser.parse_args()
    
    # Create .env file if it doesn't exist
    if not os.path.exists(".env"):
        with open(".env", "w") as f:
            f.write("# API Keys\nOKX_API_KEY=\nOKX_API_SECRET=\nOKX_PASSWORD=\nTELEGRAM_BOT_TOKEN=\nTELEGRAM_CHAT_ID=\n")
        print("Created .env file template. Please fill in your API credentials.")
        exit(0)
    
    # Display startup information
    print(f"Starting trading bot with strategy: {args.strategy}")
    timeframe_info = args.timeframe if args.timeframe else "optimal timeframe for strategy"
    print(f"Timeframe: {timeframe_info}")
    print(f"Symbol: {args.symbol}")
    print(f"Trailing Profit: {'Enabled' if args.trailing_profit else 'Disabled'}")
    print(f"Note: The bot will automatically restore previous trading data from CSV if available")
    
    # Run the bot
    asyncio.run(run_bot(args.symbol, args.timeframe, args.limit, args.strategy, args.trailing_profit)) 