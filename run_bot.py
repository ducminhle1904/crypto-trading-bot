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


async def run_bot(symbol, timeframe, limit, strategies):
    """Run the trading bot with specified strategies."""
    # Initialize bot with the specified timeframe
    bot = await TradingBot(timeframe=timeframe).initialize()
    
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
        bot.add_strategy(BollingerSqueezeStrategy(timeframe=timeframe))
    
    if 'vwap' in strategies or 'all' in strategies:
        bot.add_strategy(VwapStochStrategy(timeframe=timeframe))
    
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
    
    try:
        asyncio.run(run_bot(args.symbol, args.timeframe, args.limit, strategies))
    except KeyboardInterrupt:
        print("Bot stopped by user")
    except Exception as e:
        print(f"Error: {e}") 