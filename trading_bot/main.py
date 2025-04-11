"""
Main module for the trading bot.
"""
import asyncio
import time
from datetime import datetime
from typing import Dict, List, Any, Optional

from trading_bot.config import (
    logger, DEFAULT_SYMBOL, DEFAULT_TIMEFRAME, DEFAULT_LIMIT, 
    INITIAL_BALANCE, validate_config
)
from trading_bot.exchange_client import ExchangeClient
from trading_bot.strategies.base_strategy import BaseStrategy
from trading_bot.strategies.ema_trend_strategy import EmaTrendStrategy
from trading_bot.strategies.rsi_strategy import RsiStrategy
from trading_bot.utils.logger import setup_summary_logger, log_trade_summary, send_telegram_message
from trading_bot.models import Position
from trading_bot.backtester import backtest_strategy


class TradingBot:
    """
    Main trading bot class that handles multiple strategies.
    """
    
    def __init__(self, exchange_id: str = 'okx', timeframe: str = DEFAULT_TIMEFRAME):
        """Initialize the trading bot."""
        self.exchange_client = None
        self.exchange_id = exchange_id
        self.timeframe = timeframe
        self.strategies = {}  # {strategy_name: strategy_instance}
        self.positions = {}   # {strategy_name: position_object}
        self.balances = {}    # {strategy_name: balance}
        self.trades = {}      # {strategy_name: [trade_results]}
        self.trade_id = 1
        logger.info(f"Initialized trading bot for {timeframe} timeframe")
        
    def add_strategy(self, strategy: BaseStrategy):
        """Add a strategy to the bot."""
        # Ensure strategy is using the same timeframe as the bot
        if hasattr(strategy, 'timeframe') and strategy.timeframe != self.timeframe:
            strategy.update_timeframe(self.timeframe)
            
        self.strategies[strategy.name] = strategy
        self.positions[strategy.name] = None
        self.balances[strategy.name] = INITIAL_BALANCE
        self.trades[strategy.name] = []
        logger.info(f"Added strategy: {strategy.name}")
        
    async def initialize(self):
        """Initialize the bot and its components."""
        # Initialize exchange client
        self.exchange_client = await ExchangeClient(self.exchange_id).initialize()
        
        # Initialize logging
        setup_summary_logger()
        
        # Add default strategies if none added yet
        if not self.strategies:
            self.add_strategy(EmaTrendStrategy(timeframe=self.timeframe))
            self.add_strategy(RsiStrategy(timeframe=self.timeframe))
            
        logger.info(f"Trading bot fully initialized with {len(self.strategies)} strategies")
        return self
    
    def update_timeframe(self, timeframe: str):
        """Update the timeframe for the bot and all strategies."""
        self.timeframe = timeframe
        for name, strategy in self.strategies.items():
            if hasattr(strategy, 'update_timeframe'):
                strategy.update_timeframe(timeframe)
        logger.info(f"Updated bot timeframe to {timeframe}")
        
    async def backtest_all_strategies(self, symbol: str = DEFAULT_SYMBOL, timeframe: str = None, limit: int = DEFAULT_LIMIT):
        """Run backtest for all strategies."""
        timeframe = timeframe or self.timeframe
        df = await self.exchange_client.fetch_ohlcv(symbol, timeframe, limit)
        if df is None:
            logger.error("Failed to fetch data for backtesting")
            return
            
        strategy_results = {}
        
        for name, strategy in self.strategies.items():
            logger.info(f"Backtesting strategy: {name}")
            results = await backtest_strategy(df, strategy)
            strategy_results[name] = results
            
            # Send results to telegram
            backtest_msg = (
                f"📊 <b>Backtest Results: {name}</b>\n"
                f"Trades: {results['trades']}\n"
                f"Win Rate: {results['win_rate']:.2%}\n"
                f"Avg Profit: {results['avg_profit']:.2f}%\n"
                f"Final Balance: ${results['final_balance']:.2f}\n"
                f"Max Drawdown: {results['max_drawdown']:.2f}%\n"
                f"Profit Factor: {results['profit_factor']:.2f}"
            )
            await send_telegram_message(backtest_msg)
            
        return strategy_results
    
    async def process_candle(self, df, symbol: str):
        """Process a new candle for all strategies."""
        if df is None or len(df) < 2:
            logger.warning("Insufficient data for signal processing")
            return
            
        last_candle = df.iloc[-1]
        timestamp = last_candle['timestamp']
        last_price = last_candle['close']
        
        # Market info to be shared across strategies
        market_info = (
            f"🔄 <b>Market Update</b> ({timestamp})\n"
            f"<b>{symbol}</b>: ${last_price:.2f}\n"
        )
        
        # Process each strategy
        for strategy_name, strategy in self.strategies.items():
            # Get the strategy's position and balance
            position = self.positions[strategy_name]
            balance = self.balances[strategy_name]
            
            # Calculate indicators for this strategy
            df_with_indicators = await strategy.calculate_indicators(df.copy())
            if df_with_indicators is None:
                logger.warning(f"Failed to calculate indicators for {strategy_name}")
                continue
                
            # Handle position management if we have one
            additional_close_signal = False
            if position:
                position, additional_close_signal, additional_signals = await strategy.manage_position(
                    df_with_indicators, position, balance
                )
                self.positions[strategy_name] = position
            
            # Get trading signals
            long_signal, short_signal, close_signal, long_signals, short_signals, close_signals, _ = await strategy.check_signals(
                df_with_indicators, position
            )
            
            # Combine close signals
            if additional_close_signal:
                close_signal = True
                close_signals.extend(additional_signals)
            
            # Process signals
            if long_signal and not position:
                # Enter long position
                size = await strategy.calculate_position_size(last_price, balance)
                position = Position(
                    side='long',
                    entry=last_price,
                    size=size,
                    open_time=timestamp,
                    trade_id=self.trade_id,
                    strategy_name=strategy_name
                )
                self.positions[strategy_name] = position
                
                # Log the trade entry
                log_trade_summary({
                    "timestamp": timestamp,
                    "trade_id": self.trade_id,
                    "action": "ENTRY",
                    "side": "LONG",
                    "price": last_price,
                    "size": size,
                    "entry_price": last_price,
                    "balance": balance,
                    "signals": "; ".join(long_signals),
                    "trade_status": "OPEN",
                    "strategy": strategy_name
                })
                
                # Get trade stats
                win_rate, avg_profit = self._get_trade_stats(strategy_name)
                
                # Send notification
                long_message = (
                    f"🟢<b>LONG SIGNAL - {symbol} ({strategy_name})</b>🟢\n"
                    f"Price: ${last_price:.2f}\nSize: {size:.6f} BTC\n"
                    f"Time: {timestamp}\n\n<b>Signals:</b>\n• " + "\n• ".join(long_signals) +
                    f"\n\n📈 <b>Strategy Stats</b>\nTrades: {len(self.trades[strategy_name])}\n"
                    f"Win Rate: {win_rate:.2%}\nAvg Profit: {avg_profit:.2f}%"
                )
                await send_telegram_message(long_message)
                self.trade_id += 1
                
            elif short_signal and not position:
                # Enter short position
                size = await strategy.calculate_position_size(last_price, balance)
                position = Position(
                    side='short',
                    entry=last_price,
                    size=size,
                    open_time=timestamp,
                    trade_id=self.trade_id,
                    strategy_name=strategy_name
                )
                self.positions[strategy_name] = position
                
                # Log the trade entry
                log_trade_summary({
                    "timestamp": timestamp,
                    "trade_id": self.trade_id,
                    "action": "ENTRY",
                    "side": "SHORT",
                    "price": last_price,
                    "size": size,
                    "entry_price": last_price,
                    "balance": balance,
                    "signals": "; ".join(short_signals),
                    "trade_status": "OPEN",
                    "strategy": strategy_name
                })
                
                # Get trade stats
                win_rate, avg_profit = self._get_trade_stats(strategy_name)
                
                # Send notification
                short_message = (
                    f"🔴<b>SHORT SIGNAL - {symbol} ({strategy_name})</b>🔴\n"
                    f"Price: ${last_price:.2f}\nSize: {size:.6f} BTC\n"
                    f"Time: {timestamp}\n\n<b>Signals:</b>\n• " + "\n• ".join(short_signals) +
                    f"\n\n📈 <b>Strategy Stats</b>\nTrades: {len(self.trades[strategy_name])}\n"
                    f"Win Rate: {win_rate:.2%}\nAvg Profit: {avg_profit:.2f}%"
                )
                await send_telegram_message(short_message)
                self.trade_id += 1
                
            elif close_signal and position:
                # Close position
                profit, profit_pct = await strategy.calculate_profit(position, last_price)
                
                # Calculate position duration
                position_duration = (timestamp - position.open_time).total_seconds() / 60  # in minutes
                
                # Update balance and record trade results
                balance += profit
                self.balances[strategy_name] = balance
                self.trades[strategy_name].append(profit_pct)
                
                # Log the trade exit
                log_trade_summary({
                    "timestamp": timestamp,
                    "trade_id": position.trade_id,
                    "action": "EXIT",
                    "side": position.side.upper(),
                    "price": last_price,
                    "size": position.size,
                    "entry_price": position.entry,
                    "exit_price": last_price,
                    "profit_amount": profit,
                    "profit_percent": profit_pct,
                    "balance": balance,
                    "signals": "; ".join(close_signals),
                    "position_duration": f"{position_duration:.2f}",
                    "trade_status": "PROFIT" if profit > 0 else "LOSS",
                    "strategy": strategy_name
                })
                
                # Get trade stats
                win_rate, avg_profit = self._get_trade_stats(strategy_name)
                
                # Send notification
                close_message = (
                    f"⚪ <b>CLOSE {position.side.upper()} - {symbol} ({strategy_name})</b> ⚪\n"
                    f"Price: ${last_price:.2f}\nProfit: ${profit:.2f} ({profit_pct:.2f}%)\n"
                    f"Time: {timestamp}\nDuration: {position_duration:.2f} minutes\n\n<b>Signals:</b>\n• " + 
                    "\n• ".join(close_signals) +
                    f"\n\n📈 <b>Strategy Stats</b>\nTrades: {len(self.trades[strategy_name])}\n"
                    f"Win Rate: {win_rate:.2%}\nAvg Profit: {avg_profit:.2f}%"
                )
                await send_telegram_message(close_message)
                self.positions[strategy_name] = None
                
            # Add strategy-specific info to the market update
            strategy_indicators = df_with_indicators.iloc[-1]
            market_info += f"\n<b>{strategy_name}</b> - Balance: ${balance:.2f}\n"
            
            if strategy_name == "ema_trend_strategy":
                market_info += (
                    f"EMA({strategy.ema_short}): {strategy_indicators.get('ema_short', 0):.2f}, "
                    f"EMA({strategy.ema_long}): {strategy_indicators.get('ema_long', 0):.2f}\n"
                    f"Trend: {strategy_indicators.get('trend_slope', 0):.2f}, "
                    f"RSI: {strategy_indicators.get('fast_rsi', 0):.2f}"
                )
            elif strategy_name == "rsi_strategy":
                market_info += (
                    f"RSI({strategy.rsi_period}): {strategy_indicators.get('rsi', 0):.2f}, "
                    f"EMA({strategy.ema_period}): {strategy_indicators.get('ema', 0):.2f}"
                )
            
            # Add position info if exists
            if position:
                if position.side == 'long':
                    unrealized_profit = (last_price - position.entry) / position.entry * 100
                else:
                    unrealized_profit = (position.entry - last_price) / position.entry * 100
                    
                position_duration = (timestamp - position.open_time).total_seconds() / 60
                
                market_info += (
                    f"\nPosition: {position.side.upper()}, "
                    f"Entry: ${position.entry:.2f}, "
                    f"P/L: {unrealized_profit:.2f}%, "
                    f"Duration: {position_duration:.1f}m"
                )
        
        # Send market update occasionally (every hour)
        if int(time.time()) % (3600) < 60:
            await send_telegram_message(market_info)
    
    def _get_trade_stats(self, strategy_name: str):
        """Calculate trade statistics for a strategy."""
        trades = self.trades.get(strategy_name, [])
        if not trades:
            return 0.0, 0.0
            
        win_trades = [t for t in trades if t > 0]
        win_rate = len(win_trades) / len(trades) if trades else 0
        avg_profit = sum(trades) / len(trades) if trades else 0
        
        return win_rate, avg_profit
    
    async def run(self, symbol: str = DEFAULT_SYMBOL, timeframe: str = None, limit: int = DEFAULT_LIMIT):
        """Run the trading bot with multiple strategies."""
        try:
            # Use provided timeframe or default to the one set in the constructor
            if timeframe and timeframe != self.timeframe:
                self.update_timeframe(timeframe)
                
            # Send startup message
            await send_telegram_message(
                f"🤖 <b>Multi-Strategy Trading Bot Started</b>\n"
                f"Monitoring {symbol} on {self.timeframe} timeframe\n"
                f"Strategies: {', '.join(self.strategies.keys())}"
            )
            
            # Run backtest to evaluate strategies
            await self.backtest_all_strategies(symbol, self.timeframe, limit)
            
            # Main trading loop
            while True:
                # Wait for next candle
                wait_time = await self.exchange_client.get_next_candle_time(self.timeframe)
                logger.info(f"Waiting {wait_time:.2f} seconds for next candle...")
                await asyncio.sleep(wait_time)
                
                # Fetch latest data
                latest_df = await self.exchange_client.fetch_ohlcv(symbol, self.timeframe, limit)
                
                # Process the candle
                await self.process_candle(latest_df, symbol)
                
        except Exception as e:
            logger.critical(f"Critical error in main loop: {e}")
            await send_telegram_message(f"❌ <b>Bot Error</b>\n{str(e)}")
        finally:
            if self.exchange_client:
                await self.exchange_client.close()
            logger.info("Bot shutdown complete")


async def main():
    """Entry point for the trading bot."""
    # Validate configuration
    if not validate_config():
        logger.error("Invalid configuration. Exiting.")
        return
        
    try:
        # Initialize bot with default timeframe
        bot = await TradingBot(timeframe=DEFAULT_TIMEFRAME).initialize()
        
        # Add strategies (default strategies are added in initialize)
        # You can add custom strategies here:
        # bot.add_strategy(CustomStrategy())
        
        # Run the bot
        await bot.run()
        
    except KeyboardInterrupt:
        logger.info("Bot stopped by user")
    except Exception as e:
        logger.critical(f"Unhandled exception: {e}")


if __name__ == "__main__":
    asyncio.run(main()) 