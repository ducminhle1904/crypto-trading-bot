"""
Main module for the trading bot.
"""
import asyncio
import time
from datetime import datetime
from typing import Dict, List, Any, Optional

from trading_bot.config import (
    logger, DEFAULT_SYMBOL, DEFAULT_TIMEFRAME, DEFAULT_LIMIT, 
    INITIAL_BALANCE, validate_config, set_active_strategy
)
from trading_bot.exchange_client import ExchangeClient
from trading_bot.strategies.base_strategy import BaseStrategy
from trading_bot.strategies.ema_trend_strategy import EmaTrendStrategy
from trading_bot.strategies.rsi_strategy import RsiStrategy
from trading_bot.utils.logger import (
    setup_summary_logger, log_trade_summary, send_telegram_message,
    load_performance_metrics, load_previous_trades
)
from trading_bot.models import Position
from trading_bot.backtester import backtest_strategy


class TradingBot:
    """
    Main trading bot class that handles a single strategy.
    """
    
    def __init__(self, exchange_id: str = 'okx', timeframe: str = DEFAULT_TIMEFRAME, frequent_updates: bool = False):
        """Initialize the trading bot."""
        self.exchange_client = None
        self.exchange_id = exchange_id
        self.timeframe = timeframe
        self.strategy = None  # Single strategy instead of multiple
        self.position = None  # Single position
        self.balance = INITIAL_BALANCE  # Single balance
        self.trades = []  # List of trades for the single strategy
        self.trade_id = 1
        
        # Remove frequent updates mode - always use the strategy's timeframe for both fetching and analysis
        self.frequent_updates = False  # Always set to False
        self.fetch_timeframe = timeframe  # Always equal to timeframe
        self.analysis_timeframe = timeframe  # Original timeframe for analysis
        
        logger.info(f"Initialized trading bot for {timeframe} timeframe")
        
    def set_strategy(self, strategy: BaseStrategy):
        """Set the active strategy for the bot."""
        # Ensure strategy is using the same timeframe as the bot, or vice versa
        if hasattr(strategy, 'timeframe'):
            if self.analysis_timeframe is None:
                # Bot doesn't have a timeframe, use the strategy's
                self.analysis_timeframe = strategy.timeframe
                self.timeframe = strategy.timeframe
                self.fetch_timeframe = strategy.timeframe
                logger.info(f"Set bot timeframe to strategy's optimal: {self.timeframe}")
            elif self.analysis_timeframe != strategy.timeframe:
                # Strategy has a different timeframe than the bot, update strategy
                strategy.update_timeframe(self.analysis_timeframe)
        
        # Set the strategy
        self.strategy = strategy
        
        # Set the active strategy name in config for file naming
        strategy_name = getattr(strategy, 'name', 'unknown_strategy')
        set_active_strategy(strategy_name)
        
        # Reset trade data for this strategy
        self.position = None
        self.balance = INITIAL_BALANCE
        self.trades = []
        logger.info(f"Set active strategy: {strategy.name}, timeframe: {self.analysis_timeframe}")
        
    async def initialize(self):
        """Initialize the bot and its components."""
        # Initialize exchange client
        self.exchange_client = await ExchangeClient(self.exchange_id).initialize()
        
        # Initialize logging
        setup_summary_logger()
        
        # Add default strategy if none set
        if not self.strategy:
            if self.timeframe:
                self.set_strategy(EmaTrendStrategy(timeframe=self.timeframe))
            else:
                # Use strategy's default timeframe
                strategy = EmaTrendStrategy()  
                self.timeframe = strategy.timeframe  # Get strategy's default timeframe
                self.analysis_timeframe = strategy.timeframe
                self.fetch_timeframe = strategy.timeframe
                self.set_strategy(strategy)
                logger.info(f"Using default strategy with its optimal timeframe: {self.timeframe}")
        
        # Ensure we have a valid timeframe
        if not self.timeframe:
            logger.warning("No timeframe set during initialization, using default")
            from trading_bot.config import DEFAULT_TIMEFRAME
            self.timeframe = DEFAULT_TIMEFRAME
            self.analysis_timeframe = DEFAULT_TIMEFRAME
            self.fetch_timeframe = DEFAULT_TIMEFRAME
            if hasattr(self.strategy, 'update_timeframe'):
                self.strategy.update_timeframe(self.timeframe)
        
        # Load performance metrics
        load_performance_metrics()
        
        # Load previous trades from CSV file
        previous_data = load_previous_trades()
        
        # Show loaded trade statistics
        if self.strategy and self.strategy.name in previous_data['trades']:
            trade_list = previous_data['trades'][self.strategy.name]
            if trade_list:
                logger.info(f"Loaded {len(trade_list)} previous trades for {self.strategy.name}")
                self.trades = trade_list
                
                # Restore balance if available
                if self.strategy.name in previous_data['balances']:
                    self.balance = previous_data['balances'][self.strategy.name]
                    logger.info(f"Restored balance for {self.strategy.name}: ${self.balance:.2f}")
        
        # Restore open position if any
        if self.strategy and self.strategy.name in previous_data['positions']:
            position_data = previous_data['positions'][self.strategy.name]
            try:
                # Convert the position data to a Position object
                position = Position(
                    side=position_data['side'],
                    entry=position_data['entry'],
                    size=position_data['size'],
                    open_time=position_data['open_time'],
                    trade_id=position_data['trade_id'],
                    strategy_name=self.strategy.name
                )
                
                # Set trailing_stop if it exists
                if 'trailing_stop' in position_data:
                    position.trailing_stop = position_data['trailing_stop']
                
                # Set open_candles if it exists
                if 'open_candles' in position_data:
                    position.open_candles = position_data['open_candles']
                else:
                    position.open_candles = 0  # Default value
                
                self.position = position
                logger.info(f"Restored open {position.side} position for {self.strategy.name} at price {position.entry:.2f} from previous session")
            except Exception as e:
                logger.error(f"Failed to restore position for {self.strategy.name}: {str(e)}")
                self.position = None
        
        # Update the trade ID counter
        if previous_data['trade_id'] > self.trade_id:
            self.trade_id = previous_data['trade_id']
            logger.info(f"Resumed trade ID counter at {self.trade_id}")
        
        logger.info(f"Trading bot fully initialized with strategy: {self.strategy.name}")
        return self
    
    def update_timeframe(self, timeframe: str):
        """Update the timeframe for the bot and strategy."""
        self.timeframe = timeframe
        self.analysis_timeframe = timeframe
        self.fetch_timeframe = timeframe
        if self.strategy and hasattr(self.strategy, 'update_timeframe'):
            self.strategy.update_timeframe(timeframe)
        logger.info(f"Updated bot timeframe to {timeframe}")
        
    async def backtest_strategy(self, symbol: str = DEFAULT_SYMBOL, timeframe: str = None, limit: int = DEFAULT_LIMIT):
        """Run backtest for the active strategy."""
        # Use provided timeframe, or bot's timeframe, or strategy's optimal timeframe
        if timeframe is None:
            if self.timeframe is None and self.strategy and hasattr(self.strategy, 'timeframe'):
                timeframe = self.strategy.timeframe
                logger.info(f"Using strategy's optimal timeframe for backtesting: {timeframe}")
            else:
                timeframe = self.analysis_timeframe
                
        df = await self.exchange_client.fetch_ohlcv(symbol, timeframe, limit)
        if df is None:
            logger.error("Failed to fetch data for backtesting")
            return
            
        if not self.strategy:
            logger.error("No strategy set for backtesting")
            return
            
        logger.info(f"Backtesting strategy: {self.strategy.name}")
        results = await backtest_strategy(df, self.strategy)
            
        # Send results to telegram
        backtest_msg = (
            f"📊 <b>Backtest Results: {self.strategy.name}</b>\n"
            f"Trades: {results['trades']}\n"
            f"Win Rate: {results['win_rate']:.2%}\n"
            f"Avg Profit: {results['avg_profit']:.2f}%\n"
            f"Final Balance: ${results['final_balance']:.2f}\n"
            f"Max Drawdown: {results['max_drawdown']:.2f}%\n"
            f"Profit Factor: {results['profit_factor']:.2f}"
        )
        await send_telegram_message(backtest_msg)
            
        return results
    
    async def process_candle(self, df, symbol: str):
        """Process a new candle for the active strategy."""
        if df is None:
            logger.error("No data received for signal processing")
            await send_telegram_message(f"❌ <b>Data Error</b>\nFailed to receive valid market data for {symbol}. Will retry on next candle.")
            return
            
        if len(df) < 2:
            logger.warning(f"Insufficient data for signal processing: {len(df)} candles")
            await send_telegram_message(f"⚠️ <b>Data Warning</b>\nInsufficient data points ({len(df)}) for {symbol}. Need at least 2 candles.")
            return
            
        if not self.strategy:
            logger.error("No strategy set for processing candle")
            return
            
        try:
            last_candle = df.iloc[-1]
            timestamp = last_candle['timestamp']
            last_price = last_candle['close']
            
            # Market info to be shared
            market_info = (
                f"🔄 <b>Market Update</b> ({timestamp})\n"
                f"<b>{symbol}</b>: ${last_price:.2f}\n"
                f"Strategy: {self.strategy.name}\n"
            )
            
            # Calculate indicators for this strategy
            df_with_indicators = await self.strategy.calculate_indicators(df.copy())
            if df_with_indicators is None:
                logger.warning(f"Failed to calculate indicators for {self.strategy.name}")
                await send_telegram_message(f"⚠️ <b>Indicator Error</b>\nCould not calculate indicators for {self.strategy.name}")
                return
                
            # Handle position management if we have one
            additional_close_signal = False
            if self.position:
                # Make sure restored positions have necessary attributes
                if not hasattr(self.position, 'trailing_stop') or self.position.trailing_stop is None:
                    logger.info(f"Setting initial trailing stop for restored position in {self.strategy.name}")
                    # We'll let the strategy's manage_position method set the trailing stop
                
                if not hasattr(self.position, 'open_candles'):
                    self.position.open_candles = 0
                
                # Manage the position normally
                self.position, additional_close_signal, additional_signals = await self.strategy.manage_position(
                    df_with_indicators, self.position, self.balance
                )
            
            # Get trading signals
            long_signal, short_signal, close_signal, long_signals, short_signals, close_signals, _ = await self.strategy.check_signals(
                df_with_indicators, self.position
            )
            
            # Combine close signals
            if additional_close_signal:
                close_signal = True
                close_signals.extend(additional_signals)
            
            # Execute orders based on signals
            if long_signal and not self.position:
                # Calculate position size
                position_size = await self.strategy.calculate_position_size(last_price, self.balance)
                if position_size <= 0:
                    logger.warning(f"Invalid position size calculated: {position_size}")
                    return
                    
                # Open long position
                self.position = Position(
                    side='long',
                    entry=last_price,
                    size=position_size,
                    open_time=timestamp,
                    trade_id=self.trade_id,
                    strategy_name=self.strategy.name
                )
                
                self.trade_id += 1
                
                # Log the trade entry
                log_trade_summary({
                    "timestamp": timestamp,
                    "trade_id": self.position.trade_id,
                    "action": "ENTRY",
                    "side": "LONG",
                    "price": last_price,
                    "size": position_size,
                    "entry_price": last_price,
                    "exit_price": None,
                    "profit_amount": None,
                    "profit_percent": None,
                    "balance": self.balance,
                    "signals": "; ".join(long_signals),
                    "position_duration": "0",
                    "trade_status": "OPEN",
                    "strategy": self.strategy.name
                })
                
                # Send telegram notification
                entry_msg = (
                    f"🟢 <b>LONG Entry: {self.strategy.name}</b>\n"
                    f"Price: ${last_price:.2f}\n"
                    f"Size: {position_size:.6f}\n"
                    f"Balance: ${self.balance:.2f}\n\n"
                    f"<u>Signals</u>:\n"
                )
                for signal in long_signals:
                    entry_msg += f"• {signal}\n"
                    
                await send_telegram_message(entry_msg)
                
            elif short_signal and not self.position:
                # Calculate position size
                position_size = await self.strategy.calculate_position_size(last_price, self.balance)
                if position_size <= 0:
                    logger.warning(f"Invalid position size calculated: {position_size}")
                    return
                    
                # Open short position
                self.position = Position(
                    side='short',
                    entry=last_price,
                    size=position_size,
                    open_time=timestamp,
                    trade_id=self.trade_id,
                    strategy_name=self.strategy.name
                )
                
                self.trade_id += 1
                
                # Log the trade entry
                log_trade_summary({
                    "timestamp": timestamp,
                    "trade_id": self.position.trade_id,
                    "action": "ENTRY",
                    "side": "SHORT",
                    "price": last_price,
                    "size": position_size,
                    "entry_price": last_price,
                    "exit_price": None,
                    "profit_amount": None,
                    "profit_percent": None,
                    "balance": self.balance,
                    "signals": "; ".join(short_signals),
                    "position_duration": "0",
                    "trade_status": "OPEN",
                    "strategy": self.strategy.name
                })
                
                # Send telegram notification
                entry_msg = (
                    f"🔴 <b>SHORT Entry: {self.strategy.name}</b>\n"
                    f"Price: ${last_price:.2f}\n"
                    f"Size: {position_size:.6f}\n"
                    f"Balance: ${self.balance:.2f}\n\n"
                    f"<u>Signals</u>:\n"
                )
                for signal in short_signals:
                    entry_msg += f"• {signal}\n"
                    
                await send_telegram_message(entry_msg)
                
            elif close_signal and self.position:
                # Close position
                profit, profit_pct = await self.strategy.calculate_profit(self.position, last_price)
                
                # Calculate position duration
                position_duration = (timestamp - self.position.open_time).total_seconds() / 60  # in minutes
                
                # Update balance and record trade results
                self.balance += profit
                self.trades.append(profit_pct)
                
                # Log the trade exit
                log_trade_summary({
                    "timestamp": timestamp,
                    "trade_id": self.position.trade_id,
                    "action": "EXIT",
                    "side": self.position.side.upper(),
                    "price": last_price,
                    "size": self.position.size,
                    "entry_price": self.position.entry,
                    "exit_price": last_price,
                    "profit_amount": profit,
                    "profit_percent": profit_pct,
                    "balance": self.balance,
                    "signals": "; ".join(close_signals),
                    "position_duration": f"{position_duration:.2f}",
                    "trade_status": "PROFIT" if profit > 0 else "LOSS",
                    "strategy": self.strategy.name
                })
                
                # Get trade stats
                win_rate, avg_profit = self._get_trade_stats()
                
                # Determine trade color based on profit
                color = "🟩" if profit > 0 else "🟥"
                
                # Send telegram notification
                exit_msg = (
                    f"{color} <b>{self.position.side.upper()} Exit: {self.strategy.name}</b>\n"
                    f"Entry: ${self.position.entry:.2f}\n"
                    f"Exit: ${last_price:.2f}\n"
                    f"Profit: ${profit:.2f} ({profit_pct:.2f}%)\n"
                    f"Balance: ${self.balance:.2f}\n"
                    f"Duration: {position_duration:.1f} minutes\n"
                    f"Win Rate: {win_rate:.2%}\n"
                    f"Avg Profit: {avg_profit:.2f}%\n\n"
                    f"<u>Exit Signals</u>:\n"
                )
                for signal in close_signals:
                    exit_msg += f"• {signal}\n"
                    
                await send_telegram_message(exit_msg)
                
                # Clear the position
                self.position = None
                
            # Add position info if exists
            if self.position:
                if self.position.side == 'long':
                    unrealized_profit = (last_price - self.position.entry) / self.position.entry * 100
                else:
                    unrealized_profit = (self.position.entry - last_price) / self.position.entry * 100
                    
                position_duration = (timestamp - self.position.open_time).total_seconds() / 60
                
                market_info += (
                    f"\nPosition: {self.position.side.upper()}, "
                    f"Entry: ${self.position.entry:.2f}, "
                    f"P/L: {unrealized_profit:.2f}% "
                )
                
                # Add highest profit seen if using trailing profit
                if hasattr(self.position, 'trailing_profit_activated') and self.position.trailing_profit_activated:
                    market_info += f"(Peak: {self.position.highest_profit_pct:.2f}%), "
                    market_info += f"TP Mode: ACTIVE, "
                else:
                    market_info += f", "
                
                # Add trailing stop info
                if self.position.trailing_stop:
                    if self.position.side == 'long':
                        stop_distance = (last_price - self.position.trailing_stop) / last_price * 100
                    else:
                        stop_distance = (self.position.trailing_stop - last_price) / last_price * 100
                    market_info += f"Stop: ${self.position.trailing_stop:.2f} ({stop_distance:.2f}% away), "
                
                market_info += f"Duration: {position_duration:.1f}m"
        
            # Send market update occasionally (every hour)
            if int(time.time()) % (3600) < 60:
                await send_telegram_message(market_info)
                
        except Exception as e:
            logger.error(f"Error in process_candle: {str(e)}", exc_info=True)
            await send_telegram_message(f"⚠️ <b>Candle Processing Error</b>\n{str(e)}")
            return
    
    def _get_trade_stats(self):
        """Calculate trade statistics for the strategy."""
        if not self.trades:
            return 0.0, 0.0
            
        win_trades = [t for t in self.trades if t > 0]
        win_rate = len(win_trades) / len(self.trades) if self.trades else 0
        avg_profit = sum(self.trades) / len(self.trades) if self.trades else 0
        
        return win_rate, avg_profit
    
    async def run(self, symbol: str = DEFAULT_SYMBOL, timeframe: str = None, limit: int = DEFAULT_LIMIT):
        """Run the trading bot with a single strategy."""
        try:
            # Use provided timeframe or keep existing settings
            if timeframe:
                if timeframe != self.timeframe:
                    self.update_timeframe(timeframe)
            elif self.timeframe is None and self.strategy and hasattr(self.strategy, 'timeframe'):
                # Use strategy's optimal timeframe if bot doesn't have one set
                self.update_timeframe(self.strategy.timeframe)
                logger.info(f"Using strategy's optimal timeframe: {self.timeframe}")
                
            if not self.timeframe:
                logger.error("No timeframe set for bot or strategy")
                await send_telegram_message("❌ <b>Configuration Error</b>\nNo timeframe specified for bot or strategy")
                return
                
            # Prepare startup message with information about restored data
            has_restored_data = len(self.trades) > 0
            has_open_position = self.position is not None
            
            startup_message = (
                f"🤖 <b>Single Strategy Trading Bot Started</b>\n"
                f"Monitoring {symbol} on {self.timeframe} timeframe\n"
                f"Strategy: {self.strategy.name}"
            )
            
            # Add information about restored data if applicable
            if has_restored_data or has_open_position:
                startup_message += "\n\n<b>Restored from previous session:</b>"
                
                if has_restored_data:
                    startup_message += f"\n✅ {len(self.trades)} historical trades"
                
                if has_open_position:
                    startup_message += f"\n🔄 Open position: {self.position.side.upper()}"
            
            # Send startup message
            await send_telegram_message(startup_message)
            
            # Run backtest to evaluate strategy
            backtest_result = await self.backtest_strategy(symbol, self.timeframe, limit)
            if not backtest_result:
                await send_telegram_message("⚠️ <b>Warning</b>: Backtest did not produce results. Check data quality.")
            
            # Main trading loop
            retry_count = 0
            max_fetch_retries = 5  # Maximum number of consecutive fetch failures
            
            while True:
                try:
                    # Wait for next candle based on timeframe
                    wait_time = await self.exchange_client.get_next_candle_time(self.timeframe)
                    logger.info(f"Waiting {wait_time:.2f} seconds for next {self.timeframe} candle...")
                    await asyncio.sleep(wait_time)
                    
                    # Fetch data in the current timeframe
                    latest_df = await self.exchange_client.fetch_ohlcv(symbol, self.timeframe, limit)
                    
                    # Handle fetch failures with retry logic
                    if latest_df is None:
                        retry_count += 1
                        if retry_count >= max_fetch_retries:
                            logger.critical(f"Failed to fetch data after {max_fetch_retries} consecutive attempts")
                            await send_telegram_message(f"🚨 <b>Critical Error</b>: Failed to fetch market data after {max_fetch_retries} consecutive attempts. Bot will restart.")
                            await asyncio.sleep(60)
                            retry_count = 0
                        else:
                            logger.warning(f"Failed to fetch data (attempt {retry_count}/{max_fetch_retries}), will retry")
                            await asyncio.sleep(5)
                        continue
                    
                    # Reset retry counter on successful fetch
                    retry_count = 0
                    
                    # Standard mode - process with the single timeframe
                    await self.process_candle(latest_df, symbol)
                
                except asyncio.CancelledError:
                    logger.info("Bot execution cancelled")
                    await send_telegram_message("🛑 <b>Bot Stopped</b>: Execution cancelled by user")
                    break
                    
                except Exception as e:
                    logger.error(f"Error in trading loop: {str(e)}", exc_info=True)
                    await send_telegram_message(f"⚠️ <b>Trading Loop Error</b>: {str(e)}\nBot will continue running.")
                    await asyncio.sleep(10)  # Wait before continuing
                
        except Exception as e:
            logger.critical(f"Error in trading bot: {e}", exc_info=True)
            await send_telegram_message(f"❌ <b>Bot Error</b>\n{str(e)}")
            raise


async def main():
    """Entry point for the trading bot."""
    # Validate configuration
    if not validate_config():
        logger.error("Invalid configuration. Exiting.")
        return
        
    try:
        # Initialize bot with default timeframe
        bot = await TradingBot(timeframe=DEFAULT_TIMEFRAME).initialize()
        
        # Run the bot
        await bot.run()
        
    except KeyboardInterrupt:
        logger.info("Bot stopped by user")
    except Exception as e:
        logger.critical(f"Unhandled exception: {e}")


if __name__ == "__main__":
    asyncio.run(main()) 