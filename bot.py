import asyncio
import time
import ccxt.async_support as ccxt_async
import pandas as pd
import numpy as np
import os
import sys
import logging
import csv
from datetime import datetime
from telegram import Bot
from dotenv import load_dotenv
from scipy.stats import linregress

if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("trading_bot.log", encoding='utf-8'), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

OKX_API_KEY = os.getenv("OKX_API_KEY")
OKX_API_SECRET = os.getenv("OKX_API_SECRET")
OKX_PASSWORD = os.getenv("OKX_PASSWORD")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

if not all([OKX_API_KEY, OKX_API_SECRET, OKX_PASSWORD, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID]):
    logger.error("Missing required environment variables. Please check your .env file.")
    exit(1)

SYMBOL = "BTC/USDT:USDT"
TIMEFRAME = "3m"
LIMIT = 300
MAX_RETRIES = 3
INITIAL_BALANCE = 1000
RISK_PER_TRADE = 0.01
TRADING_FEE = 0.001
EMA_SHORT = 21
EMA_LONG = 55
TREND_WINDOW = 40

telegram_bot = Bot(token=TELEGRAM_BOT_TOKEN)

# Setup summary logger to CSV
SUMMARY_LOG_FILE = "trade_summary.csv"
SUMMARY_HEADERS = [
    "timestamp", "trade_id", "action", "side", "price", "size", "entry_price", 
    "exit_price", "profit_amount", "profit_percent", "balance", "signals", 
    "position_duration", "trade_status"
]

def setup_summary_logger():
    """Initialize the CSV summary log file with headers"""
    file_exists = os.path.isfile(SUMMARY_LOG_FILE)
    with open(SUMMARY_LOG_FILE, mode='a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(SUMMARY_HEADERS)
            logger.info(f"Created trade summary log file: {SUMMARY_LOG_FILE}")

def log_trade_summary(data):
    """Log a trade event to the summary CSV file"""
    try:
        with open(SUMMARY_LOG_FILE, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([data.get(header, "") for header in SUMMARY_HEADERS])
        logger.debug(f"Logged trade summary: {data['action']} {data.get('side', '')}")
    except Exception as e:
        logger.error(f"Error writing to trade summary log: {e}")

def calculate_ema(series, period):
    return series.ewm(span=period, adjust=False).mean()

def calculate_trendline(df, window):
    x = np.arange(window)
    slopes = []
    for i in range(len(df) - window + 1):
        y = df['close'].iloc[i:i+window].values
        slope, _, _, _, _ = linregress(x, y)
        slopes.append(slope)
    slopes = [np.nan] * (window - 1) + slopes
    return pd.Series(slopes, index=df.index)

def calculate_atr(df, period=10):
    """Calculate Average True Range for volatility"""
    high = df['high']
    low = df['low']
    close = df['close'].shift(1)
    
    tr1 = high - low
    tr2 = (high - close).abs()
    tr3 = (low - close).abs()
    
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()
    return atr

async def fetch_ohlcv(exchange, symbol, timeframe, limit, retries=0):
    try:
        ohlcv = await exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        logger.info(f"Fetched {len(df)} candles for {symbol}")
        return df
    except Exception as e:
        if retries < MAX_RETRIES:
            logger.warning(f"Error fetching OHLCV, retrying ({retries+1}/{MAX_RETRIES}): {e}")
            await asyncio.sleep(2 ** retries)
            return await fetch_ohlcv(exchange, symbol, timeframe, limit, retries + 1)
        else:
            logger.error(f"Failed to fetch OHLCV after {MAX_RETRIES} attempts: {e}")
            return None

async def calculate_indicators(df):
    try:
        if df is None or len(df) < EMA_LONG:
            logger.warning("Insufficient data for indicator calculation")
            return None
        df = df.copy()
        
        # Original indicators
        df['ema_short'] = calculate_ema(df['close'], EMA_SHORT)
        df['ema_long'] = calculate_ema(df['close'], EMA_LONG)
        df['trend_slope'] = calculate_trendline(df, TREND_WINDOW)
        
        # Add faster-responding indicators suitable for 3-minute timeframe
        # Shorter RSI for quick reversals
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.ewm(span=7, adjust=False).mean()  # Shorter and EWM for faster response
        avg_loss = loss.ewm(span=7, adjust=False).mean()
        rs = avg_gain / avg_loss
        df['fast_rsi'] = 100 - (100 / (1 + rs))
        
        # Short-term momentum
        df['momentum'] = df['close'].pct_change(3) * 100  # 9-minute momentum
        
        # Volatility measurement - shorter period for 3-min chart
        df['atr'] = calculate_atr(df, period=10)  # 30-minute volatility window
        
        return df
    except Exception as e:
        logger.error(f"Error calculating indicators: {e}")
        return None

async def calculate_position_size(price, balance):
    stop_loss_percent = 0.0075
    amount_to_risk = balance * RISK_PER_TRADE
    position_size = amount_to_risk / (price * stop_loss_percent)
    return position_size

async def check_signals(df, position=None):
    if df is None or len(df) < EMA_LONG + TREND_WINDOW:
        logger.warning("Insufficient data for signal calculation")
        return None, None, None, [], [], [], []
    
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
    
    # New confirmation indicators
    momentum_bullish = last['momentum'] > 0
    momentum_bearish = last['momentum'] < 0
    rsi_oversold = last['fast_rsi'] < 30
    rsi_overbought = last['fast_rsi'] > 70
    
    # Dynamic risk based on volatility
    volatility_factor = last['atr'] / last['close'] * 100  # ATR as percentage of price
    
    # Enhanced signal conditions
    long_confirmation = momentum_bullish or rsi_oversold
    long_condition = trend_up and ema_bullish and long_confirmation
    
    short_confirmation = momentum_bearish or rsi_overbought
    short_condition = trend_down and ema_bearish and short_confirmation
    
    if long_condition:
        long_signals.append("Trendline slope positive")
        long_signals.append(f"Slope: {last['trend_slope']:.2f}")
        long_signals.append("EMA(21) > EMA(55) and Price > EMA(21)")
        long_signals.append(f"EMA(21): {last['ema_short']:.2f}, EMA(55): {last['ema_long']:.2f}")
        if momentum_bullish:
            long_signals.append(f"Bullish momentum: {last['momentum']:.2f}%")
        if rsi_oversold:
            long_signals.append(f"Oversold RSI: {last['fast_rsi']:.2f}")
    else:
        if not trend_up:
            long_fail_reasons.append(f"No trend up: Slope ({last['trend_slope']:.2f})")
        if not ema_bullish:
            long_fail_reasons.append(f"Not bullish: Price ({last['close']:.2f}), EMA(21) ({last['ema_short']:.2f}), EMA(55) ({last['ema_long']:.2f})")
        if not long_confirmation:
            long_fail_reasons.append(f"No confirmation: Momentum ({last['momentum']:.2f}%), RSI ({last['fast_rsi']:.2f})")
    
    if short_condition:
        short_signals.append("Trendline slope negative")
        short_signals.append(f"Slope: {last['trend_slope']:.2f}")
        short_signals.append("EMA(21) < EMA(55) and Price < EMA(21)")
        short_signals.append(f"EMA(21): {last['ema_short']:.2f}, EMA(55): {last['ema_long']:.2f}")
        if momentum_bearish:
            short_signals.append(f"Bearish momentum: {last['momentum']:.2f}%")
        if rsi_overbought:
            short_signals.append(f"Overbought RSI: {last['fast_rsi']:.2f}")
    else:
        if not trend_down:
            short_fail_reasons.append(f"No trend down: Slope ({last['trend_slope']:.2f})")
        if not ema_bearish:
            short_fail_reasons.append(f"Not bearish: Price ({last['close']:.2f}), EMA(21) ({last['ema_short']:.2f}), EMA(55) ({last['ema_long']:.2f})")
        if not short_confirmation:
            short_fail_reasons.append(f"No confirmation: Momentum ({last['momentum']:.2f}%), RSI ({last['fast_rsi']:.2f})")
    
    close_condition = False
    if position:
        # Dynamic stop-loss and take-profit based on volatility
        stop_loss_pct = max(0.005, min(0.0075, volatility_factor * 0.75))
        take_profit_pct = max(0.01, min(0.015, volatility_factor * 1.5))
        
        if position['side'] == 'long':
            # Check if trailing stop exists, otherwise use initial stop loss
            if 'trailing_stop' in position:
                stop_loss = position['trailing_stop']
            else:
                stop_loss = position['entry'] * (1 - stop_loss_pct)
                
            take_profit = position['entry'] * (1 + take_profit_pct)
            reversal = trend_down or ema_bearish
            
            if last['close'] <= stop_loss:
                close_signals.append("Stop Loss Hit")
                close_condition = True
            elif last['close'] >= take_profit:
                close_signals.append("Take Profit Hit")
                close_condition = True
            elif reversal:
                close_signals.append("Bearish reversal detected")
                close_signals.append(f"Slope: {last['trend_slope']:.2f}, EMA(21): {last['ema_short']:.2f}, EMA(55): {last['ema_long']:.2f}")
                close_condition = True
                
        elif position['side'] == 'short':
            # Check if trailing stop exists, otherwise use initial stop loss
            if 'trailing_stop' in position:
                stop_loss = position['trailing_stop']
            else:
                stop_loss = position['entry'] * (1 + stop_loss_pct)
                
            take_profit = position['entry'] * (1 - take_profit_pct)
            reversal = trend_up or ema_bullish
            
            if last['close'] >= stop_loss:
                close_signals.append("Stop Loss Hit")
                close_condition = True
            elif last['close'] <= take_profit:
                close_signals.append("Take Profit Hit")
                close_condition = True
            elif reversal:
                close_signals.append("Bullish reversal detected")
                close_signals.append(f"Slope: {last['trend_slope']:.2f}, EMA(21): {last['ema_short']:.2f}, EMA(55): {last['ema_long']:.2f}")
                close_condition = True
    
    long_signal = long_condition and not position
    short_signal = short_condition and not position
    close_signal = close_condition and position
    
    if not long_signal and not position:
        logger.info(f"No long signal at {last['timestamp']}: " + "; ".join(long_fail_reasons))
    if not short_signal and not position:
        logger.info(f"No short signal at {last['timestamp']}: " + "; ".join(short_fail_reasons))
    if not close_signal and position:
        logger.info(f"No close signal at {last['timestamp']} for {position['side']}")
    
    return long_signal, short_signal, close_signal, long_signals, short_signals, close_signals, long_fail_reasons

async def send_telegram_message(message, retries=0):
    try:
        await telegram_bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=message, parse_mode='HTML')
        logger.info(f"Sent Telegram message: {message}")
        return True
    except Exception as e:
        if retries < MAX_RETRIES:
            logger.warning(f"Error sending Telegram message, retrying ({retries+1}/{MAX_RETRIES}): {e}")
            await asyncio.sleep(2 ** retries)
            return await send_telegram_message(message, retries + 1)
        else:
            logger.error(f"Failed to send Telegram message after {MAX_RETRIES} attempts: {e}")
            return False

async def get_next_candle_time(exchange, timeframe):
    try:
        time_response = await exchange.fetch_time()
        server_time = time_response / 1000
        tf_seconds = int(timeframe[:-1]) * 60
        current_candle_start = int(server_time / tf_seconds) * tf_seconds
        next_candle_start = current_candle_start + tf_seconds
        seconds_until_next = next_candle_start - server_time
        return max(seconds_until_next + 5, 5)
    except Exception as e:
        logger.error(f"Error calculating next candle time: {e}")
        return 60

async def backtest_strategy(df, initial_balance=INITIAL_BALANCE, fee=TRADING_FEE):
    if df is None or len(df) < EMA_LONG + TREND_WINDOW:
        logger.warning("Insufficient data for backtesting")
        return 0, 0, 0, initial_balance, 0
    
    df = await calculate_indicators(df.copy())
    if df is None:
        return 0, 0, 0, initial_balance, 0
    
    balance = initial_balance
    position = None
    trades = []
    equity = [initial_balance]
    
    for i in range(1, len(df)):
        temp_df = df.iloc[:i+1].copy()
        long_signal, short_signal, close_signal, long_signals, short_signals, close_signals, _ = await check_signals(temp_df, position)
        current_price = df.iloc[i]['close']
        
        if not position and long_signal:
            size = await calculate_position_size(current_price, balance)
            position = {'side': 'long', 'entry': current_price, 'size': size}
            logger.debug(f"Backtest Long at {current_price:.2f} - Reasons: {'; '.join(long_signals)}")
        
        elif not position and short_signal:
            size = await calculate_position_size(current_price, balance)
            position = {'side': 'short', 'entry': current_price, 'size': size}
            logger.debug(f"Backtest Short at {current_price:.2f} - Reasons: {'; '.join(short_signals)}")
        
        elif position and close_signal:
            exit_price = current_price
            if position['side'] == 'long':
                profit = (exit_price * (1 - fee) - position['entry'] * (1 + fee)) * position['size']
                profit_pct = (exit_price / position['entry'] - 1) * 100
            else:  # short
                profit = (position['entry'] * (1 + fee) - exit_price * (1 - fee)) * position['size']
                profit_pct = (position['entry'] / exit_price - 1) * 100
            balance += profit
            trades.append(profit_pct)
            equity.append(balance)
            logger.debug(f"Backtest Close {position['side']} at {exit_price:.2f}, Profit: {profit_pct:.2f}% - Reasons: {'; '.join(close_signals)}")
            position = None
    
    if trades:
        win_trades = [t for t in trades if t > 0]
        win_rate = len(win_trades) / len(trades)
        avg_profit = sum(trades) / len(trades)
        max_drawdown = min(0, min(np.diff(equity))) / initial_balance * 100
    else:
        win_rate = avg_profit = max_drawdown = 0
    
    logger.info(f"Backtest: {len(trades)} trades, Win rate: {win_rate:.2%}, "
                f"Avg profit: {avg_profit:.2f}%, Final balance: {balance:.2f}, "
                f"Max drawdown: {max_drawdown:.2f}%")
    return len(trades), win_rate, avg_profit, balance, max_drawdown

async def manage_position(df, position, balance):
    """Manages existing positions with trailing stops and dynamic exits"""
    if not position:
        return position, False, []
    
    last = df.iloc[-1]
    close_signals = []
    close_condition = False
    
    # Initialize if first check of this position
    if 'trailing_stop' not in position:
        # Calculate initial stop-loss based on volatility
        volatility_pct = last['atr'] / last['close'] * 100 if 'atr' in last else 0.5
        stop_loss_pct = max(0.005, min(0.0075, volatility_pct * 0.75))
        
        if position['side'] == 'long':
            position['trailing_stop'] = position['entry'] * (1 - stop_loss_pct)
        else:
            position['trailing_stop'] = position['entry'] * (1 + stop_loss_pct)
        
        position['open_time'] = df.iloc[-1]['timestamp']
        position['open_candles'] = 0
    else:
        # Increment candle counter
        position['open_candles'] = position.get('open_candles', 0) + 1
    
    # For 3-minute timeframe, implement quick trailing stop activation
    if position['side'] == 'long':
        breakeven_level = position['entry'] * 1.003  # 0.3% profit
        stop_loss = position['trailing_stop']
        
        # Move to breakeven after small profit
        if last['close'] > breakeven_level and position['trailing_stop'] < position['entry']:
            position['trailing_stop'] = position['entry']
            close_signals.append("Moved stop-loss to breakeven")
        # More aggressive trailing as profit increases
        elif last['close'] > position['entry'] * 1.006:  # 0.6% profit
            potential_stop = max(position['entry'], last['close'] * 0.997)  # 0.3% below current price
            if potential_stop > position['trailing_stop']:
                position['trailing_stop'] = potential_stop
                close_signals.append(f"Updated trailing stop to {potential_stop:.2f}")
    
    elif position['side'] == 'short':
        breakeven_level = position['entry'] * 0.997  # 0.3% profit
        stop_loss = position['trailing_stop']
        
        # Move to breakeven after small profit
        if last['close'] < breakeven_level and position['trailing_stop'] > position['entry']:
            position['trailing_stop'] = position['entry']
            close_signals.append("Moved stop-loss to breakeven")
        # More aggressive trailing as profit increases
        elif last['close'] < position['entry'] * 0.994:  # 0.6% profit
            potential_stop = min(position['entry'], last['close'] * 1.003)  # 0.3% above current price
            if potential_stop < position['trailing_stop']:
                position['trailing_stop'] = potential_stop
                close_signals.append(f"Updated trailing stop to {potential_stop:.2f}")
    
    # For 3-minute timeframe - expire positions that last too long (10 candles = 30 minutes)
    if position.get('open_candles', 0) > 10:
        close_signals.append(f"Position time limit reached ({position['open_candles']} candles)")
        close_condition = True
    
    return position, close_condition, close_signals

async def main():
    try:
        exchange = ccxt_async.okx({
            'apiKey': OKX_API_KEY,
            'secret': OKX_API_SECRET,
            'password': OKX_PASSWORD,
            'enableRateLimit': True,
            'adjustForTimeDifference': True,
        })
        
        # Setup the summary logger
        setup_summary_logger()
        
        logger.info(f"Starting futures trading bot for {SYMBOL} on {TIMEFRAME} timeframe...")
        await send_telegram_message(f"🤖 <b>Futures Trading Bot Started</b>\nMonitoring {SYMBOL} on {TIMEFRAME} timeframe (Long/Short)")
        
        df = await fetch_ohlcv(exchange, SYMBOL, TIMEFRAME, LIMIT)
        if df is not None:
            df = await calculate_indicators(df)
            num_trades, win_rate, avg_profit, final_balance, max_drawdown = await backtest_strategy(df)
            backtest_msg = (
                f"📊 <b>Initial Backtest Results</b>\n"
                f"Trades: {num_trades}\nWin Rate: {win_rate:.2%}\n"
                f"Avg Profit: {avg_profit:.2f}%\nFinal Balance: ${final_balance:.2f}\n"
                f"Max Drawdown: {max_drawdown:.2f}%"
            )
            await send_telegram_message(backtest_msg)
        
        balance = INITIAL_BALANCE
        position = None
        trades = []  # List to store profit percentages for live trading
        trade_id = 1  # Initialize trade counter
        
        while True:
            try:
                wait_time = await get_next_candle_time(exchange, TIMEFRAME)
                logger.info(f"Waiting {wait_time:.2f} seconds for next candle...")
                await asyncio.sleep(wait_time)
                
                latest_df = await fetch_ohlcv(exchange, SYMBOL, TIMEFRAME, LIMIT)
                if latest_df is not None:
                    latest_df = await calculate_indicators(latest_df)
                    if latest_df is not None:
                        # First manage existing position with trailing stops
                        additional_close_signal = False
                        if position:
                            position, additional_close_signal, additional_signals = await manage_position(latest_df, position, balance)
                        
                        # Get standard signals
                        long_signal, short_signal, close_signal, long_signals, short_signals, close_signals, _ = await check_signals(latest_df, position)
                        
                        # Combine close signals
                        if additional_close_signal:
                            close_signal = True
                            close_signals.extend(additional_signals)
                        
                        last_price = latest_df.iloc[-1]['close']
                        timestamp = latest_df.iloc[-1]['timestamp']
                        
                        market_info = (
                            f"🔄 <b>Market Update</b> ({timestamp})\n"
                            f"<b>{SYMBOL}</b>: ${last_price:.2f}\n"
                            f"EMA(21): {latest_df.iloc[-1]['ema_short']:.2f}\n"
                            f"EMA(55): {latest_df.iloc[-1]['ema_long']:.2f}\n"
                            f"Trend Slope: {latest_df.iloc[-1]['trend_slope']:.2f}\n"
                            f"Fast RSI: {latest_df.iloc[-1]['fast_rsi']:.2f}\n"
                            f"Momentum: {latest_df.iloc[-1]['momentum']:.2f}%\n"
                            f"Balance: ${balance:.2f}"
                        )
                        
                        # Summary function
                        def get_trade_summary():
                            total_trades = len(trades)
                            win_rate = len([t for t in trades if t > 0]) / total_trades if total_trades > 0 else 0
                            avg_profit = sum(trades) / total_trades if total_trades > 0 else 0
                            return total_trades, win_rate, avg_profit
                        
                        if long_signal and not position:
                            size = await calculate_position_size(last_price, balance)
                            position = {
                                'side': 'long', 
                                'entry': last_price, 
                                'size': size,
                                'open_time': timestamp,
                                'trade_id': trade_id
                            }
                            
                            # Log the trade entry to summary
                            log_trade_summary({
                                "timestamp": timestamp,
                                "trade_id": trade_id,
                                "action": "ENTRY",
                                "side": "LONG",
                                "price": last_price,
                                "size": size,
                                "entry_price": last_price,
                                "balance": balance,
                                "signals": "; ".join(long_signals),
                                "trade_status": "OPEN"
                            })
                            
                            total_trades, win_rate, avg_profit = get_trade_summary()
                            long_message = (
                                f"🟢<b>LONG SIGNAL - {SYMBOL}</b>🟢\n"
                                f"Price: ${last_price:.2f}\nSize: {size:.6f} BTC\n"
                                f"Time: {timestamp}\n\n<b>Signals:</b>\n• " + "\n• ".join(long_signals) +
                                f"\n\n📈 <b>Trade Summary</b>\nTrades: {total_trades}\nWin Rate: {win_rate:.2%}\nAvg Profit: {avg_profit:.2f}%"
                            )
                            logger.info(f"Long signal at {last_price:.2f} USDT - Trades: {total_trades}, Win Rate: {win_rate:.2%}, Avg Profit: {avg_profit:.2f}%")
                            await send_telegram_message(long_message)
                            trade_id += 1
                        
                        elif short_signal and not position:
                            size = await calculate_position_size(last_price, balance)
                            position = {
                                'side': 'short', 
                                'entry': last_price, 
                                'size': size,
                                'open_time': timestamp,
                                'trade_id': trade_id
                            }
                            
                            # Log the trade entry to summary
                            log_trade_summary({
                                "timestamp": timestamp,
                                "trade_id": trade_id,
                                "action": "ENTRY",
                                "side": "SHORT",
                                "price": last_price,
                                "size": size,
                                "entry_price": last_price,
                                "balance": balance,
                                "signals": "; ".join(short_signals),
                                "trade_status": "OPEN"
                            })
                            
                            total_trades, win_rate, avg_profit = get_trade_summary()
                            short_message = (
                                f"🔴<b>SHORT SIGNAL - {SYMBOL}</b>🔴\n"
                                f"Price: ${last_price:.2f}\nSize: {size:.6f} BTC\n"
                                f"Time: {timestamp}\n\n<b>Signals:</b>\n• " + "\n• ".join(short_signals) +
                                f"\n\n📈 <b>Trade Summary</b>\nTrades: {total_trades}\nWin Rate: {win_rate:.2%}\nAvg Profit: {avg_profit:.2f}%"
                            )
                            logger.info(f"Short signal at {last_price:.2f} USDT - Trades: {total_trades}, Win Rate: {win_rate:.2%}, Avg Profit: {avg_profit:.2f}%")
                            await send_telegram_message(short_message)
                            trade_id += 1
                        
                        elif close_signal and position:
                            exit_price = last_price
                            if position['side'] == 'long':
                                profit = (exit_price * (1 - TRADING_FEE) - position['entry'] * (1 + TRADING_FEE)) * position['size']
                                profit_pct = (exit_price / position['entry'] - 1) * 100
                            else:  # short
                                profit = (position['entry'] * (1 + TRADING_FEE) - exit_price * (1 - TRADING_FEE)) * position['size']
                                profit_pct = (position['entry'] / exit_price - 1) * 100
                            
                            # Calculate position duration
                            position_duration = (timestamp - position['open_time']).total_seconds() / 60  # in minutes
                            
                            balance += profit
                            trades.append(profit_pct)  # Add trade result to list
                            
                            # Log the trade exit to summary
                            log_trade_summary({
                                "timestamp": timestamp,
                                "trade_id": position.get('trade_id', 'unknown'),
                                "action": "EXIT",
                                "side": position['side'].upper(),
                                "price": exit_price,
                                "size": position['size'],
                                "entry_price": position['entry'],
                                "exit_price": exit_price,
                                "profit_amount": profit,
                                "profit_percent": profit_pct,
                                "balance": balance,
                                "signals": "; ".join(close_signals),
                                "position_duration": f"{position_duration:.2f}",
                                "trade_status": "PROFIT" if profit > 0 else "LOSS"
                            })
                            
                            total_trades, win_rate, avg_profit = get_trade_summary()
                            close_message = (
                                f"⚪ <b>CLOSE {position['side'].upper()} - {SYMBOL}</b> ⚪\n"
                                f"Price: ${last_price:.2f}\nProfit: ${profit:.2f} ({profit_pct:.2f}%)\n"
                                f"Time: {timestamp}\nDuration: {position_duration:.2f} minutes\n\n<b>Signals:</b>\n• " + "\n• ".join(close_signals) +
                                f"\n\n📈 <b>Trade Summary</b>\nTrades: {total_trades}\nWin Rate: {win_rate:.2%}\nAvg Profit: {avg_profit:.2f}%"
                            )
                            logger.info(f"Close {position['side']} at {last_price:.2f} USDT, Profit: ${profit:.2f} ({profit_pct:.2f}%) - Trades: {total_trades}, Win Rate: {win_rate:.2%}, Avg Profit: {avg_profit:.2f}%")
                            await send_telegram_message(close_message)
                            position = None
                        
                        elif int(time.time()) % (3600) < 60:  # Hourly updates
                            # Log current market state
                            if position:
                                # Calculate unrealized P/L
                                if position['side'] == 'long':
                                    unrealized_profit = (last_price * (1 - TRADING_FEE) - position['entry'] * (1 + TRADING_FEE)) * position['size']
                                    unrealized_pct = (last_price / position['entry'] - 1) * 100
                                else:  # short
                                    unrealized_profit = (position['entry'] * (1 + TRADING_FEE) - last_price * (1 - TRADING_FEE)) * position['size']
                                    unrealized_pct = (position['entry'] / last_price - 1) * 100
                                
                                position_duration = (timestamp - position['open_time']).total_seconds() / 60
                                
                                # Add position info to market update
                                market_info += f"\n\n<b>Current Position:</b> {position['side'].upper()}\n"
                                market_info += f"Entry: ${position['entry']:.2f}\n"
                                market_info += f"Current P/L: ${unrealized_profit:.2f} ({unrealized_pct:.2f}%)\n"
                                market_info += f"Duration: {position_duration:.2f} minutes"
                                
                                # Log to summary as update
                                log_trade_summary({
                                    "timestamp": timestamp,
                                    "trade_id": position.get('trade_id', 'unknown'),
                                    "action": "UPDATE",
                                    "side": position['side'].upper(),
                                    "price": last_price,
                                    "size": position['size'],
                                    "entry_price": position['entry'],
                                    "profit_amount": unrealized_profit,
                                    "profit_percent": unrealized_pct,
                                    "balance": balance,
                                    "position_duration": f"{position_duration:.2f}",
                                    "trade_status": "OPEN"
                                })
                            
                            await send_telegram_message(market_info)
                
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                await asyncio.sleep(30)
    
    except Exception as e:
        logger.critical(f"Critical error in main function: {e}")
    finally:
        await exchange.close()
        logger.info("Bot shutdown complete")

if __name__ == "__main__":
    if not os.path.exists(".env"):
        with open(".env", "w") as f:
            f.write("# API Keys\nOKX_API_KEY=\nOKX_API_SECRET=\nOKX_PASSWORD=\nTELEGRAM_BOT_TOKEN=\nTELEGRAM_CHAT_ID=\n")
        print("Created .env file template. Please fill in your API credentials.")
        exit(0)
        
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Bot stopped by user")
    except Exception as e:
        logger.critical(f"Unhandled exception: {e}")