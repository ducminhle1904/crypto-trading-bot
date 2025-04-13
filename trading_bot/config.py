"""
Configuration module for the trading bot.
"""
import os
import logging
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("trading_bot.log", encoding='utf-8'), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Fix for Windows event loop policy
if sys.platform == 'win32':
    import asyncio
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Exchange API credentials
OKX_API_KEY = os.getenv("OKX_API_KEY")
OKX_API_SECRET = os.getenv("OKX_API_SECRET")
OKX_PASSWORD = os.getenv("OKX_PASSWORD")

# Telegram settings
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# Default trading parameters
DEFAULT_SYMBOL = "BTC/USDT:USDT"
DEFAULT_TIMEFRAME = "3m"
DEFAULT_LIMIT = 150
MAX_RETRIES = 3
INITIAL_BALANCE = 1000
RISK_PER_TRADE = 0.01
TRADING_FEE = 0.002

# Base filenames (these will be modified based on strategy)
BASE_LOG_FILE = "trading_bot.log"
BASE_SUMMARY_FILE = "trade_summary.csv"
BASE_PERFORMANCE_FILE = "strategy_performance.json"

# Summary log headers
SUMMARY_HEADERS = [
    "timestamp", "trade_id", "action", "side", "price", "size", "entry_price", 
    "exit_price", "profit_amount", "profit_percent", "balance", "signals", 
    "position_duration", "trade_status", "strategy"
]

# Current active strategy name (will be set by the bot)
ACTIVE_STRATEGY = None

def get_strategy_log_file(strategy_name=None):
    """Get the log filename for the specified strategy."""
    if not strategy_name:
        return BASE_LOG_FILE
    return f"trading_bot_{strategy_name}.log"

def get_strategy_summary_file(strategy_name=None):
    """Get the trade summary CSV filename for the specified strategy."""
    if not strategy_name:
        return BASE_SUMMARY_FILE
    return f"trade_summary_{strategy_name}.csv"

def get_strategy_performance_file(strategy_name=None):
    """Get the performance JSON filename for the specified strategy."""
    if not strategy_name:
        return BASE_PERFORMANCE_FILE
    return f"strategy_performance_{strategy_name}.json"

def set_active_strategy(strategy_name):
    """Set the active strategy name globally."""
    global ACTIVE_STRATEGY
    ACTIVE_STRATEGY = strategy_name
    # Update the logging configuration to use strategy-specific log file
    for handler in logger.handlers:
        if isinstance(handler, logging.FileHandler):
            handler.close()
            logger.removeHandler(handler)
    
    log_file = get_strategy_log_file(strategy_name)
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    logger.info(f"Switched logging to strategy-specific file: {log_file}")

def validate_config():
    """Validate that all required configuration variables are present."""
    if not all([OKX_API_KEY, OKX_API_SECRET, OKX_PASSWORD, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID]):
        logger.error("Missing required environment variables. Please check your .env file.")
        return False
    return True 