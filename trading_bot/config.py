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
DEFAULT_LIMIT = 300
MAX_RETRIES = 3
INITIAL_BALANCE = 1000
RISK_PER_TRADE = 0.01
TRADING_FEE = 0.001

# Summary log settings
SUMMARY_LOG_FILE = "trade_summary.csv"
SUMMARY_HEADERS = [
    "timestamp", "trade_id", "action", "side", "price", "size", "entry_price", 
    "exit_price", "profit_amount", "profit_percent", "balance", "signals", 
    "position_duration", "trade_status", "strategy"
]

def validate_config():
    """Validate that all required configuration variables are present."""
    if not all([OKX_API_KEY, OKX_API_SECRET, OKX_PASSWORD, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID]):
        logger.error("Missing required environment variables. Please check your .env file.")
        return False
    return True 