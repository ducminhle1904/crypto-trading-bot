"""
Logging utilities for trading bot.
"""
import csv
import os
import asyncio
from datetime import datetime
from typing import Dict, Any, List

from telegram import Bot
from trading_bot.config import (
    logger, MAX_RETRIES, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID,
    SUMMARY_LOG_FILE, SUMMARY_HEADERS
)

# Initialize telegram bot
telegram_bot = Bot(token=TELEGRAM_BOT_TOKEN)


def setup_summary_logger():
    """Initialize the CSV summary log file with headers"""
    file_exists = os.path.isfile(SUMMARY_LOG_FILE)
    with open(SUMMARY_LOG_FILE, mode='a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(SUMMARY_HEADERS)
            logger.info(f"Created trade summary log file: {SUMMARY_LOG_FILE}")


def log_trade_summary(data: Dict[str, Any]):
    """Log a trade event to the summary CSV file"""
    try:
        with open(SUMMARY_LOG_FILE, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([data.get(header, "") for header in SUMMARY_HEADERS])
        logger.debug(f"Logged trade summary: {data['action']} {data.get('side', '')}")
    except Exception as e:
        logger.error(f"Error writing to trade summary log: {e}")


async def send_telegram_message(message: str, retries: int = 0) -> bool:
    """Send message to Telegram chat"""
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