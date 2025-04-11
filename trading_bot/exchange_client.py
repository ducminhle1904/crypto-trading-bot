"""
Exchange client for interacting with cryptocurrency exchanges.
"""
import asyncio
import pandas as pd
import ccxt.async_support as ccxt_async
from typing import Optional, Dict, Any

from trading_bot.config import (
    logger, OKX_API_KEY, OKX_API_SECRET, OKX_PASSWORD, MAX_RETRIES
)


class ExchangeClient:
    """Client for interacting with cryptocurrency exchanges."""
    
    def __init__(self, exchange_id: str = 'okx'):
        """Initialize the exchange client."""
        self.exchange_id = exchange_id
        self.exchange = None
        
    async def initialize(self):
        """Initialize the exchange connection."""
        if self.exchange_id == 'okx':
            self.exchange = ccxt_async.okx({
                'apiKey': OKX_API_KEY,
                'secret': OKX_API_SECRET,
                'password': OKX_PASSWORD,
                'enableRateLimit': True,
                'adjustForTimeDifference': True,
            })
            logger.info(f"Initialized {self.exchange_id} exchange connection")
        else:
            raise ValueError(f"Unsupported exchange: {self.exchange_id}")
        
        return self
        
    async def close(self):
        """Close the exchange connection."""
        if self.exchange:
            await self.exchange.close()
            logger.info(f"Closed {self.exchange_id} exchange connection")
            
    async def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int, retries: int = 0) -> Optional[pd.DataFrame]:
        """Fetch OHLCV data from the exchange."""
        try:
            ohlcv = await self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            logger.info(f"Fetched {len(df)} candles for {symbol}")
            return df
        except Exception as e:
            if retries < MAX_RETRIES:
                logger.warning(f"Error fetching OHLCV, retrying ({retries+1}/{MAX_RETRIES}): {e}")
                await asyncio.sleep(2 ** retries)
                return await self.fetch_ohlcv(symbol, timeframe, limit, retries + 1)
            else:
                logger.error(f"Failed to fetch OHLCV after {MAX_RETRIES} attempts: {e}")
                return None
                
    async def get_next_candle_time(self, timeframe: str) -> float:
        """Calculate time until next candle."""
        try:
            time_response = await self.exchange.fetch_time()
            server_time = time_response / 1000
            tf_seconds = int(timeframe[:-1]) * 60
            current_candle_start = int(server_time / tf_seconds) * tf_seconds
            next_candle_start = current_candle_start + tf_seconds
            seconds_until_next = next_candle_start - server_time
            return max(seconds_until_next + 5, 5)  # Add 5-second buffer
        except Exception as e:
            logger.error(f"Error calculating next candle time: {e}")
            return 60  # Default wait time if calculation fails 