"""
Exchange client for interacting with cryptocurrency exchanges.
"""
import asyncio
import pandas as pd
import ccxt.async_support as ccxt_async
from typing import Optional, Dict, Any
from datetime import datetime

from trading_bot.config import (
    logger, OKX_API_KEY, OKX_API_SECRET, OKX_PASSWORD, MAX_RETRIES, DEFAULT_TIMEFRAME
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
            # If timeframe is None, use a default timeframe
            if timeframe is None:
                logger.warning(f"No timeframe provided for fetch_ohlcv, using default: {DEFAULT_TIMEFRAME}")
                timeframe = DEFAULT_TIMEFRAME
                
            # Fetch OHLCV data from the exchange
            ohlcv = await self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            
            # Validate the returned data
            if ohlcv is None or len(ohlcv) == 0:
                logger.warning(f"Empty OHLCV data received for {symbol} on {timeframe}")
                if retries < MAX_RETRIES:
                    logger.info(f"Retrying fetch_ohlcv ({retries+1}/{MAX_RETRIES})")
                    await asyncio.sleep(2 ** retries)
                    return await self.fetch_ohlcv(symbol, timeframe, limit, retries + 1)
                else:
                    logger.error(f"Failed to fetch valid OHLCV data after {MAX_RETRIES} attempts")
                    return None
            
            # Create DataFrame from valid data
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            # Final validation check
            if len(df) < 2:
                logger.warning(f"Insufficient data points in OHLCV response: {len(df)} candles")
                if retries < MAX_RETRIES:
                    logger.info(f"Retrying fetch_ohlcv ({retries+1}/{MAX_RETRIES})")
                    await asyncio.sleep(2 ** retries)
                    return await self.fetch_ohlcv(symbol, timeframe, limit, retries + 1)
                else:
                    logger.error(f"Insufficient data after {MAX_RETRIES} attempts")
                    return None
            
            logger.info(f"Fetched {len(df)} candles for {symbol} on {timeframe}")
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
            # If timeframe is None, use a default timeframe
            if timeframe is None:
                logger.warning(f"No timeframe provided for get_next_candle_time, using default: {DEFAULT_TIMEFRAME}")
                timeframe = DEFAULT_TIMEFRAME
                
            time_response = await self.exchange.fetch_time()
            server_time = time_response / 1000
            
            # Handle different timeframe formats
            if timeframe.endswith('m'):
                # Minutes format
                tf_seconds = int(timeframe[:-1]) * 60
            elif timeframe.endswith('h'):
                # Hours format
                tf_seconds = int(timeframe[:-1]) * 60 * 60
            elif timeframe.endswith('d'):
                # Days format
                tf_seconds = int(timeframe[:-1]) * 60 * 60 * 24
            else:
                logger.warning(f"Unrecognized timeframe format: {timeframe}, defaulting to 5 minutes")
                tf_seconds = 300  # Default to 5 minutes
                
            current_candle_start = int(server_time / tf_seconds) * tf_seconds
            next_candle_start = current_candle_start + tf_seconds
            seconds_until_next = next_candle_start - server_time
            
            logger.info(f"Next {timeframe} candle in {seconds_until_next:.2f} seconds (at {datetime.fromtimestamp(next_candle_start)})")
            return max(seconds_until_next + 5, 5)  # Add 5-second buffer
        except Exception as e:
            logger.error(f"Error calculating next candle time: {e}")
            return 60  # Default wait time if calculation fails 