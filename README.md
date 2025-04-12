# Crypto Trading Bot

A modular, extensible crypto trading bot with strategy-specific logging and performance tracking.

## Features

- Run trading strategies with optimal parameters for different timeframes
- Strategy-specific logging and performance tracking
- Position management with trailing stops and automatic take-profit
- Backtesting capabilities
- Telegram notifications for trade events and performance reports
- Comprehensive CSV-based trade logging
- Technical indicators with automatic parameter adjustment by timeframe
- Support for OKX exchange (more exchanges can be added)

## Included Strategies

- **RSI Strategy**: Uses RSI oversold/overbought conditions with EMA filter

  - Dynamic parameter adjustment by timeframe
  - ATR-based trailing stops
  - Profit targets scaled by volatility

- **EMA Trend Strategy**: Uses EMA crossovers, trend slope, and RSI confirmations

## Installation

1. Clone the repository
2. Install required packages:
   ```
   pip install -r requirements.txt
   ```
3. Configure your API keys in the `.env` file:
   ```
   OKX_API_KEY=your_api_key
   OKX_API_SECRET=your_api_secret
   OKX_PASSWORD=your_api_password
   TELEGRAM_BOT_TOKEN=your_telegram_bot_token
   TELEGRAM_CHAT_ID=your_telegram_chat_id
   ```

## Usage

### Running the Bot

Run the bot with default strategy:

```
python run_bot.py
```

Run with a specific strategy:

```
python run_bot.py --strategy rsi
```

Customize the trading symbol and timeframe:

```
python run_bot.py --symbol "ETH/USDT:USDT" --timeframe 5m --strategy rsi
```

### Command Line Arguments

- `--symbol`: Trading symbol (default: BTC/USDT:USDT)
- `--timeframe`: Candle timeframe (default: 3m)
- `--limit`: Number of candles to fetch (default: 300)
- `--strategy`: Strategy to run: ema, rsi (default: ema)

## Strategy Details

### RSI Strategy

The RSI strategy uses these core components:

- RSI indicator to identify oversold/overbought conditions
- EMA filter to ensure we're trading with the trend
- ATR for dynamic trailing stops

Parameters are automatically adjusted based on timeframe:

- 1m-5m: Short-term scalping with tighter parameters
- 15m-1h: Medium-term with balanced parameters
- 4h-1d: Longer-term with wider parameters

### EMA Trend Strategy

Uses moving average crossovers with RSI confirmation.

## Performance Tracking

The bot includes a comprehensive performance tracking system:

- Strategy-specific log files
- Performance metrics saved to JSON
- Trade summaries in CSV format
- Telegram performance reports

Key metrics tracked per strategy:

- Win rate
- Average profit
- Profit factor
- Max drawdown
- Direction bias (long vs short performance)

## Logging System

Trade data is logged to strategy-specific files:

- `trading_bot_{strategy_name}.log` - General logging
- `trade_summary_{strategy_name}.csv` - Trade records
- `strategy_performance_{strategy_name}.json` - Performance metrics

## Creating Custom Strategies

To create a custom strategy:

1. Create a new file in the `trading_bot/strategies` directory
2. Extend the `BaseStrategy` class
3. Implement the required methods:
   - `calculate_indicators()`
   - `check_signals()`
   - `manage_position()`

Example skeleton:

```python
from trading_bot.strategies.base_strategy import BaseStrategy

class YourStrategy(BaseStrategy):
    def __init__(self, timeframe=None):
        super().__init__(name="your_strategy", timeframe=timeframe)
        # Initialize your parameters

    async def calculate_indicators(self, df):
        # Calculate indicators and add to dataframe
        return df

    async def check_signals(self, df, position=None):
        # Generate trading signals
        return long_signal, short_signal, close_signal, long_signals, short_signals, close_signals, {}

    async def manage_position(self, df, position, balance):
        # Manage open positions (trailing stops, etc.)
        return position, close_signal, close_signals
```

## Project Structure

- `trading_bot/`: Main package
  - `config.py`: Configuration settings and file naming utilities
  - `main.py`: Main bot class
  - `backtester.py`: Backtesting engine
  - `exchange_client.py`: Exchange API wrapper
  - `strategies/`: Strategy implementations
    - `base_strategy.py`: Abstract base class for all strategies
    - `rsi_strategy.py`: RSI strategy implementation
    - `ema_trend_strategy.py`: EMA trend strategy implementation
  - `utils/`: Utility functions
    - `logger.py`: Logging and performance tracking system
- `run_bot.py`: Command-line interface
- `.env`: Environment variables (API keys)

## License

MIT License
