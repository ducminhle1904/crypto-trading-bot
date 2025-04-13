# Crypto Trading Bot

A modular, extensible crypto trading bot with strategy-specific logging and performance tracking.

## Features

- Run trading strategies with their optimal timeframes by default
- Strategy-specific logging and performance tracking
- Position management with trailing stops and automatic take-profit
- Frequent updates by default (5-minute checks with any timeframe strategy)
- Backtesting capabilities
- Telegram notifications for trade events and performance reports
- Comprehensive CSV-based trade logging
- Technical indicators with automatic parameter adjustment by timeframe
- Support for OKX exchange (more exchanges can be added)

## Included Strategies

- **RSI Strategy (1h)**: Uses RSI oversold/overbought conditions with EMA filter

  - Dynamic parameter adjustment by timeframe
  - ATR-based trailing stops
  - Profit targets scaled by volatility

- **EMA Trend Strategy (4h)**: Uses EMA crossovers, trend slope, and RSI confirmations

  - Trailing profit management
  - Dynamic stop-loss based on volatility

- **Bollinger Squeeze Strategy (15m)**: Identifies consolidation patterns followed by volatility expansions

  - Perfect for scalping explosive moves
  - RSI divergence for confirmation
  - ATR-based trailing stops

- **VWAP Stochastic Strategy (5m)**: Uses VWAP for intraday value area and Stochastic for overbought/oversold levels

  - Optimized for intraday scalping
  - Fast profit-taking with small targets
  - Dynamic stop management

- **Dow Theory EMA Strategy (4h)**: Combines Dow Theory principles with EMA34 and EMA89 indicators
  - Longer-term trend following
  - Volume confirmation for trend strength
  - Higher profit targets for trend trades

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

Run the bot with default strategy (EMA Trend):

```
python run_bot.py
```

Run with a specific strategy (using its optimal timeframe):

```
python run_bot.py --strategy vwap
```

Override the strategy's optimal timeframe:

```
python run_bot.py --symbol "ETH/USDT:USDT" --strategy rsi --timeframe 15m
```

Disable frequent updates mode (only fetch data at strategy's timeframe):

```
python run_bot.py --strategy ema --no-frequent-updates
```

### Command Line Arguments

- `--symbol`: Trading symbol (default: BTC/USDT:USDT)
- `--timeframe`: Override strategy's optimal timeframe (optional)
- `--limit`: Number of candles to fetch (default: 300)
- `--strategy`: Strategy to run:
  - `ema` (4h timeframe)
  - `rsi` (1h timeframe)
  - `squeeze` (15m timeframe)
  - `vwap` (5m timeframe)
  - `dow` (4h timeframe)
- `--trailing-profit`: Enable trailing profit (default: enabled)
- `--no-trailing-profit`: Disable trailing profit and use fixed take-profit targets
- `--frequent-updates`: Enable frequent updates mode (default: enabled)
- `--no-frequent-updates`: Disable frequent updates mode and only fetch data at strategy timeframe intervals

## Strategy Optimal Timeframes

Each strategy has been calibrated for an optimal timeframe:

| Strategy          | Optimal Timeframe | Best For           |
| ----------------- | ----------------- | ------------------ |
| EMA Trend         | 4h                | Swing trading      |
| RSI               | 1h                | Intraday reversals |
| Bollinger Squeeze | 15m               | Breakout scalping  |
| VWAP Stochastic   | 5m                | Intraday scalping  |
| Dow EMA           | 4h                | Trend following    |

## Frequent Updates Mode

The bot now runs in a special frequent updates mode by default that:

1. Fetches data every 5 minutes regardless of the strategy's timeframe
2. Performs full analysis (signal generation) only when a complete candle of the strategy's timeframe closes
3. Manages positions (trailing stops, breakeven adjustments) with the more frequent 5-minute data
4. Provides more responsive risk management while preserving the strategy's optimal parameters

This mode is especially useful for strategies with longer timeframes like EMA Trend (4h) or Dow EMA (4h), allowing them to manage risk more actively while still making trading decisions at their optimal timeframe.

If you want to disable this feature and only fetch data at the strategy's timeframe intervals, use the `--no-frequent-updates` flag:

```
python run_bot.py --strategy ema --no-frequent-updates
```

## Strategy Details

### RSI Strategy (1h)

The RSI strategy uses these core components:

- RSI indicator to identify oversold/overbought conditions
- EMA filter to ensure we're trading with the trend
- ATR for dynamic trailing stops

Parameters are automatically adjusted based on timeframe:

- 1m-5m: Short-term scalping with tighter parameters
- 15m-1h: Medium-term with balanced parameters
- 4h-1d: Longer-term with wider parameters

### EMA Trend Strategy (4h)

Uses moving average crossovers with RSI confirmation.

- Default EMAs: 18 and 42 periods
- Trend slope calculation over 20 periods
- RSI for momentum confirmation
- 2.5% profit target and 1.5% stop loss by default

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
