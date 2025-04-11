# Multi-Strategy Crypto Trading Bot

A modular, extensible crypto trading bot that can run multiple strategies simultaneously.

## Features

- Run multiple trading strategies at the same time
- Independent position management for each strategy
- Backtesting capabilities
- Telegram notifications
- CSV-based trade logging
- Automated deployment of various indicators
- Support for OKX exchange (more exchanges can be added)
- Dynamic parameter adjustment based on timeframe

## Included Strategies

- **EMA Trend Strategy**: Uses EMA crossovers, trend slope, and RSI to generate signals
- **RSI Strategy**: Uses RSI oversold/overbought conditions with EMA filter
- **Bollinger Squeeze Strategy**: Identifies volatility contractions followed by breakouts (great for scalping)
- **VWAP + Stochastic Strategy**: Uses VWAP as intraday value area and Stochastic for trade timing (excellent for day trading)

## Day Trading Optimizations

All strategies include special optimizations for day trading and scalping:

- Tighter stop-loss levels
- Faster trailing stops
- Quick breakeven movements
- Smaller profit targets
- Shorter trade durations
- Faster-responding indicators
- Auto-adjustment based on timeframe (1m-5m optimized for scalping)

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

Run the bot with default settings (all strategies):

```
python run_bot.py
```

Run specific strategies:

```
python run_bot.py --strategies ema,rsi,squeeze,vwap
```

Run just the day trading strategies on 1-minute chart:

```
python run_bot.py --strategies squeeze,vwap --timeframe 1m
```

Customize the trading symbol and timeframe:

```
python run_bot.py --symbol "ETH/USDT:USDT" --timeframe 5m
```

### Command Line Arguments

- `--symbol`: Trading symbol (default: BTC/USDT:USDT)
- `--timeframe`: Candle timeframe (default: 3m)
- `--limit`: Number of candles to fetch (default: 300)
- `--strategies`: Strategies to run (comma-separated): ema,rsi,squeeze,vwap,all (default: all)

## Strategy Recommendations

For day trading and scalping:

- Use 1m-3m timeframes
- Try the `squeeze` and `vwap` strategies
- Focus on liquid pairs like BTC/USDT, ETH/USDT

For swing trading:

- Use 1h-4h timeframes
- Try the `ema` and `rsi` strategies
- Parameter adjustments happen automatically based on timeframe

## Creating Custom Strategies

To create a custom strategy:

1. Create a new file in the `trading_bot/strategies` directory
2. Extend the `BaseStrategy` class
3. Implement the abstract methods:
   - `calculate_indicators()`
   - `check_signals()`
   - `manage_position()`
4. Add your strategy to the bot:

   ```python
   from trading_bot.strategies.your_strategy import YourStrategy

   bot.add_strategy(YourStrategy())
   ```

## Project Structure

- `trading_bot/`: Main package
  - `config.py`: Configuration and environment variables
  - `models.py`: Data models for positions and trades
  - `exchange_client.py`: Exchange API client
  - `backtester.py`: Backtesting functionality
  - `main.py`: Main bot orchestrator
  - `strategies/`: Strategy implementations
    - `base_strategy.py`: Abstract base class for strategies
    - `ema_trend_strategy.py`: EMA trend strategy
    - `rsi_strategy.py`: RSI strategy
    - `bollinger_squeeze_strategy.py`: Bollinger Squeeze strategy for scalping
    - `vwap_stoch_strategy.py`: VWAP + Stochastic strategy for day trading
  - `utils/`: Utility functions
    - `indicators.py`: Technical indicators
    - `logger.py`: Logging and notifications
- `run_bot.py`: Command-line interface to run the bot

## License

MIT License
