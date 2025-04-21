# Crypto Trading Bot

A modular, extensible crypto trading bot with strategy-specific logging and performance tracking.

## Features

- Run trading strategies with their optimal timeframes by default
- Strategy-specific logging and performance tracking
- Position management with trailing stops and automatic take-profit
- Backtesting capabilities
- Telegram notifications for trade events and performance reports
- Comprehensive CSV-based trade logging
- Technical indicators with automatic parameter adjustment by timeframe
- Support for OKX exchange (more exchanges can be added)

## Included Strategies

- **RSI Strategy (15m)**: Uses RSI oversold/overbought conditions with EMA filter

  - Dynamic parameter adjustment by timeframe
  - ATR-based trailing stops
  - Profit targets scaled by volatility

- **EMA Trend Strategy (1h)**: Uses EMA crossovers, trend slope, and RSI confirmations

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

- **Volume Profile VWAP Strategy (5m)**: Combines VWAP with Volume Profile analysis

  - Identifies high-probability entry/exit zones based on volume distribution
  - Tighter stops near high-volume nodes
  - Perfect for day trading with enhanced signal accuracy
  - More aggressive profit locking with adaptive trailing stops

- **Volume Profile Bollinger RSI Strategy (30m)**: Integrates Volume Profile with Bollinger Bands and RSI

  - Identifies key volume levels for high-probability reversals
  - Uses Bollinger Band compression/expansion for volatility timing
  - RSI for momentum confirmation
  - Multi-level trailing profit system with progressive lockout levels
  - Adaptive stop-loss tightening near high-volume zones

- **Multi-Timeframe Strategy (15m)**: Analyzes three timeframes simultaneously for high-probability setups
  - Primary timeframe (15m): For signal generation and position management
  - Higher timeframe (1h): For trend confirmation and alignment
  - Lower timeframe (5m): For precise entry and exit timing
  - Multi-timeframe exit signals with weighted score system
  - Dynamic trailing stops adjusted based on timeframe alignment
  - Adaptive parameters based on volatility conditions

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

Try the new Volume Profile VWAP strategy:

```
python run_bot.py --strategy vpvwap
```

Try the Volume Profile Bollinger RSI strategy:

```
python run_bot.py --strategy vpbb
```

Try the Multi-Timeframe strategy:

```
python run_bot.py --strategy mtf
```

Override the strategy's optimal timeframe:

```
python run_bot.py --symbol "ETH/USDT:USDT" --strategy rsi --timeframe 1h
```

### Command Line Arguments

- `--symbol`: Trading symbol (default: BTC/USDT:USDT)
- `--timeframe`: Override strategy's optimal timeframe (optional)
- `--limit`: Number of candles to fetch (default: 300)
- `--strategy`: Strategy to run:
  - `ema` (1h timeframe)
  - `rsi` (15m timeframe)
  - `squeeze` (15m timeframe)
  - `vwap` (5m timeframe)
  - `dow` (4h timeframe)
  - `vpvwap` (5m timeframe)
  - `vpbb` (30m timeframe)
  - `mtf` (15m timeframe)
- `--trailing-profit`: Enable trailing profit (default: enabled)
- `--no-trailing-profit`: Disable trailing profit and use fixed take-profit targets

## Strategy Optimal Timeframes

Each strategy has been calibrated for an optimal timeframe:

| Strategy                     | Optimal Timeframe | Best For                       |
| ---------------------------- | ----------------- | ------------------------------ |
| EMA Trend                    | 1h                | Swing trading                  |
| RSI                          | 15m               | Intraday reversals             |
| Bollinger Squeeze            | 15m               | Breakout scalping              |
| VWAP Stochastic              | 5m                | Intraday scalping              |
| Dow EMA                      | 4h                | Trend following                |
| Volume Profile VWAP          | 5m                | Day trading w/ better entries  |
| Volume Profile Bollinger RSI | 30m               | Reversals at key volume levels |
| Multi-Timeframe              | 15m               | High-probability setups        |

## Multi-Timeframe Strategy

The Multi-Timeframe Strategy combines analysis from three timeframes to generate more reliable trading signals with stronger confirmation:

1. **Primary Timeframe (15m)** - Used for main signal generation and position management
2. **Higher Timeframe (1h)** - Used for trend confirmation and filtering out weak setups
3. **Lower Timeframe (5m)** - Used for optimizing entries and exits with better timing

### Signal Generation Logic

The strategy looks for alignment across all three timeframes:

- **Higher Timeframe**: Must confirm the overall trend direction (given 1.5x weight)
- **Primary Timeframe**: Must show a specific entry opportunity (given 1.0x weight)
- **Lower Timeframe**: Must confirm momentum in the right direction (given 0.5x weight)

Signals are only generated when there's sufficient agreement across timeframes, resulting in an alignment score of at least 2 out of 3 possible points.

### Key Features

- **Trend Alignment Score**: Weighted 3-point system that measures agreement across timeframes
- **Adaptive Position Management**: Adjusts stop losses and take profits based on trend strength
- **Dynamic Breakeven Levels**: Moves to breakeven faster in high volatility conditions
- **Multi-Timeframe Exit Signals**: Looks for reversal patterns across all timeframes
- **Extended Holding Time**: Holds positions longer when trend alignment is strong
- **Volatility-Aware Trailing**: Tightens trailing stops when price extends beyond Bollinger Bands

This strategy provides stronger confirmation than single-timeframe strategies, reducing false signals and improving overall performance across different market conditions.

## Volume Profile Bollinger RSI Strategy

The Volume Profile Bollinger RSI Strategy combines three powerful components:

1. **Volume Profile** - Identifies key price levels with significant historical volume
2. **Bollinger Bands** - Measures volatility expansion/contraction cycles
3. **RSI (Relative Strength Index)** - Provides momentum confirmation

This strategy specifically looks for high-probability reversal and continuation patterns where:

- Price interacts with significant volume nodes (POC or Value Area boundaries)
- Bollinger Bands indicate either compression (ready to expand) or price reaching band extremes
- RSI confirms momentum direction and potential overbought/oversold conditions

Key features:

- **Volume-aware stops** - Tightens stop losses when price is near high-volume zones
- **Multi-level profit locking** - Progressive trailing profit system that locks in gains at 30%, 50%, 70%, and 90% of target
- **Adaptive parameters** - Automatically adjusts RSI sensitivity, profit targets, and stop distance based on timeframe
- **Timeframe flexibility** - Works well across 15m, 30m, and 1h timeframes

The strategy generates balanced trade signals that respect volume distribution while capturing volatility expansion moves. Perfect for trading reversals at key institutional levels.

## Volume Profile VWAP Strategy

The Volume Profile VWAP Strategy combines the power of:

1. **VWAP (Volume Weighted Average Price)** - Dynamic intraday value level
2. **Volume Profile** - Shows trading volume distribution at different price levels
3. **Stochastic Oscillator** - Momentum indicator for overbought/oversold conditions

Volume Profile provides significant enhancements by identifying:

- **POC (Point of Control)** - The price level with the highest trading volume
- **Value Area** - Price range containing 70% of total volume
- **High Volume Nodes** - Areas of significant trading activity that often act as support/resistance

This strategy specifically targets high-probability trades by combining:

- Standard VWAP signals with Volume Profile confirmation
- More reliable entries near high-volume levels
- Tighter stops near Volume Profile POC (high-liquidity areas)
- More aggressive trailing profit to lock in gains faster

This strategy generates fewer but higher-quality signals, perfect for day trading.

## Strategy Details

### RSI Strategy (15m)

The RSI strategy uses these core components:

- RSI indicator to identify oversold/overbought conditions (RSI period: 10)
- EMA filter to ensure we're trading with the trend (EMA period: 34)
- ATR for dynamic trailing stops with tighter settings for 15m timeframe
- More sensitive RSI levels (30/70) for faster signal generation
- 0.8% profit target for shorter-term trades

Parameters are automatically adjusted based on timeframe:

- 1m-5m: Short-term scalping with tighter parameters
- 15m-1h: Medium-term with balanced parameters
- 4h-1d: Longer-term with wider parameters

### EMA Trend Strategy (1h)

Uses moving average crossovers with RSI confirmation and dynamic parameters optimized for different timeframes:

For 1h timeframe (default):

- EMAs: 13 and 34 periods
- Trend slope calculation over 15 periods
- RSI period: 10 with 70/30 levels
- 2.0% profit target and 1.2% stop loss
- Strong momentum confirmation filters

For shorter timeframes (1m-15m):

- Faster EMAs: 9 and 21 periods
- Tighter stops and more aggressive profit targets
- More responsive RSI (period: 7) with wider levels (75/25)

For higher timeframes (4h+):

- Longer EMAs: 18 and 42 periods
- Expanded trend window (20 periods)
- Standard RSI (period: 14)
- Higher profit targets (2.5%) with wider stop losses (1.5%)

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
