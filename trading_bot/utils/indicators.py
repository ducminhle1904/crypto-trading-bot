"""
Technical indicators for trading strategies.
"""
import pandas as pd
import numpy as np
from scipy.stats import linregress
from typing import Optional


def calculate_ema(series: pd.Series, period: int) -> pd.Series:
    """Calculate Exponential Moving Average."""
    return series.ewm(span=period, adjust=False).mean()


def calculate_sma(series: pd.Series, period: int) -> pd.Series:
    """Calculate Simple Moving Average."""
    return series.rolling(window=period).mean()


def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Calculate Relative Strength Index."""
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    
    # Handle division by zero
    rs = pd.Series(np.where(avg_loss == 0, 100, avg_gain / avg_loss), index=series.index)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_macd(series: pd.Series, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9) -> pd.DataFrame:
    """Calculate MACD (Moving Average Convergence Divergence)."""
    fast_ema = calculate_ema(series, fast_period)
    slow_ema = calculate_ema(series, slow_period)
    macd_line = fast_ema - slow_ema
    signal_line = calculate_ema(macd_line, signal_period)
    histogram = macd_line - signal_line
    
    return pd.DataFrame({
        'macd': macd_line,
        'signal': signal_line,
        'histogram': histogram
    }, index=series.index)


def calculate_bollinger_bands(series: pd.Series, period: int = 20, std_dev: float = 2.0) -> pd.DataFrame:
    """Calculate Bollinger Bands."""
    middle_band = calculate_sma(series, period)
    std = series.rolling(window=period).std()
    upper_band = middle_band + (std * std_dev)
    lower_band = middle_band - (std * std_dev)
    
    return pd.DataFrame({
        'upper': upper_band,
        'middle': middle_band,
        'lower': lower_band
    }, index=series.index)


def calculate_trendline(df: pd.DataFrame, window: int) -> pd.Series:
    """Calculate trendline slope."""
    x = np.arange(window)
    slopes = []
    for i in range(len(df) - window + 1):
        y = df['close'].iloc[i:i+window].values
        slope, _, _, _, _ = linregress(x, y)
        slopes.append(slope)
    slopes = [np.nan] * (window - 1) + slopes
    return pd.Series(slopes, index=df.index)


def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate Average True Range for volatility."""
    high = df['high']
    low = df['low']
    close = df['close'].shift(1)
    
    tr1 = high - low
    tr2 = (high - close).abs()
    tr3 = (low - close).abs()
    
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()
    return atr


def calculate_momentum(series: pd.Series, period: int = 10) -> pd.Series:
    """Calculate momentum indicator."""
    return series.pct_change(period) * 100
    

def apply_standard_indicators(df: pd.DataFrame, 
                             ema_short: int = 21, 
                             ema_long: int = 55,
                             trend_window: int = 40,
                             rsi_period: int = 7,
                             atr_period: int = 10, 
                             momentum_period: int = 3) -> Optional[pd.DataFrame]:
    """Apply standard set of indicators to a dataframe."""
    try:
        if df is None or len(df) < ema_long:
            return None
            
        df = df.copy()
        
        # Calculate EMAs
        df['ema_short'] = calculate_ema(df['close'], ema_short)
        df['ema_long'] = calculate_ema(df['close'], ema_long)
        
        # Calculate trendline
        df['trend_slope'] = calculate_trendline(df, trend_window)
        
        # Calculate RSI
        df['fast_rsi'] = calculate_rsi(df['close'], rsi_period)
        
        # Calculate Momentum
        df['momentum'] = calculate_momentum(df['close'], momentum_period)
        
        # Calculate ATR
        df['atr'] = calculate_atr(df, atr_period)
        
        return df
    except Exception as e:
        print(f"Error applying indicators: {e}")
        return None 