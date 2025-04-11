"""
Technical indicators for trading strategies.
"""
import pandas as pd
import numpy as np
from scipy.stats import linregress
from typing import Optional, Tuple


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
    
    bandwidth = (upper_band - lower_band) / middle_band * 100
    
    return pd.DataFrame({
        'upper': upper_band,
        'middle': middle_band,
        'lower': lower_band,
        'bandwidth': bandwidth
    }, index=series.index)


def calculate_keltner_channels(df: pd.DataFrame, period: int = 20, atr_mult: float = 2.0) -> pd.DataFrame:
    """Calculate Keltner Channels."""
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    middle_line = calculate_ema(typical_price, period)
    atr = calculate_atr(df, period)
    
    upper_band = middle_line + (atr * atr_mult)
    lower_band = middle_line - (atr * atr_mult)
    
    return pd.DataFrame({
        'upper': upper_band,
        'middle': middle_line,
        'lower': lower_band
    }, index=df.index)


def calculate_bollinger_squeeze(df: pd.DataFrame, bb_period: int = 20, bb_std: float = 2.0, 
                               kc_period: int = 20, kc_mult: float = 1.5) -> pd.Series:
    """Calculate Bollinger Bands squeeze indicator."""
    bb = calculate_bollinger_bands(df['close'], bb_period, bb_std)
    kc = calculate_keltner_channels(df, kc_period, kc_mult)
    
    # Squeeze is when Bollinger Bands are inside Keltner Channels
    squeeze = (bb['lower'] > kc['lower']) & (bb['upper'] < kc['upper'])
    
    return squeeze


def calculate_vwap(df: pd.DataFrame) -> pd.Series:
    """
    Calculate Volume Weighted Average Price.
    
    This is especially useful for intraday trading.
    """
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    vwap = (typical_price * df['volume']).cumsum() / df['volume'].cumsum()
    return vwap


def calculate_stochastic(df: pd.DataFrame, k_period: int = 14, d_period: int = 3, smooth_k: int = 3) -> Tuple[pd.Series, pd.Series]:
    """
    Calculate Stochastic Oscillator.
    
    Returns:
        Tuple containing %K and %D lines
    """
    # Calculate %K
    low_min = df['low'].rolling(window=k_period).min()
    high_max = df['high'].rolling(window=k_period).max()
    
    # Handle division by zero
    denominator = high_max - low_min
    denominator = denominator.replace(0, np.nan)
    
    k = 100 * ((df['close'] - low_min) / denominator)
    
    # Apply smoothing to %K if specified
    if smooth_k > 1:
        k = k.rolling(window=smooth_k).mean()
    
    # Calculate %D (signal line) which is SMA of %K
    d = k.rolling(window=d_period).mean()
    
    return k, d


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


def calculate_rsi_divergence(df: pd.DataFrame, window: int = 14, lookback: int = 5) -> Tuple[pd.Series, pd.Series]:
    """
    Detect RSI divergence.
    
    Bullish divergence: Price makes lower lows while RSI makes higher lows
    Bearish divergence: Price makes higher highs while RSI makes lower highs
    
    Args:
        df: DataFrame with price data
        window: Window length for RSI calculation
        lookback: Number of bars to look back for divergence pattern
        
    Returns:
        Tuple of Series for bullish and bearish divergence
    """
    # Calculate RSI if not already present
    if 'rsi' not in df.columns:
        df['rsi'] = calculate_rsi(df['close'], window)
    
    bullish_divergence = pd.Series(False, index=df.index)
    bearish_divergence = pd.Series(False, index=df.index)
    
    # We need at least 2*lookback periods of data
    if len(df) < 2 * lookback:
        return bullish_divergence, bearish_divergence
    
    for i in range(lookback, len(df)):
        # Get the slice of data to check for divergence
        current_slice = df.iloc[i-lookback:i+1]
        
        # Find local price lows and highs
        price_min_idx = current_slice['low'].idxmin()
        price_max_idx = current_slice['high'].idxmax()
        
        # Corresponding RSI values
        current_price = current_slice.iloc[-1]['close']
        current_rsi = current_slice.iloc[-1]['rsi']
        
        # The previous low/high price and RSI
        prev_min_slice = current_slice.loc[:price_min_idx]
        if not prev_min_slice.empty and len(prev_min_slice) > 1:
            prev_low_price = prev_min_slice['low'].min()
            prev_low_rsi = df.loc[prev_min_slice['low'].idxmin(), 'rsi']
            
            # Bullish divergence: lower price low but higher RSI low
            if (current_price < prev_low_price) and (current_rsi > prev_low_rsi):
                bullish_divergence.iloc[i] = True
        
        prev_max_slice = current_slice.loc[:price_max_idx]
        if not prev_max_slice.empty and len(prev_max_slice) > 1:
            prev_high_price = prev_max_slice['high'].max()
            prev_high_rsi = df.loc[prev_max_slice['high'].idxmax(), 'rsi']
            
            # Bearish divergence: higher price high but lower RSI high
            if (current_price > prev_high_price) and (current_rsi < prev_high_rsi):
                bearish_divergence.iloc[i] = True
    
    return bullish_divergence, bearish_divergence


def apply_bollinger_squeeze_indicators(df: pd.DataFrame,
                                      bb_length: int = 20,
                                      bb_std: float = 2.0,
                                      kc_length: int = 20,
                                      kc_atr_mult: float = 1.5,
                                      macd_fast: int = 12,
                                      macd_slow: int = 26,
                                      macd_signal: int = 9,
                                      rsi_period: int = 14,
                                      rsi_divergence_lookback: int = 5) -> Optional[pd.DataFrame]:
    """Apply indicators for the Bollinger Squeeze strategy."""
    try:
        if df is None or len(df) < max(bb_length, kc_length, macd_slow, rsi_period + rsi_divergence_lookback):
            return None
            
        df = df.copy()
        
        # Calculate Bollinger Bands
        bb = calculate_bollinger_bands(df['close'], bb_length, bb_std)
        df['bb_upper'] = bb['upper']
        df['bb_middle'] = bb['middle']
        df['bb_lower'] = bb['lower']
        df['bb_bandwidth'] = bb['bandwidth']
        
        # Calculate Keltner Channels
        kc = calculate_keltner_channels(df, kc_length, kc_atr_mult)
        df['kc_upper'] = kc['upper']
        df['kc_middle'] = kc['middle']
        df['kc_lower'] = kc['lower']
        
        # Calculate Squeeze indicator
        df['squeeze'] = calculate_bollinger_squeeze(df, bb_length, bb_std, kc_length, kc_atr_mult)
        
        # Calculate squeeze release
        df['squeeze_release'] = df['squeeze'].shift(1) & ~df['squeeze']
        
        # Calculate MACD
        macd = calculate_macd(df['close'], macd_fast, macd_slow, macd_signal)
        df['macd'] = macd['macd']
        df['macd_signal'] = macd['signal']
        df['macd_histogram'] = macd['histogram']
        
        # Calculate RSI
        df['rsi'] = calculate_rsi(df['close'], rsi_period)
        
        # Calculate RSI Divergence
        df['bullish_divergence'], df['bearish_divergence'] = calculate_rsi_divergence(
            df, window=rsi_period, lookback=rsi_divergence_lookback
        )
        
        # Calculate ATR
        df['atr'] = calculate_atr(df, 14)
        
        return df
    except Exception as e:
        print(f"Error applying Bollinger Squeeze indicators: {e}")
        return None


def apply_vwap_stoch_indicators(df: pd.DataFrame,
                               stoch_k: int = 14,
                               stoch_d: int = 3,
                               stoch_smooth: int = 3,
                               ema_period: int = 8) -> Optional[pd.DataFrame]:
    """Apply indicators for the VWAP + Stochastic strategy."""
    try:
        if df is None or len(df) < max(stoch_k, ema_period):
            return None
            
        df = df.copy()
        
        # Calculate VWAP
        df['vwap'] = calculate_vwap(df)
        
        # Calculate Stochastic Oscillator
        k, d = calculate_stochastic(df, stoch_k, stoch_d, stoch_smooth)
        df['stoch_k'] = k
        df['stoch_d'] = d
        
        # Calculate Stochastic conditions
        df['stoch_overbought'] = (df['stoch_k'] > 80) & (df['stoch_d'] > 80)
        df['stoch_oversold'] = (df['stoch_k'] < 20) & (df['stoch_d'] < 20)
        df['stoch_cross_up'] = (df['stoch_k'] > df['stoch_d']) & (df['stoch_k'].shift(1) <= df['stoch_d'].shift(1))
        df['stoch_cross_down'] = (df['stoch_k'] < df['stoch_d']) & (df['stoch_k'].shift(1) >= df['stoch_d'].shift(1))
        
        # Calculate EMA
        df['ema'] = calculate_ema(df['close'], ema_period)
        
        # Calculate price relative to VWAP
        df['above_vwap'] = df['close'] > df['vwap']
        df['below_vwap'] = df['close'] < df['vwap']
        
        # Calculate VWAP crossovers
        df['vwap_cross_up'] = (df['close'] > df['vwap']) & (df['close'].shift(1) <= df['vwap'].shift(1))
        df['vwap_cross_down'] = (df['close'] < df['vwap']) & (df['close'].shift(1) >= df['vwap'].shift(1))
        
        # Calculate ATR
        df['atr'] = calculate_atr(df, 14)
        
        return df
    except Exception as e:
        print(f"Error applying VWAP + Stochastic indicators: {e}")
        return None 