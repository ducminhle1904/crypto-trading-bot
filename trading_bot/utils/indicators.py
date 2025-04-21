"""
Technical indicators for trading strategies.
"""
import pandas as pd
import numpy as np
from scipy.stats import linregress
from typing import Optional, Tuple, Dict, List


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


def calculate_volume_profile(df: pd.DataFrame, num_bins: int = 20, lookback_periods: int = 100) -> Dict:
    """
    Calculate Volume Profile.
    
    Args:
        df: DataFrame with OHLCV data
        num_bins: Number of price levels to divide the range into
        lookback_periods: Number of periods to look back for volume profile calculation
        
    Returns:
        Dictionary containing volume profile data including POC, value area, and volume nodes
    """
    # Use only the lookback periods
    data = df.iloc[-lookback_periods:].copy() if len(df) > lookback_periods else df.copy()
    
    if 'volume' not in data.columns or len(data) < 5:
        # Return empty results if no volume data
        return {
            'poc_price': None,
            'value_area_high': None,
            'value_area_low': None,
            'volume_nodes': [],
            'high_volume_nodes': []
        }
    
    # Get min and max prices in the range
    price_min = data['low'].min()
    price_max = data['high'].max()
    price_range = price_max - price_min
    
    # Avoid division by zero for very tight ranges
    if price_range <= 0:
        price_range = data['close'].std() * 4
        if price_range <= 0:
            price_range = 1.0
        price_min = data['close'].mean() - price_range/2
        price_max = data['close'].mean() + price_range/2
    
    # Create price bins
    bin_size = price_range / num_bins
    price_bins = [price_min + i * bin_size for i in range(num_bins + 1)]
    
    # Initialize volume count for each bin
    volume_profile = {i: 0 for i in range(num_bins)}
    
    # Calculate volume for each bin
    for _, row in data.iterrows():
        # Process each candle and distribute its volume
        candle_min = min(row['open'], row['close'])
        candle_max = max(row['open'], row['close'])
        
        # Find which bins this candle spans
        min_bin = max(0, int((candle_min - price_min) / bin_size))
        max_bin = min(num_bins - 1, int((candle_max - price_min) / bin_size))
        
        # Distribute volume proportionally across bins
        bin_count = max(1, max_bin - min_bin + 1)  # Avoid division by zero
        vol_per_bin = row['volume'] / bin_count
        
        for bin_idx in range(min_bin, max_bin + 1):
            volume_profile[bin_idx] += vol_per_bin
    
    # Convert to list of (price, volume) tuples
    volume_nodes = [(price_min + (i + 0.5) * bin_size, vol) for i, vol in volume_profile.items()]
    
    # Sort by volume to find point of control (POC)
    sorted_nodes = sorted(volume_nodes, key=lambda x: x[1], reverse=True)
    
    # POC is the price level with highest volume
    poc_price = sorted_nodes[0][0] if sorted_nodes else data['close'].iloc[-1]
    
    # Calculate value area (70% of volume)
    total_volume = sum(vol for _, vol in volume_nodes)
    value_area_target = total_volume * 0.7
    
    value_area_nodes = []
    current_vol_sum = 0
    
    for price, vol in sorted_nodes:
        value_area_nodes.append((price, vol))
        current_vol_sum += vol
        if current_vol_sum >= value_area_target:
            break
    
    # Get high and low of value area
    if value_area_nodes:
        value_area_prices = [price for price, _ in value_area_nodes]
        value_area_high = max(value_area_prices)
        value_area_low = min(value_area_prices)
    else:
        value_area_high = data['high'].iloc[-1]
        value_area_low = data['low'].iloc[-1]
    
    # Identify high volume nodes (nodes with significantly above average volume)
    avg_volume = total_volume / num_bins
    high_volume_nodes = [(price, vol) for price, vol in volume_nodes if vol > avg_volume * 1.5]
    
    return {
        'poc_price': poc_price,
        'value_area_high': value_area_high,
        'value_area_low': value_area_low,
        'volume_nodes': volume_nodes,
        'high_volume_nodes': high_volume_nodes
    }


def volume_profile_signals(df: pd.DataFrame, vp_data: Dict) -> Tuple[bool, bool, List[str]]:
    """
    Generate signals based on Volume Profile analysis.
    This function is designed to enhance existing signals, not replace them.
    
    Returns:
        Tuple containing (potential_long, potential_short, reasons)
    """
    if vp_data['poc_price'] is None:
        return False, False, ["No Volume Profile data available"]
    
    last_close = df['close'].iloc[-1]
    last_high = df['high'].iloc[-1]
    last_low = df['low'].iloc[-1]
    
    potential_long = False
    potential_short = False
    reasons = []
    
    # Check for price near POC
    poc_price = vp_data['poc_price']
    near_poc = abs(last_close - poc_price) / poc_price < 0.005  # Within 0.5% of POC
    
    # Check for price at value area boundaries
    at_value_area_low = abs(last_low - vp_data['value_area_low']) / vp_data['value_area_low'] < 0.005
    at_value_area_high = abs(last_high - vp_data['value_area_high']) / vp_data['value_area_high'] < 0.005
    
    # Price below value area (potential bounce or breakdown)
    if last_close < vp_data['value_area_low']:
        potential_long = True
        reasons.append(f"Price below value area: potential bounce at {vp_data['value_area_low']:.2f}")
    
    # Price above value area (potential pullback or breakout)
    if last_close > vp_data['value_area_high']:
        potential_short = True
        reasons.append(f"Price above value area: potential pullback from {vp_data['value_area_high']:.2f}")
    
    # Price near POC (potential reversal)
    if near_poc:
        reasons.append(f"Price near Point of Control: {poc_price:.2f} (high liquidity area)")
        
        # Check if we're near POC but just crossed it
        prev_close = df['close'].iloc[-2] if len(df) > 1 else last_close
        if prev_close < poc_price and last_close > poc_price:
            potential_long = True
            reasons.append("Crossed POC from below: bullish")
        elif prev_close > poc_price and last_close < poc_price:
            potential_short = True
            reasons.append("Crossed POC from above: bearish")
    
    # Bouncing from value area edge
    if at_value_area_low:
        potential_long = True
        reasons.append(f"Price at value area low: {vp_data['value_area_low']:.2f} (support)")
    
    if at_value_area_high:
        potential_short = True
        reasons.append(f"Price at value area high: {vp_data['value_area_high']:.2f} (resistance)")
    
    return potential_long, potential_short, reasons


def apply_volume_profile_indicators(df: pd.DataFrame, 
                                   num_bins: int = 20, 
                                   lookback_periods: int = 100) -> Optional[pd.DataFrame]:
    """
    Apply Volume Profile analysis to a dataframe.
    This adds volume profile features to enhance existing signals rather than creating new ones.
    
    Returns:
        DataFrame with added volume profile indicators
    """
    try:
        if df is None or len(df) < 10:
            return None
            
        df = df.copy()
        
        # Calculate Volume Profile
        vp_data = calculate_volume_profile(df, num_bins, lookback_periods)
        
        # Add Volume Profile data to dataframe
        df['vp_poc'] = vp_data['poc_price']
        df['vp_vah'] = vp_data['value_area_high']
        df['vp_val'] = vp_data['value_area_low']
        
        # Calculate distance to POC as percentage
        if vp_data['poc_price']:
            df['vp_poc_dist'] = (df['close'] - vp_data['poc_price']) / vp_data['poc_price'] * 100
        else:
            df['vp_poc_dist'] = 0
            
        # Calculate if price is inside or outside value area
        df['vp_in_value_area'] = (df['close'] >= vp_data['value_area_low']) & (df['close'] <= vp_data['value_area_high'])
        
        # Calculate distance to value area edges
        if vp_data['value_area_high'] and vp_data['value_area_low']:
            df['vp_vah_dist'] = (df['close'] - vp_data['value_area_high']) / vp_data['value_area_high'] * 100
            df['vp_val_dist'] = (df['close'] - vp_data['value_area_low']) / vp_data['value_area_low'] * 100
        else:
            df['vp_vah_dist'] = 0
            df['vp_val_dist'] = 0
        
        # Add additional volume profile signals
        potential_long, potential_short, _ = volume_profile_signals(df, vp_data)
        df['vp_potential_long'] = potential_long
        df['vp_potential_short'] = potential_short
        
        return df
    except Exception as e:
        print(f"Error applying volume profile indicators: {e}")
        return df  # Return original dataframe if error 


def calculate_ssl_channel(df, period=10):
    """
    Calculate SSL Channel (Buy Stop Line and Sell Stop Line) indicators.
    
    Args:
        df: DataFrame with OHLC data
        period: Period for the SMA calculation
        
    Returns:
        DataFrame with added BSL and SSL columns
    """
    sma_high = df['high'].rolling(window=period).mean()
    sma_low = df['low'].rolling(window=period).mean()
    
    # Initialize the BSL/SSL columns
    df['ssl_up'] = 0.0
    df['ssl_down'] = 0.0
    
    # Initial values
    df.at[period, 'ssl_down'] = sma_high[period]
    df.at[period, 'ssl_up'] = sma_low[period]
    
    # Calculate SSL Channel
    for i in range(period+1, len(df)):
        if df.at[i-1, 'close'] > df.at[i-1, 'ssl_down']:
            df.at[i, 'ssl_down'] = sma_low[i]
            df.at[i, 'ssl_up'] = sma_high[i]
        else:
            df.at[i, 'ssl_down'] = sma_high[i]
            df.at[i, 'ssl_up'] = sma_low[i]
    
    # Determine trend based on SSL Channel
    df['ssl_trend'] = 0  # 1 for bullish, -1 for bearish, 0 for neutral
    for i in range(period+1, len(df)):
        if df.at[i, 'ssl_up'] > df.at[i, 'ssl_down']:
            df.at[i, 'ssl_trend'] = 1  # Bullish
        elif df.at[i, 'ssl_up'] < df.at[i, 'ssl_down']:
            df.at[i, 'ssl_trend'] = -1  # Bearish
    
    return df 