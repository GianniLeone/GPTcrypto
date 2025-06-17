import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import logging
from datetime import datetime

# Configure logging
logger = logging.getLogger("CryptoBot.TechnicalIndicators")

def calculate_rsi(prices: List[float], period: int = 14) -> Optional[float]:
    """Calculate RSI (Relative Strength Index)"""
    if len(prices) < period + 1:
        return None
    
    price_series = pd.Series(prices)
    delta = price_series.diff()
    gains = delta.where(delta > 0, 0)
    losses = -delta.where(delta < 0, 0)
    avg_gains = gains.rolling(window=period).mean()
    avg_losses = losses.rolling(window=period).mean()
    rs = avg_gains / avg_losses
    rsi = 100 - (100 / (1 + rs))
    
    return float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else None

def calculate_macd(prices: List[float], fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, Optional[float]]:
    """Calculate MACD (Moving Average Convergence Divergence)"""
    if len(prices) < slow + signal:
        return {"macd": None, "macd_signal": None, "macd_hist": None}
    
    price_series = pd.Series(prices)
    ema_fast = price_series.ewm(span=fast).mean()
    ema_slow = price_series.ewm(span=slow).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal).mean()
    histogram = macd_line - signal_line
    
    return {
        "macd": float(macd_line.iloc[-1]) if not pd.isna(macd_line.iloc[-1]) else None,
        "macd_signal": float(signal_line.iloc[-1]) if not pd.isna(signal_line.iloc[-1]) else None,
        "macd_hist": float(histogram.iloc[-1]) if not pd.isna(histogram.iloc[-1]) else None
    }

def calculate_moving_averages(prices: List[float], periods: List[int] = [20, 50, 200]) -> Dict[str, Optional[float]]:
    """Calculate Simple Moving Averages for multiple periods"""
    if not prices:
        return {f"ma_{period}": None for period in periods}
    
    price_series = pd.Series(prices)
    ma_dict = {}
    
    for period in periods:
        if len(prices) >= period:
            ma_value = price_series.rolling(window=period).mean().iloc[-1]
            ma_dict[f"ma_{period}"] = float(ma_value) if not pd.isna(ma_value) else None
        else:
            ma_dict[f"ma_{period}"] = None
    
    return ma_dict

def analyze_candlestick_patterns(ohlcv_data: List[List[float]]) -> Dict[str, Any]:
    """
    Analyze candlestick patterns from OHLCV data
    
    Args:
        ohlcv_data: List of [timestamp, open, high, low, close, volume] arrays
        
    Returns:
        Dictionary with candlestick analysis
    """
    if not ohlcv_data or len(ohlcv_data) < 2:
        return {
            "candlestick_signal": "insufficient_data",
            "pattern": "none",
            "body_to_wick_ratio": None,
            "volume_spike": False,
            "candle_color": "neutral"
        }
    
    try:
        # Get the latest candle
        latest_candle = ohlcv_data[-1]
        prev_candle = ohlcv_data[-2] if len(ohlcv_data) >= 2 else None
        
        # Extract OHLCV values
        timestamp, open_price, high, low, close, volume = latest_candle
        
        # Calculate candle characteristics
        body_size = abs(close - open_price)
        total_range = high - low
        upper_wick = high - max(open_price, close)
        lower_wick = min(open_price, close) - low
        
        # Avoid division by zero
        if total_range == 0:
            body_to_wick_ratio = 0
        else:
            body_to_wick_ratio = body_size / total_range
        
        # Determine candle color
        candle_color = "green" if close > open_price else "red" if close < open_price else "neutral"
        
        # Check for volume spike (compare with previous candle)
        volume_spike = False
        if prev_candle and len(prev_candle) > 5:
            prev_volume = prev_candle[5]
            if prev_volume > 0:
                volume_spike = volume > prev_volume * 1.5  # 50% increase
        
        # Detect basic patterns
        pattern = detect_candlestick_pattern(latest_candle, prev_candle)
        
        # Generate overall signal
        signal = generate_candlestick_signal(pattern, candle_color, body_to_wick_ratio, volume_spike)
        
        return {
            "candlestick_signal": signal,
            "pattern": pattern,
            "body_to_wick_ratio": round(body_to_wick_ratio, 3),
            "volume_spike": volume_spike,
            "candle_color": candle_color,
            "body_size": round(body_size, 2),
            "upper_wick": round(upper_wick, 2),
            "lower_wick": round(lower_wick, 2),
            "total_range": round(total_range, 2)
        }
        
    except Exception as e:
        logger.error(f"Error analyzing candlestick patterns: {str(e)}")
        return {
            "candlestick_signal": "error",
            "pattern": "none",
            "body_to_wick_ratio": None,
            "volume_spike": False,
            "candle_color": "neutral"
        }

def detect_candlestick_pattern(current_candle: List[float], prev_candle: List[float] = None) -> str:
    """
    Detect specific candlestick patterns
    
    Args:
        current_candle: [timestamp, open, high, low, close, volume]
        prev_candle: Previous candle data for multi-candle patterns
        
    Returns:
        Pattern name string
    """
    if not current_candle or len(current_candle) < 5:
        return "none"
    
    try:
        _, open_price, high, low, close, volume = current_candle
        
        body_size = abs(close - open_price)
        total_range = high - low
        upper_wick = high - max(open_price, close)
        lower_wick = min(open_price, close) - low
        
        # Avoid division by zero
        if total_range == 0:
            return "none"
        
        # Doji pattern (small body relative to range)
        if body_size < total_range * 0.1:  # Body is less than 10% of range
            if upper_wick > body_size * 3 and lower_wick > body_size * 3:
                return "doji"
            elif upper_wick > body_size * 5:
                return "dragonfly_doji"
            elif lower_wick > body_size * 5:
                return "gravestone_doji"
        
        # Hammer pattern (small body, long lower wick, short upper wick)
        if (lower_wick > body_size * 2 and 
            upper_wick < body_size * 0.5 and 
            body_size > total_range * 0.2):
            return "hammer"
        
        # Inverted hammer (small body, long upper wick, short lower wick)
        if (upper_wick > body_size * 2 and 
            lower_wick < body_size * 0.5 and 
            body_size > total_range * 0.2):
            return "inverted_hammer"
        
        # Spinning top (small body, both wicks present)
        if (body_size < total_range * 0.3 and 
            upper_wick > body_size * 0.5 and 
            lower_wick > body_size * 0.5):
            return "spinning_top"
        
        # Marubozu (large body, minimal wicks)
        if (body_size > total_range * 0.9 and 
            upper_wick < total_range * 0.05 and 
            lower_wick < total_range * 0.05):
            return "marubozu"
        
        # Engulfing pattern (requires previous candle)
        if prev_candle and len(prev_candle) >= 5:
            prev_open = prev_candle[1]
            prev_close = prev_candle[4]
            
            # Bullish engulfing
            if (prev_close < prev_open and  # Previous candle was red
                close > open_price and      # Current candle is green
                open_price < prev_close and # Current opens below prev close
                close > prev_open):         # Current closes above prev open
                return "bullish_engulfing"
            
            # Bearish engulfing
            if (prev_close > prev_open and  # Previous candle was green
                close < open_price and      # Current candle is red
                open_price > prev_close and # Current opens above prev close
                close < prev_open):         # Current closes below prev open
                return "bearish_engulfing"
        
        return "none"
        
    except Exception as e:
        logger.error(f"Error detecting candlestick pattern: {str(e)}")
        return "none"

def generate_candlestick_signal(pattern: str, candle_color: str, body_to_wick_ratio: float, volume_spike: bool) -> str:
    """
    Generate overall candlestick signal based on pattern and characteristics
    
    Args:
        pattern: Detected candlestick pattern
        candle_color: green/red/neutral
        body_to_wick_ratio: Ratio of body size to total wick size
        volume_spike: Whether there was a volume spike
        
    Returns:
        Signal: bullish/bearish/neutral
    """
    # Pattern-based signals
    bullish_patterns = ["hammer", "bullish_engulfing", "dragonfly_doji"]
    bearish_patterns = ["inverted_hammer", "bearish_engulfing", "gravestone_doji"]
    neutral_patterns = ["doji", "spinning_top", "marubozu", "none"]
    
    signal_score = 0
    
    # Pattern influence
    if pattern in bullish_patterns:
        signal_score += 2
    elif pattern in bearish_patterns:
        signal_score -= 2
    
    # Color influence
    if candle_color == "green":
        signal_score += 1
    elif candle_color == "red":
        signal_score -= 1
    
    # Body-to-wick ratio influence (strong body = stronger signal)
    if body_to_wick_ratio and body_to_wick_ratio > 0.7:
        if candle_color == "green":
            signal_score += 1
        elif candle_color == "red":
            signal_score -= 1
    
    # Volume spike amplifies signal
    if volume_spike:
        signal_score = signal_score * 1.5
    
    # Convert score to signal
    if signal_score >= 2:
        return "bullish"
    elif signal_score <= -2:
        return "bearish"
    else:
        return "neutral"

def analyze_ma_trend(ma_values: Dict[str, Optional[float]]) -> str:
    """Analyze moving average trend"""
    ma_20 = ma_values.get("ma_20")
    ma_50 = ma_values.get("ma_50")
    ma_200 = ma_values.get("ma_200")
    
    if not all([ma_20, ma_50, ma_200]):
        if ma_20 and ma_50:
            if ma_20 > ma_50:
                return "short_term_bullish"
            elif ma_20 < ma_50:
                return "short_term_bearish"
        return "insufficient_data"
    
    if ma_20 > ma_50 > ma_200:
        return "strong_bullish"
    elif ma_20 > ma_50 and ma_50 < ma_200:
        return "mixed_bullish"
    elif ma_20 < ma_50 < ma_200:
        return "strong_bearish"
    elif ma_20 < ma_50 and ma_50 > ma_200:
        return "mixed_bearish"
    else:
        return "neutral"

def detect_ma_crossovers(prices: List[float], current_price: float) -> Dict[str, str]:
    """Detect moving average crossovers"""
    if len(prices) < 50:
        return {"ma_cross_20_50": "insufficient_data", "ma_cross_50_200": "insufficient_data"}
    
    price_series = pd.Series(prices + [current_price])
    ma_20 = price_series.rolling(window=20).mean()
    ma_50 = price_series.rolling(window=50).mean()
    
    crossovers = {}
    
    # Check 20/50 crossover
    if len(ma_20) >= 2 and len(ma_50) >= 2:
        ma_20_prev = ma_20.iloc[-2]
        ma_20_curr = ma_20.iloc[-1]
        ma_50_prev = ma_50.iloc[-2]
        ma_50_curr = ma_50.iloc[-1]
        
        if ma_20_prev <= ma_50_prev and ma_20_curr > ma_50_curr:
            crossovers["ma_cross_20_50"] = "golden_cross"
        elif ma_20_prev >= ma_50_prev and ma_20_curr < ma_50_curr:
            crossovers["ma_cross_20_50"] = "death_cross"
        elif ma_20_curr > ma_50_curr:
            crossovers["ma_cross_20_50"] = "20_above_50"
        else:
            crossovers["ma_cross_20_50"] = "20_below_50"
    else:
        crossovers["ma_cross_20_50"] = "insufficient_data"
    
    # Check 50/200 crossover
    if len(prices) >= 200:
        ma_200 = price_series.rolling(window=200).mean()
        
        if len(ma_200) >= 2:
            ma_50_prev = ma_50.iloc[-2]
            ma_50_curr = ma_50.iloc[-1]
            ma_200_prev = ma_200.iloc[-2]
            ma_200_curr = ma_200.iloc[-1]
            
            if ma_50_prev <= ma_200_prev and ma_50_curr > ma_200_curr:
                crossovers["ma_cross_50_200"] = "golden_cross"
            elif ma_50_prev >= ma_200_prev and ma_50_curr < ma_200_curr:
                crossovers["ma_cross_50_200"] = "death_cross"
            elif ma_50_curr > ma_200_curr:
                crossovers["ma_cross_50_200"] = "50_above_200"
            else:
                crossovers["ma_cross_50_200"] = "50_below_200"
        else:
            crossovers["ma_cross_50_200"] = "insufficient_data"
    else:
        crossovers["ma_cross_50_200"] = "insufficient_data"
    
    return crossovers

def interpret_rsi(rsi_value: Optional[float]) -> str:
    """Interpret RSI value"""
    if rsi_value is None:
        return "insufficient_data"
    
    if rsi_value >= 70:
        return "overbought"
    elif rsi_value <= 30:
        return "oversold"
    elif rsi_value >= 60:
        return "bullish"
    elif rsi_value <= 40:
        return "bearish"
    else:
        return "neutral"

def interpret_macd(macd_data: Dict[str, Optional[float]]) -> str:
    """Interpret MACD signals"""
    macd = macd_data.get("macd")
    signal = macd_data.get("macd_signal")
    hist = macd_data.get("macd_hist")
    
    if not all([macd, signal, hist]):
        return "insufficient_data"
    
    if macd > signal and hist > 0:
        if hist > 0.5:
            return "strong_bullish"
        else:
            return "bullish"
    elif macd < signal and hist < 0:
        if hist < -0.5:
            return "strong_bearish"
        else:
            return "bearish"
    else:
        return "neutral"

def get_indicators_for_asset(asset: str, prices: List[float], current_price: float = None, 
                           ohlcv_data: List[List[float]] = None) -> Dict[str, Any]:
    """
    Calculate comprehensive technical indicators including candlestick analysis
    
    Args:
        asset: Asset symbol
        prices: List of historical closing prices
        current_price: Current price
        ohlcv_data: OHLCV data for candlestick analysis
        
    Returns:
        Dictionary with all technical indicators
    """
    if not prices:
        logger.warning(f"No price data available for {asset}")
        return {
            "rsi": None,
            "rsi_signal": "insufficient_data",
            "macd": None,
            "macd_signal": None,
            "macd_hist": None,
            "macd_interpretation": "insufficient_data",
            "ma_20": None,
            "ma_50": None,
            "ma_200": None,
            "ma_trend": "insufficient_data",
            "ma_cross_20_50": "insufficient_data",
            "ma_cross_50_200": "insufficient_data",
            "candlestick_signal": "insufficient_data",
            "candlestick_pattern": "none",
            "technical_summary": "insufficient_data"
        }
    
    if current_price is None:
        current_price = prices[-1]
    
    logger.debug(f"Calculating technical indicators for {asset} with {len(prices)} price points")
    
    # Calculate traditional indicators
    rsi = calculate_rsi(prices)
    rsi_signal = interpret_rsi(rsi)
    
    macd_data = calculate_macd(prices)
    macd_interpretation = interpret_macd(macd_data)
    
    ma_values = calculate_moving_averages(prices)
    ma_trend = analyze_ma_trend(ma_values)
    
    crossovers = detect_ma_crossovers(prices, current_price)
    
    # Analyze candlestick patterns
    candlestick_analysis = analyze_candlestick_patterns(ohlcv_data) if ohlcv_data else {
        "candlestick_signal": "no_ohlcv_data",
        "pattern": "none",
        "body_to_wick_ratio": None,
        "volume_spike": False,
        "candle_color": "neutral"
    }
    
    # Create technical summary including candlestick
    technical_summary = create_technical_summary(
        rsi_signal, macd_interpretation, ma_trend, 
        candlestick_analysis.get("candlestick_signal", "neutral")
    )
    
    # Compile all indicators
    indicators = {
        "rsi": round(rsi, 2) if rsi is not None else None,
        "rsi_signal": rsi_signal,
        "macd": round(macd_data["macd"], 4) if macd_data["macd"] is not None else None,
        "macd_signal": round(macd_data["macd_signal"], 4) if macd_data["macd_signal"] is not None else None,
        "macd_hist": round(macd_data["macd_hist"], 4) if macd_data["macd_hist"] is not None else None,
        "macd_interpretation": macd_interpretation,
        "ma_20": round(ma_values["ma_20"], 2) if ma_values["ma_20"] is not None else None,
        "ma_50": round(ma_values["ma_50"], 2) if ma_values["ma_50"] is not None else None,
        "ma_200": round(ma_values["ma_200"], 2) if ma_values["ma_200"] is not None else None,
        "ma_trend": ma_trend,
        "ma_cross_20_50": crossovers.get("ma_cross_20_50", "insufficient_data"),
        "ma_cross_50_200": crossovers.get("ma_cross_50_200", "insufficient_data"),
        "candlestick_signal": candlestick_analysis.get("candlestick_signal", "insufficient_data"),
        "candlestick_pattern": candlestick_analysis.get("pattern", "none"),
        "candlestick_details": {
            "body_to_wick_ratio": candlestick_analysis.get("body_to_wick_ratio"),
            "volume_spike": candlestick_analysis.get("volume_spike", False),
            "candle_color": candlestick_analysis.get("candle_color", "neutral")
        },
        "technical_summary": technical_summary,
        "data_points": len(prices),
        "timestamp": datetime.now().isoformat()
    }
    
    logger.info(f"Technical indicators calculated for {asset}: "
               f"RSI={rsi}, MACD={macd_interpretation}, Trend={ma_trend}, "
               f"Candlestick={candlestick_analysis.get('candlestick_signal')}")
    
    return indicators

def create_technical_summary(rsi_signal: str, macd_interpretation: str, ma_trend: str, 
                           candlestick_signal: str) -> str:
    """Create summary including candlestick analysis"""
    bullish_signals = 0
    bearish_signals = 0
    
    # RSI signals
    if "bullish" in rsi_signal or "oversold" in rsi_signal:
        bullish_signals += 1
    elif "bearish" in rsi_signal or "overbought" in rsi_signal:
        bearish_signals += 1
    
    # MACD signals
    if "bullish" in macd_interpretation:
        bullish_signals += 1
    elif "bearish" in macd_interpretation:
        bearish_signals += 1
    
    # MA trend signals
    if "bullish" in ma_trend:
        bullish_signals += 1
    elif "bearish" in ma_trend:
        bearish_signals += 1
    
    # Candlestick signals
    if candlestick_signal == "bullish":
        bullish_signals += 1
    elif candlestick_signal == "bearish":
        bearish_signals += 1
    
    # Determine overall signal
    if bullish_signals > bearish_signals:
        if bullish_signals >= 3:
            return "strong_bullish"
        else:
            return "bullish"
    elif bearish_signals > bullish_signals:
        if bearish_signals >= 3:
            return "strong_bearish"
        else:
            return "bearish"
    else:
        return "neutral"

# Testing
if __name__ == "__main__":
    # Test candlestick analysis
    test_ohlcv = [
        [1640995200, 47000, 48000, 46500, 47500, 1000],  # Hammer-like
        [1641081600, 47500, 47600, 46000, 46200, 1200],  # Bearish
        [1641168000, 46200, 47000, 46000, 46800, 1500],  # Bullish with volume spike
    ]
    
    candlestick_result = analyze_candlestick_patterns(test_ohlcv)
    print("Candlestick Analysis Test:")
    for key, value in candlestick_result.items():
        print(f"  {key}: {value}")
    
    # Test full indicators
    sample_prices = [100, 102, 101, 105, 107, 106, 108, 110, 109, 111, 113, 112, 115, 114, 116]
    indicators = get_indicators_for_asset("TEST", sample_prices, ohlcv_data=test_ohlcv)
    
    print("\nFull Technical Indicators Test:")
    for key, value in indicators.items():
        print(f"  {key}: {value}")