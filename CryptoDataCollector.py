import os
import time
import logging
import json
import requests
import ccxt
import pandas as pd
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta
import threading
import queue
import random
from openai import OpenAI
from technical_indicators import get_indicators_for_asset

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("crypto_data_collector.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("CryptoDataCollector")

def safe_get_env_int(key: str, default: int) -> int:
    """Safely get integer from environment variable, handling comments and whitespace"""
    try:
        value = os.getenv(key, str(default))
        if value:
            # Remove comments (everything after #) and strip whitespace
            if '#' in value:
                value = value.split('#')[0]
            value = value.strip()
        return int(value) if value else default
    except (ValueError, TypeError) as e:
        logger.warning(f"Error parsing {key}='{os.getenv(key)}', using default {default}: {e}")
        return default

class APILimiter:
    """Manages API rate limits to avoid exceeding quotas"""
    
    def __init__(self, max_calls_per_minute: int, max_calls_per_day: int):
        self.max_calls_per_minute = max_calls_per_minute
        self.max_calls_per_day = max_calls_per_day
        self.minute_calls = 0
        self.day_calls = 0
        self.last_reset_minute = datetime.now()
        self.last_reset_day = datetime.now().date()
        self.lock = threading.Lock()
    
    def can_make_call(self) -> bool:
        """Check if API call can be made within limits"""
        with self.lock:
            current_time = datetime.now()
            current_date = current_time.date()
            
            # Reset daily counter if day changed
            if current_date > self.last_reset_day:
                self.day_calls = 0
                self.last_reset_day = current_date
            
            # Reset minute counter if minute changed
            if (current_time - self.last_reset_minute).total_seconds() >= 60:
                self.minute_calls = 0
                self.last_reset_minute = current_time
            
            # Check if within limits
            if (self.minute_calls < self.max_calls_per_minute and 
                self.day_calls < self.max_calls_per_day):
                return True
            return False
    
    def increment_counters(self):
        """Increment API call counters"""
        with self.lock:
            self.minute_calls += 1
            self.day_calls += 1


class BaseDataCollector(ABC):
    """Base class for all data collectors"""
    
    def __init__(self, name: str, update_frequency: int = 60):
        """
        Args:
            name: Collector name
            update_frequency: How often to update data (in seconds)
        """
        self.name = name
        self.update_frequency = update_frequency
        self.last_update_time = None
        self.data = None
        self.error_count = 0
        self.max_errors = 3  # Max consecutive errors before fallback
        self.logger = logging.getLogger(f"DataCollector.{name}")
    
    def should_update(self) -> bool:
        """Check if data should be updated based on frequency"""
        if self.last_update_time is None:
            return True
        elapsed = (datetime.now() - self.last_update_time).total_seconds()
        return elapsed >= self.update_frequency
    
    def update(self) -> Dict[str, Any]:
        """Update data if needed and return the latest data"""
        if self.should_update():
            try:
                self.data = self._fetch_data()
                self.last_update_time = datetime.now()
                self.error_count = 0
                self.logger.info(f"Data updated successfully from {self.name}")
            except Exception as e:
                self.error_count += 1
                self.logger.error(f"Error updating data from {self.name}: {str(e)}")
                if self.error_count >= self.max_errors:
                    self.logger.warning(f"Max errors reached for {self.name}. Consider fallback.")
        return self.data
    
    @abstractmethod
    def _fetch_data(self) -> Dict[str, Any]:
        """Fetch data from the source - to be implemented by subclasses"""
        pass


class MarketDataCollector(BaseDataCollector):
    """Collector for cryptocurrency market data with fallback mechanisms"""
    
    def __init__(self, symbols: List[str], update_frequency: int = 60):
        super().__init__("MarketData", update_frequency)
        self.symbols = symbols
        self.active_source = None
        self.config = {}
        
        # Rate limiting for technical indicators - FIXED: Use safe parsing
        self.last_full_update = {}
        self.asset_rotation_index = 0
        self.assets_per_cycle = safe_get_env_int("ASSETS_PER_CYCLE", 3)
        self.full_update_interval = safe_get_env_int("FULL_UPDATE_INTERVAL", 300)  # 5 minutes
        
        # Cache for technical indicators - FIXED: Use safe parsing
        self.indicators_cache = {}
        self.cache_duration = safe_get_env_int("INDICATORS_CACHE_DURATION", 180)  # 3 minutes
        
        # Initialize data sources with rate limiters
        self.sources = {
            "coingecko_api": {
                "limiter": APILimiter(5, 100),  # 5 calls/min, 100 calls/day
                "method": self._fetch_coingecko_api,
                "priority": 1
            },
            "coinbase_api": {
                "limiter": APILimiter(10, 200),  # 10 calls/min, 200 calls/day
                "method": self._fetch_coinbase_api,
                "priority": 2
            },
            "simulated": {
                "limiter": APILimiter(1000, 100000),  # High limits for simulated
                "method": self._fetch_simulated_data,
                "priority": 3
            }
        }
        
        # Sort sources by priority
        self.source_priority = sorted(
            self.sources.keys(), 
            key=lambda x: self.sources[x]["priority"]
        )
    
    def _get_coingecko_config(self):
        """Get the appropriate CoinGecko API configuration based on key type - FIXED"""
        coingecko_api_key = self.config.get("coingecko_api_key")
        
        if not coingecko_api_key:
            # No API key - use free tier without authentication
            return {
                "base_url": "https://api.coingecko.com/api/v3", 
                "headers": {}
            }
        
        # Check if it's a Pro API key (paid plan)
        # Pro keys typically start with "CG-" and are longer, but the safest way is to check plan type
        # For now, we'll assume if key starts with "CG-" and is longer than 32 chars, it's Pro
        if coingecko_api_key.startswith("CG-") and len(coingecko_api_key) > 32:
            # Pro API key format - use pro URL
            return {
                "base_url": "https://pro-api.coingecko.com/api/v3",
                "headers": {"x-cg-pro-api-key": coingecko_api_key}
            }
        else:
            # Demo/Free API key - use standard URL regardless of format
            return {
                "base_url": "https://api.coingecko.com/api/v3",
                "headers": {"x-cg-demo-api-key": coingecko_api_key}
            }

    def _should_update_indicators(self, symbol: str) -> bool:
        """Check if indicators should be updated for this symbol"""
        if symbol not in self.indicators_cache:
            return True
        
        last_update = self.indicators_cache[symbol].get("timestamp")
        if not last_update:
            return True
        
        try:
            last_update_time = datetime.fromisoformat(last_update.replace('Z', '+00:00'))
            elapsed = (datetime.now() - last_update_time).total_seconds()
            return elapsed > self.cache_duration
        except:
            return True

    def _get_symbols_for_update(self) -> List[str]:
        """Get symbols that should be updated this cycle (rate limiting)"""
        base_symbols = [s.split('/')[0] for s in self.symbols]
        
        # Check if it's time for a full update
        now = datetime.now()
        last_full = self.last_full_update.get("timestamp")
        
        if not last_full or (now - last_full).total_seconds() > self.full_update_interval:
            logger.info("Performing full technical indicators update for all assets")
            self.last_full_update["timestamp"] = now
            return base_symbols
        
        # Otherwise, rotate through assets
        start_idx = self.asset_rotation_index
        end_idx = min(start_idx + self.assets_per_cycle, len(base_symbols))
        
        symbols_to_update = base_symbols[start_idx:end_idx]
        
        # Update rotation index
        self.asset_rotation_index = end_idx if end_idx < len(base_symbols) else 0
        
        logger.info(f"Rotating technical indicators update: {symbols_to_update}")
        return symbols_to_update

    def _fetch_historical_prices_and_ohlcv(self, symbol: str, days: int = 30) -> tuple:
        """
        Fetch both historical prices and OHLCV data for technical analysis - ENHANCED VERSION
        """
        base_symbol = symbol.split('/')[0]
        
        symbol_to_id = {
            "BTC": "bitcoin",
            "ETH": "ethereum", 
            "SOL": "solana",
            "XRP": "ripple",
            "DOGE": "dogecoin"
        }
        
        coin_id = symbol_to_id.get(base_symbol, base_symbol.lower())
        
        try:
            # Use the comprehensive API config
            api_config = self._get_coingecko_config()
            
            # Method 1: Try CoinGecko OHLC endpoint with better error handling
            success, prices, ohlcv = self._try_coingecko_ohlc(api_config, coin_id, symbol, days)
            if success:
                return prices, ohlcv
            
            # Method 2: Try market_chart endpoint with constructed OHLCV
            success, prices, ohlcv = self._try_coingecko_market_chart(api_config, coin_id, symbol, days)
            if success:
                return prices, ohlcv
                
            # Method 3: Try alternative endpoint
            success, prices, ohlcv = self._try_coingecko_alternative(api_config, coin_id, symbol, days)
            if success:
                return prices, ohlcv
            
            logger.warning(f"All OHLCV methods failed for {symbol}, using basic price data only")
            return self._fetch_historical_prices_fallback(symbol, days), []
                
        except Exception as e:
            logger.error(f"Error fetching OHLCV data for {symbol}: {str(e)}")
            # Fallback to price history
            return self._fetch_historical_prices_fallback(symbol, days), []

    def _try_coingecko_ohlc(self, api_config: dict, coin_id: str, symbol: str, days: int) -> tuple:
        """Try CoinGecko OHLC endpoint with enhanced error handling"""
        try:
            # CoinGecko OHLC endpoint - try different day parameters
            valid_days = [1, 7, 14, 30, 90, 180, 365]
            
            # Find the closest valid day value
            closest_days = min(valid_days, key=lambda x: abs(x - days))
            
            url = f"{api_config['base_url']}/coins/{coin_id}/ohlc"
            params = {
                "vs_currency": "usd",
                "days": str(closest_days)
            }
            
            logger.debug(f"Trying OHLC endpoint for {symbol}: {url} with params {params}")
            
            response = requests.get(url, params=params, headers=api_config['headers'], timeout=10)
            
            logger.debug(f"OHLC response for {symbol}: status={response.status_code}")
            
            if response.status_code == 200:
                ohlc_data = response.json()
                
                if ohlc_data and len(ohlc_data) > 0:
                    # Convert to our format: [timestamp, open, high, low, close, volume]
                    closing_prices = []
                    ohlcv_data = []
                    
                    for candle in ohlc_data:
                        if len(candle) >= 5:  # timestamp, open, high, low, close
                            timestamp, open_price, high, low, close = candle[:5]
                            closing_prices.append(float(close))
                            
                            # Add volume as 0 since CoinGecko OHLC doesn't include volume
                            ohlcv_data.append([
                                timestamp,
                                float(open_price),
                                float(high), 
                                float(low),
                                float(close),
                                0.0  # Volume not available in this endpoint
                            ])
                    
                    logger.info(f"Successfully fetched {len(closing_prices)} OHLC candles for {symbol}")
                    return True, closing_prices, ohlcv_data
                else:
                    logger.warning(f"Empty OHLC data returned for {symbol}")
                    return False, [], []
            else:
                error_text = response.text[:200] if response.text else "No error text"
                logger.warning(f"OHLC endpoint failed for {symbol}: {response.status_code} - {error_text}")
                return False, [], []
                
        except Exception as e:
            logger.warning(f"Error in OHLC method for {symbol}: {str(e)}")
            return False, [], []

    def _try_coingecko_market_chart(self, api_config: dict, coin_id: str, symbol: str, days: int) -> tuple:
        """Try market_chart endpoint and construct OHLCV from price/volume data"""
        try:
            url = f"{api_config['base_url']}/coins/{coin_id}/market_chart"
            params = {
                "vs_currency": "usd",
                "days": str(days),
                "interval": "daily" if days > 7 else "hourly"
            }
            
            logger.debug(f"Trying market_chart endpoint for {symbol}: {url}")
            
            response = requests.get(url, params=params, headers=api_config['headers'], timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                prices_data = data.get("prices", [])
                volumes_data = data.get("total_volumes", [])
                
                if prices_data and len(prices_data) > 10:  # Need reasonable amount of data
                    closing_prices = [float(price[1]) for price in prices_data]
                    
                    # Construct OHLCV data from available price and volume data
                    ohlcv_data = []
                    
                    # Group prices by time periods to create OHLC
                    time_interval = 24 * 60 * 60 * 1000 if days > 7 else 60 * 60 * 1000  # daily or hourly
                    
                    i = 0
                    while i < len(prices_data):
                        # Collect prices for this time period
                        period_prices = []
                        period_start_time = prices_data[i][0]
                        
                        # Collect all prices in this period
                        while i < len(prices_data) and (prices_data[i][0] - period_start_time) < time_interval:
                            period_prices.append(prices_data[i][1])
                            i += 1
                        
                        if len(period_prices) >= 1:
                            open_price = period_prices[0]
                            close_price = period_prices[-1]
                            high_price = max(period_prices)
                            low_price = min(period_prices)
                            
                            # Find corresponding volume
                            volume = 0.0
                            if volumes_data:
                                # Find volume entry closest to this time
                                for vol_entry in volumes_data:
                                    if abs(vol_entry[0] - period_start_time) < time_interval:
                                        volume = vol_entry[1]
                                        break
                            
                            ohlcv_data.append([
                                period_start_time,
                                float(open_price),
                                float(high_price),
                                float(low_price),
                                float(close_price),
                                float(volume)
                            ])
                    
                    logger.info(f"Constructed {len(ohlcv_data)} OHLCV candles from market_chart for {symbol}")
                    return True, closing_prices, ohlcv_data
                else:
                    logger.warning(f"Insufficient market_chart data for {symbol}")
                    return False, [], []
            else:
                logger.warning(f"Market_chart failed for {symbol}: {response.status_code}")
                return False, [], []
                
        except Exception as e:
            logger.warning(f"Error in market_chart method for {symbol}: {str(e)}")
            return False, [], []

    def _try_coingecko_alternative(self, api_config: dict, coin_id: str, symbol: str, days: int) -> tuple:
        """Try alternative approach using range endpoint or historical data"""
        try:
            # Try the range endpoint if available
            from datetime import datetime, timedelta
            
            end_time = int(datetime.now().timestamp())
            start_time = int((datetime.now() - timedelta(days=days)).timestamp())
            
            url = f"{api_config['base_url']}/coins/{coin_id}/market_chart/range"
            params = {
                "vs_currency": "usd",
                "from": str(start_time),
                "to": str(end_time)
            }
            
            logger.debug(f"Trying range endpoint for {symbol}: {url}")
            
            response = requests.get(url, params=params, headers=api_config['headers'], timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                prices_data = data.get("prices", [])
                
                if prices_data and len(prices_data) > 5:
                    closing_prices = [float(price[1]) for price in prices_data]
                    
                    # Create simple OHLCV with limited data
                    ohlcv_data = []
                    for i in range(0, len(prices_data), max(1, len(prices_data) // 20)):  # Sample data
                        if i < len(prices_data):
                            price_point = prices_data[i]
                            price = float(price_point[1])
                            
                            # Simple OHLCV where O=H=L=C (not ideal but functional)
                            ohlcv_data.append([
                                price_point[0],
                                price,  # open
                                price,  # high
                                price,  # low
                                price,  # close
                                0.0     # volume
                            ])
                    
                    logger.info(f"Alternative method created {len(ohlcv_data)} simple candles for {symbol}")
                    return True, closing_prices, ohlcv_data
                else:
                    logger.warning(f"Alternative method insufficient data for {symbol}")
                    return False, [], []
            else:
                logger.warning(f"Alternative range endpoint failed for {symbol}: {response.status_code}")
                return False, [], []
                
        except Exception as e:
            logger.warning(f"Error in alternative method for {symbol}: {str(e)}")
            return False, [], []

    def _debug_api_response(self, symbol: str, url: str, params: dict, headers: dict, response):
        """Debug helper to log API response details"""
        logger.debug(f"API Debug for {symbol}:")
        logger.debug(f"  URL: {url}")
        logger.debug(f"  Params: {params}")
        logger.debug(f"  Headers: {list(headers.keys())}")
        logger.debug(f"  Status: {response.status_code}")
        
        if response.status_code != 200:
            logger.debug(f"  Error Response: {response.text[:500]}")
        else:
            try:
                data = response.json()
                if isinstance(data, list):
                    logger.debug(f"  Response: List with {len(data)} items")
                elif isinstance(data, dict):
                    logger.debug(f"  Response: Dict with keys: {list(data.keys())}")
                else:
                    logger.debug(f"  Response: {type(data)}")
            except:
                logger.debug(f"  Response: {response.text[:200]}")

    def _fetch_historical_prices_fallback(self, symbol: str, days: int = 30) -> List[float]:
        """Fallback method to fetch just closing prices"""
        base_symbol = symbol.split('/')[0]
        
        symbol_to_id = {
            "BTC": "bitcoin",
            "ETH": "ethereum",
            "SOL": "solana", 
            "XRP": "ripple",
            "DOGE": "dogecoin"
        }
        
        coin_id = symbol_to_id.get(base_symbol, base_symbol.lower())
        
        try:
            # Use the comprehensive API config
            api_config = self._get_coingecko_config()
            
            url = f"{api_config['base_url']}/coins/{coin_id}/market_chart"
            params = {
                "vs_currency": "usd",
                "days": str(days),
                "interval": "daily" if days > 7 else "hourly"
            }
            
            response = requests.get(url, params=params, headers=api_config['headers'])
            
            if response.status_code == 200:
                data = response.json()
                prices = data.get("prices", [])
                closing_prices = [float(price[1]) for price in prices]
                
                logger.info(f"Fetched {len(closing_prices)} historical prices for {symbol}")
                return closing_prices
            else:
                logger.warning(f"Failed to fetch historical data for {symbol}: {response.status_code}")
                return []
                
        except Exception as e:
            logger.error(f"Error fetching historical prices for {symbol}: {str(e)}")
            return []

    def _fetch_data(self) -> Dict[str, Any]:
        """Try to fetch data from sources in priority order"""
        errors = {}
        
        for source_name in self.source_priority:
            source = self.sources[source_name]
            
            if source["limiter"].can_make_call():
                try:
                    self.logger.info(f"Attempting to fetch market data from {source_name}")
                    data = source["method"]()
                    source["limiter"].increment_counters()
                    self.active_source = source_name
                    self.logger.info(f"Successfully fetched market data from {source_name}")
                    return {
                        "source": source_name,
                        "timestamp": datetime.now().isoformat(),
                        "data": data
                    }
                except Exception as e:
                    error_msg = f"Error fetching from {source_name}: {str(e)}"
                    self.logger.warning(error_msg)
                    errors[source_name] = error_msg
            else:
                self.logger.warning(f"Rate limit reached for {source_name}")
        
        # If we get here, all sources failed
        error_details = json.dumps(errors, indent=2)
        self.logger.error(f"All market data sources failed. Details: {error_details}")
        raise Exception(f"All market data sources failed: {error_details}")

    def _fetch_coingecko_api(self) -> Dict[str, Any]:
        """Fetch data from CoinGecko API with technical indicators and candlestick analysis - FIXED"""
        result = {}
        
        # Extract base symbols without trading pair
        base_symbols = [s.split('/')[0] for s in self.symbols]
        
        symbol_to_id = {
            "BTC": "bitcoin",
            "ETH": "ethereum",
            "SOL": "solana",
            "XRP": "ripple", 
            "DOGE": "dogecoin"
        }
        
        # Get symbols that need indicator updates (rate limiting)
        symbols_to_update = self._get_symbols_for_update()
        
        # Get IDs for the requested symbols
        coin_ids = []
        for symbol in base_symbols:
            if symbol in symbol_to_id:
                coin_ids.append(symbol_to_id[symbol])
            else:
                coin_ids.append(symbol.lower())
                
        ids_param = ",".join(coin_ids)
        
        # Use the FIXED API config
        api_config = self._get_coingecko_config()
        
        # Build the API URL for market data
        market_url = f"{api_config['base_url']}/coins/markets?vs_currency=usd&ids={ids_param}&order=market_cap_desc&sparkline=false&price_change_percentage=24h"
        
        # Make the API request
        response = requests.get(market_url, headers=api_config['headers'])
        
        if response.status_code != 200:
            raise Exception(f"CoinGecko API returned status code {response.status_code}: {response.text}")
        
        market_data = response.json()
        
        # Map CoinGecko data to our standard format
        id_to_symbol = {v: k for k, v in symbol_to_id.items()}
        
        for item in market_data:
            coin_id = item.get("id")
            coin_symbol = item.get("symbol", "").upper()
            
            # Determine the symbol
            if coin_id in id_to_symbol:
                symbol = id_to_symbol[coin_id]
            else:
                symbol = coin_symbol
            
            # Find the full symbol in our symbols list
            matching_symbols = [s for s in self.symbols if s.startswith(symbol + "/")]
            if matching_symbols:
                full_symbol = matching_symbols[0]
            else:
                full_symbol = f"{symbol}/USDT"
            
            current_price = item.get("current_price")
            
            # FIXED: Always try to get technical indicators when needed
            technical_indicators = {}
            
            # Check if we should update indicators for this symbol
            if symbol in symbols_to_update and self._should_update_indicators(symbol):
                try:
                    logger.info(f"Updating technical indicators for {symbol}")
                    
                    # Fetch both prices and OHLCV data
                    historical_prices, ohlcv_data = self._fetch_historical_prices_and_ohlcv(full_symbol, days=365)
                    
                    if historical_prices:
                        technical_indicators = get_indicators_for_asset(
                            symbol, historical_prices, current_price, ohlcv_data
                        )
                        
                        # Cache the indicators
                        self.indicators_cache[symbol] = {
                            "indicators": technical_indicators,
                            "timestamp": datetime.now().isoformat()
                        }
                        
                        logger.info(f"Updated and cached technical indicators for {symbol}: {technical_indicators.get('technical_summary', 'neutral')}")
                    else:
                        logger.warning(f"No historical data for {symbol}, using cached indicators if available")
                        
                except Exception as e:
                    logger.error(f"Error calculating technical indicators for {symbol}: {str(e)}")
            
            # ALWAYS use cached indicators if available (not just when updating)
            if symbol in self.indicators_cache:
                cached_data = self.indicators_cache[symbol]
                if cached_data and "indicators" in cached_data:
                    technical_indicators = cached_data["indicators"]
                    logger.debug(f"Using cached technical indicators for {symbol}")
            
            result[full_symbol] = {
                "price": current_price,
                "volume_24h": item.get("total_volume"),
                "change_24h_percent": item.get("price_change_percentage_24h"),
                "high_24h": item.get("high_24h"),
                "low_24h": item.get("low_24h"),
                "market_cap": item.get("market_cap"),
                "last_updated": item.get("last_updated"),
                "technical_indicators": technical_indicators
            }
        
        # Handle symbols that weren't found in the market data
        for symbol in self.symbols:
            if symbol not in result:
                base_symbol = symbol.split('/')[0]
                logger.warning(f"No data from CoinGecko for {symbol}, trying simple price endpoint")
                
                coin_id = symbol_to_id.get(base_symbol, base_symbol.lower())
                simple_price_url = f"{api_config['base_url']}/simple/price?ids={coin_id}&vs_currencies=usd&include_24hr_vol=true&include_24hr_change=true"
                
                try:
                    simple_response = requests.get(simple_price_url, headers=api_config['headers'])
                    
                    if simple_response.status_code == 200:
                        simple_data = simple_response.json()
                        
                        if coin_id in simple_data:
                            current_price = simple_data[coin_id].get("usd")
                            
                            # Get technical indicators if this symbol should be updated
                            technical_indicators = {}
                            if (base_symbol in symbols_to_update and 
                                self._should_update_indicators(base_symbol)):
                                
                                try:
                                    historical_prices, ohlcv_data = self._fetch_historical_prices_and_ohlcv(symbol, days=200)
                                    
                                    if historical_prices:
                                        technical_indicators = get_indicators_for_asset(
                                            base_symbol, historical_prices, current_price, ohlcv_data
                                        )
                                        
                                        # Cache the indicators
                                        self.indicators_cache[base_symbol] = {
                                            "indicators": technical_indicators,
                                            "timestamp": datetime.now().isoformat()
                                        }
                                        
                                except Exception as e:
                                    logger.error(f"Error calculating technical indicators for {base_symbol}: {str(e)}")
                            
                            # Use cached indicators if available
                            if base_symbol in self.indicators_cache:
                                cached_data = self.indicators_cache[base_symbol]
                                if cached_data and "indicators" in cached_data:
                                    technical_indicators = cached_data["indicators"]
                            
                            result[symbol] = {
                                "price": current_price,
                                "volume_24h": simple_data[coin_id].get("usd_24h_vol"),
                                "change_24h_percent": simple_data[coin_id].get("usd_24h_change"),
                                "high_24h": None,
                                "low_24h": None,
                                "technical_indicators": technical_indicators
                            }
                            
                except Exception as e:
                    logger.error(f"Error fetching simple price data for {symbol}: {str(e)}")
        
        return result
    
    def _fetch_coinbase_api(self) -> Dict[str, Any]:
        """Fetch data from Coinbase API directly - IMPROVED"""
        result = {}
        base_url = "https://api.coinbase.com/v2"
        
        for symbol in self.symbols:
            try:
                # Convert CCXT symbol format to Coinbase format (e.g., BTC/USDT -> BTC-USD)
                # Coinbase uses USD instead of USDT
                cb_symbol = symbol.replace("/USDT", "-USD").replace("/", "-")
                
                # Get current price
                price_url = f"{base_url}/prices/{cb_symbol}/spot"
                price_response = requests.get(price_url)
                
                if price_response.status_code == 200:
                    price_data = price_response.json()
                    current_price = float(price_data["data"]["amount"])
                    
                    # Try to get 24h data
                    buy_url = f"{base_url}/prices/{cb_symbol}/buy"
                    sell_url = f"{base_url}/prices/{cb_symbol}/sell"
                    
                    buy_response = requests.get(buy_url)
                    sell_response = requests.get(sell_url)
                    
                    # Calculate a rough spread as a proxy for volatility
                    spread = 0
                    if buy_response.status_code == 200 and sell_response.status_code == 200:
                        buy_price = float(buy_response.json()["data"]["amount"])
                        sell_price = float(sell_response.json()["data"]["amount"])
                        spread = abs(buy_price - sell_price) / current_price * 100
                    
                    # Get base symbol for technical indicators
                    base_symbol = symbol.split('/')[0]
                    symbols_to_update = self._get_symbols_for_update()
                    
                    # Get technical indicators if needed
                    technical_indicators = {}
                    if (base_symbol in symbols_to_update and 
                        self._should_update_indicators(base_symbol)):
                        
                        try:
                            historical_prices, ohlcv_data = self._fetch_historical_prices_and_ohlcv(symbol, days=200)
                            
                            if historical_prices:
                                technical_indicators = get_indicators_for_asset(
                                    base_symbol, historical_prices, current_price, ohlcv_data
                                )
                                
                                # Cache the indicators
                                self.indicators_cache[base_symbol] = {
                                    "indicators": technical_indicators,
                                    "timestamp": datetime.now().isoformat()
                                }
                                
                        except Exception as e:
                            logger.error(f"Error calculating technical indicators for {base_symbol}: {str(e)}")
                    
                    # Use cached indicators if available
                    if base_symbol in self.indicators_cache:
                        cached_data = self.indicators_cache[base_symbol]
                        if cached_data and "indicators" in cached_data:
                            technical_indicators = cached_data["indicators"]
                    
                    result[symbol] = {
                        "price": current_price,
                        "currency": price_data["data"]["currency"],
                        "volume_24h": None,  # Coinbase public API doesn't provide volume easily
                        "change_24h_percent": None,  # Would need historical data
                        "spread_percent": spread,
                        "technical_indicators": technical_indicators
                    }
                else:
                    logger.warning(f"Failed to get Coinbase price for {symbol}: {price_response.status_code}")
                    
            except Exception as e:
                logger.error(f"Error fetching Coinbase data for {symbol}: {str(e)}")
        
        return result
    
    def _fetch_simulated_data(self) -> Dict[str, Any]:
        """Generate simulated data with OHLCV for candlestick analysis"""
        self.logger.warning("Using simulated market data as fallback")
        result = {}
        
        for symbol in self.symbols:
            base_price = {
                "BTC/USDT": 50000,
                "ETH/USDT": 3000,
                "XRP/USDT": 0.5,
                "SOL/USDT": 100,
                "DOGE/USDT": 0.1
            }.get(symbol, 10)
            
            # Add some randomness
            price = base_price * (1 + random.uniform(-0.02, 0.02))
            
            # Generate simulated OHLCV data for candlestick analysis
            ohlcv = []
            current_time = int(time.time())
            for i in range(60):
                candle_time = current_time - (59 - i) * 60
                base_candle_price = base_price * (1 + random.uniform(-0.05, 0.05))
                
                # Create realistic OHLC with some patterns
                open_price = base_candle_price * (1 + random.uniform(-0.02, 0.02))
                close_price = base_candle_price * (1 + random.uniform(-0.02, 0.02))
                high_price = max(open_price, close_price) * (1 + random.uniform(0, 0.03))
                low_price = min(open_price, close_price) * (1 - random.uniform(0, 0.03))
                volume = random.uniform(10, 100)
                
                candle = [
                    candle_time,
                    open_price,
                    high_price,
                    low_price,
                    close_price,
                    volume
                ]
                ohlcv.append(candle)
            
            # Calculate technical indicators using simulated data
            base_symbol = symbol.split('/')[0]
            historical_prices = [candle[4] for candle in ohlcv]  # closing prices
            
            technical_indicators = {}
            try:
                technical_indicators = get_indicators_for_asset(
                    base_symbol, historical_prices, price, ohlcv
                )
            except Exception as e:
                self.logger.error(f"Error calculating simulated technical indicators for {base_symbol}: {str(e)}")
                technical_indicators = {}
            
            result[symbol] = {
                "price": price,
                "volume_24h": random.uniform(1000000, 10000000),
                "change_24h_percent": random.uniform(-5, 5),
                "high_24h": price * 1.05,
                "low_24h": price * 0.95,
                "ohlcv_1h": ohlcv,
                "technical_indicators": technical_indicators,
                "is_simulated": True
            }
        
        return result
    
    def get_active_source(self) -> str:
        """Return the name of the currently active data source"""
        return self.active_source


class SentimentDataCollector(BaseDataCollector):
    """Collector for cryptocurrency sentiment data with fallback mechanisms - FIXED VERSION"""
    
    def __init__(self, symbols: List[str], news_api_key: str = None, openai_api_key: str = None):
        """
        Args:
            symbols: List of cryptocurrency symbols to collect sentiment for
            news_api_key: API key for NewsAPI
            openai_api_key: API key for OpenAI (for fallback sentiment analysis)
        """
        super().__init__("SentimentData", update_frequency=300)  # Update every 5 minutes by default
        self.symbols = [s.split('/')[0] for s in symbols]  # Extract base currency (BTC from BTC/USDT)
        self.news_api_key = news_api_key
        self.openai_api_key = openai_api_key
        
        # Initialize data sources with rate limiters
        self.sources = {
            "news_api": {
                "limiter": APILimiter(max_calls_per_minute=5, max_calls_per_day=100),
                "method": self._fetch_news_api,
                "priority": 1
            },
            "reddit_api": {
                "limiter": APILimiter(max_calls_per_minute=10, max_calls_per_day=200),
                "method": self._fetch_reddit_api,
                "priority": 2
            },
            "gpt_sentiment": {
                "limiter": APILimiter(max_calls_per_minute=3, max_calls_per_day=50),
                "method": self._fetch_gpt_sentiment,
                "priority": 3
            },
            "simulated": {
                "limiter": APILimiter(max_calls_per_minute=1000, max_calls_per_day=100000),
                "method": self._fetch_simulated_sentiment,
                "priority": 4
            }
        }
        
        # Sort sources by priority
        self.source_priority = sorted(
            self.sources.keys(), 
            key=lambda x: self.sources[x]["priority"]
        )
    
    def _fetch_data(self) -> Dict[str, Any]:
        """Try to fetch data from sources in priority order"""
        errors = {}
        
        for source_name in self.source_priority:
            source = self.sources[source_name]
            
            if source["limiter"].can_make_call():
                try:
                    self.logger.info(f"Attempting to fetch sentiment data from {source_name}")
                    data = source["method"]()
                    source["limiter"].increment_counters()
                    self.logger.info(f"Successfully fetched sentiment data from {source_name}")
                    return {
                        "source": source_name,
                        "timestamp": datetime.now().isoformat(),
                        "data": data
                    }
                except Exception as e:
                    error_msg = f"Error fetching from {source_name}: {str(e)}"
                    self.logger.warning(error_msg)
                    errors[source_name] = error_msg
            else:
                self.logger.warning(f"Rate limit reached for {source_name}")
        
        # If we get here, all sources failed
        error_details = json.dumps(errors, indent=2)
        self.logger.error(f"All sentiment data sources failed. Details: {error_details}")
        raise Exception(f"All sentiment data sources failed: {error_details}")
    
    def _safe_get_string(self, data: Dict[str, Any], key: str, default: str = "") -> str:
        """Safely extract string value from dictionary, handling None values"""
        try:
            value = data.get(key, default)
            if value is None:
                return default
            if isinstance(value, str):
                return value.strip()
            # Convert non-string values to string
            return str(value).strip()
        except Exception:
            return default
    
    def _clean_article_data(self, article: Dict[str, Any]) -> Dict[str, Any]:
        """Clean article data to ensure no None values cause issues downstream"""
        cleaned = {}
        
        # Define expected string fields and their defaults
        string_fields = {
            "title": "No title",
            "description": "No description", 
            "content": "No content",
            "url": "",
            "urlToImage": "",
            "publishedAt": "",
            "search_query": "",
            "fetched_at": ""
        }
        
        # Clean string fields
        for field, default in string_fields.items():
            cleaned[field] = self._safe_get_string(article, field, default)
        
        # Handle source object
        source = article.get("source", {})
        if isinstance(source, dict):
            cleaned["source"] = {
                "id": self._safe_get_string(source, "id", "unknown"),
                "name": self._safe_get_string(source, "name", "Unknown Source")
            }
        else:
            cleaned["source"] = {"id": "unknown", "name": "Unknown Source"}
        
        # Handle other fields
        cleaned["coin_mentions"] = article.get("coin_mentions", [])
        
        return cleaned
    
    def _calculate_recency_weight(self, published_at_str: str) -> float:
        """Calculate weight based on article recency"""
        if not published_at_str:
            return 1.0
        
        try:
            # Handle various date formats
            if 'T' in published_at_str:
                if published_at_str.endswith('Z'):
                    published_time = datetime.fromisoformat(published_at_str.replace('Z', '+00:00'))
                else:
                    published_time = datetime.fromisoformat(published_at_str)
            else:
                # Try parse with several formats
                for fmt in ['%Y-%m-%d %H:%M:%S', '%Y-%m-%d']:
                    try:
                        published_time = datetime.strptime(published_at_str, fmt)
                        break
                    except:
                        continue
                else:
                    return 1.0  # Default if no format matched
            
            # Calculate time difference
            now = datetime.now()
            if published_time.tzinfo:
                now = datetime.now(published_time.tzinfo)
                
            hours_ago = (now - published_time).total_seconds() / 3600
            
            # Articles in the last 6 hours get higher weight
            if hours_ago <= 6:
                return 1.5
            # Articles in the last 24 hours get normal weight
            elif hours_ago <= 24:
                return 1.0
            # Older articles get reduced weight
            else:
                return max(0.5, 1.0 - (hours_ago - 24) / 72)  # Gradually decrease weight up to 0.5
        except Exception as e:
            self.logger.debug(f"Error calculating recency weight: {str(e)}")
            return 1.0  # Default weight if parsing fails
    
    def _calculate_article_sentiment(self, article: Dict[str, Any], symbol: str) -> Optional[float]:
        """Calculate sentiment score for an article - FIXED with better error handling"""
        try:
            # FIXED: Safe text extraction
            title = self._safe_get_string(article, "title").lower()
            description = self._safe_get_string(article, "description").lower()
            combined_text = f"{title} {description}".strip()
            
            if not combined_text:
                return 0  # Neutral if no text
            
            # Enhanced keyword lists for better sentiment detection
            positive_keywords = [
                "bullish", "surge", "rally", "gain", "growth", "soar", "moon", "up", "rise",
                "adoption", "partnership", "upgrade", "breakthrough", "success", "positive",
                "optimistic", "recover", "boost", "strong", "high", "approve", "launch",
                "buy", "pump", "bull", "breakthrough", "milestone", "institutional"
            ]
            
            negative_keywords = [
                "bearish", "crash", "fall", "plunge", "drop", "decline", "down", "loss",
                "sell-off", "dump", "collapse", "fear", "concern", "warn", "risk", "hack",
                "regulation", "ban", "investigation", "fraud", "scam", "trouble", "weak",
                "sell", "bear", "correction", "bubble", "volatile", "uncertain"
            ]
            
            # Count keyword occurrences with weighting
            positive_count = 0
            negative_count = 0
            
            for word in positive_keywords:
                count = combined_text.count(word)
                positive_count += count
            
            for word in negative_keywords:
                count = combined_text.count(word)
                negative_count += count
            
            # Calculate sentiment score with recency weighting
            recency_weight = self._calculate_recency_weight(article.get("publishedAt", ""))
            
            if positive_count > negative_count:
                sentiment_strength = min(positive_count / 3.0, 1.0)  # Cap at 1.0
                return sentiment_strength * 0.7 * recency_weight  # Scale and apply recency
            elif negative_count > positive_count:
                sentiment_strength = min(negative_count / 3.0, 1.0)  # Cap at 1.0
                return -sentiment_strength * 0.7 * recency_weight  # Scale and apply recency
            else:
                return 0  # Neutral
                
        except Exception as e:
            self.logger.error(f"Error calculating sentiment for article: {str(e)}")
            return None
    
    def _fetch_news_api(self) -> Dict[str, Any]:
        """Fetch broad news that could impact crypto markets - FIXED VERSION"""
        if not self.news_api_key:
            raise ValueError("NewsAPI key not provided")
        
        result = {}
        base_url = "https://newsapi.org/v2/everything"
        
        # Broader search terms to catch all market-moving news
        search_queries = [
            # Direct crypto queries
            "cryptocurrency OR bitcoin OR ethereum",
            
            # Macroeconomic factors
            "federal reserve OR interest rates OR inflation",
            "SEC regulation OR financial policy",
            
            # Geopolitical events
            "sanctions OR trade war OR geopolitical",
            
            # Technology and adoption
            "blockchain adoption OR crypto partnership",
            "hack OR security breach cryptocurrency",
            
            # Market conditions
            "stock market crash OR recession OR economic crisis"
        ]
        
        all_articles = []
        seen_urls = set()  # Avoid duplicates
        
        # Fetch articles for each query
        for query in search_queries:
            params = {
                "q": query,
                "sortBy": "publishedAt",
                "language": "en",
                "pageSize": 5,  # Limit per query to avoid too many articles
                "apiKey": self.news_api_key,
                "from": (datetime.now() - timedelta(hours=24)).isoformat()  # Last 24 hours
            }
            
            try:
                response = requests.get(base_url, params=params, timeout=10)
                
                if response.status_code != 200:
                    self.logger.warning(f"NewsAPI returned status {response.status_code} for query '{query}'")
                    continue
                
                data = response.json()
                
                if data.get("status") == "ok":
                    articles = data.get("articles", [])
                    for article in articles:
                        try:
                            # FIXED: Robust null checking
                            url = self._safe_get_string(article, "url")
                            if url and url not in seen_urls:
                                seen_urls.add(url)
                                
                                # Add metadata for GPT analysis - FIXED with safe string handling
                                article["search_query"] = query
                                article["fetched_at"] = datetime.now().isoformat()
                                
                                # FIXED: Safe text extraction with null checks
                                title = self._safe_get_string(article, "title")
                                description = self._safe_get_string(article, "description")
                                text = f"{title} {description}".lower().strip()
                                
                                coin_mentions = []
                                
                                coin_keywords = {
                                    "BTC": ["bitcoin", "btc"],
                                    "ETH": ["ethereum", "eth"],
                                    "SOL": ["solana", "sol"],
                                    "XRP": ["ripple", "xrp"],
                                    "DOGE": ["dogecoin", "doge"]
                                }
                                
                                # FIXED: Safe keyword matching
                                if text:  # Only process if we have valid text
                                    for coin, keywords in coin_keywords.items():
                                        if any(keyword in text for keyword in keywords):
                                            coin_mentions.append(coin)
                                
                                article["coin_mentions"] = coin_mentions
                                
                                # FIXED: Clean up article data to ensure no None values cause issues later
                                article = self._clean_article_data(article)
                                all_articles.append(article)
                        except Exception as e:
                            self.logger.warning(f"Error processing article: {str(e)}")
                            continue
                            
                else:
                    self.logger.warning(f"NewsAPI error for query '{query}': {data.get('message', 'Unknown error')}")
                    
            except Exception as e:
                self.logger.warning(f"Error fetching news for query '{query}': {str(e)}")
        
        # Sort by publish time and limit to configured amount - FIXED: Use safe parsing
        all_articles.sort(key=lambda x: x.get("publishedAt", ""), reverse=True)
        limited_articles = all_articles[:safe_get_env_int('NEWS_ARTICLE_LIMIT', 15)]
        
        self.logger.info(f"Fetched {len(limited_articles)} unique news articles from {len(all_articles)} total")
        
        # Store articles for news impact analysis
        result["articles"] = limited_articles
        result["article_count"] = len(limited_articles)
        result["timestamp"] = datetime.now().isoformat()
        
        # FIXED: Enhanced sentiment analysis with better error handling
        for symbol in self.symbols:
            try:
                # Filter articles mentioning this symbol
                symbol_articles = [a for a in limited_articles if symbol in a.get("coin_mentions", [])]
                
                if symbol_articles:
                    # FIXED: Safe sentiment scoring
                    sentiment_scores = []
                    for article in symbol_articles:
                        score = self._calculate_article_sentiment(article, symbol)
                        if score is not None:
                            sentiment_scores.append(score)
                    
                    avg_sentiment = sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0
                else:
                    avg_sentiment = 0
                
                result[symbol] = {
                    "overall_sentiment": avg_sentiment,
                    "sentiment_category": self._categorize_sentiment(avg_sentiment),
                    "article_count": len(symbol_articles),
                    "confidence": min(1.0, len(symbol_articles) / 5)  # More articles = higher confidence
                }
                
            except Exception as e:
                self.logger.error(f"Error processing sentiment for {symbol}: {str(e)}")
                # Provide default values if processing fails
                result[symbol] = {
                    "overall_sentiment": 0,
                    "sentiment_category": "neutral",
                    "article_count": 0,
                    "confidence": 0
                }
        
        return result
    
    def _post_mentions_crypto(self, post: Dict[str, Any], crypto_name: str, symbol: str) -> bool:
        """Check if Reddit post mentions the specified cryptocurrency - FIXED"""
        try:
            post_data = post.get("data", {})
            title = self._safe_get_string(post_data, "title").lower()
            
            if not title:
                return False
            
            # Check for various mentions
            mentions = [
                crypto_name.lower() in title,
                symbol.lower() in title,
                f"${symbol.lower()}" in title
            ]
            
            return any(mentions)
            
        except Exception as e:
            self.logger.debug(f"Error checking crypto mention in post: {str(e)}")
            return False
    
    def _analyze_reddit_post(self, post: Dict[str, Any], crypto_data: Dict[str, str], symbol: str) -> Optional[Dict[str, Any]]:
        """Analyze individual Reddit post for sentiment - FIXED"""
        try:
            post_data_obj = post.get("data", {})
            title = self._safe_get_string(post_data_obj, "title").lower()
            
            if not title:
                return None
            
            # Enhanced keyword lists for more varied sentiment detection
            positive_keywords = [
                "bullish", "moon", "gain", "profit", "buy", "hodl", "up", "lambo",
                "adoption", "opportunity", "potential", "support", "partnership", "pump",
                "optimistic", "promising", "recovery", "milestone", "upgrade", "innovation",
                "rise", "surge", "rally", "strong", "positive", "good", "great"
            ]
            negative_keywords = [
                "bearish", "crash", "sell", "dump", "loss", "down", "fud", "scam",
                "sell-off", "correction", "risk", "bubble", "regulation", "concern",
                "warning", "troubled", "pessimistic", "uncertainty", "investigation", "ban",
                "fall", "drop", "decline", "weak", "bad", "terrible", "fear"
            ]
            
            # Check if post is directly relevant
            crypto_name = crypto_data['name']
            directly_relevant = (
                crypto_name.lower() in title or 
                symbol.lower() in title or
                f"${symbol.lower()}" in title
            )
            
            # Count positive and negative keywords
            positive_count = sum(1 for word in positive_keywords if word in title)
            negative_count = sum(1 for word in negative_keywords if word in title)
            
            # Upvote data as sentiment indicator - with safe conversion
            try:
                upvote_ratio = float(post_data_obj.get("upvote_ratio", 0.5))
                upvotes = int(post_data_obj.get("ups", 0))
            except (ValueError, TypeError):
                upvote_ratio = 0.5
                upvotes = 0
            
            # Age of the post (newer posts get higher weight)
            try:
                created_utc = float(post_data_obj.get("created_utc", 0))
                post_age_hours = (datetime.now().timestamp() - created_utc) / 3600
                recency_weight = 1.5 if post_age_hours < 12 else 1.0 if post_age_hours < 48 else 0.7
            except (ValueError, TypeError):
                recency_weight = 1.0
            
            # Relevance weight
            relevance_weight = 1.5 if directly_relevant else 1.0
            # Subreddit weight (dedicated subreddit gets more weight)
            subreddit_weight = 1.2 if post_data_obj.get("subreddit") == crypto_data['subreddit'] else 1.0
            
            # Combined score calculation
            if positive_count == 0 and negative_count == 0:
                # No sentiment keywords found - use upvote ratio as a weak signal
                word_sentiment = (upvote_ratio - 0.5) * 0.5  # Scale from -0.25 to 0.25
            else:
                word_sentiment = (positive_count - negative_count) / (positive_count + negative_count + 1)
            
            upvote_sentiment = (upvote_ratio - 0.5) * 2  # Scale from -1 to 1
            
            # Weighted score components
            word_weight = 0.6
            upvote_weight = 0.4
            
            base_score = (word_sentiment * word_weight) + (upvote_sentiment * upvote_weight)
            weighted_score = base_score * recency_weight * relevance_weight * subreddit_weight
            
            # Scale by post popularity (max upvotes considered is 1000)
            popularity_factor = min(1.0, upvotes / 1000) * 0.5 + 0.5  # From 0.5 to 1.0
            final_score = weighted_score * popularity_factor
            
            return {
                "title": self._safe_get_string(post_data_obj, "title"),
                "url": f"https://www.reddit.com{self._safe_get_string(post_data_obj, 'permalink')}",
                "upvotes": upvotes,
                "upvote_ratio": upvote_ratio,
                "sentiment_score": final_score,
                "subreddit": self._safe_get_string(post_data_obj, "subreddit"),
                "created_utc": post_data_obj.get("created_utc")
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing Reddit post: {str(e)}")
            return None
    
    def _fetch_reddit_api(self) -> Dict[str, Any]:
        """Fetch sentiment data from Reddit API with enhanced analysis - FIXED VERSION"""
        result = {}
        
        for symbol in self.symbols:
            try:
                # Get crypto name and subreddit from symbol
                crypto_data = {
                    "BTC": {"name": "Bitcoin", "subreddit": "bitcoin"},
                    "ETH": {"name": "Ethereum", "subreddit": "ethereum"},
                    "XRP": {"name": "Ripple", "subreddit": "ripple"},
                    "SOL": {"name": "Solana", "subreddit": "solana"},
                    "DOGE": {"name": "Dogecoin", "subreddit": "dogecoin"}
                }.get(symbol, {"name": symbol, "subreddit": "cryptocurrency"})
                
                # Reddit API endpoints (using public endpoints without auth for simplicity)
                # Try both dedicated subreddit and cryptocurrency subreddit
                subreddits = [crypto_data['subreddit'], "cryptocurrency"]
                all_posts = []
                
                for subreddit in subreddits:
                    url = f"https://www.reddit.com/r/{subreddit}/hot.json?limit=25"
                    headers = {"User-Agent": "Crypto Sentiment Analyzer v1.0"}
                    
                    try:
                        response = requests.get(url, headers=headers, timeout=10)
                        
                        if response.status_code != 200:
                            self.logger.warning(f"Reddit API returned status {response.status_code} for {subreddit}")
                            continue
                        
                        data = response.json()
                        
                        posts = data.get("data", {}).get("children", [])
                        if subreddit == "cryptocurrency":
                            # Filter posts from cryptocurrency subreddit to only include relevant ones
                            posts = [
                                post for post in posts 
                                if self._post_mentions_crypto(post, crypto_data['name'], symbol)
                            ]
                        
                        # Add subreddit info to each post
                        for post in posts:
                            post_data = post.get("data", {})
                            post_data["subreddit"] = subreddit
                        
                        all_posts.extend(posts)
                    except Exception as e:
                        self.logger.warning(f"Error fetching Reddit data for {subreddit}: {str(e)}")
                
                # FIXED: Enhanced sentiment analysis with better error handling
                sentiment_scores = []
                post_data = []
                
                for post in all_posts:
                    try:
                        post_analysis = self._analyze_reddit_post(post, crypto_data, symbol)
                        if post_analysis:
                            sentiment_scores.append(post_analysis["sentiment_score"])
                            post_data.append(post_analysis)
                    except Exception as e:
                        self.logger.warning(f"Error analyzing Reddit post: {str(e)}")
                        continue
                
                # Calculate overall sentiment
                if sentiment_scores:
                    avg_sentiment = sum(sentiment_scores) / len(sentiment_scores)
                    # Calculate confidence based on post count
                    confidence = min(1.0, len(sentiment_scores) / 10)
                else:
                    avg_sentiment = 0
                    confidence = 0
                
                result[symbol] = {
                    "overall_sentiment": avg_sentiment,
                    "sentiment_category": self._categorize_sentiment(avg_sentiment),
                    "post_count": len(all_posts),
                    "posts": sorted(post_data, key=lambda x: x.get("sentiment_score", 0), reverse=True)[:5],
                    "confidence": confidence,
                    "timestamp": datetime.now().isoformat()
                }
                
                # Log the sentiment results for debugging
                self.logger.debug(f"Reddit sentiment for {symbol}: {avg_sentiment:.2f} "
                                 f"({result[symbol]['sentiment_category']}) "
                                 f"with {confidence:.2f} confidence")
                
            except Exception as e:
                self.logger.error(f"Error processing Reddit sentiment for {symbol}: {str(e)}")
                # Provide default values if processing fails
                result[symbol] = {
                    "overall_sentiment": 0,
                    "sentiment_category": "neutral",
                    "post_count": 0,
                    "posts": [],
                    "confidence": 0,
                    "timestamp": datetime.now().isoformat()
                }
        
        return result
    
    def _fetch_gpt_sentiment(self) -> Dict[str, Any]:
        """Use GPT to analyze sentiment based on recent news"""
        if not self.openai_api_key:
            raise ValueError("OpenAI API key not provided")
        
        client = OpenAI(api_key=self.openai_api_key)
        result = {}
        
        # First get some recent news headlines for context
        recent_news = {}
        for symbol in self.symbols:
            crypto_name = {
                "BTC": "Bitcoin",
                "ETH": "Ethereum",
                "XRP": "Ripple",
                "SOL": "Solana",
                "DOGE": "Dogecoin"
            }.get(symbol, symbol)
            
            # Use a free news API if available, or web search, or manually curated headlines
            search_url = f"https://api.coingecko.com/api/v3/coins/{crypto_name.lower()}/status_updates"
            try:
                response = requests.get(search_url)
                data = response.json()
                updates = data.get("status_updates", [])
                headlines = [update.get("description") for update in updates[:5]]
            except:
                # Fallback to manually defined recent headlines
                headlines = [
                    f"{crypto_name} shows volatility amid market uncertainty",
                    f"Analysts divided on {crypto_name}'s short-term prospects",
                    f"Trading volume for {crypto_name} increases by 15%"
                ]
            
            recent_news[symbol] = headlines
        
        # Ask GPT for sentiment analysis
        for symbol in self.symbols:
            crypto_name = {
                "BTC": "Bitcoin",
                "ETH": "Ethereum",
                "XRP": "Ripple",
                "SOL": "Solana",
                "DOGE": "Dogecoin"
            }.get(symbol, symbol)
            
            headlines = recent_news.get(symbol, [])
            headlines_text = "\n".join(headlines)
            
            prompt = f"""
            Analyze the sentiment for {crypto_name} ({symbol}) based on these recent headlines:
            
            {headlines_text}
            
            Rate the sentiment on a scale from -1 (extremely negative) to 1 (extremely positive).
            Provide your assessment in JSON format with these fields:
            - overall_sentiment: a number between -1 and 1
            - sentiment_category: one of ["very_bearish", "bearish", "neutral", "bullish", "very_bullish"]
            - rationale: brief explanation of your assessment
            - confidence: a number between 0 and 1 indicating how confident you are
            """
            
            try:
                # Check if the model supports JSON response format
                json_format_models = ["gpt-4-turbo", "gpt-3.5-turbo", "gpt-4-0125-preview", "gpt-4-1106-preview"]
                gpt_model = os.getenv("GPT_MODEL", "gpt-3.5-turbo")
                use_json_format = gpt_model in json_format_models
                
                api_params = {
                    "model": gpt_model,
                    "messages": [
                        {"role": "system", "content": "You are a cryptocurrency market sentiment analyzer that provides detailed analysis."},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.3
                }
                
                # Only add response_format for supported models
                if use_json_format:
                    api_params["response_format"] = {"type": "json_object"}
                
                response = client.chat.completions.create(**api_params)
                
                content = response.choices[0].message.content
                
                # Try to parse as JSON
                try:
                    gpt_response = json.loads(content)
                except json.JSONDecodeError:
                    # If JSON parsing fails, extract sentiment manually
                    self.logger.warning(f"Failed to parse GPT response as JSON for {symbol}")
                    
                    # Simple extraction using regexes or string searches would go here
                    # For now, default to neutral
                    gpt_response = {
                        "overall_sentiment": 0,
                        "sentiment_category": "neutral",
                        "rationale": "Could not parse sentiment from response",
                        "confidence": 0.3
                    }
                
                result[symbol] = {
                    "overall_sentiment": gpt_response.get("overall_sentiment", 0),
                    "sentiment_category": gpt_response.get("sentiment_category", "neutral"),
                    "rationale": gpt_response.get("rationale", ""),
                    "confidence": gpt_response.get("confidence", 0.5),
                    "headlines_analyzed": headlines,
                    "timestamp": datetime.now().isoformat()
                }
            except Exception as e:
                # Fallback if GPT response isn't valid JSON
                self.logger.error(f"Error in GPT analysis for {symbol}: {str(e)}")
                result[symbol] = {
                    "overall_sentiment": 0,
                    "sentiment_category": "neutral",
                    "rationale": "Could not analyze sentiment reliably",
                    "confidence": 0.3,
                    "timestamp": datetime.now().isoformat()
                }
        
        return result
    
    def _fetch_simulated_sentiment(self) -> Dict[str, Any]:
        """Generate simulated sentiment data as a last resort"""
        self.logger.warning("Using simulated sentiment data as fallback")
        result = {}
        
        for symbol in self.symbols:
            # Generate varied sentiment scores rather than uniform values
            # Use a distribution skewed slightly positive for crypto sentiment
            sentiment_score = random.betavariate(2, 1.8) * 2 - 1  # Range from -1 to 1, slightly positive-biased
            
            # Add some per-symbol variance to make results different
            symbol_variance = {
                "BTC": 0.1,    # Bitcoin tends to be slightly more positive
                "ETH": 0.05,   # Ethereum slightly positive
                "SOL": 0.0,    # Solana neutral
                "XRP": -0.05,  # XRP slightly negative
                "DOGE": 0.15   # Doge more positive (meme coin enthusiasm)
            }.get(symbol, 0)
            
            adjusted_score = max(-1, min(1, sentiment_score + symbol_variance))
            
            # Confidence based on symbol with some randomness
            base_confidence = {
                "BTC": 0.7,  # Higher confidence for major coins
                "ETH": 0.7,
                "SOL": 0.5,
                "XRP": 0.6,
                "DOGE": 0.5
            }.get(symbol, 0.4)
            
            confidence = min(1.0, max(0.1, base_confidence + random.uniform(-0.1, 0.1)))
            
            result[symbol] = {
                "overall_sentiment": adjusted_score,
                "sentiment_category": self._categorize_sentiment(adjusted_score),
                "confidence": confidence,
                "is_simulated": True,
                "timestamp": datetime.now().isoformat()
            }
        
        return result
    
    def _categorize_sentiment(self, score: float) -> str:
        """Convert a numerical sentiment score to a category"""
        if score < -0.6:
            return "very_bearish"
        elif score < -0.2:
            return "bearish"
        elif score <= 0.2:
            return "neutral"
        elif score <= 0.6:
            return "bullish"
        else:
            return "very_bullish"

class FearGreedIndexCollector(BaseDataCollector):
    """Collector for the Crypto Fear & Greed Index"""
    
    def __init__(self, update_frequency: int = 3600):  # Default: update hourly
        """
        Initialize Fear & Greed Index collector
        
        Args:
            update_frequency: How often to update data (in seconds)
        """
        super().__init__("FearGreedIndex", update_frequency)
        
        # Sources with rate limiters
        self.sources = {
            "alternative_me": {
                "limiter": APILimiter(5, 100),  # Fixed: positional args
                "method": self._fetch_alternative_me,
                "priority": 1
            },
            "simulated": {
                "limiter": APILimiter(1000, 100000),  # Fixed: positional args
                "method": self._fetch_simulated_data,
                "priority": 2
            }
        }
        
        # Sort sources by priority
        self.source_priority = sorted(
            self.sources.keys(), 
            key=lambda x: self.sources[x]["priority"]
        )
    
    def _fetch_data(self) -> Dict[str, Any]:
        """Try to fetch data from sources in priority order"""
        errors = {}
        
        for source_name in self.source_priority:
            source = self.sources[source_name]
            
            if source["limiter"].can_make_call():
                try:
                    self.logger.info(f"Attempting to fetch Fear & Greed data from {source_name}")
                    data = source["method"]()
                    source["limiter"].increment_counters()
                    self.logger.info(f"Successfully fetched Fear & Greed data from {source_name}")
                    return {
                        "source": source_name,
                        "timestamp": datetime.now().isoformat(),
                        "data": data
                    }
                except Exception as e:
                    error_msg = f"Error fetching from {source_name}: {str(e)}"
                    self.logger.warning(error_msg)
                    errors[source_name] = error_msg
            else:
                self.logger.warning(f"Rate limit reached for {source_name}")
        
        # If we get here, all sources failed
        error_details = json.dumps(errors, indent=2)
        self.logger.error(f"All Fear & Greed data sources failed. Details: {error_details}")
        raise Exception(f"All Fear & Greed data sources failed: {error_details}")
    
    def _fetch_alternative_me(self) -> Dict[str, Any]:
        """Fetch Fear & Greed Index data from the Alternative.me API"""
        url = "https://api.alternative.me/fng/"
        response = requests.get(url)
        
        if response.status_code != 200:
            raise Exception(f"API returned status code {response.status_code}")
        
        data = response.json()
        
        # Process the response data
        if not data.get("data") or len(data["data"]) == 0:
            raise Exception("No Fear & Greed Index data available")
        
        # Get the latest value
        latest = data["data"][0]
        
        # Map value to category
        value = int(latest.get("value", 50))
        
        # Define category ranges
        categories = {
            (0, 20): "extreme_fear",
            (20, 40): "fear",
            (40, 60): "neutral",
            (60, 80): "greed",
            (80, 101): "extreme_greed"
        }
        
        # Determine category
        category = None
        for range_key, cat_value in categories.items():
            if range_key[0] <= value < range_key[1]:
                category = cat_value
                break
        
        # Build structured response
        result = {
            "value": value,
            "value_classification": latest.get("value_classification", "Unknown"),
            "category": category,
            "time_until_update": latest.get("time_until_update", "Unknown")
        }
        
        return result
    
    def _fetch_simulated_data(self) -> Dict[str, Any]:
        """Generate simulated Fear & Greed Index data as a fallback"""
        self.logger.warning("Using simulated Fear & Greed data as fallback")
        
        # Generate a random value between 0 and 100
        value = random.randint(0, 100)
        
        # Determine category and classification
        if value < 20:
            category = "extreme_fear"
            classification = "Extreme Fear"
        elif value < 40:
            category = "fear"
            classification = "Fear"
        elif value < 60:
            category = "neutral"
            classification = "Neutral"
        elif value < 80:
            category = "greed"
            classification = "Greed"
        else:
            category = "extreme_greed"
            classification = "Extreme Greed"
        
        # Build simulated response
        result = {
            "value": value,
            "value_classification": classification,
            "category": category,
            "time_until_update": "24 hours",
            "is_simulated": True
        }
        
        return result


class CryptoDataCollector:
    """Main data collector that orchestrates all specific collectors"""
    
    def __init__(self, symbols: List[str], config: Dict[str, Any] = None):
        """
        Args:
            symbols: List of cryptocurrency symbols to collect data for (e.g. ['BTC/USDT', 'ETH/USDT'])
            config: Configuration options including API keys and update frequencies
        """
        self.symbols = symbols
        self.config = config or {}
        self.logger = logging.getLogger("CryptoDataCollector")
        
        # Initialize market data collector with CoinGecko as primary source
        self.market_collector = MarketDataCollector(
            symbols=symbols,
            update_frequency=self.config.get("market_update_frequency", 60)
        )
        # Provide config to MarketDataCollector for CoinGecko API key
        self.market_collector.config = self.config
        
        # Initialize sentiment collector
        self.sentiment_collector = SentimentDataCollector(
            symbols=symbols,
            news_api_key=self.config.get("news_api_key"),
            openai_api_key=self.config.get("openai_api_key")
        )
        
        # Initialize Fear & Greed Index collector
        self.fear_greed_collector = FearGreedIndexCollector(
            update_frequency=self.config.get("fear_greed_update_frequency", 3600)  # Default: update hourly
        )
        
        # Queue for storing collected data
        self.data_queue = queue.Queue(maxsize=1000)
        
        # Flag for controlling the collection thread
        self.running = False
        self.collection_thread = None
    
    def start_collection(self):
        """Start collecting data in a separate thread"""
        if self.running:
            self.logger.warning("Data collection is already running")
            return
        
        self.running = True
        self.collection_thread = threading.Thread(target=self._collection_loop)
        self.collection_thread.daemon = True
        self.collection_thread.start()
        self.logger.info("Data collection started")
    
    def stop_collection(self):
        """Stop the data collection thread"""
        self.running = False
        if self.collection_thread:
            self.collection_thread.join(timeout=5)
            self.logger.info("Data collection stopped")
    
    def _collection_loop(self):
        """Main collection loop that runs in a separate thread"""
        while self.running:
            try:
                # Collect market data
                market_data = self.market_collector.update()
                if market_data:
                    self.data_queue.put({
                        "type": "market",
                        "data": market_data,
                        "timestamp": datetime.now().isoformat()
                    })
                
                # Collect sentiment data
                sentiment_data = self.sentiment_collector.update()
                if sentiment_data:
                    self.data_queue.put({
                        "type": "sentiment",
                        "data": sentiment_data,
                        "timestamp": datetime.now().isoformat()
                    })
                
                # Collect fear & greed data
                try:
                    fear_greed_data = self.fear_greed_collector.update()
                    if fear_greed_data:
                        self.data_queue.put({
                            "type": "fear_greed",
                            "data": fear_greed_data,
                            "timestamp": datetime.now().isoformat()
                        })
                except Exception as e:
                    self.logger.error(f"Error collecting fear & greed data: {str(e)}")
                
                # Sleep for a short time before checking again
                time.sleep(1)
            except Exception as e:
                self.logger.error(f"Error in collection loop: {str(e)}")
                time.sleep(5)  # Sleep longer after an error
    
    def get_latest_data(self) -> Dict[str, Any]:
        """Get the latest collected data for all data types"""
        latest_data = {
            "market": self.market_collector.data,
            "sentiment": self.sentiment_collector.data,
            "fear_greed": self.fear_greed_collector.data,
            "timestamp": datetime.now().isoformat()
        }
        return latest_data
    
    def get_data_by_type(self, data_type: str) -> Optional[Dict[str, Any]]:
        """Get the latest data for a specific data type"""
        if data_type == "market":
            return self.market_collector.data
        elif data_type == "sentiment":
            return self.sentiment_collector.data
        elif data_type == "fear_greed":
            return self.fear_greed_collector.data
        else:
            self.logger.error(f"Unknown data type: {data_type}")
            return None
    
    def get_data_from_queue(self, max_items: int = 100) -> List[Dict[str, Any]]:
        """Get multiple data items from the queue (non-blocking)"""
        items = []
        for _ in range(min(max_items, self.data_queue.qsize())):
            try:
                items.append(self.data_queue.get_nowait())
                self.data_queue.task_done()
            except queue.Empty:
                break
        return items
    
    def format_data_for_gpt(self) -> Dict[str, Any]:
        """Format the latest data for GPT analysis including technical indicators"""
        latest_data = self.get_latest_data()
        
        # Format market data with technical indicators
        market_summary = {}
        if latest_data.get("market"):
            market_data = latest_data["market"].get("data", {})
            for symbol, data in market_data.items():
                # Basic market data
                symbol_data = {
                    "price": data.get("price"),
                    "change_24h_percent": data.get("change_24h_percent"),
                    "volume_24h": data.get("volume_24h")
                }
                
                # Add technical indicators if available
                technical_indicators = data.get("technical_indicators", {})
                if technical_indicators:
                    symbol_data["technical_indicators"] = technical_indicators
                
                market_summary[symbol] = symbol_data
        
        # FIXED: Format sentiment data - handle base symbols correctly
        sentiment_summary = {}
        if latest_data.get("sentiment"):
            sentiment_data = latest_data["sentiment"].get("data", {})
            # Extract base symbols from self.symbols (e.g., "BTC" from "BTC/USDT")
            base_symbols = [s.split('/')[0] for s in self.symbols]
            
            for symbol, data in sentiment_data.items():
                # Check if this is a base symbol we track (BTC, ETH, SOL, etc.)
                if symbol in base_symbols:
                    # Handle both old and new data structures
                    if isinstance(data, dict):
                        sentiment_summary[symbol] = {
                            "sentiment_category": data.get("sentiment_category"),
                            "overall_sentiment": data.get("overall_sentiment"),
                            "confidence": data.get("confidence", 0)  # Add confidence for display
                        }
        
        # Format fear & greed data
        fear_greed_summary = {}
        if latest_data.get("fear_greed"):
            fear_greed_data = latest_data["fear_greed"].get("data", {})
            fear_greed_summary = {
                "value": fear_greed_data.get("value"),
                "category": fear_greed_data.get("category"),
                "classification": fear_greed_data.get("value_classification")
            }
        
        return {
            "market": market_summary,
            "sentiment": sentiment_summary,
            "fear_greed": fear_greed_summary,
            "timestamp": latest_data.get("timestamp")
        }
    
    def get_active_market_source(self) -> str:
        """Return the currently active market data source"""
        return self.market_collector.get_active_source()


# Example usage
def main():
    # Configuration
    config = {
        "market_update_frequency": 60,     # seconds
        "sentiment_update_frequency": 300,  # seconds
        "fear_greed_update_frequency": 3600, # seconds (hourly)
        "news_api_key": "YOUR_NEWS_API_KEY",
        "openai_api_key": "YOUR_OPENAI_API_KEY",
        "coingecko_api_key": "YOUR_COINGECKO_API_KEY"  # Optional, can use free tier without this
    }
    
    # Symbols to track
    symbols = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT", "DOGE/USDT"]
    
    # Initialize the collector
    collector = CryptoDataCollector(symbols=symbols, config=config)
    
    # Start data collection
    collector.start_collection()
    
    # Simulate running for a short time
    try:
        print("Data collection started. Press Ctrl+C to stop.")
        print(f"Active market data source: {collector.get_active_market_source()}")
        
        # Get and print data every 5 seconds for demo
        for _ in range(3):
            time.sleep(5)
            latest_data = collector.get_latest_data()
            
            # Format for GPT and print
            formatted_data = collector.format_data_for_gpt()
            print("\nCurrent data summary:")
            print(json.dumps(formatted_data, indent=2))
            
            # In a real application, you might send this to GPT for analysis
            # or pass it to other components of your trading bot
    
    except KeyboardInterrupt:
        pass
    finally:
        # Stop data collection
        collector.stop_collection()
        print("\nData collection stopped.")


if __name__ == "__main__":
    main()