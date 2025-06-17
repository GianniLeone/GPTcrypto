import os
import logging
import requests
import time
import hmac
import hashlib
import uuid
from typing import Dict, Any, Optional
from dotenv import load_dotenv
from decimal import Decimal, ROUND_DOWN
from urllib.parse import urlencode

# Import the position manager
from position_manager import get_position_manager, record_trade, is_in_cooldown

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("crypto_bot.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("CryptoBot.Portfolio")

class BinanceAPI:
    """Binance REST API client with trading capabilities"""
    
    def __init__(self, api_key: str, api_secret: str, testnet: bool = False):
        self.api_key = api_key
        self.api_secret = api_secret
        self.testnet = testnet
        
        # Use testnet or live endpoints
        if testnet:
            self.base_url = "https://testnet.binance.vision"
            logger.info("🧪 Using Binance TESTNET")
        else:
            self.base_url = "https://api.binance.com"
            logger.info("💰 Using Binance LIVE trading")
        
        # Symbol mapping for Binance format
        self.symbol_map = {
            "BTC": "BTCUSDT",
            "ETH": "ETHUSDT", 
            "SOL": "SOLUSDT",
            "XRP": "XRPUSDT",
            "DOGE": "DOGEUSDT"
        }
    
    def _generate_signature(self, query_string: str) -> str:
        """Generate HMAC SHA256 signature for Binance API"""
        return hmac.new(
            self.api_secret.encode('utf-8'),
            query_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
    
    def _make_request(self, method: str, endpoint: str, params: Dict[str, Any] = None, 
                     signed: bool = False) -> Dict[str, Any]:
        """Make authenticated request to Binance API"""
        if params is None:
            params = {}
        
        # Add timestamp for signed requests
        if signed:
            params['timestamp'] = int(time.time() * 1000)
        
        # Create query string
        query_string = urlencode(params)
        
        # Generate signature for signed requests
        if signed:
            signature = self._generate_signature(query_string)
            query_string += f"&signature={signature}"
        
        # Build URL
        url = f"{self.base_url}{endpoint}"
        if query_string:
            url += f"?{query_string}"
        
        # Prepare headers
        headers = {
            'X-MBX-APIKEY': self.api_key,
            'Content-Type': 'application/json'
        }
        
        try:
            if method.upper() == 'GET':
                response = requests.get(url, headers=headers, timeout=10)
            elif method.upper() == 'POST':
                response = requests.post(url, headers=headers, timeout=30)
            elif method.upper() == 'DELETE':
                response = requests.delete(url, headers=headers, timeout=10)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")
            
            logger.debug(f"Binance API {method} {endpoint}: {response.status_code}")
            
            if response.status_code == 200:
                return response.json()
            else:
                error_text = response.text
                logger.error(f"Binance API error: {response.status_code} - {error_text}")
                return {"error": f"API error: {response.status_code} - {error_text}"}
                
        except Exception as e:
            logger.error(f"Error making Binance API request: {str(e)}")
            return {"error": str(e)}
    
    def get_account_info(self) -> Dict[str, Any]:
        """Get account information including balances"""
        return self._make_request('GET', '/api/v3/account', signed=True)
    
    def get_balances(self) -> Dict[str, float]:
        """Get account balances in a simplified format"""
        try:
            account_info = self.get_account_info()
            
            if "error" in account_info:
                logger.error(f"Failed to get account info: {account_info['error']}")
                return {}
            
            balances = {}
            
            for balance_info in account_info.get('balances', []):
                asset = balance_info.get('asset', '')
                free_balance = float(balance_info.get('free', '0'))
                
                if free_balance > 0:
                    # Convert USDT to USD for consistency with existing code
                    if asset == 'USDT':
                        asset = 'USD'
                    balances[asset] = free_balance
            
            logger.info(f"Successfully fetched balances for {len(balances)} assets")
            return balances
            
        except Exception as e:
            logger.error(f"Error fetching Binance balances: {str(e)}")
            return {}
    
    def get_symbol_info(self, symbol: str) -> Dict[str, Any]:
        """Get trading rules for a symbol"""
        try:
            response = self._make_request('GET', '/api/v3/exchangeInfo')
            
            if "error" in response:
                return {}
            
            for symbol_info in response.get('symbols', []):
                if symbol_info.get('symbol') == symbol:
                    return symbol_info
            
            return {}
            
        except Exception as e:
            logger.error(f"Error getting symbol info for {symbol}: {str(e)}")
            return {}
    
    def place_market_order(self, symbol: str, side: str, quantity: str = None, 
                          quote_order_qty: str = None) -> Dict[str, Any]:
        """
        Place a market order on Binance
        
        Args:
            symbol: Trading pair (e.g., "BTCUSDT")
            side: "BUY" or "SELL"
            quantity: Amount of base asset (for SELL orders)
            quote_order_qty: Amount of quote asset (for BUY orders)
        """
        try:
            params = {
                'symbol': symbol,
                'side': side.upper(),
                'type': 'MARKET'
            }
            
            # For BUY orders, use quoteOrderQty (spend USDT amount)
            # For SELL orders, use quantity (sell crypto amount)
            if side.upper() == "BUY" and quote_order_qty:
                params['quoteOrderQty'] = quote_order_qty
            elif side.upper() == "SELL" and quantity:
                params['quantity'] = quantity
            else:
                return {"error": "Missing quantity parameter for order"}
            
            logger.info(f"Placing {side.upper()} order: {symbol} {params}")
            
            response = self._make_request('POST', '/api/v3/order', params, signed=True)
            
            if "error" not in response:
                logger.info(f"✅ Order placed successfully: {response.get('orderId', 'Unknown ID')}")
            
            return response
            
        except Exception as e:
            logger.error(f"❌ Error placing order: {str(e)}")
            return {"error": str(e)}
    
    def buy_crypto(self, symbol: str, usd_amount: float) -> Dict[str, Any]:
        """
        Buy cryptocurrency with USDT
        
        Args:
            symbol: Crypto symbol (e.g., "BTC", "ETH")
            usd_amount: USD amount to spend
        """
        # Convert symbol to Binance format
        binance_symbol = self.symbol_map.get(symbol, f"{symbol}USDT")
        
        # Format amount with proper precision (Binance uses USDT)
        quote_order_qty = f"{usd_amount:.2f}"
        
        logger.info(f"🟢 EXECUTING BUY: ${quote_order_qty} USDT worth of {symbol}")
        
        result = self.place_market_order(
            symbol=binance_symbol,
            side="BUY",
            quote_order_qty=quote_order_qty
        )
        
        return result
    
    def sell_crypto(self, symbol: str, crypto_amount: float) -> Dict[str, Any]:
        """
        Sell cryptocurrency for USDT
        
        Args:
            symbol: Crypto symbol (e.g., "BTC", "ETH")
            crypto_amount: Amount of crypto to sell
        """
        # Convert symbol to Binance format
        binance_symbol = self.symbol_map.get(symbol, f"{symbol}USDT")
        
        # Get symbol info to determine precision
        symbol_info = self.get_symbol_info(binance_symbol)
        
        # Format amount with proper precision
        if symbol_info:
            # Find the step size for the base asset
            step_size = None
            for filter_info in symbol_info.get('filters', []):
                if filter_info.get('filterType') == 'LOT_SIZE':
                    step_size = float(filter_info.get('stepSize', '0'))
                    break
            
            if step_size and step_size > 0:
                # Round down to the nearest step size
                precision = len(str(step_size).split('.')[-1].rstrip('0'))
                quantity = f"{crypto_amount:.{precision}f}"
            else:
                # Default precision based on symbol
                if symbol == "BTC":
                    quantity = f"{crypto_amount:.5f}"
                elif symbol in ["ETH", "SOL"]:
                    quantity = f"{crypto_amount:.4f}"
                else:
                    quantity = f"{crypto_amount:.3f}"
        else:
            # Fallback precision
            if symbol == "BTC":
                quantity = f"{crypto_amount:.5f}"
            elif symbol in ["ETH", "SOL"]:
                quantity = f"{crypto_amount:.4f}"
            else:
                quantity = f"{crypto_amount:.3f}"
        
        logger.info(f"🔴 EXECUTING SELL: {quantity} {symbol}")
        
        result = self.place_market_order(
            symbol=binance_symbol,
            side="SELL",
            quantity=quantity
        )
        
        return result
    
    def get_current_price(self, symbol: str) -> Optional[float]:
        """Get current price for a symbol"""
        try:
            binance_symbol = self.symbol_map.get(symbol, f"{symbol}USDT")
            
            response = self._make_request('GET', '/api/v3/ticker/price', 
                                        {'symbol': binance_symbol})
            
            if "error" not in response and 'price' in response:
                return float(response['price'])
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting price for {symbol}: {str(e)}")
            return None

def create_simulated_exchange() -> object:
    """Create a simulated exchange for testing"""
    class SimulatedExchange:
        def __init__(self):
            self.name = "simulated_binance"
            
        def get_balances(self):
            return {
                'USD': 148.75,  # Using USD instead of USDT for consistency
                'BTC': 0.001,
                'ETH': 0.05
            }
        
        def buy_crypto(self, symbol: str, usd_amount: float) -> Dict[str, Any]:
            logger.info(f"🎭 SIMULATED BUY: ${usd_amount:.2f} USDT worth of {symbol}")
            return {
                "orderId": f"simulated_{int(time.time())}",
                "symbol": f"{symbol}USDT",
                "side": "BUY",
                "type": "MARKET",
                "status": "FILLED",
                "quoteOrderQty": f"{usd_amount:.2f}",
                "simulated": True
            }
        
        def sell_crypto(self, symbol: str, crypto_amount: float) -> Dict[str, Any]:
            logger.info(f"🎭 SIMULATED SELL: {crypto_amount:.6f} {symbol}")
            return {
                "orderId": f"simulated_{int(time.time())}",
                "symbol": f"{symbol}USDT", 
                "side": "SELL",
                "type": "MARKET",
                "status": "FILLED",
                "quantity": f"{crypto_amount:.6f}",
                "simulated": True
            }
        
        def get_current_price(self, symbol: str) -> Optional[float]:
            # Simulated prices
            prices = {
                'BTC': 104000,
                'ETH': 2500, 
                'SOL': 156,
                'XRP': 2.13,
                'DOGE': 0.19
            }
            return prices.get(symbol)
    
    return SimulatedExchange()

def initialize_binance_client(api_key: str, api_secret: str) -> Optional[BinanceAPI]:
    """Initialize Binance API client with trading capabilities"""
    
    # Check if we should use simulated trading
    use_simulated = os.getenv('USE_SIMULATED_TRADING', 'false').lower() == 'true'
    if use_simulated:
        logger.info("🎭 Using simulated trading (USE_SIMULATED_TRADING is true)")
        return create_simulated_exchange()
    
    # Check if we should use testnet
    use_testnet = os.getenv('USE_BINANCE_TESTNET', 'false').lower() == 'true'
    
    try:
        logger.info("🚀 Connecting to Binance API with trading capabilities...")
        
        # Initialize the Binance API client
        client = BinanceAPI(api_key, api_secret, testnet=use_testnet)
        
        # Test the connection
        account_info = client.get_account_info()
        if account_info and 'balances' in account_info:
            logger.info(f"✅ Successfully connected to Binance {'Testnet' if use_testnet else 'Live'}! "
                       f"Account status: {account_info.get('accountType', 'SPOT')}")
            return client
        else:
            logger.error("❌ Failed to fetch account info from Binance API")
            logger.info("🎭 Falling back to simulated trading")
            return create_simulated_exchange()
            
    except Exception as e:
        logger.error(f"❌ Failed to initialize Binance trading client: {str(e)}")
        logger.info("🎭 Falling back to simulated trading")
        return create_simulated_exchange()

def analyze_action_type(action: str, analysis_result: Dict[str, Any]) -> Dict[str, str]:
    """
    Analyze the action type and determine execution strategy
    
    Enhanced to handle position reduction scenarios
    """
    pm = get_position_manager()
    asset = analysis_result.get('asset')
    amount_percentage = analysis_result.get('amount_percentage', 0)
    analysis_type = analysis_result.get('analysis_type', '')
    
    if action.lower() == "hold":
        return {
            "execution_type": "hold",
            "action_description": "Hold all positions",
            "reasoning": "No action needed"
        }
    
    elif action.lower() == "buy":
        return {
            "execution_type": "full_buy", 
            "action_description": f"Buy {amount_percentage}% allocation in {asset}",
            "reasoning": "Opening new position"
        }
    
    elif action.lower() == "sell":
        # Check if this is a position reduction vs full exit
        open_positions = pm.get_open_positions()
        
        if asset in open_positions:
            if amount_percentage < 100:
                return {
                    "execution_type": "position_reduction",
                    "action_description": f"Reduce {asset} position by {amount_percentage}%",
                    "reasoning": f"Partial exit due to {analysis_type.replace('_', ' ')}"
                }
            else:
                return {
                    "execution_type": "full_exit",
                    "action_description": f"Fully exit {asset} position",
                    "reasoning": f"Complete exit due to {analysis_type.replace('_', ' ')}"
                }
        else:
            return {
                "execution_type": "short_sell",
                "action_description": f"Short sell {amount_percentage}% in {asset}",
                "reasoning": "Opening short position"
            }
    
    else:
        return {
            "execution_type": "unknown",
            "action_description": f"Unknown action: {action}",
            "reasoning": "Unrecognized action type"
        }

def execute_trade(client, action: str, asset: str, amount_percentage: float, 
                 portfolio_data: Dict[str, float], confidence: float = 0.0,
                 rationale: str = "", analysis_result: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Enhanced trade execution with support for position reductions and market-responsive actions
    
    Args:
        client: Binance client
        action: "buy", "sell", or "hold"
        asset: Asset symbol (e.g., "BTC")
        amount_percentage: Percentage of available funds/holdings to trade
        portfolio_data: Current portfolio balances
        confidence: GPT confidence level (for position tracking)
        rationale: GPT reasoning (for position tracking)
        analysis_result: Full analysis result for enhanced tracking
        
    Returns:
        Enhanced trade execution result
    """
    # Analyze the action type for better logging and tracking
    action_analysis = analyze_action_type(action, analysis_result or {})
    execution_type = action_analysis["execution_type"]
    action_description = action_analysis["action_description"]
    
    logger.info(f"📋 TRADE ANALYSIS: {action_description}")
    logger.info(f"📋 REASONING: {action_analysis['reasoning']}")
    
    if action.lower() == "hold":
        logger.info("💤 Action is HOLD - no trade execution needed")
        
        # Still record the hold decision for tracking
        record_trade(asset, action, confidence, rationale, amount_percentage, executed=False)
        
        return {
            "action": "hold",
            "executed": False,
            "message": "No trade needed",
            "execution_type": execution_type
        }
    
    # CHECK COOLDOWN FIRST
    if is_in_cooldown(asset):
        pm = get_position_manager()
        remaining = pm.get_cooldown_remaining(asset)
        remaining_minutes = int(remaining.total_seconds() / 60) if remaining else 0
        
        logger.warning(f"🚫 TRADE BLOCKED: {asset} is in cooldown for {remaining_minutes} more minutes")
        logger.warning(f"🚫 Attempted: {action_description}")
        
        return {
            "action": action,
            "executed": False,
            "blocked_by_cooldown": True,
            "remaining_cooldown_minutes": remaining_minutes,
            "execution_type": execution_type,
            "message": f"Trade blocked - {asset} in cooldown for {remaining_minutes} minutes"
        }
    
    try:
        trade_result = None
        
        if action.lower() == "buy":
            trade_result = _execute_buy_order(client, asset, amount_percentage, portfolio_data, 
                                           confidence, rationale, execution_type, analysis_result)
            
        elif action.lower() == "sell":
            trade_result = _execute_sell_order(client, asset, amount_percentage, portfolio_data, 
                                            confidence, rationale, execution_type, analysis_result)
        
        return trade_result
    
    except Exception as e:
        logger.error(f"❌ Error executing trade: {str(e)}")
        
        # Record the error
        record_trade(asset, action, confidence, rationale, amount_percentage, executed=False)
        
        return {
            "action": action,
            "executed": False,
            "error": str(e),
            "execution_type": execution_type
        }

def _execute_buy_order(client, asset: str, amount_percentage: float, portfolio_data: Dict[str, float],
                      confidence: float, rationale: str, execution_type: str, 
                      analysis_result: Dict[str, Any] = None) -> Dict[str, Any]:
    """Execute buy order with enhanced tracking"""
    
    # Calculate USD amount to spend (Binance uses USDT but we treat as USD)
    available_usd = portfolio_data.get("USD", 0)
    usd_to_spend = available_usd * (amount_percentage / 100)
    
    if usd_to_spend < 10.0:  # Binance minimum ~$10 trade
        logger.warning(f"⚠️ Trade amount too small: ${usd_to_spend:.2f} (minimum ~$10 for Binance)")
        
        # Record the failed attempt
        record_trade(asset, "buy", confidence, rationale, amount_percentage, executed=False)
        
        return {
            "action": "buy",
            "executed": False,
            "message": "Trade amount too small for Binance",
            "execution_type": execution_type
        }
    
    logger.info(f"🟢 EXECUTING BUY: {amount_percentage}% of ${available_usd:.2f} = ${usd_to_spend:.2f} in {asset}")
    logger.info(f"🎯 EXECUTION TYPE: {execution_type}")
    
    result = client.buy_crypto(asset, usd_to_spend)
    
    if "error" not in result:
        logger.info(f"✅ BUY ORDER EXECUTED: ${usd_to_spend:.2f} worth of {asset}")
        
        trade_result = {
            "action": "buy",
            "executed": True,
            "asset": asset,
            "usd_amount": usd_to_spend,
            "result": result,
            "execution_type": execution_type
        }
        
        # Enhanced trade recording with analysis context
        record_trade(
            asset, "buy", confidence, rationale, amount_percentage, 
            executed=True, trade_result=trade_result,
            market_data=analysis_result.get('market_data') if analysis_result else None,
            sentiment_data=analysis_result.get('sentiment_data') if analysis_result else None,
            fear_greed_data=analysis_result.get('fear_greed_data') if analysis_result else None,
            conviction_score=analysis_result.get('conviction_score') if analysis_result else None,
            news_driven=analysis_result.get('news_driven', False) if analysis_result else False
        )
        
        return trade_result
        
    else:
        logger.error(f"❌ BUY ORDER FAILED: {result['error']}")
        
        # Record failed trade attempt
        record_trade(asset, "buy", confidence, rationale, amount_percentage, executed=False)
        
        return {
            "action": "buy",
            "executed": False,
            "error": result["error"],
            "execution_type": execution_type
        }

def _execute_sell_order(client, asset: str, amount_percentage: float, portfolio_data: Dict[str, float],
                       confidence: float, rationale: str, execution_type: str, 
                       analysis_result: Dict[str, Any] = None) -> Dict[str, Any]:
    """Execute sell order with enhanced support for position reductions"""
    
    # Calculate crypto amount to sell
    available_crypto = portfolio_data.get(asset, 0)
    
    if available_crypto <= 0:
        logger.warning(f"⚠️ No {asset} available to sell")
        
        # Record the failed attempt
        record_trade(asset, "sell", confidence, rationale, amount_percentage, executed=False)
        
        return {
            "action": "sell",
            "executed": False,
            "message": f"No {asset} to sell",
            "execution_type": execution_type
        }
    
    crypto_to_sell = available_crypto * (amount_percentage / 100)
    
    # Enhanced logging based on execution type
    if execution_type == "position_reduction":
        logger.info(f"📉 EXECUTING POSITION REDUCTION: {amount_percentage}% of {available_crypto:.6f} = {crypto_to_sell:.6f} {asset}")
        logger.info(f"🎯 KEEPING: {(available_crypto - crypto_to_sell):.6f} {asset} ({100-amount_percentage}% of position)")
    elif execution_type == "full_exit":
        logger.info(f"🔴 EXECUTING FULL EXIT: {amount_percentage}% of {available_crypto:.6f} = {crypto_to_sell:.6f} {asset}")
        logger.info(f"🎯 EXITING COMPLETE POSITION")
    else:
        logger.info(f"🔴 EXECUTING SELL: {amount_percentage}% of {available_crypto:.6f} = {crypto_to_sell:.6f} {asset}")
    
    logger.info(f"🎯 EXECUTION TYPE: {execution_type}")
    
    result = client.sell_crypto(asset, crypto_to_sell)
    
    if "error" not in result:
        if execution_type == "position_reduction":
            logger.info(f"✅ POSITION REDUCTION EXECUTED: {crypto_to_sell:.6f} {asset} sold ({amount_percentage}%)")
            logger.info(f"📈 REMAINING POSITION: {(available_crypto - crypto_to_sell):.6f} {asset}")
        else:
            logger.info(f"✅ SELL ORDER EXECUTED: {crypto_to_sell:.6f} {asset}")
        
        trade_result = {
            "action": "sell",
            "executed": True,
            "asset": asset,
            "crypto_amount": crypto_to_sell,
            "remaining_crypto": available_crypto - crypto_to_sell,
            "reduction_percentage": amount_percentage,
            "result": result,
            "execution_type": execution_type
        }
        
        # Enhanced trade recording with analysis context
        record_trade(
            asset, "sell", confidence, rationale, amount_percentage, 
            executed=True, trade_result=trade_result,
            market_data=analysis_result.get('market_data') if analysis_result else None,
            sentiment_data=analysis_result.get('sentiment_data') if analysis_result else None,
            fear_greed_data=analysis_result.get('fear_greed_data') if analysis_result else None,
            conviction_score=analysis_result.get('conviction_score') if analysis_result else None,
            news_driven=analysis_result.get('news_driven', False) if analysis_result else False
        )
        
        return trade_result
        
    else:
        logger.error(f"❌ SELL ORDER FAILED: {result['error']}")
        
        # Record failed trade attempt
        record_trade(asset, "sell", confidence, rationale, amount_percentage, executed=False)
        
        return {
            "action": "sell",
            "executed": False,
            "error": result["error"],
            "execution_type": execution_type
        }

def fetch_balances(client) -> Dict[str, float]:
    """Fetch balances from Binance API"""
    try:
        if hasattr(client, 'get_balances'):
            balances = client.get_balances()
            logger.info(f"Successfully fetched balances for {len(balances)} assets")
            return balances
        else:
            logger.error("Client does not support get_balances method")
            return {}
        
    except Exception as e:
        logger.error(f"Error fetching balances: {str(e)}")
        return {}

def calculate_usd_values_with_market_data(client, balances: Dict[str, float], 
                                         market_prices: Dict[str, float] = None) -> Dict[str, float]:
    """Calculate USD values using market data or current exchange rates"""
    try:
        usd_values = {}
        
        for currency, amount in balances.items():
            if amount <= 0:
                continue
            
            # Handle USD and stablecoins
            if currency in ['USD', 'USDT', 'USDC', 'BUSD', 'DAI']:
                usd_values[currency] = amount
                continue
            
            # Try to use market data first
            if market_prices and currency in market_prices:
                price = market_prices[currency]
                usd_values[currency] = amount * price
                logger.debug(f"Used market data for {currency}: {price}")
                continue
            
            # Try to get current price from Binance
            if hasattr(client, 'get_current_price'):
                try:
                    current_price = client.get_current_price(currency)
                    if current_price:
                        usd_values[currency] = amount * current_price
                        logger.debug(f"Used Binance price for {currency}: {current_price}")
                        continue
                except Exception as e:
                    logger.debug(f"Failed to get Binance price for {currency}: {str(e)}")
            
            # Fallback prices
            fallback_prices = {
                'BTC': 104000,
                'ETH': 2500,
                'SOL': 156,
                'XRP': 2.13,
                'DOGE': 0.19
            }
            
            if currency in fallback_prices:
                price = fallback_prices[currency]
                usd_values[currency] = amount * price
                logger.debug(f"Used fallback price for {currency}: {price}")
            else:
                logger.warning(f"No price data available for {currency}")
                usd_values[currency] = 0
        
        return usd_values
        
    except Exception as e:
        logger.error(f"Error calculating USD values: {str(e)}")
        return {}

def extract_market_prices_from_data(market_data: Dict[str, Any]) -> Dict[str, float]:
    """Extract current prices from market data for portfolio valuation"""
    prices = {}
    
    if not market_data:
        return prices
    
    for symbol, data in market_data.items():
        if isinstance(data, dict) and 'price' in data:
            base_currency = symbol.split('/')[0] if '/' in symbol else symbol
            price = data['price']
            
            if isinstance(price, (int, float)) and price > 0:
                prices[base_currency] = price
    
    return prices

def get_focused_portfolio_state(market_data: Dict[str, Any] = None) -> Dict[str, float]:
    """Get portfolio state FOCUSED on main trading assets only"""
    
    # Define our 4 main trading assets + USD
    MAIN_ASSETS = {'BTC', 'ETH', 'SOL', 'DOGE', 'USD', 'USDT', 'USDC', 'DAI'}
    
    try:
        load_dotenv()
        
        api_key = os.getenv('BINANCE_API_KEY')
        api_secret = os.getenv('BINANCE_API_SECRET')
        
        if not api_key or not api_secret:
            logger.warning("Binance API credentials not found. Using simulated portfolio.")
            # Return simulated focused portfolio
            return {
                'USD': 10000.0,
                'BTC': 109538.0,
                'ETH': 2862.57,
                'SOL': 498.21,
                'DOGE': 515.79
            }
        
        # Initialize client
        client = initialize_binance_client(api_key, api_secret)
        if not client:
            logger.error("Failed to initialize Binance client")
            return {}
        
        # Fetch ALL balances but filter immediately
        all_balances = fetch_balances(client)
        if not all_balances:
            logger.warning("No balance data available")
            return {}
        
        # FILTER: Only keep our main trading assets
        focused_balances = {}
        for currency, amount in all_balances.items():
            if currency in MAIN_ASSETS and amount > 0:
                focused_balances[currency] = amount
        
        logger.info(f"Focused on {len(focused_balances)} main assets (filtered from {len(all_balances)} total)")
        
        # Extract market prices for our assets only
        market_prices = {}
        if market_data:
            for symbol, data in market_data.items():
                if isinstance(data, dict) and 'price' in data:
                    base_currency = symbol.split('/')[0]
                    if base_currency in MAIN_ASSETS:
                        market_prices[base_currency] = data['price']
        
        # Calculate USD values for focused assets only
        usd_values = {}
        for currency, amount in focused_balances.items():
            if currency in ['USD', 'USDT', 'USDC', 'DAI']:
                usd_values[currency] = amount
            elif currency in market_prices:
                usd_values[currency] = amount * market_prices[currency]
            else:
                # Fallback prices for main assets
                fallback_prices = {
                    'BTC': 109538.0,
                    'ETH': 2862.57, 
                    'SOL': 498.21,
                    'DOGE': 0.20
                }
                if currency in fallback_prices:
                    usd_values[currency] = amount * fallback_prices[currency]
        
        # Clean logging - only show our main assets
        total_value = sum(usd_values.values())
        logger.info(f"Focused Portfolio Value: ${total_value:.2f}")
        
        pm = get_position_manager()
        open_positions = pm.get_open_positions()
        
        for currency in ['BTC', 'ETH', 'SOL', 'DOGE', 'USD']:
            if currency in usd_values:
                value = usd_values[currency]
                percentage = (value / total_value) * 100 if total_value > 0 else 0
                position_indicator = " 📈" if currency in open_positions else ""
                logger.info(f"  {currency}: ${value:.2f} ({percentage:.1f}%){position_indicator}")
        
        return usd_values
        
    except Exception as e:
        logger.error(f"Error getting focused portfolio: {str(e)}")
        return {}


def display_focused_portfolio(portfolio_data: Dict[str, float]):
    """Display clean focused portfolio - ONLY main trading assets"""
    if not portfolio_data:
        return
    
    MAIN_ASSETS = ['BTC', 'ETH', 'SOL', 'DOGE', 'USD', 'USDT', 'USDC']
    
    # Filter to only show our main assets
    focused_data = {k: v for k, v in portfolio_data.items() if k in MAIN_ASSETS}
    total_value = sum(focused_data.values())
    
    if total_value == 0:
        print("\n💼 FOCUSED PORTFOLIO: No main assets detected")
        return
    
    print(f"\n💼 FOCUSED PORTFOLIO: ${total_value:,.2f}")
    
    # Sort by value but keep main cryptos first
    crypto_assets = [(k, v) for k, v in focused_data.items() if k in ['BTC', 'ETH', 'SOL', 'DOGE']]
    stable_assets = [(k, v) for k, v in focused_data.items() if k in ['USD', 'USDT', 'USDC', 'DAI']]
    
    # Sort each group by value
    crypto_assets.sort(key=lambda x: x[1], reverse=True)
    stable_assets.sort(key=lambda x: x[1], reverse=True)
    
    # Display cryptos first
    for currency, value in crypto_assets:
        percentage = (value / total_value) * 100
        currency_emoji = {"BTC": "₿", "ETH": "Ξ", "SOL": "◎", "DOGE": "Ð"}.get(currency, "💰")
        
        # Position indicators
        pm = get_position_manager()
        position_indicator = ""
        if currency in pm.get_open_positions():
            position_indicator = " 📈"
        elif pm.is_in_cooldown(currency):
            remaining = pm.get_cooldown_remaining(currency)
            remaining_minutes = int(remaining.total_seconds() / 60) if remaining else 0
            position_indicator = f" 🚫({remaining_minutes}m)"
        
        print(f"  {currency_emoji} {currency}: ${value:,.2f} ({percentage:.1f}%){position_indicator}")
    
    # Display stables
    for currency, value in stable_assets:
        percentage = (value / total_value) * 100
        print(f"  💵 {currency}: ${value:,.2f} ({percentage:.1f}%)")

# Compatibility function - update main.py to use initialize_binance_client
def initialize_coinbase_client(api_key: str, api_secret: str):
    """DEPRECATED: Use initialize_binance_client instead"""
    logger.warning("initialize_coinbase_client is deprecated. Using Binance client instead.")
    return initialize_binance_client(api_key, api_secret)

# Test the enhanced trading system with Binance
if __name__ == "__main__":
    print("=== Testing Enhanced Portfolio with Binance API ===")
    
    # Test portfolio connection
    portfolio = get_focused_portfolio_state()
    if portfolio:
        print(f"✅ Portfolio connected: ${sum(portfolio.values()):.2f}")
        
        # Test trade execution with position management (simulated)
        os.environ['USE_SIMULATED_TRADING'] = 'true'
        os.environ['TRADE_COOLDOWN_MINUTES'] = '1'  # 1 minute for testing
        
        client = initialize_binance_client("test", "test")
        
        # Test buy order
        result1 = execute_trade(
            client=client,
            action="buy",
            asset="BTC",
            amount_percentage=50,
            portfolio_data=portfolio,
            confidence=0.85,
            rationale="Test buy order with strong signals"
        )
        
        print(f"Test buy result: {result1}")
        
        # Test position reduction (partial sell)
        result2 = execute_trade(
            client=client,
            action="sell",
            asset="BTC", 
            amount_percentage=30,  # Reduce by 30%
            portfolio_data={"BTC": 0.001, "USD": 100},  # Simulate BTC holdings
            confidence=0.75,
            rationale="Partial exit due to bearish technicals",
            analysis_result={
                "analysis_type": "market_responsive_position_review",
                "conviction_score": 7.5
            }
        )
        
        print(f"Test position reduction result: {result2}")
        
        # Test current price fetching
        if hasattr(client, 'get_current_price'):
            btc_price = client.get_current_price('BTC')
            print(f"Current BTC price: ${btc_price}")
        
        # Show position manager stats
        pm = get_position_manager()
        stats = pm.get_trading_stats()
        print(f"Trading stats: {stats}")
        
    else:
        print("❌ Could not connect to portfolio")