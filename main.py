import os
import time
import json
import logging
from datetime import datetime, timedelta
from dotenv import load_dotenv
from CryptoDataCollector import CryptoDataCollector
from gpt_analysis import analyze_with_strategic_conviction, process_news_batch
from portfolio import get_focused_portfolio_state, initialize_binance_client, execute_trade

# Import position manager
from position_manager import get_position_manager, get_cooldown_status, get_trading_stats

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("crypto_bot.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("CryptoBot.AutoTrading")

MAIN_TRADING_ASSETS = ['BTC', 'ETH', 'SOL', 'DOGE']

def setup_focused_logging():
    """Setup logging focused on main trading assets"""
    # Reduce portfolio spam
    logging.getLogger("CryptoBot.Portfolio").setLevel(logging.WARNING)
    
    # Keep important trading logs
    logging.getLogger("CryptoBot.AutoTrading").setLevel(logging.INFO)
    logging.getLogger("CryptoBot.StrategicGPT").setLevel(logging.INFO)
    
    # Reduce technical indicator spam
    logging.getLogger("CryptoBot.TechnicalIndicators").setLevel(logging.WARNING)
    logging.getLogger("CryptoDataCollector").setLevel(logging.WARNING)

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

def safe_get_env_bool(key: str, default: bool) -> bool:
    """Safely get boolean from environment variable, handling comments"""
    try:
        value = os.getenv(key, str(default)).lower()
        if '#' in value:
            value = value.split('#')[0]
        value = value.strip()
        return value in ['true', '1', 'yes', 'on']
    except Exception as e:
        logger.warning(f"Error parsing {key}='{os.getenv(key)}', using default {default}: {e}")
        return default

def enhanced_gpt_analysis_execution(formatted_data, portfolio_data, last_news_impact):
    """Enhanced GPT analysis FOCUSED on 4 main assets"""
    logger.info("🧠 Running focused GPT analysis (BTC, ETH, SOL, DOGE)")
    
    # Get the enhanced analysis result
    from gpt_analysis import analyze_with_strategic_conviction
    analysis_result = analyze_with_strategic_conviction(formatted_data, portfolio_data, last_news_impact)
    
    # FOCUSED portfolio display
    if portfolio_data:
        from portfolio import display_focused_portfolio
        display_focused_portfolio(portfolio_data)
    
    # Enhanced execution with analysis context
    trade_result = None
    auto_trading_enabled = safe_get_env_bool('ENABLE_AUTO_TRADING', True)
    
    if auto_trading_enabled:
        logger.info("🤖 EXECUTING FOCUSED AUTOMATIC TRADE")
        
        # Initialize Binance trading client
        api_key = os.getenv("BINANCE_API_KEY")
        api_secret = os.getenv("BINANCE_API_SECRET")
        trading_client = initialize_binance_client(api_key, api_secret)
        
        if trading_client:
            trade_result = execute_trade(
                client=trading_client,
                action=analysis_result.get("action", "hold"),
                asset=analysis_result.get("asset"),
                amount_percentage=analysis_result.get("amount_percentage", 0),
                portfolio_data=portfolio_data,
                confidence=analysis_result.get("confidence", 0.0),
                rationale=analysis_result.get("rationale", ""),
                analysis_result=analysis_result
            )
        else:
            logger.error("Failed to initialize Binance trading client")
    else:
        logger.info("Auto-trading disabled - showing recommendation only")
    
    return analysis_result, trade_result

def print_startup_banner():
    """Print a clear startup banner for auto-trading bot"""
    print("\n" + "🤖" * 50)
    print("🤖" + " " * 46 + "🤖")
    print("🤖" + " " * 8 + "AUTO-TRADING CRYPTO BOT STARTED" + " " * 7 + "🤖")
    print("🤖" + " " * 15 + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + " " * 15 + "🤖")
    print("🤖" + " " * 46 + "🤖")
    print("🤖" * 50)
    print()

def safe_format_number(value, format_str="{:.2f}"):
    """Safely format a number, handling None/null values"""
    if value is None:
        return "N/A"
    try:
        if isinstance(value, (int, float)):
            return format_str.format(value)
        return str(value)
    except:
        return str(value)

def get_sentiment_display(formatted_data):
    """Extract and format sentiment data for display"""
    sentiment_data = formatted_data.get("sentiment", {})
    
    if not sentiment_data:
        return "⏳ Processing news articles..."
    
    sentiment_lines = []
    for symbol in MAIN_TRADING_ASSETS:  # Use MAIN_TRADING_ASSETS instead of hardcoded list
        if symbol in sentiment_data:
            sentiment_info = sentiment_data[symbol]
            category = sentiment_info.get("sentiment_category", "neutral")
            score = sentiment_info.get("overall_sentiment", 0)
            confidence = sentiment_info.get("confidence", 0)
            
            # Get emoji for sentiment
            if category in ["very_bullish", "bullish"]:
                emoji = "🟢"
            elif category in ["very_bearish", "bearish"]:
                emoji = "🔴"
            else:
                emoji = "🟡"
            
            category_display = category.replace('_', ' ').title()
            score_display = f"{score:+.2f}" if isinstance(score, (int, float)) else "N/A"
            confidence_display = f"{confidence:.0%}" if isinstance(confidence, (int, float)) else "N/A"
            
            sentiment_lines.append(f"  {emoji} {symbol}: {category_display} ({score_display}, {confidence_display} conf)")
    
    if sentiment_lines:
        return "\n".join(sentiment_lines)
    else:
        return "⏳ Processing news articles..."

def display_position_manager_status():
    """Display position manager status including cooldowns and open positions"""
    try:
        pm = get_position_manager()
        
        # Get trading statistics
        stats = get_trading_stats()
        
        # Get cooldown status
        cooldown_status = get_cooldown_status()
        active_cooldowns = {symbol: info for symbol, info in cooldown_status.items() if info['in_cooldown']}
        
        # Get open positions
        open_positions = pm.get_open_positions()
        
        print(f"\n🏦 POSITION MANAGER STATUS:")
        print(f"  📊 Total Trades: {stats['executed_trades']} executed ({stats['buy_trades']} buys, {stats['sell_trades']} sells)")
        print(f"  📈 Open Positions: {len(open_positions)}")
        print(f"  ⏰ Cooldown Period: {stats['cooldown_period_minutes']} minutes")
        
        # Show active cooldowns
        if active_cooldowns:
            print(f"  🚫 Active Cooldowns:")
            for symbol, info in active_cooldowns.items():
                remaining_minutes = info['remaining_minutes']
                print(f"    {symbol}: {remaining_minutes} minutes remaining")
        else:
            print(f"  ✅ No active cooldowns - all symbols available for trading")
        
        # Show open positions
        if open_positions:
            print(f"  📈 Open Positions:")
            for symbol, position in open_positions.items():
                entry_time = datetime.fromisoformat(position.entry_timestamp)
                hours_held = (datetime.now() - entry_time).total_seconds() / 3600
                print(f"    {symbol}: {position.entry_action.upper()} @ {entry_time.strftime('%H:%M')} ({hours_held:.1f}h ago)")
        
        # Show recent trading activity
        recent_trades = pm.get_recent_trades(24)
        if recent_trades:
            print(f"  📋 Last 24h: {len(recent_trades)} trades")
        
    except Exception as e:
        logger.error(f"Error displaying position manager status: {str(e)}")
        print(f"  ❌ Error loading position data: {str(e)}")

def display_data_update(formatted_data, update_count):
    """Display formatted data update - FOCUSED on 4 main assets"""
    current_time = datetime.now().strftime('%H:%M:%S')
    
    print(f"\n{'─' * 60}")
    print(f"📊 UPDATE #{update_count} │ {current_time} │ FOCUSED TRADING")
    print(f"{'─' * 60}")
    
    # FOCUSED: Only show our 4 main trading assets
    print("\n💰 MAIN ASSETS:")
    market_data = formatted_data.get("market", {})
    
    if market_data:
        pm = get_position_manager()
        
        for symbol in ["BTC/USDT", "ETH/USDT", "SOL/USDT", "DOGE/USDT"]:
            if symbol in market_data:
                data = market_data[symbol]
                base_symbol = symbol.split('/')[0]
                price = safe_format_number(data.get("price"), "${:.2f}")
                change = data.get("change_24h_percent")
                
                # Format change
                if change is not None and isinstance(change, (int, float)):
                    change_str = f"{change:+.2f}%"
                    change_emoji = "🟢" if change > 0 else "🔴" if change < 0 else "🟡"
                else:
                    change_str = "N/A"
                    change_emoji = "⚪"
                
                # Position indicators
                cooldown_indicator = ""
                if pm.is_in_cooldown(base_symbol):
                    remaining = pm.get_cooldown_remaining(base_symbol)
                    remaining_minutes = int(remaining.total_seconds() / 60) if remaining else 0
                    cooldown_indicator = f" 🚫({remaining_minutes}m)"
                elif base_symbol in pm.get_open_positions():
                    cooldown_indicator = " 📈"
                
                print(f"  {change_emoji} {symbol}: {price} ({change_str}){cooldown_indicator}")
                
                # Technical indicators (compact)
                technical = data.get("technical_indicators", {})
                if technical and isinstance(technical, dict):
                    rsi = technical.get("rsi")
                    tech_summary = technical.get("technical_summary", "neutral")
                    
                    rsi_display = ""
                    if rsi is not None:
                        if rsi > 70:
                            rsi_display = f"RSI: {rsi:.0f} 🔴"
                        elif rsi < 30:
                            rsi_display = f"RSI: {rsi:.0f} 🟢"
                        else:
                            rsi_display = f"RSI: {rsi:.0f} 🟡"
                    
                    summary_emoji = "🟢" if "bullish" in tech_summary else "🔴" if "bearish" in tech_summary else "🟡"
                    tech_display = f"{rsi_display} | {tech_summary.replace('_', ' ').title()} {summary_emoji}"
                    print(f"    📊 {tech_display}")
    
    # FOCUSED: Only show main asset sentiment
    print(f"\n🗞️ SENTIMENT:")
    sentiment_data = formatted_data.get("sentiment", {})
    
    for symbol in MAIN_TRADING_ASSETS:
        if symbol in sentiment_data:
            sentiment_info = sentiment_data[symbol]
            category = sentiment_info.get("sentiment_category", "neutral")
            score = sentiment_info.get("overall_sentiment", 0)
            confidence = sentiment_info.get("confidence", 0)
            
            emoji = "🟢" if "bullish" in category else "🔴" if "bearish" in category else "🟡"
            score_display = f"{score:+.2f}" if isinstance(score, (int, float)) else "0.00"
            conf_display = f"{confidence:.0%}" if isinstance(confidence, (int, float)) else "0%"
            
            print(f"  {emoji} {symbol}: {category.replace('_', ' ').title()} ({score_display}, {conf_display})")
    
    # Fear & Greed (compact)
    fear_greed_data = formatted_data.get("fear_greed", {})
    if fear_greed_data and fear_greed_data.get("value") is not None:
        value = fear_greed_data.get("value", 50)
        category = fear_greed_data.get("category", "neutral")
        fg_emoji = "🤑" if value >= 80 else "😊" if value >= 60 else "😐" if value >= 40 else "😰" if value >= 20 else "😱"
        print(f"\n🎭 FEAR & GREED: {value}/100 ({category.replace('_', ' ').title()}) {fg_emoji}")
    
    # Show position status every 5 updates (compact)
    if update_count % 5 == 0:
        display_focused_position_status()


def display_focused_position_status():
    """Display focused position status for main assets only"""
    try:
        pm = get_position_manager()
        stats = get_trading_stats()
        
        print(f"\n🏦 TRADING STATUS:")
        print(f"  📊 Trades: {stats['executed_trades']} executed | Open: {stats['open_positions']} | Cooldown: {stats['cooldown_period_minutes']}m")
        
        # Active cooldowns for main assets only
        cooldown_status = get_cooldown_status()
        main_cooldowns = {s: info for s, info in cooldown_status.items() 
                         if s in MAIN_TRADING_ASSETS and info['in_cooldown']}
        
        if main_cooldowns:
            cooldown_list = [f"{s}({info['remaining_minutes']}m)" for s, info in main_cooldowns.items()]
            print(f"  🚫 Cooldowns: {', '.join(cooldown_list)}")
        else:
            print(f"  ✅ All main assets available for trading")
            
    except Exception as e:
        logger.error(f"Error displaying focused status: {str(e)}")

def display_gpt_analysis_with_execution(analysis, trade_result=None):
    """Display GPT analysis and trade execution results with enhanced formatting including position history"""
    print(f"\n{'🤖' * 25}")
    print("🤖" + " " * 21 + "🤖")
    print("🤖" + "   GPT AUTO-TRADING RESULT    " + "🤖")
    print("🤖" + " " * 21 + "🤖")
    print(f"{'🤖' * 25}")
    
    # Show if this is news-driven with urgency (enhanced from original)
    if analysis.get('news_driven'):
        print("📰 NEWS-DRIVEN SIGNAL 📰")
        urgency = analysis.get('urgency', 'medium')
        urgency_emoji = {"high": "🔴", "medium": "🟡", "low": "🟢"}
        print(f"⚡ URGENCY: {urgency_emoji.get(urgency, '🟡')} {urgency.upper()}")
        print()
    
    # Show consistency information if available (NEW FEATURE)
    if analysis.get('consistency_score') is not None:
        consistency_score = analysis.get('consistency_score', 1.0)
        consistency_explanation = analysis.get('consistency_explanation', '')
        
        if consistency_score < 1.0:
            print("🧠 POSITION HISTORY ANALYSIS 🧠")
            consistency_emoji = "🟢" if consistency_score >= 0.8 else "🟡" if consistency_score >= 0.5 else "🔴"
            print(f"📊 CONSISTENCY: {consistency_emoji} {consistency_score:.0%}")
            print(f"💭 {consistency_explanation}")
            
            if analysis.get('original_confidence') is not None:
                original_conf = analysis.get('original_confidence', 0)
                print(f"⚙️ Confidence adjusted: {original_conf:.0%} → {analysis.get('confidence', 0):.0%}")
            print()
    
    # Action with appropriate emoji (enhanced from original)
    action = analysis['action'].upper()
    action_emoji = {"BUY": "🟢", "SELL": "🔴", "HOLD": "🟡"}
    print(f"🎯 ACTION: {action_emoji.get(action, '⚪')} {action}")
    
    # Show asset and percentage for buy/sell recommendations (enhanced from original)
    if analysis['action'] in ['buy', 'sell'] and analysis.get('asset'):
        asset_emoji = {"BTC": "₿", "ETH": "Ξ", "SOL": "◎", "DOGE": "Ð"}  # Removed XRP
        asset = analysis.get('asset')
        amount = analysis.get('amount_percentage', 0)
        
        print(f"💰 ASSET: {asset_emoji.get(asset, '💰')} {asset}")
        
        if analysis['action'] == 'buy':
            print(f"💵 AMOUNT: {amount}% of available funds")
        else:
            print(f"💵 AMOUNT: {amount}% of {asset} holdings")
    
    # Confidence with visual bar (enhanced from original)
    confidence = analysis.get('confidence', 0)
    if isinstance(confidence, (int, float)):
        confidence_pct = confidence * 100
        bar_length = int(confidence * 20)  # 20 character bar
        confidence_bar = "█" * bar_length + "░" * (20 - bar_length)
        print(f"📊 CONFIDENCE: {confidence_pct:.0f}% [{confidence_bar}]")
    
    # EXECUTION RESULTS (enhanced with cooldown info)
    print(f"\n{'⚡' * 20}")
    print("⚡ EXECUTION RESULTS:")
    
    if trade_result:
        if trade_result.get("blocked_by_cooldown", False):
            remaining_minutes = trade_result.get("remaining_cooldown_minutes", 0)
            print(f"🚫 TRADE BLOCKED BY COOLDOWN!")
            print(f"⏰ {analysis.get('asset')} available for trading in {remaining_minutes} minutes")
            print(f"🎯 Attempted: {trade_result['action'].upper()} {analysis.get('asset', '')}")
            
        elif trade_result.get("executed", False):
            if trade_result["action"] == "buy":
                usd_amount = trade_result.get("usd_amount", 0)
                asset = trade_result.get("asset", "")
                print(f"✅ BUY ORDER EXECUTED!")
                print(f"💰 Purchased: ${usd_amount:.2f} worth of {asset}")
                
                if trade_result.get("result", {}).get("simulated"):
                    print("🎭 (Simulated Trade)")
                else:
                    order_id = trade_result.get("result", {}).get("orderId", "Unknown")
                    print(f"📋 Order ID: {order_id}")
                    
            elif trade_result["action"] == "sell":
                crypto_amount = trade_result.get("crypto_amount", 0)
                asset = trade_result.get("asset", "")
                print(f"✅ SELL ORDER EXECUTED!")
                print(f"💰 Sold: {crypto_amount:.6f} {asset}")
                
                if trade_result.get("result", {}).get("simulated"):
                    print("🎭 (Simulated Trade)")
                else:
                    order_id = trade_result.get("result", {}).get("orderId", "Unknown")
                    print(f"📋 Order ID: {order_id}")
        else:
            print(f"❌ TRADE EXECUTION FAILED")
            error_msg = trade_result.get("error", trade_result.get("message", "Unknown error"))
            print(f"💥 Reason: {error_msg}")
    else:
        print("💤 No trade execution needed (HOLD recommendation)")
    
    print(f"{'⚡' * 20}")
    
    # Rationale with better formatting (enhanced from original)
    rationale = analysis.get('rationale', 'No rationale provided')
    print(f"\n💭 RATIONALE:")
    
    # Split long rationale into multiple lines for better readability
    words = rationale.split()
    lines = []
    current_line = ""
    
    for word in words:
        if len(current_line + " " + word) <= 70:  # Max 70 chars per line
            current_line += " " + word if current_line else word
        else:
            lines.append(current_line)
            current_line = word
    
    if current_line:
        lines.append(current_line)
    
    for line in lines:
        print(f"   {line}")
    
    # Show if analysis included specific technical signals (enhanced from original)
    if 'candlestick' in rationale.lower():
        print("\n🕯️ INCLUDES CANDLESTICK ANALYSIS")
    if 'news' in rationale.lower() or analysis.get('news_driven'):
        print("📰 INCLUDES NEWS IMPACT ANALYSIS")
    if analysis.get('consistency_score') is not None and analysis.get('consistency_score') < 1.0:
        print("🧠 INCLUDES POSITION HISTORY ANALYSIS")
    
    print(f"\n{'🤖' * 25}")

def display_startup_sequence(config, symbols):
    """Display enhanced startup information"""
    print_startup_banner()
    
    print("⚙️ CONFIGURATION:")
    print(f"  📊 Tracked assets: {', '.join([s.split('/')[0] for s in symbols])}")
    print(f"  ⏰ Data updates: Every {safe_get_env_int('DISPLAY_UPDATE_INTERVAL', 5)}s")
    print(f"  🤖 GPT analysis: Every {safe_get_env_int('GPT_ANALYSIS_INTERVAL', 60)}s") 
    print(f"  📰 News analysis: Every {safe_get_env_int('NEWS_UPDATE_FREQUENCY', 1200)}s")
    print(f"  🔄 Auto-trading: {'🟢 ENABLED' if safe_get_env_bool('ENABLE_AUTO_TRADING', True) else '🔴 DISABLED'}")
    print(f"  ⏰ Trade cooldown: {safe_get_env_int('TRADE_COOLDOWN_MINUTES', 120)} minutes")
    print(f"  🧠 Position history: {'🟢 ENABLED' if True else '🔴 DISABLED'}")  # Always enabled now
    print(f"  🐛 Debug dumps: {'🟢 ENABLED' if safe_get_env_bool('SHOW_DEBUG_DUMP', False) else '🔴 DISABLED'}")  # NEW
    
    # API status (UPDATED for Binance)
    print(f"\n🔑 API STATUS:")
    api_keys = {
        "OpenAI": "✅" if config.get("openai_api_key") else "❌",
        "NewsAPI": "✅" if config.get("news_api_key") else "❌", 
        "CoinGecko": "✅" if config.get("coingecko_api_key") else "🆓 Free",
        "Binance Trading": "✅" if config.get("binance_api_key") else "❌"  # CHANGED from Coinbase
    }
    
    for api, status in api_keys.items():
        print(f"  {api}: {status}")
    
    # Trading mode (UPDATED)
    use_testnet = safe_get_env_bool('USE_BINANCE_TESTNET', False)
    trading_mode = "🎭 SIMULATED" if safe_get_env_bool('USE_SIMULATED_TRADING', False) else "🧪 TESTNET" if use_testnet else "💰 LIVE"
    print(f"\n🎮 TRADING MODE: {trading_mode}")
    
    if trading_mode == "💰 LIVE":
        print("  ⚠️  LIVE TRADING ENABLED - REAL MONEY AT RISK!")
        print("  💡 Set USE_SIMULATED_TRADING=true in .env for testing")
        print("  💡 Set USE_BINANCE_TESTNET=true for testnet trading")
    elif trading_mode == "🧪 TESTNET":
        print("  ✅ Testnet mode - safe testing with fake money")
        print("  💡 Set USE_SIMULATED_TRADING=true for full simulation")
        print("  💡 Set USE_BINANCE_TESTNET=false for live trading")
    else:
        print("  ✅ Safe testing mode - no real trades")
        print("  💡 Set USE_SIMULATED_TRADING=false for live trading")
        print("  💡 Set USE_BINANCE_TESTNET=true for testnet trading")
    
    # Position manager status
    print(f"\n🏦 POSITION MANAGER:")
    try:
        pm = get_position_manager()
        stats = get_trading_stats()
        print(f"  📊 Historical trades: {stats['total_trades']}")
        print(f"  📈 Open positions: {stats['open_positions']}")
        print(f"  ⏰ Cooldown period: {stats['cooldown_period_minutes']} minutes")
        print(f"  💾 Data directory: {pm.data_dir}")
        print(f"  🧠 Trading memory: 24h lookback with consistency scoring")
    except Exception as e:
        print(f"  ❌ Error loading position manager: {str(e)}")
    
    print(f"\n🎮 CONTROLS:")
    print(f"  Press Ctrl+C to stop gracefully")
    print(f"  Logs saved to: crypto_bot.log")
    print(f"  Position data: data/")
    print(f"  💡 Add SHOW_DEBUG_DUMP=true to .env for JSON debug dumps")
    print(f"  💡 Set DEBUG_DUMP_INTERVAL=X to control dump frequency")
    
    print(f"\n{'🤖' * 50}")
    print("🤖 AUTO-TRADING BOT IS NOW RUNNING - READY TO TRADE! 🤖")
    print(f"{'🤖' * 50}\n")

def load_configuration():
    """Load configuration from environment variables"""
    logger.info("Loading configuration from .env file")
    load_dotenv()
    
    config = {
        # Use safe parsing for all integer values - FIXED ALL LINES
        "market_update_frequency": safe_get_env_int("MARKET_UPDATE_FREQUENCY", 60),
        "sentiment_update_frequency": safe_get_env_int("SENTIMENT_UPDATE_FREQUENCY", 300),
        "fear_greed_update_frequency": safe_get_env_int("FEAR_GREED_UPDATE_FREQUENCY", 3600),
        
        # String values (no parsing needed)
        "news_api_key": os.getenv("NEWS_API_KEY"),
        "openai_api_key": os.getenv("OPENAI_API_KEY"),
        "coingecko_api_key": os.getenv("COINGECKO_API_KEY"),
        
        # Binance API credentials
        "binance_api_key": os.getenv("BINANCE_API_KEY"),
        "binance_api_secret": os.getenv("BINANCE_API_SECRET"),
    }
    
    # Load tracking symbols - FIXED: Remove XRP to match MAIN_TRADING_ASSETS
    default_symbols = "BTC/USDT,ETH/USDT,SOL/USDT,DOGE/USDT"  # Removed XRP/USDT
    symbols_str = os.getenv("TRACKING_SYMBOLS", default_symbols)
    symbols = [s.strip() for s in symbols_str.split(",")]
    
    logger.info(f"Configured to track symbols: {', '.join(symbols)}")
    return config, symbols

def display_enhanced_status_countdown(last_analysis_result, time_since_last_analysis, 
                time_since_news_analysis, gpt_analysis_interval, 
                news_analysis_interval, auto_trading_enabled):
    """Enhanced countdown display with market-responsive context"""
    if not last_analysis_result:
        return
    
    # Calculate countdowns
    next_analysis_in = gpt_analysis_interval - time_since_last_analysis
    next_news_in = news_analysis_interval - time_since_news_analysis
    
    action_str = last_analysis_result['action'].upper()
    asset_str = f" {last_analysis_result.get('asset', '')}" if last_analysis_result.get('asset') else ""
    
    # Enhanced flags
    news_flag = " [NEWS-DRIVEN]" if last_analysis_result.get('news_driven') else ""
    consistency_flag = " [HISTORY-ADJUSTED]" if last_analysis_result.get('consistency_score', 1.0) < 1.0 else ""
    
    analysis_type = last_analysis_result.get('analysis_type', 'strategic')
    type_flag = ""
    if analysis_type == 'market_responsive_position_review':
        trigger = last_analysis_result.get('position_review_trigger', '')
        type_flag = f" [POSITION REVIEW: {trigger.replace('_', ' ').upper()}]"
    elif analysis_type == 'opportunity_scan':
        type_flag = " [OPPORTUNITY SCAN]"
    
    # Enhanced countdown display
    analysis_mins = int(next_analysis_in // 60)
    analysis_secs = int(next_analysis_in % 60)
    news_mins = int(next_news_in // 60)
    
    print(f"\n⏰ ENHANCED AUTO-TRADING STATUS:")
    print(f"  🎯 Last Signal: {action_str}{asset_str}{news_flag}{consistency_flag}{type_flag}")
    
    # Show conviction and confidence from last analysis
    conviction = last_analysis_result.get('conviction_score', 5.0)
    confidence = last_analysis_result.get('confidence', 0.0) * 100
    print(f"  💪 Last Conviction: {conviction:.1f}/10 (Confidence: {confidence:.0f}%)")
    
    print(f"  🤖 Next GPT Analysis: {analysis_mins}m {analysis_secs}s")
    print(f"  📰 Next News Check: {news_mins}m")
    print(f"  🔄 Auto-Trading: {'🟢 ACTIVE' if auto_trading_enabled else '🔴 DISABLED'}")
    print(f"  🧠 Market Responsiveness: 🟢 ACTIVE")
    
    # Show market condition awareness
    market_change = last_analysis_result.get('market_condition_change', '')
    if market_change and market_change != 'stable':
        print(f"  📊 Market Conditions: {market_change.replace('_', ' ').title()}")
    
    # Show quick cooldown status
    cooldown_status = get_cooldown_status()
    active_cooldowns = {symbol: info for symbol, info in cooldown_status.items() if info['in_cooldown']}
    if active_cooldowns:
        cooldown_summary = ", ".join([f"{symbol}({info['remaining_minutes']}m)" 
                                    for symbol, info in active_cooldowns.items()])
        print(f"  🚫 Active Cooldowns: {cooldown_summary}")

def main():
    """Main entry point - FOCUSED on 4 main trading assets"""
    setup_focused_logging()  # Add this first
    
    logger.info("Starting FOCUSED AUTO-TRADING Bot (BTC, ETH, SOL, DOGE)")
    
    try:
        # Load configuration
        config, symbols = load_configuration()
        
        # VALIDATION: Ensure we're only tracking main assets (updated with clearer messaging)
        expected_symbols = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "DOGE/USDT"]
        if len(symbols) != 4 or not all(s.split('/')[0] in MAIN_TRADING_ASSETS for s in symbols):
            logger.info(f"📝 Symbols configured: {symbols}")
            logger.info(f"🎯 Using focused trading assets: {expected_symbols}")
            symbols = expected_symbols
        
        logger.info(f"🎯 FOCUSED TRADING: {', '.join([s.split('/')[0] for s in symbols])}")
        
        # Initialize position manager early
        pm = get_position_manager()
        logger.info("Position Manager with trading history initialized successfully")
        
        # Display startup sequence
        display_startup_sequence(config, symbols)
        
        # Validate required API keys (UPDATED)
        missing_keys = []
        required_keys = ["news_api_key", "openai_api_key", "binance_api_key", "binance_api_secret"]  # CHANGED
        for key_name in required_keys:
            if not config.get(key_name):
                missing_keys.append(key_name)
        
        if missing_keys:
            logger.warning(f"Missing API keys: {', '.join(missing_keys)}. Some features may be limited.")
            if "binance_api_key" in missing_keys or "binance_api_secret" in missing_keys:  # CHANGED
                logger.warning("⚠️  Trading will be simulated without Binance API keys")
        
        # Check if auto-trading is enabled
        auto_trading_enabled = safe_get_env_bool('ENABLE_AUTO_TRADING', True)
        if not auto_trading_enabled:
            logger.info("Auto-trading is disabled - will only show recommendations")
        
        # Initialize Binance trading client (UPDATED)
        api_key = config.get("binance_api_key")
        api_secret = config.get("binance_api_secret")
        trading_client = initialize_binance_client(api_key, api_secret)
        
        if not trading_client:
            logger.error("Failed to initialize Binance trading client")
            return
        
        # Initialize data collector
        logger.info("Initializing CryptoDataCollector")
        collector = CryptoDataCollector(symbols=symbols, config=config)
        
        # Start data collection
        logger.info("Starting data collection process")
        collector.start_collection()
        
        # Set update intervals - ALL USING SAFE PARSING
        data_display_interval = safe_get_env_int("DISPLAY_UPDATE_INTERVAL", 5)
        gpt_analysis_interval = safe_get_env_int("GPT_ANALYSIS_INTERVAL", 60)
        news_analysis_interval = safe_get_env_int("NEWS_UPDATE_FREQUENCY", 1200)
        
        logger.info(f"Data updates: every {data_display_interval}s")
        logger.info(f"GPT analysis: every {gpt_analysis_interval}s")
        logger.info(f"News analysis: every {news_analysis_interval}s")
        logger.info(f"Auto-trading: {'ENABLED' if auto_trading_enabled else 'DISABLED'}")
        logger.info(f"Trade cooldown: {safe_get_env_int('TRADE_COOLDOWN_MINUTES', 120)} minutes")
        logger.info(f"Position history: ENABLED with consistency scoring")
        
        update_count = 0
        last_gpt_analysis_time = datetime.now()
        last_news_analysis_time = datetime.now() - timedelta(seconds=news_analysis_interval)
        last_analysis_result = None
        last_news_impact = None
        
        # Main trading loop with position management and history
        while True:
            time.sleep(data_display_interval)
            current_time = datetime.now()
            update_count += 1
            
            try:
                # Get latest data
                latest_data = collector.get_latest_data()
                formatted_data = collector.format_data_for_gpt()
                
                # Display the data update (includes position status every 5 updates)
                display_data_update(formatted_data, update_count)
                logger.info(f"Data update #{update_count} completed")
                
                # Check if we need to analyze news
                time_since_news_analysis = (current_time - last_news_analysis_time).total_seconds()
                if time_since_news_analysis >= news_analysis_interval:
                    logger.info("Running news impact analysis")
                    print(f"\n🔍 ANALYZING NEWS FOR TRADING OPPORTUNITIES...")
                    
                    sentiment_data = collector.get_data_by_type("sentiment")
                    if sentiment_data and sentiment_data.get("data", {}).get("articles"):
                        news_articles = sentiment_data["data"]["articles"]
                        portfolio_data = get_focused_portfolio_state(formatted_data.get("market", {}))
                        
                        news_impact = process_news_batch(news_articles, formatted_data, portfolio_data)
                        last_news_impact = news_impact
                        last_news_analysis_time = current_time
                        
                        # Display news summary
                        summary = news_impact.get("summary", {})
                        print(f"📰 NEWS SUMMARY: {summary.get('relevant_articles', 0)} relevant articles")
                        if news_impact.get("best_trading_opportunity"):
                            print(f"⚡ Trading opportunity detected with {news_impact['best_trading_opportunity'].get('confidence', 0):.0%} confidence")
                
                # Run GPT analysis and execute trades with position history
                time_since_last_analysis = (current_time - last_gpt_analysis_time).total_seconds()
                if time_since_last_analysis >= gpt_analysis_interval:
                    logger.info("Running GPT analysis with position history and trade execution")
                    print(f"\n🔮 RUNNING GPT ANALYSIS & AUTO-TRADING WITH POSITION HISTORY...")
                    
                    # Get current portfolio
                    portfolio_data = get_focused_portfolio_state(formatted_data.get("market", {}))
                    if portfolio_data:
                        total_value = sum(portfolio_data.values())
                        print(f"\n💼 CURRENT PORTFOLIO: ${total_value:,.2f}")
                        for currency, value in sorted(portfolio_data.items(), key=lambda x: x[1], reverse=True):
                            percentage = (value / total_value) * 100 if total_value > 0 else 0
                            currency_emoji = {"BTC": "₿", "ETH": "Ξ", "SOL": "◎", "DOGE": "Ð", "USD": "💵"}.get(currency, "💰")  # Removed XRP
                            
                            # Add position indicator
                            position_indicator = ""
                            if currency in pm.get_open_positions():
                                position_indicator = " 📈"
                            elif pm.is_in_cooldown(currency):
                                remaining = pm.get_cooldown_remaining(currency)
                                remaining_minutes = int(remaining.total_seconds() / 60) if remaining else 0
                                position_indicator = f" 🚫({remaining_minutes}m)"
                            
                            print(f"  {currency_emoji} {currency}: ${value:,.2f} ({percentage:.1f}%){position_indicator}")
                    
                    # Enhanced GPT analysis with market responsiveness
                    analysis_result, trade_result = enhanced_gpt_analysis_execution(
                        formatted_data, portfolio_data, last_news_impact
                    )
                    last_analysis_result = analysis_result
                    last_gpt_analysis_time = current_time

                    # Enhanced logging for different analysis types
                    analysis_type = analysis_result.get('analysis_type', 'strategic')
                    if analysis_type == 'market_responsive_position_review':
                        trigger = analysis_result.get('position_review_trigger', 'unknown')
                        logger.info(f"🔄 Position review completed: {trigger}")
                        
                        if trade_result and trade_result.get('execution_type') == 'position_reduction':
                            reduction_pct = trade_result.get('reduction_percentage', 0)
                            asset = analysis_result.get('asset', '')
                            logger.info(f"📉 Position reduction executed: {asset} reduced by {reduction_pct}%")

                    if analysis_result.get("news_driven"):
                        logger.info(f"📰 News-driven trade signal executed with urgency: {analysis_result.get('urgency', 'medium')}")

                    if trade_result and trade_result.get("executed"):
                        execution_type = trade_result.get('execution_type', 'unknown')
                        logger.info(f"✅ TRADE EXECUTED ({execution_type}): {trade_result['action'].upper()} {trade_result.get('asset', '')} "
                                f"- Amount: {trade_result.get('usd_amount', trade_result.get('crypto_amount', 0))}")
                    elif trade_result and trade_result.get("blocked_by_cooldown"):
                        logger.info(f"🚫 TRADE BLOCKED: {analysis_result.get('asset')} in cooldown "
                                f"for {trade_result.get('remaining_cooldown_minutes', 0)} minutes")
                    else:
                        logger.info("Auto-trading disabled - showing recommendation only")
                        # Still record the recommendation for tracking
                        if analysis_result.get("action") != "hold":
                            pm.record_trade(
                                symbol=analysis_result.get("asset", ""),
                                action=analysis_result.get("action", "hold"),
                                confidence=analysis_result.get("confidence", 0.0),
                                rationale=analysis_result.get("rationale", ""),
                                amount_percentage=analysis_result.get("amount_percentage", 0),
                                executed=False
                            )
                    
                    # Display results with position history analysis
                    display_gpt_analysis_with_execution(analysis_result, trade_result)
                    
                    if analysis_result.get("news_driven"):
                        logger.info(f"NEWS-DRIVEN trade signal executed with urgency: {analysis_result.get('urgency', 'medium')}")
                    
                    if trade_result and trade_result.get("executed"):
                        logger.info(f"✅ TRADE EXECUTED: {trade_result['action'].upper()} {trade_result.get('asset', '')} "
                                   f"- Amount: {trade_result.get('usd_amount', trade_result.get('crypto_amount', 0))}")
                    elif trade_result and trade_result.get("blocked_by_cooldown"):
                        logger.info(f"🚫 TRADE BLOCKED: {analysis_result.get('asset')} in cooldown "
                                   f"for {trade_result.get('remaining_cooldown_minutes', 0)} minutes")
                
                # Show status countdown with enhanced position info
                elif last_analysis_result:
                    # Call enhanced status countdown
                    display_enhanced_status_countdown(
                        last_analysis_result, time_since_last_analysis, time_since_news_analysis,
                        gpt_analysis_interval, news_analysis_interval, auto_trading_enabled
                    )
                
                # Optional detailed data dump (controlled by environment variable)
                show_debug_dump = safe_get_env_bool('SHOW_DEBUG_DUMP', False)
                debug_dump_interval = safe_get_env_int('DEBUG_DUMP_INTERVAL', 50)  # Every 50 updates by default
                
                if show_debug_dump and update_count % debug_dump_interval == 0:
                    print(f"\n{'📋' * 20}")
                    print(f"📋 DEBUG DATA DUMP #{update_count}")
                    print(f"{'📋' * 20}")
                    print(json.dumps(formatted_data, indent=2, default=str))
                    print(f"{'📋' * 20}")
                
                # Show detailed position manager stats every 20 updates (much less frequent)
                if update_count % 20 == 0:
                    try:
                        trading_stats = get_trading_stats()
                        print(f"\n📊 TRADING STATS UPDATE #{update_count}:")
                        print(f"  📊 Total: {trading_stats['executed_trades']} executed trades")
                        print(f"  📈 Open: {trading_stats['open_positions']} positions")
                        print(f"  💪 Avg Conviction: {trading_stats['avg_conviction_score']}/10")
                        print(f"  🎯 Success Rate: {trading_stats['symbols_traded']} symbols traded")
                        
                        # Show recent trading decisions for context (more compact)
                        recent_trades = pm.get_recent_trades(hours=6)  # Last 6 hours only
                        if recent_trades:
                            print(f"  🧠 Recent (6h): {len(recent_trades)} trades")
                            for trade in recent_trades[-2:]:  # Show only last 2 trades
                                trade_time = datetime.fromisoformat(trade.timestamp)
                                time_ago = (datetime.now() - trade_time).total_seconds() / 3600
                                conviction = getattr(trade, 'conviction_score', trade.confidence * 10)
                                print(f"    {trade.symbol} {trade.action.upper()}: {time_ago:.1f}h ago ({conviction:.1f}/10)")
                    except Exception as e:
                        logger.error(f"Error showing trading stats: {str(e)}")
                    
            except Exception as e:
                logger.error(f"Error in update #{update_count}: {str(e)}", exc_info=True)
                print(f"❌ Error in update #{update_count}: {str(e)}")
    
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt detected, shutting down gracefully")
        print(f"\n{'🛑' * 20}")
        print("🛑 SHUTTING DOWN AUTO-TRADING BOT...")
        print(f"{'🛑' * 20}")
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        print(f"\n❌ Unexpected Error: {str(e)}")
    finally:
        # Stop data collection
        if 'collector' in locals():
            logger.info("Stopping data collection")
            collector.stop_collection()
            logger.info("Data collection stopped successfully")
            print("✅ Data collection stopped successfully.")
        
        # Export position manager data before shutdown
        try:
            pm = get_position_manager()
            export_path = pm.export_trade_history()
            logger.info(f"📊 Trade history exported to: {export_path}")
            print(f"📊 Trade history exported to: {export_path}")
        except Exception as e:
            logger.error(f"Error exporting trade history: {str(e)}")
        
        logger.info("Auto-Trading Crypto Bot with Position History shutdown complete")
        print(f"\n{'👋' * 20}")
        print("👋 AUTO-TRADING BOT SHUTDOWN COMPLETE")
        print(f"{'👋' * 20}\n")

if __name__ == "__main__":
    main()