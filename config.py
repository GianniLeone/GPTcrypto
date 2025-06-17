# Configuration with Position Manager settings and Binance API
config = {
    "market_update_frequency": 60,      # seconds
    "sentiment_update_frequency": 300,  # seconds
    "fear_greed_update_frequency": 3600, # seconds (hourly)
    "news_api_key": "YOUR_NEWS_API_KEY",
    "openai_api_key": "YOUR_OPENAI_API_KEY",
    "coingecko_api_key": "YOUR_COINGECKO_API_KEY",  # Optional, can use free tier
    
    # Binance API credentials (replace Coinbase)
    "binance_api_key": "YOUR_BINANCE_API_KEY",       # Binance API key
    "binance_api_secret": "YOUR_BINANCE_API_SECRET", # Binance API secret
    
    # Position Manager settings
    "trade_cooldown_minutes": 120,      # Cooldown period between trades per asset
    "position_data_dir": "data",        # Directory to store position data
    "max_trade_history_days": 30,       # Keep trade history for N days
    
    # Trading settings
    "enable_auto_trading": True,        # Enable/disable automatic trade execution
    "use_simulated_trading": False,     # Use simulated trades for testing
    "use_binance_testnet": False,       # Use Binance testnet for testing (set to True for testing)
    "max_position_percentage": 15,      # Maximum % of portfolio per position
    "min_trade_usd": 10.0,             # Minimum trade amount in USD (Binance minimum ~$10)
    
    # Analysis settings
    "gpt_analysis_interval": 60,        # seconds between GPT analyses
    "news_update_frequency": 1200,      # seconds between news analyses (20 min)
    "display_update_interval": 5,       # seconds between data display updates
    "max_gpt_queries_per_hour": 10,     # Rate limit for GPT API calls
    
    # Technical analysis settings
    "assets_per_cycle": 3,              # Number of assets to update technical indicators per cycle
    "full_update_interval": 300,        # seconds for full technical analysis update
    "indicators_cache_duration": 180,   # seconds to cache technical indicators
}

# Symbols to track (Binance format internally converts these)
symbols = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT", "DOGE/USDT"]

# Example usage with Position Manager integration and Binance API
if __name__ == "__main__":
    # Initialize and start the collector with position management
    from CryptoDataCollector import CryptoDataCollector
    from position_manager import get_position_manager
    
    # Initialize position manager first
    position_manager = get_position_manager()
    print(f"Position Manager initialized with {position_manager.cooldown_minutes}min cooldown")
    
    # Initialize the data collector
    collector = CryptoDataCollector(symbols=symbols, config=config)
    collector.start_collection()
    
    # Get the latest data
    latest_data = collector.get_latest_data()
    
    # Get data formatted for GPT analysis
    gpt_data = collector.format_data_for_gpt()
    
    # Example: Record a simulated trade
    position_manager.record_trade(
        symbol="BTC",
        action="buy", 
        confidence=0.85,
        rationale="Strong technical signals and positive sentiment",
        amount_percentage=25,
        executed=False  # Simulated for example
    )
    
    # Check cooldown status
    cooldown_status = position_manager.get_cooldown_status()
    print(f"Cooldown status: {cooldown_status}")
    
    # Get trading statistics
    stats = position_manager.get_trading_stats()
    print(f"Trading stats: {stats}")
    
    # Stop collection when done
    collector.stop_collection()
    
    # Export trade history
    export_path = position_manager.export_trade_history()
    print(f"Trade history exported to: {export_path}")
    
    print("\n🔄 MIGRATION TO BINANCE COMPLETE!")
    print("📝 Update your .env file with:")
    print("   BINANCE_API_KEY=your_binance_api_key")
    print("   BINANCE_API_SECRET=your_binance_api_secret")
    print("   USE_BINANCE_TESTNET=true  # For testing")
    print("   USE_SIMULATED_TRADING=false  # For live trading")