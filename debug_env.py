#!/usr/bin/env python3
"""
Debug script to check .env file loading
Run this to diagnose dotenv issues
"""

import os
import sys
from pathlib import Path

def debug_env_file():
    """Debug .env file loading issues"""
    
    print("🔍 DEBUGGING .env FILE LOADING")
    print("="*50)
    
    # 1. Check current directory
    current_dir = Path.cwd()
    print(f"📁 Current directory: {current_dir}")
    
    # 2. Check if .env file exists
    env_file = current_dir / ".env"
    print(f"📄 Looking for .env at: {env_file}")
    print(f"📄 .env file exists: {env_file.exists()}")
    
    if not env_file.exists():
        print("\n❌ .env file NOT FOUND!")
        print("Create a .env file in the same directory as main.py")
        return False
    
    # 3. Check file permissions and encoding
    try:
        with open(env_file, 'r', encoding='utf-8') as f:
            content = f.read()
        print(f"📄 .env file readable: ✅")
        print(f"📄 .env file size: {len(content)} characters")
    except UnicodeDecodeError:
        print("❌ .env file has encoding issues! Try saving as UTF-8")
        return False
    except Exception as e:
        print(f"❌ Cannot read .env file: {e}")
        return False
    
    # 4. Show first few lines (without revealing secrets)
    print(f"\n📄 First few lines of .env file:")
    lines = content.strip().split('\n')
    for i, line in enumerate(lines[:10]):  # Show first 10 lines
        if '=' in line:
            key = line.split('=')[0]
            value = line.split('=', 1)[1]
            # Hide the actual value for security
            masked_value = '*' * min(len(value), 10) if value else '(empty)'
            print(f"  {i+1}: {key}={masked_value}")
        else:
            print(f"  {i+1}: {line}")
    
    if len(lines) > 10:
        print(f"  ... and {len(lines) - 10} more lines")
    
    # 5. Check for common syntax issues
    print(f"\n🔍 CHECKING FOR COMMON ISSUES:")
    
    issues = []
    for i, line in enumerate(lines, 1):
        line = line.strip()
        if not line or line.startswith('#'):
            continue
            
        if ':' in line and '=' not in line:
            issues.append(f"Line {i}: Uses ':' instead of '=' - {line[:30]}...")
        
        if '=' not in line:
            issues.append(f"Line {i}: No '=' found - {line[:30]}...")
        
        if line.count('=') > 1 and not (line.startswith('"') or line.startswith("'")):
            issues.append(f"Line {i}: Multiple '=' without quotes - {line[:30]}...")
    
    if issues:
        print("❌ SYNTAX ISSUES FOUND:")
        for issue in issues:
            print(f"  • {issue}")
    else:
        print("✅ No obvious syntax issues found")
    
    # 6. Test loading with dotenv
    print(f"\n🧪 TESTING DOTENV LOADING:")
    
    try:
        from dotenv import load_dotenv, find_dotenv
        
        # Try to find .env file
        found_env = find_dotenv()
        print(f"📄 find_dotenv() result: '{found_env}'")
        
        # Try to load with verbose output
        result = load_dotenv(verbose=True)
        print(f"📄 load_dotenv() result: {result}")
        
        # Try with override=True
        result2 = load_dotenv(override=True, verbose=True)
        print(f"📄 load_dotenv(override=True) result: {result2}")
        
    except ImportError:
        print("❌ python-dotenv not installed!")
        print("Run: pip install python-dotenv")
        return False
    except Exception as e:
        print(f"❌ Error loading dotenv: {e}")
        return False
    
    # 7. Check specific environment variables
    print(f"\n🔍 CHECKING SPECIFIC VARIABLES:")
    
    test_vars = [
        'USE_SIMULATED_TRADING',
        'OPENAI_API_KEY', 
        'NEWS_API_KEY',
        'BINANCE_API_KEY',
        'BINANCE_API_SECRET'
    ]
    
    for var in test_vars:
        value = os.getenv(var)
        if value:
            # Mask the value for security
            if len(value) > 10:
                masked = value[:4] + '*' * 6 + value[-4:]
            else:
                masked = '*' * len(value)
            print(f"  ✅ {var}: {masked}")
        else:
            print(f"  ❌ {var}: Not found")
    
    # 8. Check if in simulation mode
    sim_mode = os.getenv('USE_SIMULATED_TRADING', '').lower()
    print(f"\n🎭 SIMULATION MODE CHECK:")
    print(f"  USE_SIMULATED_TRADING value: '{sim_mode}'")
    print(f"  Is simulation mode: {sim_mode in ['true', '1', 'yes']}")
    
    return True

def create_sample_env():
    """Create a sample .env file"""
    sample_content = """# Crypto Trading Bot Configuration
# IMPORTANT: Set USE_SIMULATED_TRADING=true for safe testing

# TRADING MODE (Set to true for safe testing)
USE_SIMULATED_TRADING=true
USE_BINANCE_TESTNET=false
ENABLE_AUTO_TRADING=true

# API KEYS (Replace with your actual keys)
OPENAI_API_KEY=your_openai_key_here
NEWS_API_KEY=your_news_key_here
BINANCE_API_KEY=your_binance_key_here
BINANCE_API_SECRET=your_binance_secret_here
COINGECKO_API_KEY=

# TRADING SETTINGS
TRACKING_SYMBOLS=BTC/USDT,ETH/USDT,SOL/USDT,DOGE/USDT
TRADE_COOLDOWN_MINUTES=120
MAX_POSITION_PERCENTAGE=15
MIN_TRADE_USD=10.0

# UPDATE INTERVALS (seconds)
DISPLAY_UPDATE_INTERVAL=5
GPT_ANALYSIS_INTERVAL=60
MARKET_UPDATE_FREQUENCY=60
SENTIMENT_UPDATE_FREQUENCY=300
NEWS_UPDATE_FREQUENCY=1200
FEAR_GREED_UPDATE_FREQUENCY=3600

# GPT SETTINGS
GPT_MODEL=gpt-4
MAX_GPT_QUERIES_PER_HOUR=10

# DEBUG SETTINGS
SHOW_DEBUG_DUMP=false
DEBUG_DUMP_INTERVAL=50
"""
    
    env_file = Path('.env')
    if env_file.exists():
        backup = Path('.env.backup')
        print(f"📄 Backing up existing .env to {backup}")
        env_file.rename(backup)
    
    try:
        with open(env_file, 'w', encoding='utf-8') as f:
            f.write(sample_content)
        print(f"✅ Created sample .env file")
        return True
    except Exception as e:
        print(f"❌ Failed to create .env file: {e}")
        return False

if __name__ == "__main__":
    success = debug_env_file()
    
    if not success:
        print(f"\n🔧 WOULD YOU LIKE TO CREATE A SAMPLE .env FILE?")
        response = input("Create sample .env file? (y/N): ").lower().strip()
        if response in ['y', 'yes']:
            create_sample_env()
            print(f"\n✅ Sample .env file created!")
            print(f"📝 Edit it with your real API keys, then run main.py again")