import os
import json
import logging
import statistics
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import threading

# Configure logging
logger = logging.getLogger("CryptoBot.PositionManager")

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

class TradeAction(Enum):
    """Enum for trade actions"""
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"

class ConvictionLevel(Enum):
    """Enum for conviction levels"""
    VERY_LOW = 1
    LOW = 3
    MEDIUM = 5
    HIGH = 7
    VERY_HIGH = 9

@dataclass
class TradeRecord:
    """Enhanced data class for storing comprehensive trade information"""
    symbol: str
    action: str
    timestamp: str
    confidence: float
    rationale: str
    amount_percentage: float
    conviction_score: float = 5.0  # 1-10 scale
    usd_amount: Optional[float] = None
    crypto_amount: Optional[float] = None
    executed: bool = False
    simulated: bool = False
    order_id: Optional[str] = None
    entry_price: Optional[float] = None
    exit_price: Optional[float] = None
    market_conditions: Optional[Dict[str, Any]] = None
    technical_summary: str = "neutral"
    sentiment_category: str = "neutral"
    fear_greed_value: Optional[int] = None
    news_driven: bool = False
    emergency_override: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TradeRecord':
        """Create TradeRecord from dictionary"""
        # Handle legacy data that might not have all fields
        required_fields = ['symbol', 'action', 'timestamp', 'confidence', 'rationale', 'amount_percentage']
        for field in required_fields:
            if field not in data:
                data[field] = "" if field in ['symbol', 'action', 'rationale'] else 0
        
        return cls(**{k: v for k, v in data.items() if k in cls.__annotations__})

@dataclass
class Position:
    """Enhanced data class for tracking current positions"""
    symbol: str
    entry_action: str
    entry_timestamp: str
    entry_confidence: float
    entry_rationale: str
    amount_percentage: float
    entry_conviction_score: float = 5.0
    usd_amount: Optional[float] = None
    crypto_amount: Optional[float] = None
    entry_price: Optional[float] = None
    is_open: bool = True
    exit_timestamp: Optional[str] = None
    exit_action: Optional[str] = None
    exit_rationale: Optional[str] = None
    exit_price: Optional[float] = None
    max_conviction_seen: float = 5.0
    min_conviction_seen: float = 5.0
    conviction_snapshots_count: int = 0
    position_pnl: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Position':
        """Create Position from dictionary"""
        # Handle legacy data
        required_fields = ['symbol', 'entry_action', 'entry_timestamp', 'entry_confidence', 'entry_rationale', 'amount_percentage']
        for field in required_fields:
            if field not in data:
                data[field] = "" if field in ['symbol', 'entry_action', 'entry_rationale'] else 0
        
        return cls(**{k: v for k, v in data.items() if k in cls.__annotations__})

@dataclass
class ConvictionSnapshot:
    """Track conviction and market conditions at different points in time"""
    timestamp: str
    symbol: str
    conviction_score: float
    confidence: float
    action_considered: str
    market_conditions: Dict[str, Any]
    technical_summary: str
    sentiment_category: str
    fear_greed_value: Optional[int]
    analysis_type: str = "regular"
    position_exists: bool = False
    consistency_score: Optional[float] = None
    signal_changes_6h: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ConvictionSnapshot':
        return cls(**data)

@dataclass
class ConsistencyAnalysis:
    """Results of conviction consistency analysis"""
    symbol: str
    consistency_score: float
    conviction_trend: str
    signal_changes: int
    snapshots_analyzed: int
    avg_conviction: float
    conviction_volatility: float
    recommendation: str
    time_period_hours: int
    technical_signals: List[str]
    sentiment_signals: List[str]
    last_updated: str
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

class PositionManager:
    """
    Enhanced Position Manager with strategic conviction-based trading logic
    """
    
    def __init__(self, data_dir: str = "data", cooldown_minutes: int = 120):
        """
        Initialize Enhanced PositionManager
        
        Args:
            data_dir: Directory to store persistent data
            cooldown_minutes: Base cooldown period in minutes
        """
        self.data_dir = data_dir
        self.base_cooldown_minutes = cooldown_minutes
        self.cooldown_period = timedelta(minutes=cooldown_minutes)
        
        # Dynamic cooldown bounds
        self.min_cooldown_minutes = max(30, cooldown_minutes // 4)  # Min 30 min
        self.max_cooldown_minutes = cooldown_minutes * 3  # Max 6 hours default
        
        # Thread lock for thread-safe operations
        self._lock = threading.Lock()
        
        # Storage files
        self.trade_history_file = os.path.join(data_dir, "trade_history.json")
        self.positions_file = os.path.join(data_dir, "positions.json")
        self.cooldowns_file = os.path.join(data_dir, "cooldowns.json")
        self.conviction_file = os.path.join(data_dir, "conviction_snapshots.json")
        self.consistency_file = os.path.join(data_dir, "consistency_analysis.json")
        
        # In-memory storage
        self.trade_history: List[TradeRecord] = []
        self.positions: Dict[str, Position] = {}
        self.cooldowns: Dict[str, Dict[str, Any]] = {}  # Enhanced cooldown tracking
        self.conviction_snapshots: List[ConvictionSnapshot] = []
        self.consistency_cache: Dict[str, ConsistencyAnalysis] = {}
        
        # Ensure data directory exists
        os.makedirs(data_dir, exist_ok=True)
        
        # Load existing data
        self._load_all_data()
        
        # Log initialization
        logger.info(f"Enhanced PositionManager initialized:")
        logger.info(f"  Base cooldown: {cooldown_minutes} min (range: {self.min_cooldown_minutes}-{self.max_cooldown_minutes})")
        logger.info(f"  Trade history: {len(self.trade_history)} records")
        logger.info(f"  Open positions: {len([p for p in self.positions.values() if p.is_open])}")
        logger.info(f"  Conviction snapshots: {len(self.conviction_snapshots)}")
        logger.info(f"  Active cooldowns: {len([s for s, c in self.cooldowns.items() if self._is_cooldown_active(s)])}")
    
    def _load_all_data(self):
        """Load all persistent data from disk"""
        try:
            # Load trade history
            if os.path.exists(self.trade_history_file):
                with open(self.trade_history_file, 'r') as f:
                    trade_data = json.load(f)
                    self.trade_history = [TradeRecord.from_dict(trade) for trade in trade_data]
                logger.info(f"Loaded {len(self.trade_history)} trades from {self.trade_history_file}")
            
            # Load positions
            if os.path.exists(self.positions_file):
                with open(self.positions_file, 'r') as f:
                    positions_data = json.load(f)
                    self.positions = {
                        symbol: Position.from_dict(pos_data) 
                        for symbol, pos_data in positions_data.items()
                    }
                logger.info(f"Loaded {len(self.positions)} positions from {self.positions_file}")
            
            # Load enhanced cooldowns
            if os.path.exists(self.cooldowns_file):
                with open(self.cooldowns_file, 'r') as f:
                    self.cooldowns = json.load(f)
                # Convert legacy simple cooldowns to enhanced format
                for symbol, cooldown_data in self.cooldowns.items():
                    if isinstance(cooldown_data, str):
                        # Legacy format - convert to new format
                        self.cooldowns[symbol] = {
                            "last_trade_timestamp": cooldown_data,
                            "cooldown_duration_minutes": self.base_cooldown_minutes,
                            "conviction_score": 5.0,
                            "consistency_score": 0.5
                        }
                logger.info(f"Loaded {len(self.cooldowns)} cooldown records")
            
            # Load conviction snapshots
            if os.path.exists(self.conviction_file):
                with open(self.conviction_file, 'r') as f:
                    conviction_data = json.load(f)
                    self.conviction_snapshots = [
                        ConvictionSnapshot.from_dict(snapshot) 
                        for snapshot in conviction_data
                    ]
                logger.info(f"Loaded {len(self.conviction_snapshots)} conviction snapshots")
            
            # Load consistency cache
            if os.path.exists(self.consistency_file):
                with open(self.consistency_file, 'r') as f:
                    consistency_data = json.load(f)
                    self.consistency_cache = {
                        symbol: ConsistencyAnalysis(**analysis_data)
                        for symbol, analysis_data in consistency_data.items()
                    }
                logger.info(f"Loaded consistency cache for {len(self.consistency_cache)} symbols")
                
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            # Initialize with empty data if loading fails
            self.trade_history = []
            self.positions = {}
            self.cooldowns = {}
            self.conviction_snapshots = []
            self.consistency_cache = {}
    
    def _save_all_data(self):
        """Save all data to disk"""
        try:
            with self._lock:
                # Save trade history
                with open(self.trade_history_file, 'w') as f:
                    trade_data = [trade.to_dict() for trade in self.trade_history]
                    json.dump(trade_data, f, indent=2, default=str)
                
                # Save positions
                with open(self.positions_file, 'w') as f:
                    positions_data = {
                        symbol: position.to_dict() 
                        for symbol, position in self.positions.items()
                    }
                    json.dump(positions_data, f, indent=2, default=str)
                
                # Save enhanced cooldowns
                with open(self.cooldowns_file, 'w') as f:
                    json.dump(self.cooldowns, f, indent=2, default=str)
                
                # Save conviction snapshots (keep last 48 hours only)
                cutoff_time = datetime.now() - timedelta(hours=48)
                active_snapshots = [
                    s for s in self.conviction_snapshots
                    if datetime.fromisoformat(s.timestamp) > cutoff_time
                ]
                self.conviction_snapshots = active_snapshots
                
                with open(self.conviction_file, 'w') as f:
                    conviction_data = [snapshot.to_dict() for snapshot in self.conviction_snapshots]
                    json.dump(conviction_data, f, indent=2, default=str)
                
                # Save consistency cache
                with open(self.consistency_file, 'w') as f:
                    consistency_data = {
                        symbol: analysis.to_dict() 
                        for symbol, analysis in self.consistency_cache.items()
                    }
                    json.dump(consistency_data, f, indent=2, default=str)
                    
        except Exception as e:
            logger.error(f"Error saving data: {str(e)}")
    
    def _is_cooldown_active(self, symbol: str) -> bool:
        """Check if cooldown is currently active for symbol"""
        if symbol not in self.cooldowns:
            return False
        
        try:
            cooldown_data = self.cooldowns[symbol]
            if isinstance(cooldown_data, str):
                # Legacy format
                last_trade_time = datetime.fromisoformat(cooldown_data)
                cooldown_duration = self.cooldown_period
            else:
                # New format
                last_trade_time = datetime.fromisoformat(cooldown_data["last_trade_timestamp"])
                cooldown_duration = timedelta(minutes=cooldown_data.get("cooldown_duration_minutes", self.base_cooldown_minutes))
            
            current_time = datetime.now()
            time_since_trade = current_time - last_trade_time
            
            is_cooling = time_since_trade < cooldown_duration
            
            if not is_cooling:
                # Cooldown period has passed, remove from cooldowns
                with self._lock:
                    del self.cooldowns[symbol]
                    self._save_all_data()
            
            return is_cooling
            
        except Exception as e:
            logger.error(f"Error checking cooldown for {symbol}: {str(e)}")
            return False
    
    def is_in_cooldown(self, symbol: str) -> bool:
        """Public interface for cooldown checking"""
        return self._is_cooldown_active(symbol)
    
    def get_cooldown_remaining(self, symbol: str) -> Optional[timedelta]:
        """Get remaining cooldown time for a symbol"""
        if not self.is_in_cooldown(symbol):
            return None
        
        try:
            cooldown_data = self.cooldowns[symbol]
            if isinstance(cooldown_data, str):
                last_trade_time = datetime.fromisoformat(cooldown_data)
                cooldown_duration = self.cooldown_period
            else:
                last_trade_time = datetime.fromisoformat(cooldown_data["last_trade_timestamp"])
                cooldown_duration = timedelta(minutes=cooldown_data.get("cooldown_duration_minutes", self.base_cooldown_minutes))
            
            current_time = datetime.now()
            elapsed = current_time - last_trade_time
            remaining = cooldown_duration - elapsed
            
            return remaining if remaining.total_seconds() > 0 else None
            
        except Exception as e:
            logger.error(f"Error calculating cooldown remaining for {symbol}: {str(e)}")
            return None
    
    def record_conviction_snapshot(self, symbol: str, conviction_score: float, confidence: float,
                                 action_considered: str, market_data: Dict[str, Any], 
                                 sentiment_data: Dict[str, Any], fear_greed_data: Dict[str, Any] = None,
                                 analysis_type: str = "regular"):
        """
        Record a snapshot of market conditions and conviction at a point in time
        """
        timestamp = datetime.now().isoformat()
        
        # Extract relevant market conditions
        symbol_market_data = market_data.get(f"{symbol}/USDT", market_data.get(symbol, {}))
        technical_indicators = symbol_market_data.get("technical_indicators", {})
        
        snapshot = ConvictionSnapshot(
            timestamp=timestamp,
            symbol=symbol,
            conviction_score=conviction_score,
            confidence=confidence,
            action_considered=action_considered,
            market_conditions={
                "price": symbol_market_data.get("price"),
                "change_24h": symbol_market_data.get("change_24h_percent"),
                "volume_24h": symbol_market_data.get("volume_24h"),
                "rsi": technical_indicators.get("rsi"),
                "macd_interpretation": technical_indicators.get("macd_interpretation"),
                "ma_trend": technical_indicators.get("ma_trend"),
                "candlestick_signal": technical_indicators.get("candlestick_signal")
            },
            technical_summary=technical_indicators.get("technical_summary", "neutral"),
            sentiment_category=sentiment_data.get(symbol, {}).get("sentiment_category", "neutral"),
            fear_greed_value=fear_greed_data.get("value") if fear_greed_data else None,
            analysis_type=analysis_type,
            position_exists=symbol in self.positions and self.positions[symbol].is_open
        )
        
        with self._lock:
            self.conviction_snapshots.append(snapshot)
            
            # Update position conviction tracking if position exists
            if symbol in self.positions and self.positions[symbol].is_open:
                position = self.positions[symbol]
                position.max_conviction_seen = max(position.max_conviction_seen, conviction_score)
                position.min_conviction_seen = min(position.min_conviction_seen, conviction_score)
                position.conviction_snapshots_count += 1
            
            # Trigger consistency analysis update
            self._update_consistency_analysis(symbol)
            
            self._save_all_data()
        
        logger.info(f"📸 Conviction snapshot recorded for {symbol}: {conviction_score:.1f}/10 ({action_considered})")
    
    def _update_consistency_analysis(self, symbol: str):
        """Update consistency analysis for a symbol"""
        try:
            analysis = self._analyze_conviction_consistency(symbol, hours_lookback=6)
            self.consistency_cache[symbol] = analysis
        except Exception as e:
            logger.error(f"Error updating consistency analysis for {symbol}: {str(e)}")
    
    def _analyze_conviction_consistency(self, symbol: str, hours_lookback: int = 6) -> ConsistencyAnalysis:
        """
        Analyze how consistent conviction and market signals have been over time
        """
        cutoff_time = datetime.now() - timedelta(hours=hours_lookback)
        
        # Get recent snapshots for this symbol
        recent_snapshots = [
            s for s in self.conviction_snapshots
            if s.symbol == symbol and datetime.fromisoformat(s.timestamp) > cutoff_time
        ]
        
        if len(recent_snapshots) < 2:
            return ConsistencyAnalysis(
                symbol=symbol,
                consistency_score=1.0,
                conviction_trend="stable",
                signal_changes=0,
                snapshots_analyzed=len(recent_snapshots),
                avg_conviction=5.0,
                conviction_volatility=0.0,
                recommendation="insufficient_data",
                time_period_hours=hours_lookback,
                technical_signals=[],
                sentiment_signals=[],
                last_updated=datetime.now().isoformat()
            )
        
        # Sort by timestamp
        recent_snapshots.sort(key=lambda x: x.timestamp)
        
        # Analyze conviction trend
        conviction_scores = [s.conviction_score for s in recent_snapshots]
        conviction_trend = self._calculate_conviction_trend(conviction_scores)
        
        # Analyze signal consistency
        technical_signals = [s.technical_summary for s in recent_snapshots]
        sentiment_signals = [s.sentiment_category for s in recent_snapshots]
        
        # Count signal changes
        tech_changes = sum(1 for i in range(1, len(technical_signals)) 
                          if technical_signals[i] != technical_signals[i-1])
        sentiment_changes = sum(1 for i in range(1, len(sentiment_signals)) 
                               if sentiment_signals[i] != sentiment_signals[i-1])
        
        total_signal_changes = tech_changes + sentiment_changes
        
        # Calculate consistency score
        max_possible_changes = (len(recent_snapshots) - 1) * 2
        consistency_score = 1.0 - (total_signal_changes / max_possible_changes) if max_possible_changes > 0 else 1.0
        
        # Generate recommendation
        recommendation = self._generate_consistency_recommendation(
            consistency_score, conviction_trend, total_signal_changes, conviction_scores
        )
        
        # Calculate statistics
        avg_conviction = statistics.mean(conviction_scores)
        conviction_volatility = statistics.stdev(conviction_scores) if len(conviction_scores) > 1 else 0.0
        
        return ConsistencyAnalysis(
            symbol=symbol,
            consistency_score=consistency_score,
            conviction_trend=conviction_trend,
            signal_changes=total_signal_changes,
            snapshots_analyzed=len(recent_snapshots),
            avg_conviction=avg_conviction,
            conviction_volatility=conviction_volatility,
            recommendation=recommendation,
            time_period_hours=hours_lookback,
            technical_signals=technical_signals,
            sentiment_signals=sentiment_signals,
            last_updated=datetime.now().isoformat()
        )
    
    def _calculate_conviction_trend(self, conviction_scores: List[float]) -> str:
        """Calculate conviction trend from scores"""
        if len(conviction_scores) < 3:
            return "stable"
        
        # Split into first and second half
        mid_point = len(conviction_scores) // 2
        first_half = conviction_scores[:mid_point]
        second_half = conviction_scores[mid_point:]
        
        first_avg = statistics.mean(first_half)
        second_avg = statistics.mean(second_half)
        
        difference = second_avg - first_avg
        
        if difference > 1.0:
            return "strengthening"
        elif difference < -1.0:
            return "weakening"
        else:
            return "stable"
    
    def _generate_consistency_recommendation(self, consistency_score: float, conviction_trend: str,
                                           signal_changes: int, conviction_scores: List[float]) -> str:
        """Generate recommendation based on consistency analysis"""
        avg_conviction = statistics.mean(conviction_scores)
        
        # High consistency + strong conviction = maintain
        if consistency_score >= 0.8 and avg_conviction >= 7.0:
            return "maintain_position"
        
        # Good consistency + decent conviction = monitor
        elif consistency_score >= 0.6 and avg_conviction >= 5.0:
            return "monitor_closely"
        
        # Weakening trend with low consistency = consider exit
        elif conviction_trend == "weakening" and consistency_score < 0.5:
            return "consider_exit"
        
        # High volatility in signals = review needed
        elif signal_changes >= 4:
            return "detailed_review"
        
        # Default to monitoring
        else:
            return "monitor_closely"
    
    def analyze_conviction_consistency(self, symbol: str, hours_lookback: int = 6) -> Dict[str, Any]:
        """
        Public interface for conviction consistency analysis
        """
        # Check if we have recent cached analysis
        if symbol in self.consistency_cache:
            cached_analysis = self.consistency_cache[symbol]
            last_updated = datetime.fromisoformat(cached_analysis.last_updated)
            if (datetime.now() - last_updated).total_seconds() < 300:  # 5 minutes cache
                return cached_analysis.to_dict()
        
        # Perform fresh analysis
        analysis = self._analyze_conviction_consistency(symbol, hours_lookback)
        self.consistency_cache[symbol] = analysis
        
        return analysis.to_dict()
    
    def calculate_dynamic_cooldown(self, symbol: str, conviction_score: float, 
                                 consistency_data: Dict[str, Any]) -> int:
        """
        Calculate dynamic cooldown based on conviction and market consistency
        """
        base_cooldown = self.base_cooldown_minutes
        
        # Conviction multiplier: Higher conviction = longer cooldown (stick with decision)
        conviction_multiplier = 0.5 + (conviction_score / 10.0)  # 0.6x to 1.5x
        
        # Consistency multiplier: More consistent signals = longer cooldown
        consistency_score = consistency_data.get("consistency_score", 0.5)
        consistency_multiplier = 0.7 + (consistency_score * 0.6)  # 0.7x to 1.3x
        
        # Trend multiplier: Strengthening conviction = longer cooldown
        trend = consistency_data.get("conviction_trend", "stable")
        trend_multiplier = {
            "strengthening": 1.3,
            "stable": 1.0,
            "weakening": 0.7
        }.get(trend, 1.0)
        
        # Signal volatility penalty: Many signal changes = shorter cooldown
        signal_changes = consistency_data.get("signal_changes", 0)
        volatility_multiplier = max(0.6, 1.0 - (signal_changes * 0.1))
        
        # Calculate final cooldown
        dynamic_cooldown = int(
            base_cooldown * conviction_multiplier * consistency_multiplier * 
            trend_multiplier * volatility_multiplier
        )
        
        # Apply bounds
        dynamic_cooldown = max(self.min_cooldown_minutes, min(dynamic_cooldown, self.max_cooldown_minutes))
        
        logger.info(f"⏰ Dynamic cooldown for {symbol}: {dynamic_cooldown}min "
                   f"(conviction: {conviction_score:.1f}, consistency: {consistency_score:.2f}, "
                   f"trend: {trend}, changes: {signal_changes})")
        
        return dynamic_cooldown
    
    def should_override_cooldown(self, symbol: str, new_conviction: float, 
                               new_action: str, reason: str = "") -> bool:
        """
        Determine if cooldown should be overridden for high-conviction situations
        """
        if not self.is_in_cooldown(symbol):
            return True
        
        # Get the last trade for this symbol
        symbol_trades = self.get_trades_by_symbol(symbol)
        if not symbol_trades:
            return True
        
        last_trade = max(symbol_trades, key=lambda x: x.timestamp)
        
        # Override conditions with stricter thresholds:
        
        # 1. Very high conviction (9+) with opposite action
        if (new_conviction >= 9.0 and 
            ((last_trade.action == "buy" and new_action == "sell") or 
             (last_trade.action == "sell" and new_action == "buy"))):
            logger.warning(f"🚨 COOLDOWN OVERRIDE: Very high conviction reversal for {symbol} "
                          f"({new_conviction:.1f}/10)")
            return True
        
        # 2. Emergency exit signal (conviction 9+ for sell when holding)
        if new_action == "sell" and new_conviction >= 9.0:
            logger.warning(f"🚨 COOLDOWN OVERRIDE: Emergency exit signal for {symbol} "
                          f"({new_conviction:.1f}/10)")
            return True
        
        # 3. News-driven emergency (very high conviction + news keywords)
        emergency_keywords = ["crash", "pump", "breaking", "emergency", "liquidation", "halt", "suspend"]
        if (new_conviction >= 8.5 and 
            any(keyword in reason.lower() for keyword in emergency_keywords)):
            logger.warning(f"🚨 COOLDOWN OVERRIDE: Emergency news for {symbol}: {reason}")
            return True
        
        # 4. Significant time has passed (more than 50% of original cooldown)
        cooldown_data = self.cooldowns.get(symbol)
        if cooldown_data and not isinstance(cooldown_data, str):
            original_duration = cooldown_data.get("cooldown_duration_minutes", self.base_cooldown_minutes)
            remaining = self.get_cooldown_remaining(symbol)
            if remaining:
                elapsed_percentage = 1.0 - (remaining.total_seconds() / (original_duration * 60))
                if elapsed_percentage > 0.5 and new_conviction >= 8.0:
                    logger.warning(f"🚨 COOLDOWN OVERRIDE: Partial cooldown + high conviction for {symbol}")
                    return True
        
        return False
    
    def record_trade(self, symbol: str, action: str, confidence: float, rationale: str, 
                    amount_percentage: float, executed: bool = False, 
                    trade_result: Dict[str, Any] = None,
                    market_data: Dict[str, Any] = None,
                    sentiment_data: Dict[str, Any] = None,
                    fear_greed_data: Dict[str, Any] = None,
                    conviction_score: float = None,
                    news_driven: bool = False) -> bool:
        """
        Enhanced trade recording with conviction tracking and dynamic cooldowns
        """
        # Calculate conviction score if not provided
        if conviction_score is None:
            conviction_score = confidence * 10  # Convert 0-1 to 1-10
        
        # Record conviction snapshot first
        if market_data and sentiment_data:
            self.record_conviction_snapshot(
                symbol, conviction_score, confidence, action,
                market_data, sentiment_data, fear_greed_data, 
                analysis_type="trade_decision"
            )
        
        # Check for cooldown override if needed
        emergency_override = False
        if action.lower() in ['buy', 'sell'] and self.is_in_cooldown(symbol):
            should_override = self.should_override_cooldown(symbol, conviction_score, action, rationale)
            
            if not should_override:
                remaining = self.get_cooldown_remaining(symbol)
                remaining_minutes = int(remaining.total_seconds() / 60) if remaining else 0
                logger.warning(f"🚫 TRADE BLOCKED: {symbol} cooldown not overridden ({remaining_minutes}m remaining)")
                logger.warning(f"🚫 Blocked: {action.upper()} {symbol} ({conviction_score:.1f}/10)")
                return False
            else:
                emergency_override = True
                logger.warning(f"⚡ COOLDOWN OVERRIDDEN for {symbol}")
        
        timestamp = datetime.now().isoformat()
        
        # Extract trade details from result
        usd_amount = trade_result.get("usd_amount") if trade_result else None
        crypto_amount = trade_result.get("crypto_amount") if trade_result else None
        simulated = trade_result.get("result", {}).get("simulated", False) if trade_result else False
        order_id = None
        entry_price = None
        
        if trade_result and trade_result.get("result"):
            success_response = trade_result["result"].get("success_response", {})
            order_id = success_response.get("order_id")
        
        # Extract current price if available
        if market_data:
            symbol_market_data = market_data.get(f"{symbol}/USDT", market_data.get(symbol, {}))
            entry_price = symbol_market_data.get("price")
        
        # Create enhanced trade record
        trade_record = TradeRecord(
            symbol=symbol,
            action=action.lower(),
            timestamp=timestamp,
            confidence=confidence,
            rationale=rationale,
            amount_percentage=amount_percentage,
            conviction_score=conviction_score,
            usd_amount=usd_amount,
            crypto_amount=crypto_amount,
            executed=executed,
            simulated=simulated,
            order_id=order_id,
            entry_price=entry_price,
            market_conditions=market_data.get(f"{symbol}/USDT", {}) if market_data else None,
            technical_summary=market_data.get(f"{symbol}/USDT", {}).get("technical_indicators", {}).get("technical_summary", "neutral") if market_data else "neutral",
            sentiment_category=sentiment_data.get(symbol, {}).get("sentiment_category", "neutral") if sentiment_data else "neutral",
            fear_greed_value=fear_greed_data.get("value") if fear_greed_data else None,
            news_driven=news_driven,
            emergency_override=emergency_override
        )
        
        with self._lock:
            # Add to trade history
            self.trade_history.append(trade_record)
            
            # Update positions for buy/sell actions
            if action.lower() == 'buy' and executed:
                self._open_position(trade_record)
            elif action.lower() == 'sell' and executed:
                self._close_position(trade_record)
            
            # Set dynamic cooldown for executed buy/sell actions
            if action.lower() in ['buy', 'sell'] and executed:
                consistency_data = self.analyze_conviction_consistency(symbol, hours_lookback=6)
                dynamic_cooldown_minutes = self.calculate_dynamic_cooldown(symbol, conviction_score, consistency_data)
                
                # Store enhanced cooldown data
                self.cooldowns[symbol] = {
                    "last_trade_timestamp": timestamp,
                    "cooldown_duration_minutes": dynamic_cooldown_minutes,
                    "conviction_score": conviction_score,
                    "consistency_score": consistency_data.get("consistency_score", 0.5),
                    "original_action": action.lower(),
                    "emergency_override": emergency_override
                }
                
                logger.info(f"⏰ {symbol} dynamic cooldown: {dynamic_cooldown_minutes} minutes")
            
            # Save all data
            self._save_all_data()
        
        # Log the trade
        action_emoji = {"buy": "🟢", "sell": "🔴", "hold": "🟡"}.get(action.lower(), "⚪")
        execution_status = "EXECUTED" if executed else "SIMULATED" if simulated else "PLANNED"
        override_flag = " [OVERRIDE]" if emergency_override else ""
        
        logger.info(f"📝 TRADE RECORDED: {action_emoji} {action.upper()} {symbol} "
                   f"({conviction_score:.1f}/10) - {execution_status}{override_flag}")
        
        return True
    
    def _open_position(self, trade_record: TradeRecord):
        """Open a new position from a buy trade"""
        position = Position(
            symbol=trade_record.symbol,
            entry_action=trade_record.action,
            entry_timestamp=trade_record.timestamp,
            entry_confidence=trade_record.confidence,
            entry_rationale=trade_record.rationale,
            amount_percentage=trade_record.amount_percentage,
            entry_conviction_score=trade_record.conviction_score,
            usd_amount=trade_record.usd_amount,
            crypto_amount=trade_record.crypto_amount,
            entry_price=trade_record.entry_price,
            is_open=True,
            max_conviction_seen=trade_record.conviction_score,
            min_conviction_seen=trade_record.conviction_score,
            conviction_snapshots_count=1
        )
        
        self.positions[trade_record.symbol] = position
        logger.info(f"📈 POSITION OPENED: {trade_record.symbol} "
                   f"(${trade_record.usd_amount:.2f}, {trade_record.conviction_score:.1f}/10)" 
                   if trade_record.usd_amount else f"({trade_record.conviction_score:.1f}/10)")
    
    def _close_position(self, trade_record: TradeRecord):
        """Close an existing position from a sell trade"""
        if trade_record.symbol in self.positions:
            position = self.positions[trade_record.symbol]
            position.is_open = False
            position.exit_timestamp = trade_record.timestamp
            position.exit_action = trade_record.action
            position.exit_rationale = trade_record.rationale
            position.exit_price = trade_record.entry_price  # Current price when selling
            
            # Calculate simple P&L if we have prices
            if position.entry_price and trade_record.entry_price:
                price_change_pct = ((trade_record.entry_price - position.entry_price) / position.entry_price) * 100
                position.position_pnl = price_change_pct
            
            logger.info(f"📉 POSITION CLOSED: {trade_record.symbol} "
                       f"({trade_record.crypto_amount:.6f} sold, P&L: {position.position_pnl:+.2f}%)" 
                       if trade_record.crypto_amount and position.position_pnl else f"(closed)")
        else:
            logger.warning(f"⚠️ SELL WITHOUT OPEN POSITION: {trade_record.symbol}")
    
    def get_open_positions(self) -> Dict[str, Position]:
        """Get all currently open positions"""
        return {symbol: pos for symbol, pos in self.positions.items() if pos.is_open}
    
    def get_closed_positions(self) -> Dict[str, Position]:
        """Get all closed positions"""
        return {symbol: pos for symbol, pos in self.positions.items() if not pos.is_open}
    
    def get_recent_trades(self, hours: int = 24) -> List[TradeRecord]:
        """Get trades from the last N hours"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        recent_trades = []
        for trade in self.trade_history:
            try:
                trade_time = datetime.fromisoformat(trade.timestamp)
                if trade_time > cutoff_time:
                    recent_trades.append(trade)
            except Exception as e:
                logger.debug(f"Error parsing trade timestamp: {str(e)}")
        
        return sorted(recent_trades, key=lambda x: x.timestamp, reverse=True)
    
    def get_trades_by_symbol(self, symbol: str) -> List[TradeRecord]:
        """Get all trades for a specific symbol"""
        return [trade for trade in self.trade_history if trade.symbol == symbol]
    
    def get_cooldown_status(self) -> Dict[str, Dict[str, Any]]:
        """Get enhanced cooldown status for all symbols"""
        status = {}
        
        for symbol, cooldown_data in self.cooldowns.items():
            if self.is_in_cooldown(symbol):
                remaining = self.get_cooldown_remaining(symbol)
                remaining_minutes = int(remaining.total_seconds() / 60) if remaining else 0
                
                if isinstance(cooldown_data, str):
                    # Legacy format
                    status[symbol] = {
                        "in_cooldown": True,
                        "remaining_minutes": remaining_minutes,
                        "last_trade": cooldown_data,
                        "cooldown_type": "legacy"
                    }
                else:
                    # Enhanced format
                    status[symbol] = {
                        "in_cooldown": True,
                        "remaining_minutes": remaining_minutes,
                        "last_trade": cooldown_data["last_trade_timestamp"],
                        "total_duration_minutes": cooldown_data.get("cooldown_duration_minutes", self.base_cooldown_minutes),
                        "conviction_score": cooldown_data.get("conviction_score", 5.0),
                        "consistency_score": cooldown_data.get("consistency_score", 0.5),
                        "original_action": cooldown_data.get("original_action", "unknown"),
                        "emergency_override": cooldown_data.get("emergency_override", False),
                        "cooldown_type": "dynamic"
                    }
            else:
                if isinstance(cooldown_data, str):
                    last_trade = cooldown_data
                else:
                    last_trade = cooldown_data.get("last_trade_timestamp", "unknown")
                
                status[symbol] = {
                    "in_cooldown": False,
                    "remaining_minutes": 0,
                    "last_trade": last_trade
                }
        
        return status
    
    def get_position_conviction_summary(self, symbol: str) -> Dict[str, Any]:
        """
        Get comprehensive conviction summary for a position
        """
        position = self.positions.get(symbol)
        if not position:
            return {"error": "No position found"}
        
        # Get latest consistency analysis
        consistency_data = self.analyze_conviction_consistency(symbol, hours_lookback=6)
        
        # Get recent snapshots
        recent_snapshots = [
            s for s in self.conviction_snapshots
            if s.symbol == symbol and 
            datetime.fromisoformat(s.timestamp) > datetime.now() - timedelta(hours=12)
        ]
        
        # Calculate position duration and performance
        entry_time = datetime.fromisoformat(position.entry_timestamp)
        hours_held = (datetime.now() - entry_time).total_seconds() / 3600
        
        return {
            "symbol": symbol,
            "entry_conviction": position.entry_conviction_score,
            "entry_confidence": position.entry_confidence,
            "hours_held": hours_held,
            "recent_conviction_avg": consistency_data.get("avg_conviction", 0),
            "conviction_trend": consistency_data.get("conviction_trend", "unknown"),
            "consistency_score": consistency_data.get("consistency_score", 0),
            "recommendation": consistency_data.get("recommendation", "unknown"),
            "snapshots_count": len(recent_snapshots),
            "signal_stability": "stable" if consistency_data.get("signal_changes", 0) <= 2 else "volatile",
            "max_conviction_seen": position.max_conviction_seen,
            "min_conviction_seen": position.min_conviction_seen,
            "total_snapshots": position.conviction_snapshots_count,
            "position_pnl": position.position_pnl,
            "entry_price": position.entry_price,
            "entry_rationale": position.entry_rationale
        }
    
    def get_trading_stats(self) -> Dict[str, Any]:
        """Get comprehensive enhanced trading statistics"""
        total_trades = len(self.trade_history)
        executed_trades = len([t for t in self.trade_history if t.executed])
        buy_trades = len([t for t in self.trade_history if t.action == 'buy' and t.executed])
        sell_trades = len([t for t in self.trade_history if t.action == 'sell' and t.executed])
        
        open_positions = self.get_open_positions()
        closed_positions = self.get_closed_positions()
        
        recent_trades = self.get_recent_trades(24)
        
        # Calculate average conviction and confidence
        executed_trade_convictions = [t.conviction_score for t in self.trade_history if t.executed and hasattr(t, 'conviction_score')]
        executed_trade_confidences = [t.confidence for t in self.trade_history if t.executed and t.confidence]
        
        avg_conviction = statistics.mean(executed_trade_convictions) if executed_trade_convictions else 0
        avg_confidence = statistics.mean(executed_trade_confidences) if executed_trade_confidences else 0
        
        # Emergency override statistics
        emergency_overrides = len([t for t in self.trade_history if hasattr(t, 'emergency_override') and t.emergency_override])
        news_driven_trades = len([t for t in self.trade_history if hasattr(t, 'news_driven') and t.news_driven])
        
        # Position performance
        closed_positions_with_pnl = [p for p in closed_positions.values() if p.position_pnl is not None]
        avg_position_pnl = statistics.mean([p.position_pnl for p in closed_positions_with_pnl]) if closed_positions_with_pnl else None
        
        # Cooldown statistics
        active_cooldowns = len([s for s in self.cooldowns.keys() if self.is_in_cooldown(s)])
        dynamic_cooldowns = len([c for c in self.cooldowns.values() if isinstance(c, dict)])
        
        return {
            "total_trades": total_trades,
            "executed_trades": executed_trades,
            "buy_trades": buy_trades,
            "sell_trades": sell_trades,
            "open_positions": len(open_positions),
            "closed_positions": len(closed_positions),
            "trades_last_24h": len(recent_trades),
            "avg_conviction_score": round(avg_conviction, 2),
            "avg_confidence": round(avg_confidence, 2),
            "symbols_traded": len(set(t.symbol for t in self.trade_history if t.executed)),
            "cooldown_period_minutes": self.base_cooldown_minutes,
            "min_cooldown_minutes": self.min_cooldown_minutes,
            "max_cooldown_minutes": self.max_cooldown_minutes,
            "active_cooldowns": active_cooldowns,
            "dynamic_cooldowns": dynamic_cooldowns,
            "emergency_overrides": emergency_overrides,
            "news_driven_trades": news_driven_trades,
            "avg_position_pnl": round(avg_position_pnl, 2) if avg_position_pnl is not None else None,
            "conviction_snapshots": len(self.conviction_snapshots),
            "consistency_cache_size": len(self.consistency_cache)
        }
    
    def force_clear_cooldown(self, symbol: str) -> bool:
        """Forcefully clear cooldown for a symbol (admin function)"""
        if symbol in self.cooldowns:
            with self._lock:
                del self.cooldowns[symbol]
                self._save_all_data()
            
            logger.warning(f"🔓 FORCE CLEARED cooldown for {symbol}")
            return True
        
        return False
    
    def update_cooldown_period(self, minutes: int):
        """Update the base cooldown period for future trades"""
        self.base_cooldown_minutes = minutes
        self.cooldown_period = timedelta(minutes=minutes)
        self.min_cooldown_minutes = max(30, minutes // 4)
        self.max_cooldown_minutes = minutes * 3
        logger.info(f"⏰ Base cooldown updated to {minutes} minutes (range: {self.min_cooldown_minutes}-{self.max_cooldown_minutes})")
    
    def clear_old_data(self, days: int = 30):
        """Clear old data while preserving important records"""
        cutoff_time = datetime.now() - timedelta(days=days)
        
        with self._lock:
            # Clear old trade history
            original_trade_count = len(self.trade_history)
            self.trade_history = [
                trade for trade in self.trade_history
                if datetime.fromisoformat(trade.timestamp) > cutoff_time
            ]
            
            # Clear old conviction snapshots (already done in save, but explicit here)
            original_snapshot_count = len(self.conviction_snapshots)
            snapshot_cutoff = datetime.now() - timedelta(hours=48)  # Keep 48 hours
            self.conviction_snapshots = [
                snapshot for snapshot in self.conviction_snapshots
                if datetime.fromisoformat(snapshot.timestamp) > snapshot_cutoff
            ]
            
            # Clear old consistency cache
            old_consistency_cache = len(self.consistency_cache)
            current_time = datetime.now()
            for symbol in list(self.consistency_cache.keys()):
                last_updated = datetime.fromisoformat(self.consistency_cache[symbol].last_updated)
                if (current_time - last_updated).total_seconds() > 86400:  # 24 hours
                    del self.consistency_cache[symbol]
            
            cleared_trades = original_trade_count - len(self.trade_history)
            cleared_snapshots = original_snapshot_count - len(self.conviction_snapshots)
            cleared_cache = old_consistency_cache - len(self.consistency_cache)
            
            if cleared_trades > 0 or cleared_snapshots > 0 or cleared_cache > 0:
                logger.info(f"🗑️ Cleared old data: {cleared_trades} trades, "
                           f"{cleared_snapshots} snapshots, {cleared_cache} cache entries")
                self._save_all_data()
    
    def export_trade_history(self, filepath: str = None) -> str:
        """Export comprehensive trading data including conviction analysis"""
        if not filepath:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = os.path.join(self.data_dir, f"enhanced_trade_export_{timestamp}.json")
        
        try:
            # Get comprehensive statistics
            stats = self.get_trading_stats()
            cooldown_status = self.get_cooldown_status()
            
            # Get conviction summaries for all positions
            position_summaries = {}
            for symbol in self.positions.keys():
                position_summaries[symbol] = self.get_position_conviction_summary(symbol)
            
            export_data = {
                "export_timestamp": datetime.now().isoformat(),
                "export_type": "enhanced_position_manager",
                "version": "2.0",
                "stats": stats,
                "cooldown_settings": {
                    "base_minutes": self.base_cooldown_minutes,
                    "min_minutes": self.min_cooldown_minutes,
                    "max_minutes": self.max_cooldown_minutes
                },
                "trade_history": [trade.to_dict() for trade in self.trade_history],
                "positions": {symbol: pos.to_dict() for symbol, pos in self.positions.items()},
                "conviction_snapshots": [snapshot.to_dict() for snapshot in self.conviction_snapshots],
                "consistency_analysis": {symbol: analysis.to_dict() for symbol, analysis in self.consistency_cache.items()},
                "cooldown_status": cooldown_status,
                "position_summaries": position_summaries
            }
            
            with open(filepath, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
            
            logger.info(f"📊 Enhanced trade history exported to {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"Error exporting trade history: {str(e)}")
            raise

# Global position manager instance
_position_manager = None

def get_position_manager() -> PositionManager:
    """Get global PositionManager instance (singleton pattern)"""
    global _position_manager
    if _position_manager is None:
        cooldown_minutes = safe_get_env_int("TRADE_COOLDOWN_MINUTES", 120)  # FIXED: Use safe parsing
        _position_manager = PositionManager(cooldown_minutes=cooldown_minutes)
    return _position_manager

# Enhanced convenience functions
def is_in_cooldown(symbol: str) -> bool:
    """Check if symbol is in cooldown"""
    return get_position_manager().is_in_cooldown(symbol)

def record_trade(symbol: str, action: str, confidence: float, rationale: str, 
                amount_percentage: float, executed: bool = False, 
                trade_result: Dict[str, Any] = None,
                market_data: Dict[str, Any] = None,
                sentiment_data: Dict[str, Any] = None,
                fear_greed_data: Dict[str, Any] = None,
                conviction_score: float = None,
                news_driven: bool = False) -> bool:
    """Record a trade with enhanced tracking"""
    return get_position_manager().record_trade(
        symbol, action, confidence, rationale, amount_percentage, executed, 
        trade_result, market_data, sentiment_data, fear_greed_data, 
        conviction_score, news_driven
    )

def get_cooldown_status() -> Dict[str, Dict[str, Any]]:
    """Get enhanced cooldown status for all symbols"""
    return get_position_manager().get_cooldown_status()

def get_trading_stats() -> Dict[str, Any]:
    """Get enhanced trading statistics"""
    return get_position_manager().get_trading_stats()

def analyze_conviction_consistency(symbol: str, hours_lookback: int = 6) -> Dict[str, Any]:
    """Analyze conviction consistency for a symbol"""
    return get_position_manager().analyze_conviction_consistency(symbol, hours_lookback)

def get_position_conviction_summary(symbol: str) -> Dict[str, Any]:
    """Get conviction summary for a position"""
    return get_position_manager().get_position_conviction_summary(symbol)

# Testing and demonstration
if __name__ == "__main__":
    print("=== Enhanced Position Manager Demo ===")
    
    # Initialize with shorter cooldown for testing
    pm = PositionManager(data_dir="test_data", cooldown_minutes=2)
    
    # Simulate market data
    test_market_data = {
        "BTC/USDT": {
            "price": 45000,
            "change_24h_percent": 2.5,
            "technical_indicators": {
                "technical_summary": "bullish",
                "rsi": 65,
                "macd_interpretation": "bullish"
            }
        }
    }
    
    test_sentiment_data = {
        "BTC": {
            "sentiment_category": "bullish",
            "overall_sentiment": 0.7
        }
    }
    
    test_fear_greed = {"value": 75}
    
    # Record some test trades with conviction tracking
    print("Recording BTC buy with high conviction...")
    pm.record_trade(
        symbol="BTC", 
        action="buy", 
        confidence=0.85, 
        rationale="Strong bullish signals across multiple indicators", 
        amount_percentage=25, 
        executed=True,
        market_data=test_market_data,
        sentiment_data=test_sentiment_data,
        fear_greed_data=test_fear_greed,
        conviction_score=8.5
    )
    
    print(f"BTC in cooldown: {pm.is_in_cooldown('BTC')}")
    
    # Try to trade again (should be blocked)
    print("Attempting second BTC trade...")
    blocked = pm.record_trade(
        symbol="BTC", 
        action="sell", 
        confidence=0.75, 
        rationale="Taking some profits", 
        amount_percentage=50, 
        executed=True,
        conviction_score=7.5
    )
    print(f"Second BTC trade blocked: {not blocked}")
    
    # Test conviction consistency analysis
    print("Analyzing conviction consistency...")
    consistency = pm.analyze_conviction_consistency("BTC")
    print(f"Consistency analysis: {consistency}")
    
    # Get enhanced stats
    stats = pm.get_trading_stats()
    print(f"Enhanced trading stats: {json.dumps(stats, indent=2)}")
    
    # Get position summary
    open_positions = pm.get_open_positions()
    if open_positions:
        for symbol in open_positions.keys():
            summary = pm.get_position_conviction_summary(symbol)
            print(f"Position summary for {symbol}: {json.dumps(summary, indent=2)}")
    
    print("Enhanced Position Manager demo completed!")