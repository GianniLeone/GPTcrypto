import json
import logging
import os
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass

# Import OpenAI for GPT API calls
from openai import OpenAI

# Import position manager for strategic trading
from position_manager import get_position_manager

# Configure logging
logger = logging.getLogger("CryptoBot.StrategicGPT")

# API rate limiting
_api_call_count = 0
_last_reset_time = datetime.now()
MAX_API_CALLS_PER_HOUR = int(os.getenv("MAX_GPT_QUERIES_PER_HOUR", "10"))
GPT_MODEL = os.getenv("GPT_MODEL", "gpt-4")

@dataclass
class AnalysisResult:
    """Structured result from strategic analysis"""
    action: str  # buy, sell, hold, reduce
    asset: Optional[str] = None
    amount_percentage: float = 0
    confidence: float = 0.0
    conviction_score: float = 5.0  # 1-10 scale
    rationale: str = ""
    analysis_type: str = "strategic"
    news_driven: bool = False
    urgency: str = "medium"
    consistency_score: Optional[float] = None
    positions_reviewed: int = 0
    rejected_signal: Optional[Dict[str, Any]] = None
    market_condition_change: str = "stable"  # NEW: track if conditions changed
    position_review_trigger: str = "none"    # NEW: what triggered the review
    timestamp: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for compatibility"""
        result = {
            "action": self.action,
            "asset": self.asset,
            "amount_percentage": self.amount_percentage,
            "confidence": self.confidence,
            "rationale": self.rationale,
            "analysis_type": self.analysis_type,
            "conviction_score": self.conviction_score,
            "timestamp": self.timestamp or datetime.now().isoformat()
        }
        
        # Add optional fields if present
        if self.news_driven:
            result["news_driven"] = True
            result["urgency"] = self.urgency
        if self.consistency_score is not None:
            result["consistency_score"] = self.consistency_score
        if self.positions_reviewed > 0:
            result["positions_reviewed"] = self.positions_reviewed
        if self.rejected_signal:
            result["rejected_signal"] = self.rejected_signal
        if self.market_condition_change != "stable":
            result["market_condition_change"] = self.market_condition_change
        if self.position_review_trigger != "none":
            result["position_review_trigger"] = self.position_review_trigger
            
        return result

class APIRateManager:
    """Manages GPT API rate limiting"""
    
    @staticmethod
    def check_api_limit() -> bool:
        global _api_call_count, _last_reset_time
        
        current_time = datetime.now()
        if (current_time - _last_reset_time).total_seconds() >= 3600:
            _api_call_count = 0
            _last_reset_time = current_time
            logger.info("GPT API counter reset")
        
        return _api_call_count < MAX_API_CALLS_PER_HOUR
    
    @staticmethod
    def increment_counter():
        global _api_call_count
        _api_call_count += 1
        logger.debug(f"GPT API calls: {_api_call_count}/{MAX_API_CALLS_PER_HOUR}")

class DynamicConvictionEngine:
    """NEW: Dynamic conviction threshold calculator based on market conditions"""
    
    def __init__(self):
        self.base_threshold = 7.0  # Default conviction requirement
        self.min_threshold = 5.5   # Minimum threshold (very bullish markets)
        self.max_threshold = 8.5   # Maximum threshold (very uncertain markets)
        
    def calculate_dynamic_threshold(self, market_data: Dict[str, Any], 
                                  sentiment_data: Dict[str, Any], 
                                  fear_greed_data: Dict[str, Any]) -> float:
        """
        Calculate dynamic conviction threshold based on current market conditions
        
        Returns: Float between min_threshold and max_threshold
        """
        try:
            # Start with base threshold
            threshold = self.base_threshold
            
            # 1. FEAR & GREED ADJUSTMENT
            fg_adjustment = self._calculate_fear_greed_adjustment(fear_greed_data)
            threshold += fg_adjustment
            
            # 2. TECHNICAL CONFLUENCE ADJUSTMENT  
            tech_adjustment = self._calculate_technical_confluence_adjustment(market_data)
            threshold += tech_adjustment
            
            # 3. SENTIMENT STRENGTH ADJUSTMENT
            sentiment_adjustment = self._calculate_sentiment_adjustment(sentiment_data)
            threshold += sentiment_adjustment
            
            # 4. VOLATILITY ADJUSTMENT
            volatility_adjustment = self._calculate_volatility_adjustment(market_data)
            threshold += volatility_adjustment
            
            # 5. MARKET STRUCTURE ADJUSTMENT
            structure_adjustment = self._calculate_market_structure_adjustment(market_data)
            threshold += structure_adjustment
            
            # Cap within bounds
            final_threshold = max(self.min_threshold, min(self.max_threshold, threshold))
            
            # Log the calculation for transparency
            adjustments = {
                "base": self.base_threshold,
                "fear_greed": fg_adjustment,
                "technical": tech_adjustment, 
                "sentiment": sentiment_adjustment,
                "volatility": volatility_adjustment,
                "structure": structure_adjustment,
                "final": final_threshold
            }
            
            logger.info(f"🎯 Dynamic conviction threshold: {final_threshold:.1f}/10")
            logger.debug(f"📊 Threshold adjustments: {adjustments}")
            
            return final_threshold
            
        except Exception as e:
            logger.error(f"Error calculating dynamic threshold: {str(e)}")
            return self.base_threshold
    
    def _calculate_fear_greed_adjustment(self, fear_greed_data: Dict[str, Any]) -> float:
        """Calculate adjustment based on Fear & Greed Index"""
        if not fear_greed_data or fear_greed_data.get("value") is None:
            return 0.0
        
        fg_value = fear_greed_data.get("value", 50)
        
        if fg_value <= 20:  # Extreme fear - great buying opportunities
            return -1.5  # Much lower threshold
        elif fg_value <= 30:  # Fear - good buying opportunities  
            return -1.0
        elif fg_value <= 40:  # Some fear - decent opportunities
            return -0.5
        elif fg_value >= 80:  # Extreme greed - be very cautious
            return +1.5  # Much higher threshold
        elif fg_value >= 70:  # Greed - be cautious
            return +1.0
        elif fg_value >= 60:  # Some greed - slightly cautious
            return +0.5
        else:  # Neutral (40-60)
            return 0.0
    
    def _calculate_technical_confluence_adjustment(self, market_data: Dict[str, Any]) -> float:
        """Calculate adjustment based on technical indicator confluence"""
        if not market_data:
            return 0.0
        
        bullish_signals = 0
        bearish_signals = 0
        total_assets = 0
        
        for symbol, data in market_data.items():
            if not isinstance(data, dict):
                continue
                
            total_assets += 1
            technical = data.get("technical_indicators", {})
            
            # Count bullish signals
            if technical.get("rsi_signal") in ["oversold"]:
                bullish_signals += 1
            if technical.get("macd_interpretation") in ["bullish", "strong_bullish"]:
                bullish_signals += 1
            if technical.get("ma_trend") in ["bullish", "strong_bullish"]:
                bullish_signals += 1
            if technical.get("candlestick_signal") == "bullish":
                bullish_signals += 1
            
            # Count bearish signals
            if technical.get("rsi_signal") in ["overbought"]:
                bearish_signals += 1
            if technical.get("macd_interpretation") in ["bearish", "strong_bearish"]:
                bearish_signals += 1
            if technical.get("ma_trend") in ["bearish", "strong_bearish"]:
                bearish_signals += 1
            if technical.get("candlestick_signal") == "bearish":
                bearish_signals += 1
        
        if total_assets == 0:
            return 0.0
        
        # Calculate confluence ratio
        total_signals = bullish_signals + bearish_signals
        if total_signals == 0:
            return 0.0
        
        bullish_ratio = bullish_signals / total_signals
        bearish_ratio = bearish_signals / total_signals
        
        # Strong bullish confluence = lower threshold
        if bullish_ratio > 0.7:
            return -0.8
        elif bullish_ratio > 0.6:
            return -0.4
        # Strong bearish confluence = higher threshold  
        elif bearish_ratio > 0.7:
            return +0.8
        elif bearish_ratio > 0.6:
            return +0.4
        else:
            return 0.0
    
    def _calculate_sentiment_adjustment(self, sentiment_data: Dict[str, Any]) -> float:
        """Calculate adjustment based on sentiment strength and consensus"""
        if not sentiment_data:
            return 0.0
        
        sentiment_scores = []
        confidence_scores = []
        
        for symbol, data in sentiment_data.items():
            if not isinstance(data, dict):
                continue
                
            sentiment_score = data.get("overall_sentiment", 0)
            confidence = data.get("confidence", 0)
            
            if confidence > 0.5:  # Only consider confident sentiment
                sentiment_scores.append(sentiment_score)
                confidence_scores.append(confidence)
        
        if not sentiment_scores:
            return 0.0
        
        # Calculate weighted average sentiment
        weighted_sentiment = sum(s * c for s, c in zip(sentiment_scores, confidence_scores))
        total_confidence = sum(confidence_scores)
        
        if total_confidence == 0:
            return 0.0
        
        avg_sentiment = weighted_sentiment / total_confidence
        avg_confidence = total_confidence / len(confidence_scores)
        
        # Strong positive sentiment with high confidence = lower threshold
        if avg_sentiment > 0.3 and avg_confidence > 0.7:
            return -0.6
        elif avg_sentiment > 0.2 and avg_confidence > 0.6:
            return -0.3
        # Strong negative sentiment = higher threshold
        elif avg_sentiment < -0.3 and avg_confidence > 0.7:
            return +0.6
        elif avg_sentiment < -0.2 and avg_confidence > 0.6:
            return +0.3
        else:
            return 0.0
    
    def _calculate_volatility_adjustment(self, market_data: Dict[str, Any]) -> float:
        """Calculate adjustment based on market volatility"""
        if not market_data:
            return 0.0
        
        high_volatility_count = 0
        total_assets = 0
        
        for symbol, data in market_data.items():
            if not isinstance(data, dict):
                continue
                
            total_assets += 1
            change_24h = abs(data.get("change_24h_percent", 0))
            
            # Consider >5% daily moves as high volatility
            if change_24h > 5.0:
                high_volatility_count += 1
        
        if total_assets == 0:
            return 0.0
        
        volatility_ratio = high_volatility_count / total_assets
        
        # High volatility = require higher conviction
        if volatility_ratio > 0.6:
            return +0.8
        elif volatility_ratio > 0.4:
            return +0.4
        # Low volatility = can be slightly more aggressive
        elif volatility_ratio < 0.2:
            return -0.2
        else:
            return 0.0
    
    def _calculate_market_structure_adjustment(self, market_data: Dict[str, Any]) -> float:
        """Calculate adjustment based on overall market structure"""
        if not market_data:
            return 0.0
        
        positive_moves = 0
        total_assets = 0
        
        for symbol, data in market_data.items():
            if not isinstance(data, dict):
                continue
                
            total_assets += 1
            change_24h = data.get("change_24h_percent", 0)
            
            if change_24h > 0:
                positive_moves += 1
        
        if total_assets == 0:
            return 0.0
        
        positive_ratio = positive_moves / total_assets
        
        # Strong market-wide momentum = lower threshold
        if positive_ratio > 0.8:
            return -0.4
        elif positive_ratio > 0.7:
            return -0.2
        # Weak market-wide performance = higher threshold
        elif positive_ratio < 0.3:
            return +0.4
        elif positive_ratio < 0.4:
            return +0.2
        else:
            return 0.0
    
    def explain_threshold_reasoning(self, threshold: float, market_data: Dict[str, Any], 
                                  sentiment_data: Dict[str, Any], 
                                  fear_greed_data: Dict[str, Any]) -> str:
        """Generate human-readable explanation of threshold calculation"""
        
        fg_value = fear_greed_data.get("value", 50) if fear_greed_data else 50
        
        explanation_parts = []
        
        # Fear & Greed explanation
        if fg_value <= 25:
            explanation_parts.append("🚨 Extreme fear creates buying opportunities")
        elif fg_value >= 75:
            explanation_parts.append("⚠️ Extreme greed requires extra caution")
        elif fg_value <= 40:
            explanation_parts.append("😰 Fearful markets favor patient buyers")
        elif fg_value >= 60:
            explanation_parts.append("😊 Greedy markets need careful timing")
        
        # Technical confluence
        bullish_count = 0
        bearish_count = 0
        for symbol, data in market_data.items():
            technical = data.get("technical_indicators", {})
            if "bullish" in str(technical.get("technical_summary", "")):
                bullish_count += 1
            elif "bearish" in str(technical.get("technical_summary", "")):
                bearish_count += 1
        
        if bullish_count > bearish_count:
            explanation_parts.append(f"📈 {bullish_count} assets show bullish technicals")
        elif bearish_count > bullish_count:
            explanation_parts.append(f"📉 {bearish_count} assets show bearish technicals")
        
        # Threshold interpretation
        if threshold < 6.0:
            explanation_parts.append("🎯 LOW threshold - Strong opportunities expected")
        elif threshold > 8.0:
            explanation_parts.append("🎯 HIGH threshold - Only exceptional trades")
        else:
            explanation_parts.append("🎯 NORMAL threshold - Standard selectivity")
        
        return " | ".join(explanation_parts)

class MarketConditionAnalyzer:
    """Analyzes if market conditions have significantly changed"""
    
    def __init__(self):
        self.pm = get_position_manager()
    
    def analyze_condition_change(self, symbol: str, current_market_data: Dict[str, Any], 
                                current_sentiment: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze if market conditions have changed significantly since position entry
        """
        position = self.pm.positions.get(symbol)
        if not position or not position.is_open:
            return {"change_detected": False, "change_type": "no_position"}
        
        # Get current technical conditions
        current_tech = current_market_data.get("technical_indicators", {})
        current_tech_summary = current_tech.get("technical_summary", "neutral")
        current_rsi_signal = current_tech.get("rsi_signal", "neutral")
        current_macd = current_tech.get("macd_interpretation", "neutral")
        current_ma_trend = current_tech.get("ma_trend", "neutral")
        
        # Get current sentiment
        current_sentiment_category = current_sentiment.get("sentiment_category", "neutral")
        
        # Compare with entry conditions if available
        entry_conditions = position.market_conditions if hasattr(position, 'market_conditions') else {}
        entry_tech_summary = getattr(position, 'technical_summary', 'unknown')
        entry_sentiment = getattr(position, 'sentiment_category', 'unknown')
        
        changes = []
        severity_score = 0
        
        # Technical indicator changes
        if entry_tech_summary != 'unknown' and current_tech_summary != entry_tech_summary:
            if self._is_bearish_flip(entry_tech_summary, current_tech_summary):
                changes.append(f"Technical flip: {entry_tech_summary} → {current_tech_summary}")
                severity_score += 3
            elif self._is_bullish_flip(entry_tech_summary, current_tech_summary):
                changes.append(f"Technical improvement: {entry_tech_summary} → {current_tech_summary}")
                severity_score -= 1
        
        # RSI condition changes
        current_rsi = current_tech.get("rsi")
        if current_rsi:
            if current_rsi > 75:  # Extreme overbought
                changes.append(f"RSI extreme overbought: {current_rsi:.1f}")
                severity_score += 2
            elif current_rsi < 25:  # Extreme oversold
                changes.append(f"RSI extreme oversold: {current_rsi:.1f}")
                severity_score -= 1
        
        # MACD changes
        if "bearish" in current_macd and position.entry_action == "buy":
            changes.append(f"MACD turned bearish: {current_macd}")
            severity_score += 2
        
        # Moving average trend changes
        if "bearish" in current_ma_trend and position.entry_action == "buy":
            changes.append(f"MA trend bearish: {current_ma_trend}")
            severity_score += 2
        
        # Sentiment changes
        if entry_sentiment != 'unknown' and current_sentiment_category != entry_sentiment:
            if self._is_sentiment_deterioration(entry_sentiment, current_sentiment_category):
                changes.append(f"Sentiment deteriorated: {entry_sentiment} → {current_sentiment_category}")
                severity_score += 1
        
        # Position duration factor
        entry_time = datetime.fromisoformat(position.entry_timestamp)
        hours_held = (datetime.now() - entry_time).total_seconds() / 3600
        
        if hours_held > 48:  # Position held for over 2 days
            severity_score += 1
            changes.append(f"Long duration position: {hours_held:.1f}h")
        
        # Determine change type
        change_type = "stable"
        if severity_score >= 4:
            change_type = "significant_deterioration"
        elif severity_score >= 2:
            change_type = "moderate_deterioration"
        elif severity_score <= -2:
            change_type = "improvement"
        
        return {
            "change_detected": len(changes) > 0,
            "change_type": change_type,
            "severity_score": severity_score,
            "changes": changes,
            "hours_held": hours_held,
            "current_conditions": {
                "technical_summary": current_tech_summary,
                "rsi_signal": current_rsi_signal,
                "macd": current_macd,
                "ma_trend": current_ma_trend,
                "sentiment": current_sentiment_category
            },
            "entry_conditions": {
                "technical_summary": entry_tech_summary,
                "sentiment": entry_sentiment
            }
        }
    
    def _is_bearish_flip(self, entry_signal: str, current_signal: str) -> bool:
        """Check if signal flipped from bullish to bearish"""
        bullish_signals = ["bullish", "strong_bullish"]
        bearish_signals = ["bearish", "strong_bearish"]
        
        return (entry_signal in bullish_signals and current_signal in bearish_signals)
    
    def _is_bullish_flip(self, entry_signal: str, current_signal: str) -> bool:
        """Check if signal flipped from bearish to bullish"""
        bullish_signals = ["bullish", "strong_bullish"]
        bearish_signals = ["bearish", "strong_bearish"]
        
        return (entry_signal in bearish_signals and current_signal in bullish_signals)
    
    def _is_sentiment_deterioration(self, entry_sentiment: str, current_sentiment: str) -> bool:
        """Check if sentiment deteriorated"""
        sentiment_scale = {
            "very_bullish": 5,
            "bullish": 4,
            "neutral": 3,
            "bearish": 2,
            "very_bearish": 1
        }
        
        entry_score = sentiment_scale.get(entry_sentiment, 3)
        current_score = sentiment_scale.get(current_sentiment, 3)
        
        return current_score < entry_score - 1  # Significant drop

class PositionAnalyzer:
    """Enhanced position analyzer with market condition awareness"""
    
    def __init__(self):
        self.pm = get_position_manager()
        self.market_analyzer = MarketConditionAnalyzer()
    
    def should_review_position(self, symbol: str, current_market_data: Dict[str, Any], 
                              current_sentiment: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Determine if position needs immediate review based on market conditions
        """
        position = self.pm.positions.get(symbol)
        if not position or not position.is_open:
            return False, "no_position"
        
        # Check market condition changes
        condition_change = self.market_analyzer.analyze_condition_change(
            symbol, current_market_data, current_sentiment
        )
        
        # Trigger review if significant deterioration
        if condition_change["change_type"] in ["significant_deterioration", "moderate_deterioration"]:
            return True, condition_change["change_type"]
        
        # Check position duration
        entry_time = datetime.fromisoformat(position.entry_timestamp)
        hours_held = (datetime.now() - entry_time).total_seconds() / 3600
        
        # Mandatory review after 24 hours
        if hours_held > 24:
            return True, "duration_review"
        
        # Check technical signals
        technical = current_market_data.get("technical_indicators", {})
        tech_summary = technical.get("technical_summary", "neutral")
        rsi_signal = technical.get("rsi_signal", "neutral")
        
        # Strong bearish signals trigger review
        if "bearish" in tech_summary and position.entry_action == "buy":
            return True, "bearish_technicals"
        
        # Extreme RSI conditions
        rsi = technical.get("rsi")
        if rsi and ((rsi > 75 and position.entry_action == "buy") or (rsi < 25 and position.entry_action == "sell")):
            return True, "extreme_rsi"
        
        return False, "no_trigger"
    
    def format_recent_decisions(self, symbols: List[str], hours: int = 6) -> str:
        """Format recent trading decisions for context"""
        try:
            recent_trades = self.pm.get_recent_trades(hours=hours)
            if not recent_trades:
                return f"No trading activity in the last {hours} hours."
            
            # Group by symbol
            base_symbols = [s.split('/')[0] for s in symbols]
            formatted_lines = []
            
            for base_symbol in base_symbols:
                symbol_trades = [t for t in recent_trades if t.symbol == base_symbol]
                
                if symbol_trades:
                    # Get most recent trade
                    latest_trade = max(symbol_trades, key=lambda x: x.timestamp)
                    
                    trade_time = datetime.fromisoformat(latest_trade.timestamp)
                    hours_ago = (datetime.now() - trade_time).total_seconds() / 3600
                    
                    status = "EXECUTED" if latest_trade.executed else "PLANNED"
                    if hasattr(latest_trade, 'conviction_score'):
                        conviction_info = f" (conviction: {latest_trade.conviction_score:.1f}/10)"
                    else:
                        conviction_info = f" (confidence: {latest_trade.confidence:.0%})"
                    
                    formatted_lines.append(
                        f"• {base_symbol}: {latest_trade.action.upper()} {hours_ago:.1f}h ago "
                        f"[{status}]{conviction_info}"
                    )
                else:
                    formatted_lines.append(f"• {base_symbol}: No recent activity")
            
            return "\n".join(formatted_lines)
            
        except Exception as e:
            logger.error(f"Error formatting recent decisions: {str(e)}")
            return "Error retrieving recent trading decisions."
    
    def format_open_positions_with_conditions(self, symbols: List[str], 
                                            market_data: Dict[str, Any]) -> str:
        """Format open positions with current market condition analysis"""
        try:
            open_positions = self.pm.get_open_positions()
            if not open_positions:
                return "No open positions."
            
            base_symbols = [s.split('/')[0] for s in symbols]
            formatted_lines = []
            
            for base_symbol in base_symbols:
                if base_symbol in open_positions:
                    position = open_positions[base_symbol]
                    
                    entry_time = datetime.fromisoformat(position.entry_timestamp)
                    hours_held = (datetime.now() - entry_time).total_seconds() / 3600
                    
                    # Get current market data for this symbol
                    symbol_market_data = None
                    for market_symbol, data_item in market_data.items():
                        if market_symbol.startswith(base_symbol + "/"):
                            symbol_market_data = data_item
                            break
                    
                    condition_line = f"• {base_symbol}: OPEN {position.entry_action.upper()} position ({hours_held:.1f}h held)"
                    
                    if symbol_market_data:
                        # Add current condition analysis
                        current_tech = symbol_market_data.get("technical_indicators", {})
                        current_summary = current_tech.get("technical_summary", "neutral")
                        current_price = symbol_market_data.get("price")
                        change_24h = symbol_market_data.get("change_24h_percent", 0)
                        
                        # Calculate simple P&L if possible
                        pnl_info = ""
                        if hasattr(position, 'entry_price') and position.entry_price and current_price:
                            if position.entry_action == "buy":
                                pnl_pct = ((current_price - position.entry_price) / position.entry_price) * 100
                                pnl_info = f" [P&L: {pnl_pct:+.1f}%]"
                        
                        condition_line += f"\n    Current: ${current_price:,.2f} ({change_24h:+.1f}% 24h), Tech: {current_summary.replace('_', ' ').title()}{pnl_info}"
                    
                    formatted_lines.append(condition_line)
            
            return "\n".join(formatted_lines)
            
        except Exception as e:
            logger.error(f"Error formatting positions: {str(e)}")
            return "Error retrieving position information."

class EnhancedPromptGenerator:
    """Enhanced prompt generator with dynamic conviction thresholds"""
    
    def __init__(self):
        self.position_analyzer = PositionAnalyzer()
        self.market_analyzer = MarketConditionAnalyzer()
        self.conviction_engine = DynamicConvictionEngine()
    
    def generate_market_responsive_position_review(self, symbol: str, position: Any, 
                                                  data: Dict[str, Any], 
                                                  condition_change: Dict[str, Any]) -> str:
        """Generate prompt specifically for market condition changes"""
        
        market_data = data.get("market", {})
        sentiment_data = data.get("sentiment", {})
        fear_greed_data = data.get("fear_greed", {})
        
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Get current market data for the symbol
        symbol_market_data = None
        for market_symbol, data_item in market_data.items():
            if market_symbol.startswith(symbol + "/"):
                symbol_market_data = data_item
                break
        
        prompt = f"""You are reviewing an OPEN {symbol} position due to changing market conditions at {current_time}.

POSITION REVIEW MODE: MARKET CONDITIONS CHANGED

═══════════════════════════════════════════════════════════════════════════════

CURRENT OPEN POSITION:
• Asset: {symbol}
• Action: {position.entry_action.upper()}
• Entry Time: {position.entry_timestamp.split('T')[1][:5]} ({condition_change.get('hours_held', 0):.1f} hours held)
• Entry Conviction: {getattr(position, 'entry_conviction_score', position.entry_confidence * 10):.1f}/10
• Entry Rationale: "{position.entry_rationale}"

MARKET CONDITION CHANGE ANALYSIS:
• Change Type: {condition_change['change_type'].replace('_', ' ').title()}
• Severity Score: {condition_change['severity_score']}/10
• Changes Detected: {len(condition_change['changes'])}"""

        # List specific changes
        if condition_change.get('changes'):
            prompt += f"""

SPECIFIC CHANGES:"""
            for change in condition_change['changes']:
                prompt += f"""
• {change}"""
        
        # Current market conditions
        if symbol_market_data:
            price = symbol_market_data.get("price")
            change_24h = symbol_market_data.get("change_24h_percent", 0)
            
            prompt += f"""

CURRENT MARKET CONDITIONS:
• Price: ${price:,.2f} ({change_24h:+.2f}% 24h)"""
            
            # Calculate P&L if possible
            if hasattr(position, 'entry_price') and position.entry_price:
                if position.entry_action == "buy":
                    pnl_pct = ((price - position.entry_price) / position.entry_price) * 100
                    prompt += f"""
• Position P&L: {pnl_pct:+.1f}% from entry price ${position.entry_price:,.2f}"""
            
            # Technical analysis
            technical = symbol_market_data.get("technical_indicators", {})
            if technical:
                tech_summary = technical.get("technical_summary", "neutral")
                rsi = technical.get("rsi")
                macd_interpretation = technical.get("macd_interpretation", "neutral")
                ma_trend = technical.get("ma_trend", "neutral")
                
                prompt += f"""
• Technical Summary: {tech_summary.replace('_', ' ').title()}"""
                if rsi:
                    rsi_signal = technical.get("rsi_signal", "neutral")
                    prompt += f"""
• RSI: {rsi:.1f} ({rsi_signal.replace('_', ' ').title()}"""
                    if rsi > 70:
                        prompt += f" - OVERBOUGHT WARNING"
                    elif rsi < 30:
                        prompt += f" - OVERSOLD"
                    prompt += ")"
                
                if macd_interpretation != "neutral":
                    prompt += f"""
• MACD: {macd_interpretation.replace('_', ' ').title()}"""
                
                if ma_trend != "neutral":
                    prompt += f"""
• Moving Average Trend: {ma_trend.replace('_', ' ').title()}"""
        
        # Sentiment analysis
        if symbol in sentiment_data:
            sentiment_info = sentiment_data[symbol]
            sentiment_category = sentiment_info.get("sentiment_category", "neutral")
            prompt += f"""
• Current Sentiment: {sentiment_category.replace('_', ' ').title()}"""
        
        # Entry vs current comparison
        entry_conditions = condition_change.get('entry_conditions', {})
        current_conditions = condition_change.get('current_conditions', {})
        
        prompt += f"""

ENTRY vs CURRENT CONDITIONS:
• Technical at Entry: {entry_conditions.get('technical_summary', 'unknown').replace('_', ' ').title()}
• Technical Now: {current_conditions.get('technical_summary', 'neutral').replace('_', ' ').title()}
• Sentiment at Entry: {entry_conditions.get('sentiment', 'unknown').replace('_', ' ').title()}
• Sentiment Now: {current_conditions.get('sentiment', 'neutral').replace('_', ' ').title()}"""
        
        # Market context
        if fear_greed_data and fear_greed_data.get("value") is not None:
            fg_value = fear_greed_data.get("value")
            fg_category = fear_greed_data.get("category", "neutral")
            prompt += f"""

MARKET CONTEXT:
• Fear & Greed Index: {fg_value}/100 ({fg_category.replace('_', ' ').title()})"""
        
        prompt += f"""

═══════════════════════════════════════════════════════════════════════════════

POSITION REVIEW DECISION:

Since market conditions have changed, evaluate whether to:

1. **HOLD**: Original thesis still valid despite changes
2. **REDUCE**: Partially close position (take some profits/cut some losses)  
3. **SELL**: Fully exit position (thesis no longer valid)

DECISION CRITERIA:
• Are the technical indicators still supportive of the {position.entry_action.upper()} position?
• Has the original investment thesis been invalidated?
• Do current conditions suggest higher risk than reward?
• Is this temporary market noise or a fundamental shift?

Respond in JSON format:
{{
    "analysis_type": "market_responsive_position_review",
    "action": "hold", "reduce", or "sell",
    "conviction_score": 1-10,
    "amount_percentage": 0-100 (if reducing: % to sell, if selling: 100),
    "reasoning": "Detailed analysis of why the decision makes sense",
    "market_condition_assessment": "How market changes affect the position",
    "original_thesis_status": "intact", "weakened", or "invalidated",
    "urgency": "low", "medium", or "high"
}}

CRITICAL: Focus on current market reality vs original entry thesis. Don't hold onto losing positions just because of sunk cost bias.
"""
        
        return prompt
    
    def generate_opportunity_prompt(self, data: Dict[str, Any], 
                                  portfolio_data: Dict[str, float]) -> str:
        """Generate prompt for scanning new opportunities with dynamic conviction threshold"""
        
        market_data = data.get("market", {})
        sentiment_data = data.get("sentiment", {})
        fear_greed_data = data.get("fear_greed", {})
        
        # Calculate dynamic conviction threshold
        dynamic_threshold = self.conviction_engine.calculate_dynamic_threshold(
            market_data, sentiment_data, fear_greed_data
        )
        
        # Get explanation for the threshold
        threshold_explanation = self.conviction_engine.explain_threshold_reasoning(
            dynamic_threshold, market_data, sentiment_data, fear_greed_data
        )
        
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        prompt = f"""You are scanning for high-conviction cryptocurrency trading opportunities at {current_time}.

STRATEGIC TRADING PHILOSOPHY:
- Quality over quantity - wait for exceptional setups
- Multiple confirmations required before entry
- DYNAMIC conviction threshold: {dynamic_threshold:.1f}/10 (adjusted for current market conditions)
- Clear risk management with defined stops and targets

🎯 CONVICTION THRESHOLD ANALYSIS:
{threshold_explanation}

CURRENT STATUS: No open positions - opportunity scanning mode

═══════════════════════════════════════════════════════════════════════════════

MARKET ANALYSIS:
"""
        
        # Market data analysis
        for symbol, data_item in market_data.items():
            base_symbol = symbol.split('/')[0]
            price = data_item.get("price")
            change_24h = data_item.get("change_24h_percent", 0)
            volume_24h = data_item.get("volume_24h")
            
            prompt += f"""
{base_symbol}: ${price:,.2f} ({change_24h:+.2f}% 24h)"""
            
            # Technical analysis
            technical = data_item.get("technical_indicators", {})
            if technical:
                tech_summary = technical.get("technical_summary", "neutral")
                rsi = technical.get("rsi")
                rsi_signal = technical.get("rsi_signal", "neutral")
                ma_trend = technical.get("ma_trend", "neutral")
                macd_interpretation = technical.get("macd_interpretation", "neutral")
                
                prompt += f"""
• Technical: {tech_summary.replace('_', ' ').title()}"""
                if rsi:
                    prompt += f", RSI: {rsi:.1f} ({rsi_signal.replace('_', ' ').title()})"
                if ma_trend != "neutral":
                    prompt += f", Trend: {ma_trend.replace('_', ' ').title()}"
                if macd_interpretation != "neutral":
                    prompt += f", MACD: {macd_interpretation.replace('_', ' ').title()}"
            
            # Sentiment
            if base_symbol in sentiment_data:
                sentiment_info = sentiment_data[base_symbol]
                sentiment_category = sentiment_info.get("sentiment_category", "neutral")
                sentiment_score = sentiment_info.get("overall_sentiment", 0)
                confidence = sentiment_info.get("confidence", 0)
                
                prompt += f"""
• Sentiment: {sentiment_category.replace('_', ' ').title()} ({sentiment_score:+.2f}, {confidence:.0%})"""
        
        # Market context
        if fear_greed_data and fear_greed_data.get("value") is not None:
            fg_value = fear_greed_data.get("value")
            fg_category = fear_greed_data.get("category", "neutral")
            prompt += f"""

MARKET CONTEXT:
• Fear & Greed Index: {fg_value}/100 ({fg_category.replace('_', ' ').title()})"""
        
        # Recent activity context
        symbols = list(market_data.keys())
        recent_decisions = self.position_analyzer.format_recent_decisions(symbols, hours=24)
        prompt += f"""

RECENT ACTIVITY (24h):
{recent_decisions}

═══════════════════════════════════════════════════════════════════════════════

DYNAMIC HIGH-CONVICTION OPPORTUNITY SCAN:

ENTRY REQUIREMENTS:
1. **Technical Confluence**: 2+ technical indicators aligned
2. **Sentiment Support**: Supportive sentiment with reasonable confidence
3. **Risk/Reward**: Clear entry, stop-loss, and profit targets
4. **Dynamic Conviction**: Must achieve {dynamic_threshold:.1f}+/10 conviction to trade (adjusted for current conditions)
5. **Timing**: No conflicting recent trades

DECISION OPTIONS:
• BUY: High-conviction setup meeting the {dynamic_threshold:.1f}/10 threshold
• WAIT: Insufficient conviction or conflicting signals

🚨 IMPORTANT: Current conviction threshold is {dynamic_threshold:.1f}/10 (not the standard 7.0/10)
This reflects current market conditions and opportunity quality.

Respond in JSON format:
{{
    "analysis_type": "dynamic_opportunity_scan", 
    "action": "buy" or "wait",
    "asset": "BTC", "ETH", etc. (if buying),
    "conviction_score": 1-10,
    "amount_percentage": 1-100 (of available capital),
    "entry_rationale": "Why this meets the {dynamic_threshold:.1f}/10 threshold",
    "risk_management": {{
        "stop_loss": "Exit criteria if wrong",
        "profit_target": "Target for profits", 
        "expected_duration": "Holding period"
    }},
    "market_condition_assessment": "How current market conditions affect this opportunity",
    "threshold_met": true/false (does this meet the {dynamic_threshold:.1f}/10 requirement?)
}}

REMEMBER: Only trade if conviction >= {dynamic_threshold:.1f}/10. Missing opportunities is better than taking low-conviction trades.
"""
        
        return prompt

class StrategicGPTAnalyzer:
    """Enhanced strategic GPT analysis engine with dynamic conviction thresholds"""
    
    def __init__(self):
        self.pm = get_position_manager()
        self.position_analyzer = PositionAnalyzer()
        self.prompt_generator = EnhancedPromptGenerator()
        self.market_analyzer = MarketConditionAnalyzer()
        self.conviction_engine = DynamicConvictionEngine()
        self.api_manager = APIRateManager()
    
    def _detect_portfolio_positions(self, portfolio_data: Dict[str, float]) -> Dict[str, Any]:
        """
        Detect significant holdings in portfolio - FOCUSED on main trading assets only
        
        Args:
            portfolio_data: Dictionary of {currency: usd_value}
            
        Returns:
            Dictionary of detected positions for MAIN ASSETS ONLY
        """
        if not portfolio_data:
            return {}
        
        # FOCUS: Only our 4 main trading assets
        MAIN_TRADING_ASSETS = {'BTC', 'ETH', 'SOL', 'DOGE'}
        
        detected_positions = {}
        
        # Calculate total portfolio value for main assets only
        main_asset_values = {k: v for k, v in portfolio_data.items() if k in MAIN_TRADING_ASSETS}
        total_main_value = sum(main_asset_values.values())
        
        if total_main_value == 0:
            logger.info("🔍 No main trading assets detected in portfolio")
            return {}
        
        # Define threshold - 5% of main asset value or $50 minimum
        min_position_threshold = max(50.0, total_main_value * 0.05)
        
        for currency, usd_value in main_asset_values.items():
            # Check if this is a significant holding in our main assets
            if usd_value >= min_position_threshold:
                # Check if it's already tracked as a formal position
                if currency not in self.pm.get_open_positions():
                    logger.info(f"📈 Detected untracked {currency} holding: ${usd_value:.2f}")
                    
                    # Create a synthetic position object for analysis
                    from position_manager import Position
                    detected_positions[currency] = Position(
                        symbol=currency,
                        entry_action="buy",  # Assume existing holdings are "buy" positions
                        entry_timestamp=datetime.now().isoformat(),
                        entry_confidence=0.6,  # Moderate confidence for existing holdings
                        entry_rationale="Pre-existing portfolio holding",
                        amount_percentage=0,
                        entry_conviction_score=6.0,  # Neutral-positive conviction
                        usd_amount=usd_value,
                        crypto_amount=None,
                        entry_price=None,
                        is_open=True
                    )
                    logger.info(f"📊 Tracking {currency} as portfolio position (${usd_value:.2f})")
        
        if detected_positions:
            logger.info(f"📊 Detected {len(detected_positions)} main asset positions")
        else:
            logger.info("✅ All main assets already tracked or below threshold")
        
        return detected_positions
    
    def analyze_market_strategically(self, data: Dict[str, Any], 
                               portfolio_data: Dict[str, float] = None,
                               news_impact: Dict[str, Any] = None) -> AnalysisResult:
        """
        Enhanced strategic analysis FOCUSED on 4 main trading assets
        """
        logger.info("🧠 Starting focused strategic analysis (BTC, ETH, SOL, DOGE)")
        
        # Check for formal open positions from position manager
        open_positions = self.pm.get_open_positions()
        
        # FOCUSED: Only check portfolio for our 4 main trading assets
        portfolio_positions = self._detect_portfolio_positions(portfolio_data)
        
        # Combine formal positions with main asset holdings only
        all_positions = {}
        all_positions.update(open_positions)
        
        # Add main asset positions that aren't already tracked
        for symbol, portfolio_position in portfolio_positions.items():
            if symbol not in all_positions:
                all_positions[symbol] = portfolio_position
        
        if all_positions:
            logger.info(f"📊 Focused position review: {len(all_positions)} main asset positions "
                    f"({len(open_positions)} formal + {len(portfolio_positions)} portfolio)")
            return self._analyze_positions_with_market_awareness(data, portfolio_data, all_positions, news_impact)
        else:
            logger.info("🔍 Opportunity scanning mode: No main asset positions")
            return self._scan_opportunities(data, portfolio_data, news_impact)
    
    def _analyze_positions_with_market_awareness(self, data: Dict[str, Any], 
                                               portfolio_data: Dict[str, float],
                                               open_positions: Dict[str, Any], 
                                               news_impact: Dict[str, Any]) -> AnalysisResult:
        """Enhanced position analysis with market condition awareness"""
        
        market_data = data.get("market", {})
        sentiment_data = data.get("sentiment", {})
        
        # Check each position for market condition changes
        position_priorities = []
        
        for symbol, position in open_positions.items():
            # Get current market data for this symbol
            symbol_market_data = None
            for market_symbol, data_item in market_data.items():
                if market_symbol.startswith(symbol + "/"):
                    symbol_market_data = data_item
                    break
            
            if not symbol_market_data:
                logger.warning(f"No market data found for {symbol}")
                continue
            
            # Check if position needs review due to market conditions
            should_review, trigger_reason = self.position_analyzer.should_review_position(
                symbol, symbol_market_data, sentiment_data.get(symbol, {})
            )
            
            if should_review:
                # Analyze market condition changes
                condition_change = self.market_analyzer.analyze_condition_change(
                    symbol, symbol_market_data, sentiment_data.get(symbol, {})
                )
                
                # Assign priority based on severity
                priority = 1  # Default low priority
                if trigger_reason in ["significant_deterioration", "bearish_technicals", "extreme_rsi"]:
                    priority = 3  # High priority
                elif trigger_reason in ["moderate_deterioration", "duration_review"]:
                    priority = 2  # Medium priority
                
                position_priorities.append((priority, symbol, position, condition_change, trigger_reason))
                
                logger.info(f"🔍 Position {symbol} needs review: {trigger_reason} (priority: {priority})")
            else:
                logger.debug(f"✅ Position {symbol} stable: {trigger_reason}")
        
        # If no positions need review, return hold
        if not position_priorities:
            logger.info("✅ All positions stable - no immediate action needed")
            return AnalysisResult(
                action="hold",
                confidence=0.8,
                conviction_score=7.0,
                rationale=f"All {len(open_positions)} positions stable. Market conditions haven't "
                         f"significantly changed from entry thesis.",
                analysis_type="position_stability_check",
                positions_reviewed=len(open_positions),
                market_condition_change="stable"
            )
        
        # Sort by priority and handle highest priority position
        position_priorities.sort(key=lambda x: x[0], reverse=True)
        priority, symbol, position, condition_change, trigger_reason = position_priorities[0]
        
        logger.info(f"🎯 Reviewing highest priority position: {symbol} ({trigger_reason})")
        
        return self._execute_market_responsive_review(
            data, portfolio_data, symbol, position, condition_change, trigger_reason
        )
    
    def _execute_market_responsive_review(self, data: Dict[str, Any], 
                                        portfolio_data: Dict[str, float],
                                        symbol: str, position: Any, 
                                        condition_change: Dict[str, Any], 
                                        trigger_reason: str) -> AnalysisResult:
        """Execute market-responsive position review with GPT"""
        
        if not self.api_manager.check_api_limit():
            logger.warning("GPT API limit reached - maintaining positions")
            return AnalysisResult(
                action="hold",
                asset=symbol,
                confidence=0.6,
                rationale="GPT API limit reached - maintaining positions for safety",
                analysis_type="api_limited"
            )
        
        try:
            # Generate market-responsive prompt
            prompt = self.prompt_generator.generate_market_responsive_position_review(
                symbol, position, data, condition_change
            )
            
            # Call GPT
            response = self._call_gpt(prompt, f"market-responsive position review for {symbol}")
            result = self._parse_position_review_response(response, symbol, trigger_reason)
            
            # Record conviction snapshot
            self.pm.record_conviction_snapshot(
                symbol, result.conviction_score, result.confidence, result.action,
                data.get("market", {}), data.get("sentiment", {}), data.get("fear_greed", {}),
                analysis_type="market_responsive_review"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error in market-responsive review: {str(e)}")
            return AnalysisResult(
                action="hold",
                asset=symbol,
                confidence=0.5,
                rationale=f"Review error - maintaining position: {str(e)}",
                analysis_type="error"
            )
    
    def _scan_opportunities(self, data: Dict[str, Any], portfolio_data: Dict[str, float],
                           news_impact: Dict[str, Any]) -> AnalysisResult:
        """Scan for new trading opportunities with dynamic conviction thresholds"""
        
        # Calculate dynamic threshold for current market conditions
        market_data = data.get("market", {})
        sentiment_data = data.get("sentiment", {})
        fear_greed_data = data.get("fear_greed", {})
        
        dynamic_threshold = self.conviction_engine.calculate_dynamic_threshold(
            market_data, sentiment_data, fear_greed_data
        )
        
        logger.info(f"🎯 Dynamic conviction threshold: {dynamic_threshold:.1f}/10")
        
        # Check recent trading activity to avoid overtrading
        recent_trades = self.pm.get_recent_trades(hours=3)
        recent_executed = [t for t in recent_trades if t.executed]
        
        if recent_executed:
            last_trade = recent_executed[0]
            hours_since = (datetime.now() - datetime.fromisoformat(last_trade.timestamp)).total_seconds() / 3600
            
            if hours_since < 2:  # Very recent trade
                logger.info(f"⏸️ Recent trade cooling period: {last_trade.symbol} {hours_since:.1f}h ago")
                return AnalysisResult(
                    action="hold",
                    confidence=0.8,
                    conviction_score=dynamic_threshold,
                    rationale=f"Strategic patience: Recent {last_trade.symbol} trade {hours_since:.1f}h ago. "
                             f"Avoiding overtrading.",
                    analysis_type="cooling_period"
                )
        
        # Check for high-conviction news opportunities
        if news_impact and news_impact.get("best_trading_opportunity"):
            opportunity = news_impact["best_trading_opportunity"]
            news_confidence = opportunity.get("confidence", 0)
            
            if news_confidence >= 0.85:  # Very high news conviction
                return self._process_news_opportunity(opportunity, data, portfolio_data)
        
        # Execute GPT opportunity scan with dynamic threshold
        if not self.api_manager.check_api_limit():
            logger.info("GPT API limit reached - strategic waiting")
            return AnalysisResult(
                action="hold",
                confidence=0.7,
                rationale="GPT API limit reached - strategic patience",
                analysis_type="api_limited"
            )
        
        return self._execute_opportunity_scan(data, portfolio_data, dynamic_threshold)
    
    def _execute_opportunity_scan(self, data: Dict[str, Any], portfolio_data: Dict[str, float],
                                 dynamic_threshold: float) -> AnalysisResult:
        """Execute GPT-based opportunity scanning with dynamic threshold"""
        
        try:
            prompt = self.prompt_generator.generate_opportunity_prompt(data, portfolio_data)
            response = self._call_gpt(prompt, "dynamic strategic opportunity scanning")
            result = self._parse_opportunity_response(response, dynamic_threshold)
            
            # Validate dynamic conviction requirements
            if result.action == "buy" and result.conviction_score < dynamic_threshold:
                logger.info(f"⏸️ Dynamic threshold not met: {result.conviction_score:.1f}/{dynamic_threshold:.1f} for {result.asset}")
                return AnalysisResult(
                    action="hold",
                    confidence=0.7,
                    conviction_score=dynamic_threshold,
                    rationale=f"Strategic patience: {result.asset} signal only {result.conviction_score:.1f}/{dynamic_threshold:.1f} "
                             f"conviction. Dynamic threshold requires {dynamic_threshold:.1f}+ in current market conditions.",
                    analysis_type="dynamic_threshold_not_met",
                    rejected_signal={
                        "asset": result.asset,
                        "action": result.action,
                        "conviction": result.conviction_score,
                        "required_threshold": dynamic_threshold
                    }
                )
            
            # Record high-conviction opportunities that meet dynamic threshold
            if result.action == "buy" and result.conviction_score >= dynamic_threshold:
                self.pm.record_conviction_snapshot(
                    result.asset, result.conviction_score, result.confidence, result.action,
                    data.get("market", {}), data.get("sentiment", {}), data.get("fear_greed", {}),
                    analysis_type="dynamic_opportunity_scan"
                )
                logger.info(f"✅ Dynamic high-conviction opportunity: {result.action} {result.asset} "
                           f"({result.conviction_score:.1f}/{dynamic_threshold:.1f})")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in dynamic opportunity scan: {str(e)}")
            return AnalysisResult(
                action="hold",
                confidence=0.7,
                rationale=f"Opportunity scan error - strategic patience: {str(e)}",
                analysis_type="error"
            )
    
    def _process_news_opportunity(self, opportunity: Dict[str, Any], data: Dict[str, Any],
                                 portfolio_data: Dict[str, float]) -> AnalysisResult:
        """Process high-conviction news-driven opportunities"""
        
        recommended_action = opportunity.get("recommended_action", {})
        confidence = opportunity.get("confidence", 0)
        asset = recommended_action.get("asset")
        action = recommended_action.get("action")
        
        if not asset or not action:
            return AnalysisResult(
                action="hold",
                confidence=0.6,
                rationale="News opportunity detected but insufficient details",
                analysis_type="news_invalid"
            )
        
        # Check for conflicts with recent trades
        recent_trades = self.pm.get_trades_by_symbol(asset)
        if recent_trades:
            last_trade = max(recent_trades, key=lambda x: x.timestamp)
            hours_since = (datetime.now() - datetime.fromisoformat(last_trade.timestamp)).total_seconds() / 3600
            
            # Check for conflicting signals within 1 hour
            if (hours_since < 1 and 
                ((last_trade.action == "buy" and action == "sell") or 
                 (last_trade.action == "sell" and action == "buy"))):
                
                if confidence < 0.95:  # Require very high confidence to override
                    return AnalysisResult(
                        action="hold",
                        asset=asset,
                        confidence=0.6,
                        rationale=f"News signal conflicts with recent {last_trade.action}. "
                                 f"Insufficient confidence ({confidence:.0%}) to override.",
                        analysis_type="news_conflict"
                    )
        
        # Process the news opportunity
        conviction_score = confidence * 10
        
        self.pm.record_conviction_snapshot(
            asset, conviction_score, confidence, action,
            data.get("market", {}), data.get("sentiment", {}), data.get("fear_greed", {}),
            analysis_type="news_driven"
        )
        
        return AnalysisResult(
            action=action,
            asset=asset,
            amount_percentage=recommended_action.get("amount_percentage", 25),
            confidence=confidence,
            conviction_score=conviction_score,
            rationale=f"HIGH-CONVICTION NEWS: {opportunity.get('rationale', '')}",
            analysis_type="news_driven",
            news_driven=True,
            urgency=recommended_action.get("urgency", "medium")
        )
    
    def _call_gpt(self, prompt: str, context: str) -> Dict[str, Any]:
        """Make GPT API call with error handling"""
        
        try:
            openai_api_key = os.getenv("OPENAI_API_KEY")
            if not openai_api_key:
                raise Exception("OpenAI API key not found")
            
            client = OpenAI(api_key=openai_api_key)
            self.api_manager.increment_counter()
            
            logger.info(f"🤖 Calling GPT for {context}")
            
            response = client.chat.completions.create(
                model=GPT_MODEL,
                messages=[
                    {"role": "system", "content": f"You are a strategic cryptocurrency trader focused on {context}."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.2
            )
            
            return response
            
        except Exception as e:
            logger.error(f"GPT API call failed: {str(e)}")
            raise
    
    def _parse_position_review_response(self, response: Dict[str, Any], symbol: str, 
                                      trigger_reason: str) -> AnalysisResult:
        """Parse market-responsive position review response from GPT"""
        
        try:
            content = response.choices[0].message.content
            result = json.loads(content)
            
            action = result.get("action", "hold")
            conviction_score = result.get("conviction_score", 5)
            amount_percentage = result.get("amount_percentage", 100 if action == "sell" else 50)
            
            # Map "reduce" action to "sell" with partial amount
            if action == "reduce":
                action = "sell"
                if amount_percentage == 0:
                    amount_percentage = 50  # Default 50% reduction
            
            return AnalysisResult(
                action=action,
                asset=symbol,
                amount_percentage=amount_percentage,
                confidence=conviction_score / 10.0,
                conviction_score=conviction_score,
                rationale=result.get("reasoning", ""),
                analysis_type="market_responsive_position_review",
                market_condition_change=result.get("market_condition_assessment", ""),
                position_review_trigger=trigger_reason,
                urgency=result.get("urgency", "medium")
            )
                
        except Exception as e:
            logger.error(f"Error parsing position review: {str(e)}")
            raise
    
    def _parse_opportunity_response(self, response: Dict[str, Any], 
                                  dynamic_threshold: float) -> AnalysisResult:
        """Parse opportunity scan response from GPT with dynamic threshold validation"""
        
        try:
            content = response.choices[0].message.content
            result = json.loads(content)
            
            action = result.get("action", "wait")
            
            if action == "wait":
                return AnalysisResult(
                    action="hold",
                    confidence=0.7,
                    conviction_score=dynamic_threshold,
                    rationale=result.get("patience_note", "Strategic patience - waiting for better opportunities"),
                    analysis_type="dynamic_opportunity_scan"
                )
            else:
                conviction_score = result.get("conviction_score", 5)
                threshold_met = result.get("threshold_met", conviction_score >= dynamic_threshold)
                
                return AnalysisResult(
                    action=action if threshold_met else "hold",
                    asset=result.get("asset"),
                    amount_percentage=result.get("amount_percentage", 25),
                    confidence=conviction_score / 10.0,
                    conviction_score=conviction_score,
                    rationale=result.get("entry_rationale", ""),
                    analysis_type="dynamic_opportunity_scan"
                )
                
        except Exception as e:
            logger.error(f"Error parsing dynamic opportunity response: {str(e)}")
            raise

# Main public interface
def analyze_with_strategic_conviction(data: Dict[str, Any], 
                                    portfolio_data: Dict[str, float] = None,
                                    news_impact: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Enhanced entry point for strategic conviction-based analysis with dynamic thresholds
    """
    try:
        analyzer = StrategicGPTAnalyzer()
        result = analyzer.analyze_market_strategically(data, portfolio_data, news_impact)
        return result.to_dict()
        
    except Exception as e:
        logger.error(f"Strategic analysis failed: {str(e)}")
        return {
            "action": "hold",
            "asset": None,
            "amount_percentage": 0,
            "confidence": 0.5,
            "conviction_score": 5.0,
            "rationale": f"Strategic analysis error: {str(e)}",
            "analysis_type": "error",
            "timestamp": datetime.now().isoformat()
        }

# Legacy compatibility functions
def validate_data(data: Dict[str, Any]) -> bool:
    """Validate input data structure"""
    if not isinstance(data, dict):
        return False
    
    required_keys = ["market"]
    return all(key in data for key in required_keys) and bool(data.get("market"))

def check_api_limit() -> bool:
    """Check API rate limit"""
    return APIRateManager.check_api_limit()

def increment_api_counter():
    """Increment API counter"""
    APIRateManager.increment_counter()

# For backwards compatibility - redirect old function calls
def analyze_market_with_consistency_check(data: Dict[str, Any], portfolio_data: Dict[str, float] = None,
                                        news_impact: Dict[str, Any] = None) -> Dict[str, Any]:
    """Legacy function - redirects to strategic analysis"""
    return analyze_with_strategic_conviction(data, portfolio_data, news_impact)

def analyze_market_with_news(data: Dict[str, Any], portfolio_data: Dict[str, float] = None,
                           news_impact: Dict[str, Any] = None) -> Dict[str, Any]:
    """Legacy function - redirects to strategic analysis"""
    return analyze_with_strategic_conviction(data, portfolio_data, news_impact)

def analyze_market(data: Dict[str, Any], portfolio_data: Dict[str, float] = None) -> Dict[str, Any]:
    """Legacy function - redirects to strategic analysis"""
    return analyze_with_strategic_conviction(data, portfolio_data, None)

# News analysis functions (simplified versions for basic functionality)
def process_news_batch(news_articles: List[Dict[str, Any]], market_data: Dict[str, Any],
                      portfolio_data: Dict[str, float] = None) -> Dict[str, Any]:
    """Simplified news processing for basic functionality"""
    logger.info(f"Processing {len(news_articles)} news articles")
    
    # Simple aggregation
    relevant_count = len([a for a in news_articles if "crypto" in str(a).lower() or "bitcoin" in str(a).lower()])
    
    return {
        "analyses": [],
        "summary": {
            "total_articles": len(news_articles),
            "relevant_articles": relevant_count,
            "overall_sentiment": "neutral"
        },
        "best_trading_opportunity": None,
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    print("Enhanced Strategic GPT Analysis Engine with Dynamic Conviction Thresholds - Ready")
    print("Key enhancements:")
    print("- Dynamic conviction thresholds based on market conditions")
    print("- Fear & Greed Index integration")
    print("- Technical confluence analysis")
    print("- Sentiment strength weighting")
    print("- Volatility and market structure awareness")
    print("- Real-time threshold explanations")
    print("- All legacy features maintained")