"""
Analytics module for tracking query performance and user behavior.
"""

import time
import threading
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field, asdict
from collections import defaultdict, deque
import logging
from datetime import datetime, timedelta

from api.cache import get_cache_stats

logger = logging.getLogger(__name__)


@dataclass
class QueryAnalytics:
    """Analytics data for a single query."""
    query_id: str
    query: str
    user_id: Optional[str]
    timestamp: float
    execution_time: float
    response_size: int
    confidence: float
    cache_hit: bool
    error_occurred: bool
    error_type: Optional[str]
    agent_usage: Dict[str, Any]
    token_usage: Dict[str, int]
    user_agent: Optional[str]
    ip_address: Optional[str]


@dataclass
class SystemMetrics:
    """System-wide performance metrics."""
    total_queries: int = 0
    successful_queries: int = 0
    failed_queries: int = 0
    average_response_time: float = 0.0
    cache_hit_rate: float = 0.0
    error_rate: float = 0.0
    active_users: int = 0
    peak_concurrent_users: int = 0
    total_tokens_used: int = 0
    average_confidence: float = 0.0


class AnalyticsCollector:
    """Collects and analyzes platform usage data."""
    
    def __init__(self, max_history: int = 10000):
        self.max_history = max_history
        self.queries: deque = deque(maxlen=max_history)
        self.metrics = SystemMetrics()
        self.lock = threading.Lock()
        
        # Real-time counters
        self.active_sessions = set()
        self.concurrent_users = 0
        self.peak_concurrent = 0
        
        # Time-based aggregations
        self.hourly_stats = defaultdict(lambda: {
            'queries': 0,
            'errors': 0,
            'avg_response_time': 0.0,
            'unique_users': set()
        })
        
        # Query pattern analysis
        self.query_patterns = defaultdict(int)
        self.user_behavior = defaultdict(lambda: {
            'total_queries': 0,
            'avg_confidence': 0.0,
            'favorite_topics': defaultdict(int)
        })
    
    def record_query(self, analytics: QueryAnalytics) -> None:
        """Record a query for analytics."""
        with self.lock:
            # Add to history
            self.queries.append(analytics)
            
            # Update metrics
            self.metrics.total_queries += 1
            
            if analytics.error_occurred:
                self.metrics.failed_queries += 1
            else:
                self.metrics.successful_queries += 1
            
            # Update response time
            if self.metrics.total_queries > 1:
                current_avg = self.metrics.average_response_time
                new_avg = (current_avg * (self.metrics.total_queries - 1) + analytics.execution_time) / self.metrics.total_queries
                self.metrics.average_response_time = new_avg
            else:
                self.metrics.average_response_time = analytics.execution_time
            
            # Update cache hit rate
            if analytics.cache_hit:
                cache_hits = sum(1 for q in self.queries if q.cache_hit)
                self.metrics.cache_hit_rate = cache_hits / len(self.queries)
            
            # Update error rate
            errors = sum(1 for q in self.queries if q.error_occurred)
            self.metrics.error_rate = errors / len(self.queries)
            
            # Update token usage
            self.metrics.total_tokens_used += sum(analytics.token_usage.values())
            
            # Update confidence
            if self.metrics.total_queries > 1:
                current_avg = self.metrics.average_confidence
                new_avg = (current_avg * (self.metrics.total_queries - 1) + analytics.confidence) / self.metrics.total_queries
                self.metrics.average_confidence = new_avg
            else:
                self.metrics.average_confidence = analytics.confidence
            
            # Track user sessions
            if analytics.user_id:
                self.active_sessions.add(analytics.user_id)
                self.metrics.active_users = len(self.active_sessions)
            
            # Update hourly stats
            hour_key = datetime.fromtimestamp(analytics.timestamp).strftime('%Y-%m-%d %H:00')
            self.hourly_stats[hour_key]['queries'] += 1
            if analytics.error_occurred:
                self.hourly_stats[hour_key]['errors'] += 1
            if analytics.user_id:
                self.hourly_stats[hour_key]['unique_users'].add(analytics.user_id)
            
            # Analyze query patterns
            words = analytics.query.lower().split()
            for word in words:
                if len(word) > 3:  # Skip short words
                    self.query_patterns[word] += 1
            
            # Track user behavior
            if analytics.user_id:
                user_data = self.user_behavior[analytics.user_id]
                user_data['total_queries'] += 1
                
                # Update average confidence
                if user_data['total_queries'] > 1:
                    current_avg = user_data['avg_confidence']
                    new_avg = (current_avg * (user_data['total_queries'] - 1) + analytics.confidence) / user_data['total_queries']
                    user_data['avg_confidence'] = new_avg
                else:
                    user_data['avg_confidence'] = analytics.confidence
    
    def get_recent_queries(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent queries for analysis."""
        with self.lock:
            recent = list(self.queries)[-limit:]
            return [asdict(q) for q in recent]
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """Get current system metrics."""
        with self.lock:
            return asdict(self.metrics)
    
    def get_hourly_stats(self, hours: int = 24) -> Dict[str, Any]:
        """Get hourly statistics."""
        with self.lock:
            now = datetime.now()
            stats = {}
            
            for i in range(hours):
                hour_key = (now - timedelta(hours=i)).strftime('%Y-%m-%d %H:00')
                if hour_key in self.hourly_stats:
                    hour_data = self.hourly_stats[hour_key]
                    stats[hour_key] = {
                        'queries': hour_data['queries'],
                        'errors': hour_data['errors'],
                        'unique_users': len(hour_data['unique_users']),
                        'error_rate': hour_data['errors'] / hour_data['queries'] if hour_data['queries'] > 0 else 0
                    }
            
            return stats
    
    def get_popular_queries(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get most popular query patterns."""
        with self.lock:
            sorted_patterns = sorted(self.query_patterns.items(), key=lambda x: x[1], reverse=True)
            return [{'pattern': pattern, 'count': count} for pattern, count in sorted_patterns[:limit]]
    
    def get_user_insights(self, user_id: str) -> Dict[str, Any]:
        """Get insights for a specific user."""
        with self.lock:
            if user_id not in self.user_behavior:
                return {}
            
            user_data = self.user_behavior[user_id]
            return {
                'total_queries': user_data['total_queries'],
                'average_confidence': user_data['avg_confidence'],
                'favorite_topics': dict(user_data['favorite_topics'])
            }
    
    def get_performance_alerts(self) -> List[Dict[str, Any]]:
        """Get performance alerts based on thresholds."""
        alerts = []
        
        with self.lock:
            # High error rate alert
            if self.metrics.error_rate > 0.1:  # 10% error rate
                alerts.append({
                    'type': 'high_error_rate',
                    'severity': 'warning',
                    'message': f'Error rate is {self.metrics.error_rate:.2%}',
                    'value': self.metrics.error_rate
                })
            
            # Slow response time alert
            if self.metrics.average_response_time > 5.0:  # 5 seconds
                alerts.append({
                    'type': 'slow_response_time',
                    'severity': 'warning',
                    'message': f'Average response time is {self.metrics.average_response_time:.2f}s',
                    'value': self.metrics.average_response_time
                })
            
            # Low cache hit rate alert
            if self.metrics.cache_hit_rate < 0.2:  # 20% cache hit rate
                alerts.append({
                    'type': 'low_cache_hit_rate',
                    'severity': 'info',
                    'message': f'Cache hit rate is {self.metrics.cache_hit_rate:.2%}',
                    'value': self.metrics.cache_hit_rate
                })
        
        return alerts


# Global analytics instance
analytics_collector = AnalyticsCollector()


async def track_query(
    query: str,
    user_id: Optional[str],
    execution_time: float,
    response_size: int,
    confidence: float,
    cache_hit: bool,
    error_occurred: bool,
    error_type: Optional[str],
    agent_usage: Dict[str, Any],
    token_usage: Dict[str, int],
    user_agent: Optional[str],
    ip_address: Optional[str]
) -> None:
    """Track a query for analytics."""
    analytics = QueryAnalytics(
        query_id=f"q_{int(time.time() * 1000)}",
        query=query,
        user_id=user_id,
        timestamp=time.time(),
        execution_time=execution_time,
        response_size=response_size,
        confidence=confidence,
        cache_hit=cache_hit,
        error_occurred=error_occurred,
        error_type=error_type,
        agent_usage=agent_usage,
        token_usage=token_usage,
        user_agent=user_agent,
        ip_address=ip_address
    )
    
    analytics_collector.record_query(analytics)


def get_analytics_summary() -> Dict[str, Any]:
    """Get comprehensive analytics summary."""
    return {
        'system_metrics': analytics_collector.get_system_metrics(),
        'hourly_stats': analytics_collector.get_hourly_stats(),
        'popular_queries': analytics_collector.get_popular_queries(),
        'performance_alerts': analytics_collector.get_performance_alerts(),
        'cache_stats': get_cache_stats()
    } 