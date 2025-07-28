"""
Advanced Analytics and Business Intelligence - MAANG Standards.

This module implements comprehensive analytics and business intelligence
following MAANG best practices for data-driven decision making.

Features:
    - Real-time analytics processing
    - Business metrics tracking
    - User behavior analysis
    - Performance analytics
    - Predictive analytics
    - A/B testing framework
    - Custom event tracking
    - Data visualization support
    - Machine learning integration
    - Anomaly detection

Analytics Categories:
    - User Engagement Metrics
    - Performance Metrics
    - Business Metrics
    - Security Metrics
    - System Health Metrics
    - Predictive Analytics

Authors:
    - Universal Knowledge Platform Engineering Team
    
Version:
    2.0.0 (2024-12-28)
"""

import asyncio
import json
import hashlib
from typing import (
    Optional, Dict, Any, List, Union, Callable,
    TypeVar, Protocol, Tuple, Set
)
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass, field
from enum import Enum
import structlog
from collections import defaultdict, deque
import statistics
import numpy as np
from scipy import stats

from api.monitoring import Counter, Histogram, Gauge
from api.cache import get_cache_manager
from api.config import get_settings

logger = structlog.get_logger(__name__)

# Type definitions
T = TypeVar('T')

# Analytics metrics
analytics_events = Counter(
    'analytics_events_total',
    'Analytics events',
    ['event_type', 'category', 'user_type']
)

user_engagement = Histogram(
    'user_engagement_duration_seconds',
    'User engagement duration',
    ['session_type', 'user_type'],
    buckets=(1, 5, 10, 30, 60, 300, 600, 1800, 3600)
)

business_metrics = {
    'revenue': Gauge('business_revenue_total', 'Total revenue', ['currency', 'period']),
    'conversion_rate': Gauge('business_conversion_rate', 'Conversion rate', ['funnel_stage']),
    'customer_satisfaction': Gauge('customer_satisfaction_score', 'Customer satisfaction', ['survey_type']),
    'retention_rate': Gauge('user_retention_rate', 'User retention rate', ['cohort', 'period'])
}

# Analytics event types
class EventType(str, Enum):
    """Types of analytics events."""
    PAGE_VIEW = "page_view"
    QUERY_SUBMITTED = "query_submitted"
    QUERY_COMPLETED = "query_completed"
    USER_LOGIN = "user_login"
    USER_LOGOUT = "user_logout"
    FEEDBACK_SUBMITTED = "feedback_submitted"
    ERROR_OCCURRED = "error_occurred"
    FEATURE_USED = "feature_used"
    PERFORMANCE_ISSUE = "performance_issue"
    SECURITY_EVENT = "security_event"

class EventCategory(str, Enum):
    """Categories of analytics events."""
    USER_BEHAVIOR = "user_behavior"
    PERFORMANCE = "performance"
    BUSINESS = "business"
    SECURITY = "security"
    SYSTEM = "system"

# Analytics event model
@dataclass
class AnalyticsEvent:
    """Analytics event model."""
    
    event_type: EventType
    category: EventCategory
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    properties: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            'event_type': self.event_type.value,
            'category': self.category.value,
            'user_id': self.user_id,
            'session_id': self.session_id,
            'timestamp': self.timestamp.isoformat(),
            'properties': self.properties,
            'metadata': self.metadata
        }

# User behavior tracking
@dataclass
class UserSession:
    """User session tracking."""
    
    session_id: str
    user_id: Optional[str]
    start_time: datetime
    end_time: Optional[datetime] = None
    events: List[AnalyticsEvent] = field(default_factory=list)
    properties: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def duration(self) -> Optional[timedelta]:
        """Calculate session duration."""
        if not self.end_time:
            return None
        return self.end_time - self.start_time
    
    @property
    def event_count(self) -> int:
        """Get total event count."""
        return len(self.events)

# Business metrics tracking
@dataclass
class BusinessMetric:
    """Business metric model."""
    
    name: str
    value: float
    unit: str
    timestamp: datetime
    dimensions: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

# Advanced analytics processor
class AnalyticsProcessor:
    """
    Advanced analytics processing system.
    
    Features:
    - Real-time event processing
    - User behavior analysis
    - Business metrics calculation
    - Predictive analytics
    - Anomaly detection
    """
    
    def __init__(self):
        """Initialize analytics processor."""
        self.settings = get_settings()
        self.cache_manager = get_cache_manager()
        
        # Event processing queues
        self.event_queue: deque = deque(maxlen=10000)
        self.batch_size = 100
        self.processing_interval = 5  # seconds
        
        # Analytics storage
        self.user_sessions: Dict[str, UserSession] = {}
        self.business_metrics: List[BusinessMetric] = []
        self.event_aggregates: Dict[str, Dict[str, Any]] = defaultdict(dict)
        
        # Processing state
        self._processing = False
        self._processor_task: Optional[asyncio.Task] = None
    
    async def start_processing(self) -> None:
        """Start analytics processing."""
        if self._processing:
            return
        
        self._processing = True
        self._processor_task = asyncio.create_task(self._process_events())
        logger.info("Analytics processing started")
    
    async def stop_processing(self) -> None:
        """Stop analytics processing."""
        self._processing = False
        if self._processor_task:
            self._processor_task.cancel()
            try:
                await self._processor_task
            except asyncio.CancelledError:
                pass
        logger.info("Analytics processing stopped")
    
    async def track_event(self, event: AnalyticsEvent) -> None:
        """Track an analytics event."""
        # Add to processing queue
        self.event_queue.append(event)
        
        # Update session if applicable
        if event.session_id:
            await self._update_session(event)
        
        # Record metrics
        analytics_events.labels(
            event_type=event.event_type.value,
            category=event.category.value,
            user_type=self._get_user_type(event.user_id)
        ).inc()
        
        # Process immediately for critical events
        if event.category in [EventCategory.SECURITY, EventCategory.PERFORMANCE]:
            await self._process_event_immediately(event)
    
    async def _update_session(self, event: AnalyticsEvent) -> None:
        """Update user session with event."""
        session_id = event.session_id
        if not session_id:
            return
        
        if session_id not in self.user_sessions:
            # Create new session
            self.user_sessions[session_id] = UserSession(
                session_id=session_id,
                user_id=event.user_id,
                start_time=event.timestamp
            )
        
        session = self.user_sessions[session_id]
        session.events.append(event)
        
        # Update session properties
        if 'query_count' not in session.properties:
            session.properties['query_count'] = 0
        
        if event.event_type == EventType.QUERY_SUBMITTED:
            session.properties['query_count'] += 1
    
    async def _process_event_immediately(self, event: AnalyticsEvent) -> None:
        """Process critical events immediately."""
        if event.category == EventCategory.SECURITY:
            await self._handle_security_event(event)
        elif event.category == EventCategory.PERFORMANCE:
            await self._handle_performance_event(event)
    
    async def _handle_security_event(self, event: AnalyticsEvent) -> None:
        """Handle security events."""
        logger.warning(
            "Security event detected",
            event_type=event.event_type.value,
            user_id=event.user_id,
            properties=event.properties
        )
        
        # Update security metrics
        security_events = self.event_aggregates.get('security', {})
        security_events[event.event_type.value] = security_events.get(event.event_type.value, 0) + 1
        self.event_aggregates['security'] = security_events
    
    async def _handle_performance_event(self, event: AnalyticsEvent) -> None:
        """Handle performance events."""
        logger.warning(
            "Performance issue detected",
            event_type=event.event_type.value,
            properties=event.properties
        )
        
        # Update performance metrics
        performance_events = self.event_aggregates.get('performance', {})
        performance_events[event.event_type.value] = performance_events.get(event.event_type.value, 0) + 1
        self.event_aggregates['performance'] = performance_events
    
    async def _process_events(self) -> None:
        """Process events in batches."""
        while self._processing:
            try:
                # Process batch of events
                batch = []
                while len(batch) < self.batch_size and self.event_queue:
                    batch.append(self.event_queue.popleft())
                
                if batch:
                    await self._process_batch(batch)
                
                await asyncio.sleep(self.processing_interval)
                
            except Exception as e:
                logger.error("Error processing analytics events", error=str(e))
                await asyncio.sleep(1)
    
    async def _process_batch(self, events: List[AnalyticsEvent]) -> None:
        """Process a batch of events."""
        # Aggregate events by type
        event_counts = defaultdict(int)
        user_activity = defaultdict(int)
        
        for event in events:
            event_counts[event.event_type.value] += 1
            if event.user_id:
                user_activity[event.user_id] += 1
        
        # Update aggregates
        for event_type, count in event_counts.items():
            self.event_aggregates['events'][event_type] = (
                self.event_aggregates['events'].get(event_type, 0) + count
            )
        
        # Calculate business metrics
        await self._calculate_business_metrics(events)
        
        # Store processed events
        await self._store_events(events)
        
        logger.debug(
            "Processed analytics batch",
            batch_size=len(events),
            event_types=dict(event_counts)
        )
    
    async def _calculate_business_metrics(self, events: List[AnalyticsEvent]) -> None:
        """Calculate business metrics from events."""
        now = datetime.now(timezone.utc)
        
        # Calculate conversion rate
        queries_submitted = sum(
            1 for e in events if e.event_type == EventType.QUERY_SUBMITTED
        )
        queries_completed = sum(
            1 for e in events if e.event_type == EventType.QUERY_COMPLETED
        )
        
        if queries_submitted > 0:
            conversion_rate = (queries_completed / queries_submitted) * 100
            business_metrics['conversion_rate'].labels(funnel_stage='query').set(conversion_rate)
        
        # Calculate user engagement
        for event in events:
            if event.event_type == EventType.USER_LOGIN:
                # Track session duration
                session_duration = event.properties.get('session_duration', 0)
                user_engagement.labels(
                    session_type='login',
                    user_type=self._get_user_type(event.user_id)
                ).observe(session_duration)
        
        # Calculate customer satisfaction
        feedback_events = [
            e for e in events if e.event_type == EventType.FEEDBACK_SUBMITTED
        ]
        
        if feedback_events:
            satisfaction_scores = [
                e.properties.get('rating', 0) for e in feedback_events
            ]
            avg_satisfaction = statistics.mean(satisfaction_scores)
            business_metrics['customer_satisfaction'].labels(
                survey_type='feedback'
            ).set(avg_satisfaction)
    
    async def _store_events(self, events: List[AnalyticsEvent]) -> None:
        """Store processed events."""
        # Store in cache for quick access
        cache_key = f"analytics:events:{datetime.now(timezone.utc).strftime('%Y%m%d')}"
        
        try:
            # Get existing events
            existing_events = await self.cache_manager.get(cache_key, [])
            if not isinstance(existing_events, list):
                existing_events = []
            
            # Add new events
            new_events = [event.to_dict() for event in events]
            existing_events.extend(new_events)
            
            # Store updated events
            await self.cache_manager.set(cache_key, existing_events, ttl=86400)  # 24 hours
            
        except Exception as e:
            logger.error("Error storing analytics events", error=str(e))
    
    def _get_user_type(self, user_id: Optional[str]) -> str:
        """Get user type for analytics."""
        if not user_id:
            return "anonymous"
        
        # Simple user type detection (can be enhanced)
        if user_id.startswith("admin"):
            return "admin"
        elif user_id.startswith("premium"):
            return "premium"
        else:
            return "standard"

# Predictive analytics
class PredictiveAnalytics:
    """
    Predictive analytics system.
    
    Features:
    - User behavior prediction
    - Performance forecasting
    - Anomaly detection
    - Trend analysis
    """
    
    def __init__(self):
        """Initialize predictive analytics."""
        self.models: Dict[str, Any] = {}
        self.predictions: Dict[str, List[float]] = defaultdict(list)
        self.anomaly_thresholds: Dict[str, float] = {}
    
    def detect_anomalies(self, data: List[float], window: int = 10) -> List[bool]:
        """Detect anomalies in time series data."""
        if len(data) < window:
            return [False] * len(data)
        
        anomalies = []
        for i in range(len(data)):
            if i < window:
                anomalies.append(False)
                continue
            
            # Calculate moving average and standard deviation
            window_data = data[i-window:i]
            mean = statistics.mean(window_data)
            std = statistics.stdev(window_data) if len(window_data) > 1 else 0
            
            # Detect anomaly (3-sigma rule)
            threshold = 3 * std
            is_anomaly = abs(data[i] - mean) > threshold
            
            anomalies.append(is_anomaly)
        
        return anomalies
    
    def forecast_trend(self, data: List[float], periods: int = 5) -> List[float]:
        """Forecast trend using simple linear regression."""
        if len(data) < 2:
            return [data[-1]] * periods if data else [0] * periods
        
        # Simple linear regression
        x = list(range(len(data)))
        y = data
        
        # Calculate slope and intercept
        n = len(x)
        sum_x = sum(x)
        sum_y = sum(y)
        sum_xy = sum(x[i] * y[i] for i in range(n))
        sum_xx = sum(x[i] * x[i] for i in range(n))
        
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x * sum_x)
        intercept = (sum_y - slope * sum_x) / n
        
        # Generate forecast
        forecast = []
        for i in range(periods):
            forecast_value = slope * (len(data) + i) + intercept
            forecast.append(max(0, forecast_value))  # Ensure non-negative
        
        return forecast
    
    def predict_user_behavior(
        self,
        user_events: List[AnalyticsEvent],
        prediction_type: str = "engagement"
    ) -> Dict[str, Any]:
        """Predict user behavior based on historical events."""
        if not user_events:
            return {"prediction": "insufficient_data"}
        
        # Extract features
        event_counts = defaultdict(int)
        session_durations = []
        query_counts = []
        
        for event in user_events:
            event_counts[event.event_type.value] += 1
            
            if event.event_type == EventType.USER_LOGIN:
                duration = event.properties.get('session_duration', 0)
                session_durations.append(duration)
            
            if event.event_type == EventType.QUERY_SUBMITTED:
                query_counts.append(1)
        
        # Make predictions based on patterns
        predictions = {}
        
        if prediction_type == "engagement":
            avg_session_duration = statistics.mean(session_durations) if session_durations else 0
            total_queries = sum(query_counts)
            
            # Simple engagement score
            engagement_score = min(100, (avg_session_duration / 3600) * 50 + total_queries * 10)
            
            predictions["engagement_score"] = engagement_score
            predictions["likely_retention"] = engagement_score > 50
        
        elif prediction_type == "usage":
            total_events = len(user_events)
            unique_days = len(set(e.timestamp.date() for e in user_events))
            
            # Usage prediction
            daily_usage = total_events / max(unique_days, 1)
            predictions["daily_usage_prediction"] = daily_usage
            predictions["usage_trend"] = "increasing" if daily_usage > 5 else "stable"
        
        return predictions

# A/B testing framework
class ABTestingFramework:
    """
    A/B testing framework for feature experimentation.
    
    Features:
    - Random assignment to variants
    - Statistical significance testing
    - Conversion tracking
    - Multivariate testing
    """
    
    def __init__(self):
        """Initialize A/B testing framework."""
        self.experiments: Dict[str, Dict[str, Any]] = {}
        self.assignments: Dict[str, str] = {}
        self.results: Dict[str, Dict[str, Any]] = defaultdict(dict)
    
    def create_experiment(
        self,
        experiment_id: str,
        variants: List[str],
        traffic_split: Optional[List[float]] = None
    ) -> None:
        """Create a new A/B test experiment."""
        if traffic_split is None:
            traffic_split = [1.0 / len(variants)] * len(variants)
        
        if len(variants) != len(traffic_split):
            raise ValueError("Variants and traffic split must have same length")
        
        self.experiments[experiment_id] = {
            'variants': variants,
            'traffic_split': traffic_split,
            'start_time': datetime.now(timezone.utc),
            'status': 'active'
        }
        
        logger.info(
            "A/B test experiment created",
            experiment_id=experiment_id,
            variants=variants,
            traffic_split=traffic_split
        )
    
    def get_variant(self, experiment_id: str, user_id: str) -> str:
        """Get assigned variant for user."""
        if experiment_id not in self.experiments:
            return 'control'  # Default variant
        
        # Check if user already has assignment
        assignment_key = f"{experiment_id}:{user_id}"
        if assignment_key in self.assignments:
            return self.assignments[assignment_key]
        
        # Create new assignment
        experiment = self.experiments[experiment_id]
        variants = experiment['variants']
        traffic_split = experiment['traffic_split']
        
        # Deterministic assignment based on user_id hash
        user_hash = int(hashlib.md5(user_id.encode()).hexdigest()[:8], 16)
        cumulative_prob = 0
        
        for i, prob in enumerate(traffic_split):
            cumulative_prob += prob
            if user_hash / (2**32) <= cumulative_prob:
                variant = variants[i]
                self.assignments[assignment_key] = variant
                return variant
        
        # Fallback to first variant
        variant = variants[0]
        self.assignments[assignment_key] = variant
        return variant
    
    def track_conversion(
        self,
        experiment_id: str,
        user_id: str,
        conversion_value: float = 1.0
    ) -> None:
        """Track conversion for A/B test."""
        variant = self.get_variant(experiment_id, user_id)
        
        if experiment_id not in self.results:
            self.results[experiment_id] = defaultdict(lambda: {
                'conversions': 0,
                'total_users': 0,
                'conversion_value': 0.0
            })
        
        self.results[experiment_id][variant]['conversions'] += 1
        self.results[experiment_id][variant]['conversion_value'] += conversion_value
    
    def get_experiment_results(self, experiment_id: str) -> Dict[str, Any]:
        """Get A/B test results with statistical significance."""
        if experiment_id not in self.results:
            return {"error": "Experiment not found"}
        
        results = self.results[experiment_id]
        experiment = self.experiments.get(experiment_id, {})
        
        # Calculate conversion rates
        conversion_rates = {}
        for variant, data in results.items():
            if data['total_users'] > 0:
                conversion_rates[variant] = data['conversions'] / data['total_users']
            else:
                conversion_rates[variant] = 0
        
        # Statistical significance test (chi-square)
        significance_results = {}
        variants = list(results.keys())
        
        if len(variants) >= 2:
            for i, variant1 in enumerate(variants):
                for variant2 in variants[i+1:]:
                    # Simple chi-square test
                    observed1 = results[variant1]['conversions']
                    observed2 = results[variant2]['conversions']
                    total1 = results[variant1]['total_users']
                    total2 = results[variant2]['total_users']
                    
                    if total1 > 0 and total2 > 0:
                        expected1 = (observed1 + observed2) * total1 / (total1 + total2)
                        expected2 = (observed1 + observed2) * total2 / (total1 + total2)
                        
                        chi_square = (
                            ((observed1 - expected1) ** 2) / expected1 +
                            ((observed2 - expected2) ** 2) / expected2
                        )
                        
                        # p-value approximation (simplified)
                        p_value = 1 - stats.chi2.cdf(chi_square, 1)
                        significance_results[f"{variant1}_vs_{variant2}"] = {
                            'p_value': p_value,
                            'significant': p_value < 0.05
                        }
        
        return {
            'experiment_id': experiment_id,
            'results': dict(results),
            'conversion_rates': conversion_rates,
            'significance': significance_results,
            'experiment_info': experiment
        }

# Global instances
_analytics_processor: Optional[AnalyticsProcessor] = None
_predictive_analytics: Optional[PredictiveAnalytics] = None
_ab_testing: Optional[ABTestingFramework] = None

def get_analytics_processor() -> AnalyticsProcessor:
    """Get global analytics processor instance."""
    global _analytics_processor
    
    if _analytics_processor is None:
        _analytics_processor = AnalyticsProcessor()
    
    return _analytics_processor

def get_predictive_analytics() -> PredictiveAnalytics:
    """Get global predictive analytics instance."""
    global _predictive_analytics
    
    if _predictive_analytics is None:
        _predictive_analytics = PredictiveAnalytics()
    
    return _predictive_analytics

def get_ab_testing() -> ABTestingFramework:
    """Get global A/B testing instance."""
    global _ab_testing
    
    if _ab_testing is None:
        _ab_testing = ABTestingFramework()
    
    return _ab_testing

# Analytics utilities
async def track_user_event(
    event_type: EventType,
    category: EventCategory,
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    properties: Optional[Dict[str, Any]] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> None:
    """Track a user analytics event."""
    processor = get_analytics_processor()
    
    event = AnalyticsEvent(
        event_type=event_type,
        category=category,
        user_id=user_id,
        session_id=session_id,
        properties=properties or {},
        metadata=metadata or {}
    )
    
    await processor.track_event(event)

async def get_analytics_summary(
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None
) -> Dict[str, Any]:
    """Get analytics summary for date range."""
    processor = get_analytics_processor()
    
    # Get cached analytics data
    cache_key = f"analytics:summary:{start_date.strftime('%Y%m%d') if start_date else 'all'}"
    cached_data = await processor.cache_manager.get(cache_key)
    
    if cached_data:
        return cached_data
    
    # Calculate summary (simplified)
    summary = {
        'total_events': len(processor.event_queue),
        'active_sessions': len(processor.user_sessions),
        'business_metrics': {
            'conversion_rate': business_metrics['conversion_rate']._value.get(),
            'customer_satisfaction': business_metrics['customer_satisfaction']._value.get()
        },
        'event_aggregates': dict(processor.event_aggregates),
        'timestamp': datetime.now(timezone.utc).isoformat()
    }
    
    # Cache summary
    await processor.cache_manager.set(cache_key, summary, ttl=3600)  # 1 hour
    
    return summary

# Export public API
__all__ = [
    # Classes
    'AnalyticsProcessor',
    'PredictiveAnalytics',
    'ABTestingFramework',
    'AnalyticsEvent',
    'UserSession',
    'BusinessMetric',
    
    # Enums
    'EventType',
    'EventCategory',
    
    # Functions
    'get_analytics_processor',
    'get_predictive_analytics',
    'get_ab_testing',
    'track_user_event',
    'get_analytics_summary',
] 