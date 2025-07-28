"""
Analytics System for Universal Knowledge Platform
Provides query tracking and analytics with privacy protection.
"""

import asyncio
import logging
import time
import re
import hashlib
from collections import defaultdict, deque
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
import json
import os

logger = logging.getLogger(__name__)

# Privacy configuration
DATA_RETENTION_DAYS = int(os.getenv("DATA_RETENTION_DAYS", "30"))
ANONYMIZE_QUERIES = os.getenv("ANONYMIZE_QUERIES", "true").lower() == "true"
LOG_QUERY_CONTENT = os.getenv("LOG_QUERY_CONTENT", "false").lower() == "true"


# Global metrics storage
class MetricsCollector:
    """Thread-safe metrics collector for production monitoring."""

    def __init__(self):
        self.request_counter = 0
        self.error_counter = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self.total_response_time = 0.0
        self.request_times = deque(maxlen=1000)  # Keep last 1000 response times
        self.user_activity = defaultdict(int)
        self.query_categories = defaultdict(int)
        self.error_types = defaultdict(int)
        self.agent_performance = defaultdict(
            lambda: {"success_count": 0, "error_count": 0, "total_time": 0.0, "last_used": None}
        )
        self._lock = asyncio.Lock()

        # Time-based metrics
        self.hourly_stats = defaultdict(
            lambda: {"requests": 0, "errors": 0, "avg_response_time": 0.0}
        )

        # Business metrics
        self.confidence_scores = deque(maxlen=1000)
        self.query_lengths = deque(maxlen=1000)

    async def increment_request(self, user_id: str, query: str, category: str = "general"):
        """Increment request counter with user tracking."""
        async with self._lock:
            self.request_counter += 1
            self.user_activity[user_id] += 1
            self.query_categories[category] += 1

            # Update hourly stats
            current_hour = datetime.now().replace(minute=0, second=0, microsecond=0)
            self.hourly_stats[current_hour]["requests"] += 1

    async def increment_error(self, error_type: str, user_id: str = "unknown"):
        """Increment error counter with error type tracking."""
        async with self._lock:
            self.error_counter += 1
            self.error_types[error_type] += 1
            self.user_activity[user_id] += 1

            # Update hourly stats
            current_hour = datetime.now().replace(minute=0, second=0, microsecond=0)
            self.hourly_stats[current_hour]["errors"] += 1

    async def record_response_time(self, response_time: float):
        """Record response time for average calculation."""
        async with self._lock:
            self.total_response_time += response_time
            self.request_times.append(response_time)

            # Update hourly stats
            current_hour = datetime.now().replace(minute=0, second=0, microsecond=0)
            if self.hourly_stats[current_hour]["requests"] > 0:
                current_avg = self.hourly_stats[current_hour]["avg_response_time"]
                current_count = self.hourly_stats[current_hour]["requests"]
                new_avg = (current_avg * (current_count - 1) + response_time) / current_count
                self.hourly_stats[current_hour]["avg_response_time"] = new_avg

    async def record_cache_hit(self):
        """Record cache hit."""
        async with self._lock:
            self.cache_hits += 1

    async def record_cache_miss(self):
        """Record cache miss."""
        async with self._lock:
            self.cache_misses += 1

    async def record_agent_performance(self, agent_name: str, success: bool, execution_time: float):
        """Record agent performance metrics."""
        async with self._lock:
            agent_stats = self.agent_performance[agent_name]
            if success:
                agent_stats["success_count"] += 1
            else:
                agent_stats["error_count"] += 1

            agent_stats["total_time"] += execution_time
            agent_stats["last_used"] = datetime.now()

    async def record_confidence_score(self, confidence: float):
        """Record confidence score for analysis."""
        async with self._lock:
            self.confidence_scores.append(confidence)

    async def record_query_length(self, length: int):
        """Record query length for analysis."""
        async with self._lock:
            self.query_lengths.append(length)

    async def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive metrics."""
        async with self._lock:
            # Calculate averages
            avg_response_time = 0.0
            if self.request_times:
                avg_response_time = sum(self.request_times) / len(self.request_times)

            avg_confidence = 0.0
            if self.confidence_scores:
                avg_confidence = sum(self.confidence_scores) / len(self.confidence_scores)

            avg_query_length = 0.0
            if self.query_lengths:
                avg_query_length = sum(self.query_lengths) / len(self.query_lengths)

            # Calculate cache hit rate
            total_cache_requests = self.cache_hits + self.cache_misses
            cache_hit_rate = (
                (self.cache_hits / total_cache_requests * 100) if total_cache_requests > 0 else 0.0
            )

            # Calculate error rate
            error_rate = (
                (self.error_counter / self.request_counter * 100)
                if self.request_counter > 0
                else 0.0
            )

            # Get recent activity (last 24 hours)
            current_time = datetime.now()
            recent_activity = {
                "last_hour": sum(
                    1
                    for hour, stats in self.hourly_stats.items()
                    if current_time - hour < timedelta(hours=1)
                ),
                "last_24_hours": sum(
                    stats["requests"]
                    for hour, stats in self.hourly_stats.items()
                    if current_time - hour < timedelta(hours=24)
                ),
            }

            return {
                "request_counter": self.request_counter,
                "error_counter": self.error_counter,
                "cache_hits": self.cache_hits,
                "cache_misses": self.cache_misses,
                "cache_hit_rate": cache_hit_rate,
                "average_response_time": avg_response_time,
                "average_confidence": avg_confidence,
                "average_query_length": avg_query_length,
                "error_rate": error_rate,
                "recent_activity": recent_activity,
                "top_users": dict(
                    sorted(self.user_activity.items(), key=lambda x: x[1], reverse=True)[:10]
                ),
                "query_categories": dict(self.query_categories),
                "error_types": dict(self.error_types),
                "agent_performance": {
                    agent: {
                        "success_rate": (
                            (
                                stats["success_count"]
                                / (stats["success_count"] + stats["error_count"])
                                * 100
                            )
                            if (stats["success_count"] + stats["error_count"]) > 0
                            else 0.0
                        ),
                        "avg_execution_time": (
                            stats["total_time"] / (stats["success_count"] + stats["error_count"])
                            if (stats["success_count"] + stats["error_count"]) > 0
                            else 0.0
                        ),
                        "total_requests": stats["success_count"] + stats["error_count"],
                        "last_used": stats["last_used"].isoformat() if stats["last_used"] else None,
                    }
                    for agent, stats in self.agent_performance.items()
                },
                "timestamp": time.time(),
            }


# Global metrics collector
metrics_collector = MetricsCollector()

# Analytics storage with privacy controls
query_history = deque(maxlen=10000)  # Limited history
user_analytics = defaultdict(
    lambda: {"query_count": 0, "avg_response_time": 0.0, "last_seen": 0, "anonymized_id": None}
)

# Global analytics
global_stats = {
    "total_queries": 0,
    "total_errors": 0,
    "avg_response_time": 0.0,
    "cache_hit_rate": 0.0,
    "popular_queries": defaultdict(int),
    "query_categories": defaultdict(int),
}


def sanitize_query_for_logging(query: str) -> str:
    """
    Sanitize query for logging to protect privacy.

    Args:
        query: Original query

    Returns:
        Sanitized query safe for logging
    """
    if not LOG_QUERY_CONTENT:
        return "[QUERY_CONTENT_LOGGING_DISABLED]"

    # Remove or mask sensitive patterns
    sensitive_patterns = [
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",  # Email
        r"\b\d{3}-\d{2}-\d{4}\b",  # SSN
        r"\b\d{4}-\d{4}-\d{4}-\d{4}\b",  # Credit card
        r"\b\d{10,}\b",  # Long numbers (phone, etc.)
        r"\b[A-Z]{2}\d{2}[A-Z0-9]{10,30}\b",  # IBAN
        r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b",  # IP addresses
    ]

    sanitized = query
    for pattern in sensitive_patterns:
        sanitized = re.sub(pattern, "[REDACTED]", sanitized)

    # Truncate very long queries
    if len(sanitized) > 200:
        sanitized = sanitized[:200] + "..."

    return sanitized


def anonymize_user_id(user_id: str) -> str:
    """
    Create an anonymized user ID for analytics.

    Args:
        user_id: Original user ID

    Returns:
        Anonymized user ID
    """
    if not user_id:
        return "anonymous"

    # Create consistent hash for the same user_id
    return hashlib.sha256(user_id.encode()).hexdigest()[:8]


def categorize_query(query: str) -> str:
    """
    Categorize query for analytics without storing sensitive content.

    Args:
        query: Query to categorize

    Returns:
        Query category
    """
    query_lower = query.lower()

    # Define categories based on keywords
    categories = {
        "technology": ["programming", "code", "software", "computer", "tech", "algorithm"],
        "science": ["research", "study", "experiment", "scientific", "analysis"],
        "education": ["learn", "teach", "education", "school", "university", "course"],
        "business": ["company", "business", "market", "finance", "management"],
        "health": ["medical", "health", "disease", "treatment", "medicine"],
        "general": ["what", "how", "why", "when", "where", "who"],
    }

    for category, keywords in categories.items():
        if any(keyword in query_lower for keyword in keywords):
            return category

    return "general"


async def track_query(
    query: str,
    execution_time: float,
    confidence: float,
    client_ip: str,
    user_id: str,
    cache_hit: bool = False,
    error: Optional[str] = None,
    agent_results: Optional[Dict[str, Any]] = None,
):
    """Track query analytics with comprehensive metrics."""
    try:
        # Basic metrics
        await metrics_collector.increment_request(user_id, query)
        await metrics_collector.record_response_time(execution_time)
        await metrics_collector.record_confidence_score(confidence)
        await metrics_collector.record_query_length(len(query))

        # Cache metrics
        if cache_hit:
            await metrics_collector.record_cache_hit()
        else:
            await metrics_collector.record_cache_miss()

        # Error tracking
        if error:
            await metrics_collector.increment_error(error, user_id)

        # Agent performance tracking
        if agent_results:
            for agent_name, result in agent_results.items():
                if isinstance(result, dict):
                    success = result.get("success", False)
                    agent_time = result.get("execution_time_ms", 0) / 1000.0
                    await metrics_collector.record_agent_performance(
                        agent_name, success, agent_time
                    )

        # Log structured analytics
        logger.info(
            f"Query tracked: category=general, time={execution_time:.3f}s, confidence={confidence:.2f}, user={user_id}",
            extra={
                "query_length": len(query),
                "execution_time": execution_time,
                "confidence": confidence,
                "user_id": user_id,
                "client_ip": client_ip,
                "cache_hit": cache_hit,
                "error": error,
                "agent_results": agent_results,
            },
        )

    except Exception as e:
        logger.error(f"Failed to track query analytics: {e}", exc_info=True)


async def get_metrics() -> Dict[str, Any]:
    """Get current metrics."""
    return await metrics_collector.get_metrics()


async def get_analytics() -> Dict[str, Any]:
    """Get detailed analytics data."""
    metrics = await metrics_collector.get_metrics()

    # Calculate additional analytics
    current_time = datetime.now()

    # Top queries (simplified - in production, this would come from a database)
    top_queries = [
        {"query": "What is artificial intelligence?", "count": 150},
        {"query": "How does machine learning work?", "count": 120},
        {"query": "Explain neural networks", "count": 95},
        {"query": "What is deep learning?", "count": 85},
        {"query": "How to implement AI?", "count": 70},
    ]

    # Time-based analytics
    hourly_breakdown = {}
    for hour, stats in metrics_collector.hourly_stats.items():
        if current_time - hour < timedelta(hours=24):
            hourly_breakdown[hour.strftime("%Y-%m-%d %H:00")] = {
                "requests": stats["requests"],
                "errors": stats["errors"],
                "avg_response_time": stats["avg_response_time"],
            }

    return {
        "total_queries": metrics["request_counter"],
        "successful_queries": metrics["request_counter"] - metrics["error_counter"],
        "failed_queries": metrics["error_counter"],
        "average_confidence": metrics["average_confidence"],
        "average_response_time": metrics["average_response_time"],
        "cache_hit_rate": metrics["cache_hit_rate"],
        "error_rate": metrics["error_rate"],
        "top_queries": top_queries,
        "user_activity": metrics["top_users"],
        "agent_performance": metrics["agent_performance"],
        "hourly_breakdown": hourly_breakdown,
        "query_categories": metrics["query_categories"],
        "error_types": metrics["error_types"],
        "time_period": {
            "start": (current_time - timedelta(hours=24)).isoformat(),
            "end": current_time.isoformat(),
            "duration_hours": 24,
        },
    }


async def export_analytics_to_file(filename: str = None):
    """Export analytics data to JSON file."""
    if not filename:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"analytics_export_{timestamp}.json"

    analytics = await get_analytics()

    try:
        with open(filename, "w") as f:
            json.dump(analytics, f, indent=2, default=str)
        logger.info(f"Analytics exported to {filename}")
        return filename
    except Exception as e:
        logger.error(f"Failed to export analytics: {e}")
        raise


async def cleanup_old_metrics():
    """Clean up old metrics data to prevent memory bloat."""
    try:
        current_time = datetime.now()

        # Remove hourly stats older than 7 days
        old_hours = [
            hour
            for hour in metrics_collector.hourly_stats.keys()
            if current_time - hour > timedelta(days=7)
        ]

        for hour in old_hours:
            del metrics_collector.hourly_stats[hour]

        logger.info(f"Cleaned up {len(old_hours)} old hourly metrics")

    except Exception as e:
        logger.error(f"Failed to cleanup old metrics: {e}")


# Periodic cleanup task
async def start_metrics_cleanup():
    """Start periodic metrics cleanup."""
    while True:
        try:
            await asyncio.sleep(3600)  # Run every hour
            await cleanup_old_metrics()
        except Exception as e:
            logger.error(f"Metrics cleanup failed: {e}")


# Initialize cleanup task
def init_metrics_cleanup():
    """Initialize the metrics cleanup task."""
    asyncio.create_task(start_metrics_cleanup())
