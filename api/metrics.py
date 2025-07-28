"""
Metrics Collection System for Universal Knowledge Platform
Provides Prometheus metrics for monitoring and observability.
"""

import time
import psutil
from typing import Dict, Any, Optional
from prometheus_client import (
    Counter,
    Gauge,
    Histogram,
    Summary,
    generate_latest,
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    multiprocess,
)
import logging
import os

logger = logging.getLogger(__name__)

# Create a custom registry for multiprocess support
registry = CollectorRegistry()

# Only use multiprocess collector if the environment variable is set
if os.getenv("PROMETHEUS_MULTIPROC_DIR"):
    multiprocess.MultiProcessCollector(registry)

# Request metrics
REQUEST_COUNT = Counter(
    "ukp_requests_total",
    "Total number of requests",
    ["method", "endpoint", "status_code"],
    registry=registry,
)

REQUEST_DURATION = Histogram(
    "ukp_request_duration_seconds",
    "Request duration in seconds",
    ["method", "endpoint"],
    registry=registry,
)

# Agent metrics
AGENT_REQUEST_COUNT = Counter(
    "ukp_agent_requests_total",
    "Total number of agent requests",
    ["agent_type", "status"],
    registry=registry,
)

AGENT_DURATION = Histogram(
    "ukp_agent_duration_seconds",
    "Agent processing duration in seconds",
    ["agent_type"],
    registry=registry,
)

# Cache metrics
CACHE_HITS = Counter(
    "ukp_cache_hits_total", "Total number of cache hits", ["cache_type"], registry=registry
)

CACHE_MISSES = Counter(
    "ukp_cache_misses_total", "Total number of cache misses", ["cache_type"], registry=registry
)

CACHE_SIZE = Gauge("ukp_cache_size", "Current cache size", ["cache_type"], registry=registry)

# Security metrics
SECURITY_THREATS = Counter(
    "ukp_security_threats_total",
    "Total number of security threats detected",
    ["threat_type", "severity"],
    registry=registry,
)

BLOCKED_REQUESTS = Counter(
    "ukp_blocked_requests_total", "Total number of blocked requests", ["reason"], registry=registry
)

# Token usage metrics
TOKEN_USAGE = Counter(
    "ukp_token_usage_total", "Total token usage", ["agent_type", "token_type"], registry=registry
)

# System metrics
SYSTEM_MEMORY = Gauge("ukp_system_memory_bytes", "System memory usage in bytes", registry=registry)

SYSTEM_CPU = Gauge("ukp_system_cpu_percent", "System CPU usage percentage", registry=registry)

ACTIVE_CONNECTIONS = Gauge(
    "ukp_active_connections", "Number of active connections", registry=registry
)

# Error metrics
ERROR_COUNT = Counter(
    "ukp_errors_total", "Total number of errors", ["error_type", "endpoint"], registry=registry
)

# Business metrics
QUERY_CONFIDENCE = Histogram(
    "ukp_query_confidence", "Query confidence scores", ["query_category"], registry=registry
)

RESPONSE_LENGTH = Histogram(
    "ukp_response_length_chars", "Response length in characters", ["endpoint"], registry=registry
)


class MetricsCollector:
    """Centralized metrics collection."""

    def __init__(self):
        self.start_time = time.time()
        self._update_system_metrics()

    def record_request(self, method: str, endpoint: str, status_code: int, duration: float):
        """Record request metrics."""
        REQUEST_COUNT.labels(method=method, endpoint=endpoint, status_code=status_code).inc()
        REQUEST_DURATION.labels(method=method, endpoint=endpoint).observe(duration)

    def record_agent_request(self, agent_type: str, status: str, duration: float):
        """Record agent request metrics."""
        AGENT_REQUEST_COUNT.labels(agent_type=agent_type, status=status).inc()
        AGENT_DURATION.labels(agent_type=agent_type).observe(duration)

    def record_cache_hit(self, cache_type: str):
        """Record cache hit."""
        CACHE_HITS.labels(cache_type=cache_type).inc()

    def record_cache_miss(self, cache_type: str):
        """Record cache miss."""
        CACHE_MISSES.labels(cache_type=cache_type).inc()

    def update_cache_size(self, cache_type: str, size: int):
        """Update cache size metric."""
        CACHE_SIZE.labels(cache_type=cache_type).set(size)

    def record_security_threat(self, threat_type: str, severity: str):
        """Record security threat."""
        SECURITY_THREATS.labels(threat_type=threat_type, severity=severity).inc()

    def record_blocked_request(self, reason: str):
        """Record blocked request."""
        BLOCKED_REQUESTS.labels(reason=reason).inc()

    def record_token_usage(self, agent_type: str, token_type: str, count: int):
        """Record token usage."""
        TOKEN_USAGE.labels(agent_type=agent_type, token_type=token_type).inc(count)

    def record_error(self, error_type: str, endpoint: str):
        """Record error."""
        ERROR_COUNT.labels(error_type=error_type, endpoint=endpoint).inc()

    def record_query_confidence(self, category: str, confidence: float):
        """Record query confidence."""
        QUERY_CONFIDENCE.labels(query_category=category).observe(confidence)

    def record_response_length(self, endpoint: str, length: int):
        """Record response length."""
        RESPONSE_LENGTH.labels(endpoint=endpoint).observe(length)

    def _update_system_metrics(self):
        """Update system metrics."""
        try:
            # Memory usage
            memory = psutil.virtual_memory()
            SYSTEM_MEMORY.set(memory.used)

            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            SYSTEM_CPU.set(cpu_percent)

            # Active connections (approximation)
            connections = len(psutil.net_connections())
            ACTIVE_CONNECTIONS.set(connections)

        except Exception as e:
            logger.warning(f"Failed to update system metrics: {e}")

    def get_metrics(self) -> str:
        """Get Prometheus metrics."""
        self._update_system_metrics()
        return generate_latest(registry)

    def get_metrics_dict(self) -> Dict[str, Any]:
        """Get metrics as dictionary for API responses."""
        return {
            "uptime_seconds": time.time() - self.start_time,
            "system": {
                "memory_used_bytes": psutil.virtual_memory().used,
                "cpu_percent": psutil.cpu_percent(),
                "active_connections": len(psutil.net_connections()),
            },
            "requests": {
                "total": REQUEST_COUNT._value.sum(),
                "duration_avg": REQUEST_DURATION._sum.sum() / max(REQUEST_DURATION._count.sum(), 1),
            },
            "cache": {
                "hits": CACHE_HITS._value.sum(),
                "misses": CACHE_MISSES._value.sum(),
                "hit_rate": CACHE_HITS._value.sum()
                / max(CACHE_HITS._value.sum() + CACHE_MISSES._value.sum(), 1),
            },
            "security": {
                "threats": SECURITY_THREATS._value.sum(),
                "blocked_requests": BLOCKED_REQUESTS._value.sum(),
            },
            "errors": {"total": ERROR_COUNT._value.sum()},
        }


# Global metrics collector
metrics_collector = MetricsCollector()


def get_metrics_collector() -> MetricsCollector:
    """Get the global metrics collector."""
    return metrics_collector


def record_request_metrics(method: str, endpoint: str, status_code: int, duration: float):
    """Record request metrics."""
    metrics_collector.record_request(method, endpoint, status_code, duration)


def record_agent_metrics(agent_type: str, status: str, duration: float):
    """Record agent metrics."""
    metrics_collector.record_agent_request(agent_type, status, duration)


def record_cache_metrics(cache_type: str, hit: bool, size: Optional[int] = None):
    """Record cache metrics."""
    if hit:
        metrics_collector.record_cache_hit(cache_type)
    else:
        metrics_collector.record_cache_miss(cache_type)

    if size is not None:
        metrics_collector.update_cache_size(cache_type, size)


def record_security_metrics(threat_type: str, severity: str, blocked: bool = False):
    """Record security metrics."""
    metrics_collector.record_security_threat(threat_type, severity)
    if blocked:
        metrics_collector.record_blocked_request(threat_type)


def record_token_metrics(agent_type: str, prompt_tokens: int, completion_tokens: int):
    """Record token usage metrics."""
    metrics_collector.record_token_usage(agent_type, "prompt", prompt_tokens)
    metrics_collector.record_token_usage(agent_type, "completion", completion_tokens)


def record_error_metrics(error_type: str, endpoint: str):
    """Record error metrics."""
    metrics_collector.record_error(error_type, endpoint)


def record_business_metrics(category: str, confidence: float, response_length: int, endpoint: str):
    """Record business metrics."""
    metrics_collector.record_query_confidence(category, confidence)
    metrics_collector.record_response_length(endpoint, response_length)
