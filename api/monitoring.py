"""
Monitoring and Metrics Module - MAANG Standards.

This module implements comprehensive monitoring, metrics collection, and
observability features following MAANG best practices.

Features:
    - Prometheus metrics with labels
    - Custom business metrics
    - Performance profiling
    - Distributed tracing
    - Health check probes
    - SLI/SLO tracking
    - Alerting rules
    - Dashboard definitions

Architecture:
    - Metric collectors for different components
    - Aggregation and windowing
    - Export to multiple backends
    - Low-overhead instrumentation

Authors:
    - Universal Knowledge Platform Engineering Team
    
Version:
    2.0.0 (2024-12-28)
"""

import time
import asyncio
import functools
import threading
import sys
from typing import (
    Optional, Dict, Any, List, Callable, Union, 
    TypeVar, Awaitable, Protocol, Type
)
from datetime import datetime, timedelta, timezone
from contextlib import contextmanager, asynccontextmanager
from dataclasses import dataclass, field
from enum import Enum
import psutil
import structlog

from prometheus_client import (
    Counter, Histogram, Gauge, Summary, Info,
    CollectorRegistry, generate_latest, push_to_gateway,
    CONTENT_TYPE_LATEST
)
from opentelemetry import trace, metrics
from opentelemetry.trace import Status, StatusCode
try:
    from opentelemetry.instrumentation.logging import LoggingInstrumentor
    LoggingInstrumentor().instrument()
except ImportError:
    logger.warning("OpenTelemetry logging instrumentation not available")

logger = structlog.get_logger(__name__)

# Type definitions
T = TypeVar('T')
F = TypeVar('F', bound=Callable[..., Any])

# Metric types
class MetricType(str, Enum):
    """Types of metrics for different use cases."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"
    INFO = "info"

# SLI types
class SLIType(str, Enum):
    """Service Level Indicator types."""
    AVAILABILITY = "availability"
    LATENCY = "latency"
    ERROR_RATE = "error_rate"
    THROUGHPUT = "throughput"
    QUALITY = "quality"

# Global registry
registry = CollectorRegistry()

# Core metrics
request_counter = Counter(
    'http_requests_total',
    'Total HTTP requests',
    ['method', 'endpoint', 'status_code', 'service'],
    registry=registry
)

request_duration = Histogram(
    'http_request_duration_seconds',
    'HTTP request duration',
    ['method', 'endpoint', 'service'],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
    registry=registry
)

active_requests = Gauge(
    'http_requests_active',
    'Active HTTP requests',
    ['endpoint', 'service'],
    registry=registry
)

error_counter = Counter(
    'errors_total',
    'Total errors',
    ['error_type', 'severity', 'service'],
    registry=registry
)

# Business metrics
user_activity = Counter(
    'user_activity_total',
    'User activity events',
    ['event_type', 'user_role'],
    registry=registry
)

query_performance = Histogram(
    'query_duration_seconds',
    'Query execution time',
    ['query_type', 'model'],
    buckets=(0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0),
    registry=registry
)

cache_metrics = {
    'hits': Counter(
        'cache_hits_total',
        'Cache hits',
        ['cache_name', 'operation'],
        registry=registry
    ),
    'misses': Counter(
        'cache_misses_total',
        'Cache misses',
        ['cache_name', 'operation'],
        registry=registry
    ),
    'evictions': Counter(
        'cache_evictions_total',
        'Cache evictions',
        ['cache_name', 'reason'],
        registry=registry
    )
}

# System metrics
system_info = Info(
    'system_info',
    'System information',
    registry=registry
)

cpu_usage = Gauge(
    'system_cpu_usage_percent',
    'CPU usage percentage',
    ['cpu_core'],
    registry=registry
)

memory_usage = Gauge(
    'system_memory_usage_bytes',
    'Memory usage in bytes',
    ['memory_type'],
    registry=registry
)

disk_usage = Gauge(
    'system_disk_usage_bytes',
    'Disk usage in bytes',
    ['mount_point', 'usage_type'],
    registry=registry
)

# Decorators
def track_time(
    metric: Optional[Histogram] = None,
    labels: Optional[Dict[str, str]] = None
) -> Callable[[F], F]:
    """
    Decorator to track execution time.
    
    Args:
        metric: Histogram metric to record to
        labels: Static labels to add
        
    Example:
        @track_time(query_performance, {"query_type": "search"})
        async def search_knowledge(query: str) -> List[Result]:
            ...
    """
    def decorator(func: F) -> F:
        metric_to_use = metric or request_duration
        
        if asyncio.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                start_time = time.time()
                try:
                    result = await func(*args, **kwargs)
                    return result
                finally:
                    duration = time.time() - start_time
                    if labels:
                        metric_to_use.labels(**labels).observe(duration)
                    else:
                        metric_to_use.observe(duration)
            return async_wrapper
        else:
            @functools.wraps(func)
            def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                start_time = time.time()
                try:
                    result = func(*args, **kwargs)
                    return result
                finally:
                    duration = time.time() - start_time
                    if labels:
                        metric_to_use.labels(**labels).observe(duration)
                    else:
                        metric_to_use.observe(duration)
            return sync_wrapper
    
    return decorator

def track_errors(
    severity: str = "medium",
    service: str = "api"
) -> Callable[[F], F]:
    """
    Decorator to track errors.
    
    Args:
        severity: Error severity level
        service: Service name
    """
    def decorator(func: F) -> F:
        if asyncio.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    error_counter.labels(
                        error_type=type(e).__name__,
                        severity=severity,
                        service=service
                    ).inc()
                    raise
            return async_wrapper
        else:
            @functools.wraps(func)
            def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    error_counter.labels(
                        error_type=type(e).__name__,
                        severity=severity,
                        service=service
                    ).inc()
                    raise
            return sync_wrapper
    
    return decorator

def track_endpoint_usage(
    func: Optional[F] = None,
    *,
    service: str = "api"
) -> Union[F, Callable[[F], F]]:
    """
    Decorator to track endpoint usage metrics.
    
    Can be used with or without parentheses:
        @track_endpoint_usage
        @track_endpoint_usage(service="auth")
    """
    def decorator(f: F) -> F:
        @functools.wraps(f)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Extract request info from FastAPI
            request = kwargs.get('request') or (args[0] if args else None)
            if hasattr(request, 'method') and hasattr(request, 'url'):
                method = request.method
                endpoint = request.url.path
                
                # Track active requests
                active_requests.labels(
                    endpoint=endpoint,
                    service=service
                ).inc()
                
                start_time = time.time()
                status_code = 200
                
                try:
                    result = await f(*args, **kwargs)
                    return result
                except Exception as e:
                    status_code = getattr(e, 'status_code', 500)
                    raise
                finally:
                    # Record metrics
                    duration = time.time() - start_time
                    
                    request_counter.labels(
                        method=method,
                        endpoint=endpoint,
                        status_code=status_code,
                        service=service
                    ).inc()
                    
                    request_duration.labels(
                        method=method,
                        endpoint=endpoint,
                        service=service
                    ).observe(duration)
                    
                    active_requests.labels(
                        endpoint=endpoint,
                        service=service
                    ).dec()
            else:
                # Fallback if no request object
                return await f(*args, **kwargs)
        
        return wrapper
    
    if func is None:
        return decorator
    else:
        return decorator(func)

# Context managers
@contextmanager
def track_operation(
    operation: str,
    labels: Optional[Dict[str, str]] = None
) -> Any:
    """
    Context manager to track operation metrics.
    
    Example:
        with track_operation("database_query", {"query_type": "select"}):
            results = await db.execute(query)
    """
    start_time = time.time()
    
    # Create operation-specific metrics if not exists
    if operation not in _operation_metrics:
        _operation_metrics[operation] = {
            'duration': Histogram(
                f'{operation}_duration_seconds',
                f'{operation} duration',
                list(labels.keys()) if labels else [],
                registry=registry
            ),
            'active': Gauge(
                f'{operation}_active',
                f'Active {operation} operations',
                list(labels.keys()) if labels else [],
                registry=registry
            )
        }
    
    metrics = _operation_metrics[operation]
    
    # Increment active operations
    if labels:
        metrics['active'].labels(**labels).inc()
    else:
        metrics['active'].inc()
    
    try:
        yield
    finally:
        # Record duration
        duration = time.time() - start_time
        if labels:
            metrics['duration'].labels(**labels).observe(duration)
            metrics['active'].labels(**labels).dec()
        else:
            metrics['duration'].observe(duration)
            metrics['active'].dec()

@asynccontextmanager
async def track_async_operation(
    operation: str,
    labels: Optional[Dict[str, str]] = None
) -> Any:
    """Async version of track_operation."""
    start_time = time.time()
    
    if operation not in _operation_metrics:
        _operation_metrics[operation] = {
            'duration': Histogram(
                f'{operation}_duration_seconds',
                f'{operation} duration',
                list(labels.keys()) if labels else [],
                registry=registry
            ),
            'active': Gauge(
                f'{operation}_active',
                f'Active {operation} operations',
                list(labels.keys()) if labels else [],
                registry=registry
            )
        }
    
    metrics = _operation_metrics[operation]
    
    if labels:
        metrics['active'].labels(**labels).inc()
    else:
        metrics['active'].inc()
    
    try:
        yield
    finally:
        duration = time.time() - start_time
        if labels:
            metrics['duration'].labels(**labels).observe(duration)
            metrics['active'].labels(**labels).dec()
        else:
            metrics['duration'].observe(duration)
            metrics['active'].dec()

# SLI/SLO tracking
@dataclass
class SLI:
    """Service Level Indicator definition."""
    
    name: str
    type: SLIType
    target: float  # SLO target (e.g., 0.999 for 99.9%)
    window: timedelta = field(default_factory=lambda: timedelta(days=30))
    
    def __post_init__(self) -> None:
        """Initialize metrics for this SLI."""
        self.success_counter = Counter(
            f'sli_{self.name}_success_total',
            f'Successful events for {self.name}',
            registry=registry
        )
        self.total_counter = Counter(
            f'sli_{self.name}_total',
            f'Total events for {self.name}',
            registry=registry
        )
        self.current_value = Gauge(
            f'sli_{self.name}_current',
            f'Current value for {self.name}',
            registry=registry
        )
    
    def record_success(self) -> None:
        """Record a successful event."""
        self.success_counter.inc()
        self.total_counter.inc()
        self._update_current_value()
    
    def record_failure(self) -> None:
        """Record a failed event."""
        self.total_counter.inc()
        self._update_current_value()
    
    def _update_current_value(self) -> None:
        """Update current SLI value."""
        # In production, this would query the time-series database
        # For now, we'll use a simple approximation
        try:
            current = self.success_counter._value.get() / self.total_counter._value.get()
            self.current_value.set(current)
        except ZeroDivisionError:
            self.current_value.set(1.0)
    
    def get_error_budget(self) -> float:
        """Get remaining error budget as percentage."""
        current = self.current_value._value.get()
        if current >= self.target:
            return 100.0
        else:
            used = (self.target - current) / (1 - self.target)
            return max(0, 100 * (1 - used))

# System metrics collector
class SystemMetricsCollector:
    """Collects system-level metrics."""
    
    def __init__(self, interval: float = 60.0) -> None:
        """
        Initialize system metrics collector.
        
        Args:
            interval: Collection interval in seconds
        """
        self.interval = interval
        self._running = False
        self._task: Optional[asyncio.Task] = None
        
        # Set system info
        system_info.info({
            'platform': psutil.LINUX if hasattr(psutil, 'LINUX') else 'unknown',
            'python_version': '.'.join(map(str, sys.version_info[:3])),
            'cpu_count': str(psutil.cpu_count()),
            'total_memory': str(psutil.virtual_memory().total)
        })
    
    async def start(self) -> None:
        """Start collecting system metrics."""
        if self._running:
            return
        
        self._running = True
        self._task = asyncio.create_task(self._collect_loop())
        logger.info("System metrics collector started")
    
    async def stop(self) -> None:
        """Stop collecting system metrics."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("System metrics collector stopped")
    
    async def _collect_loop(self) -> None:
        """Main collection loop."""
        while self._running:
            try:
                self._collect_metrics()
                await asyncio.sleep(self.interval)
            except Exception as e:
                logger.error("Error collecting system metrics", error=str(e))
                await asyncio.sleep(self.interval)
    
    def _collect_metrics(self) -> None:
        """Collect current system metrics."""
        # CPU metrics
        cpu_percent = psutil.cpu_percent(interval=1, percpu=True)
        for i, percent in enumerate(cpu_percent):
            cpu_usage.labels(cpu_core=str(i)).set(percent)
        
        # Memory metrics
        memory = psutil.virtual_memory()
        memory_usage.labels(memory_type='used').set(memory.used)
        memory_usage.labels(memory_type='available').set(memory.available)
        memory_usage.labels(memory_type='cached').set(memory.cached)
        
        # Disk metrics
        for partition in psutil.disk_partitions():
            try:
                usage = psutil.disk_usage(partition.mountpoint)
                disk_usage.labels(
                    mount_point=partition.mountpoint,
                    usage_type='used'
                ).set(usage.used)
                disk_usage.labels(
                    mount_point=partition.mountpoint,
                    usage_type='free'
                ).set(usage.free)
            except PermissionError:
                continue

# Metrics exporter
class MetricsExporter:
    """Exports metrics to various backends."""
    
    def __init__(
        self,
        push_gateway_url: Optional[str] = None,
        job_name: str = "universal_knowledge_platform"
    ) -> None:
        """
        Initialize metrics exporter.
        
        Args:
            push_gateway_url: Prometheus push gateway URL
            job_name: Job name for push gateway
        """
        self.push_gateway_url = push_gateway_url
        self.job_name = job_name
        self._export_task: Optional[asyncio.Task] = None
        self._running = False
    
    async def start(self, interval: float = 60.0) -> None:
        """Start periodic metric export."""
        if self._running:
            return
        
        self._running = True
        if self.push_gateway_url:
            self._export_task = asyncio.create_task(
                self._export_loop(interval)
            )
            logger.info("Metrics exporter started")
    
    async def stop(self) -> None:
        """Stop metric export."""
        self._running = False
        if self._export_task:
            self._export_task.cancel()
            try:
                await self._export_task
            except asyncio.CancelledError:
                pass
        logger.info("Metrics exporter stopped")
    
    async def _export_loop(self, interval: float) -> None:
        """Export metrics periodically."""
        while self._running:
            try:
                await self.export_metrics()
                await asyncio.sleep(interval)
            except Exception as e:
                logger.error("Error exporting metrics", error=str(e))
                await asyncio.sleep(interval)
    
    async def export_metrics(self) -> None:
        """Export metrics to push gateway."""
        if self.push_gateway_url:
            try:
                push_to_gateway(
                    self.push_gateway_url,
                    job=self.job_name,
                    registry=registry
                )
                logger.debug("Metrics pushed to gateway")
            except Exception as e:
                logger.error("Failed to push metrics", error=str(e))
    
    def get_metrics(self) -> bytes:
        """Get metrics in Prometheus format."""
        return generate_latest(registry)

# Global instances
_operation_metrics: Dict[str, Dict[str, Union[Histogram, Gauge]]] = {}
_system_collector = SystemMetricsCollector()
_metrics_exporter = MetricsExporter()

# SLI definitions
sli_availability = SLI(
    name="availability",
    type=SLIType.AVAILABILITY,
    target=0.999  # 99.9% availability
)

sli_latency = SLI(
    name="latency_p95",
    type=SLIType.LATENCY,
    target=0.95  # 95% of requests under threshold
)

sli_error_rate = SLI(
    name="error_rate",
    type=SLIType.ERROR_RATE,
    target=0.99  # Less than 1% errors
)

# Convenience functions
async def start_monitoring() -> None:
    """Start all monitoring components."""
    await _system_collector.start()
    await _metrics_exporter.start()
    logger.info("Monitoring started")

async def stop_monitoring() -> None:
    """Stop all monitoring components."""
    await _system_collector.stop()
    await _metrics_exporter.stop()
    logger.info("Monitoring stopped")

def get_metrics() -> bytes:
    """Get current metrics in Prometheus format."""
    return _metrics_exporter.get_metrics()

def record_user_activity(
    event_type: str,
    user_role: str = "user"
) -> None:
    """Record user activity event."""
    user_activity.labels(
        event_type=event_type,
        user_role=user_role
    ).inc()

def record_cache_hit(cache_name: str, operation: str = "get") -> None:
    """Record cache hit."""
    cache_metrics['hits'].labels(
        cache_name=cache_name,
        operation=operation
    ).inc()

def record_cache_miss(cache_name: str, operation: str = "get") -> None:
    """Record cache miss."""
    cache_metrics['misses'].labels(
        cache_name=cache_name,
        operation=operation
    ).inc()

def get_sli_dashboard() -> Dict[str, Any]:
    """Get SLI dashboard data."""
    return {
        "availability": {
            "current": sli_availability.current_value._value.get(),
            "target": sli_availability.target,
            "error_budget": sli_availability.get_error_budget()
        },
        "latency": {
            "current": sli_latency.current_value._value.get(),
            "target": sli_latency.target,
            "error_budget": sli_latency.get_error_budget()
        },
        "error_rate": {
            "current": sli_error_rate.current_value._value.get(),
            "target": sli_error_rate.target,
            "error_budget": sli_error_rate.get_error_budget()
        }
    }

# Export public API
__all__ = [
    # Decorators
    'track_time',
    'track_errors',
    'track_endpoint_usage',
    
    # Context managers
    'track_operation',
    'track_async_operation',
    
    # Metrics
    'request_counter',
    'request_duration',
    'active_requests',
    'error_counter',
    'user_activity',
    'query_performance',
    'cache_metrics',
    
    # SLIs
    'SLI',
    'sli_availability',
    'sli_latency',
    'sli_error_rate',
    
    # Functions
    'start_monitoring',
    'stop_monitoring',
    'get_metrics',
    'record_user_activity',
    'record_cache_hit',
    'record_cache_miss',
    'get_sli_dashboard',
    
    # Classes
    'SystemMetricsCollector',
    'MetricsExporter',
] 