"""
Performance Monitoring and Optimization - MAANG Standards.

This module implements comprehensive performance monitoring and optimization
following MAANG best practices for high-performance applications.

Features:
    - Real-time performance monitoring
    - Database query optimization
    - Cache performance analysis
    - Memory usage tracking
    - CPU profiling
    - Response time analysis
    - Performance alerts
    - Auto-scaling recommendations

Performance Metrics:
    - Response time percentiles (P50, P95, P99)
    - Throughput (requests per second)
    - Error rates and availability
    - Resource utilization
    - Database performance
    - Cache hit rates

Authors:
    - Universal Knowledge Platform Engineering Team
    
Version:
    2.0.0 (2024-12-28)
"""

import time
import asyncio
import psutil
import threading
from typing import (
    Optional, Dict, Any, List, Callable, TypeVar,
    AsyncGenerator, Union, Tuple
)
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from contextlib import contextmanager, asynccontextmanager
from functools import wraps
import structlog
import statistics
from collections import deque, defaultdict
import gc
import tracemalloc

from api.monitoring import Histogram, Gauge, Counter
from api.config import get_settings

logger = structlog.get_logger(__name__)

# Type definitions
T = TypeVar('T')
F = TypeVar('F', bound=Callable[..., Any])

# Performance metrics
response_time_histogram = Histogram(
    'api_response_time_seconds',
    'API response time',
    ['endpoint', 'method', 'status_code'],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0)
)

throughput_counter = Counter(
    'api_requests_total',
    'Total API requests',
    ['endpoint', 'method', 'status_code']
)

error_rate_gauge = Gauge(
    'api_error_rate',
    'API error rate percentage',
    ['endpoint', 'method']
)

memory_usage_gauge = Gauge(
    'system_memory_usage_bytes',
    'System memory usage',
    ['memory_type']
)

cpu_usage_gauge = Gauge(
    'system_cpu_usage_percent',
    'System CPU usage',
    ['cpu_core']
)

database_query_time = Histogram(
    'database_query_duration_seconds',
    'Database query duration',
    ['query_type', 'table'],
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0)
)

cache_performance = {
    'hit_rate': Gauge(
        'cache_hit_rate',
        'Cache hit rate percentage',
        ['cache_name']
    ),
    'miss_rate': Gauge(
        'cache_miss_rate',
        'Cache miss rate percentage',
        ['cache_name']
    ),
    'eviction_rate': Counter(
        'cache_evictions_total',
        'Cache evictions',
        ['cache_name', 'reason']
    )
}

# Performance thresholds
class PerformanceThresholds:
    """Performance thresholds for alerting."""
    
    # Response time thresholds (seconds)
    RESPONSE_TIME_P50 = 0.1
    RESPONSE_TIME_P95 = 0.5
    RESPONSE_TIME_P99 = 1.0
    
    # Throughput thresholds (requests per second)
    MIN_THROUGHPUT = 100
    TARGET_THROUGHPUT = 1000
    
    # Error rate thresholds (percentage)
    MAX_ERROR_RATE = 1.0
    CRITICAL_ERROR_RATE = 5.0
    
    # Resource utilization thresholds (percentage)
    MAX_CPU_USAGE = 80.0
    MAX_MEMORY_USAGE = 85.0
    MAX_DISK_USAGE = 90.0
    
    # Database performance thresholds (seconds)
    MAX_DB_QUERY_TIME = 0.1
    MAX_DB_CONNECTION_TIME = 0.05
    
    # Cache performance thresholds (percentage)
    MIN_CACHE_HIT_RATE = 80.0

# Performance data structures
@dataclass
class PerformanceMetrics:
    """Performance metrics for an endpoint."""
    
    endpoint: str
    method: str
    response_times: deque = field(default_factory=lambda: deque(maxlen=1000))
    error_count: int = 0
    total_count: int = 0
    last_updated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    @property
    def error_rate(self) -> float:
        """Calculate error rate percentage."""
        if self.total_count == 0:
            return 0.0
        return (self.error_count / self.total_count) * 100
    
    @property
    def avg_response_time(self) -> float:
        """Calculate average response time."""
        if not self.response_times:
            return 0.0
        return statistics.mean(self.response_times)
    
    @property
    def p50_response_time(self) -> float:
        """Calculate 50th percentile response time."""
        if not self.response_times:
            return 0.0
        return statistics.quantiles(list(self.response_times), n=2)[0]
    
    @property
    def p95_response_time(self) -> float:
        """Calculate 95th percentile response time."""
        if not self.response_times:
            return 0.0
        return statistics.quantiles(list(self.response_times), n=20)[18]
    
    @property
    def p99_response_time(self) -> float:
        """Calculate 99th percentile response time."""
        if not self.response_times:
            return 0.0
        return statistics.quantiles(list(self.response_times), n=100)[98]
    
    @property
    def throughput(self) -> float:
        """Calculate requests per second."""
        if not self.response_times:
            return 0.0
        # Calculate based on recent requests
        recent_times = list(self.response_times)[-100:]
        if len(recent_times) < 2:
            return 0.0
        
        time_span = recent_times[-1] - recent_times[0]
        if time_span <= 0:
            return 0.0
        
        return len(recent_times) / time_span

# Performance monitor
class PerformanceMonitor:
    """
    Comprehensive performance monitoring system.
    
    Features:
    - Real-time performance tracking
    - Resource utilization monitoring
    - Performance alerts
    - Historical data analysis
    """
    
    def __init__(self):
        """Initialize performance monitor."""
        self.metrics: Dict[str, PerformanceMetrics] = defaultdict(
            lambda: PerformanceMetrics("", "")
        )
        self.resource_stats: Dict[str, Any] = {}
        self.alerts: List[Dict[str, Any]] = []
        self._lock = threading.Lock()
        self._monitoring = False
        self._monitor_thread: Optional[threading.Thread] = None
        
        # Performance thresholds
        self.thresholds = PerformanceThresholds()
    
    def start_monitoring(self) -> None:
        """Start performance monitoring."""
        if self._monitoring:
            return
        
        self._monitoring = True
        self._monitor_thread = threading.Thread(
            target=self._monitor_resources,
            daemon=True
        )
        self._monitor_thread.start()
        logger.info("Performance monitoring started")
    
    def stop_monitoring(self) -> None:
        """Stop performance monitoring."""
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join()
        logger.info("Performance monitoring stopped")
    
    def record_request(
        self,
        endpoint: str,
        method: str,
        response_time: float,
        status_code: int
    ) -> None:
        """Record a request for performance analysis."""
        with self._lock:
            key = f"{method}:{endpoint}"
            metrics = self.metrics[key]
            metrics.endpoint = endpoint
            metrics.method = method
            metrics.response_times.append(response_time)
            metrics.total_count += 1
            metrics.last_updated = datetime.now(timezone.utc)
            
            if status_code >= 400:
                metrics.error_count += 1
            
            # Update Prometheus metrics
            response_time_histogram.labels(
                endpoint=endpoint,
                method=method,
                status_code=status_code
            ).observe(response_time)
            
            throughput_counter.labels(
                endpoint=endpoint,
                method=method,
                status_code=status_code
            ).inc()
            
            error_rate_gauge.labels(
                endpoint=endpoint,
                method=method
            ).set(metrics.error_rate)
    
    def record_database_query(
        self,
        query_type: str,
        table: str,
        duration: float
    ) -> None:
        """Record database query performance."""
        database_query_time.labels(
            query_type=query_type,
            table=table
        ).observe(duration)
    
    def record_cache_performance(
        self,
        cache_name: str,
        hit_rate: float,
        miss_rate: float,
        evictions: int = 0
    ) -> None:
        """Record cache performance metrics."""
        cache_performance['hit_rate'].labels(
            cache_name=cache_name
        ).set(hit_rate)
        
        cache_performance['miss_rate'].labels(
            cache_name=cache_name
        ).set(miss_rate)
        
        if evictions > 0:
            cache_performance['eviction_rate'].labels(
                cache_name=cache_name,
                reason="capacity"
            ).inc(evictions)
    
    def _monitor_resources(self) -> None:
        """Monitor system resources."""
        while self._monitoring:
            try:
                # CPU usage
                cpu_percent = psutil.cpu_percent(interval=1, percpu=True)
                for i, percent in enumerate(cpu_percent):
                    cpu_usage_gauge.labels(cpu_core=str(i)).set(percent)
                
                # Memory usage
                memory = psutil.virtual_memory()
                memory_usage_gauge.labels(memory_type='used').set(memory.used)
                memory_usage_gauge.labels(memory_type='available').set(memory.available)
                memory_usage_gauge.labels(memory_type='cached').set(memory.cached)
                
                # Store resource stats
                self.resource_stats = {
                    'cpu_percent': cpu_percent,
                    'memory_percent': memory.percent,
                    'memory_used': memory.used,
                    'memory_available': memory.available,
                    'timestamp': datetime.now(timezone.utc)
                }
                
                # Check for performance alerts
                self._check_alerts()
                
                time.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                logger.error("Error monitoring resources", error=str(e))
                time.sleep(30)  # Wait longer on error
    
    def _check_alerts(self) -> None:
        """Check for performance alerts."""
        alerts = []
        
        # Check resource utilization
        if self.resource_stats.get('memory_percent', 0) > self.thresholds.MAX_MEMORY_USAGE:
            alerts.append({
                'type': 'high_memory_usage',
                'severity': 'warning',
                'message': f"Memory usage is {self.resource_stats['memory_percent']:.1f}%",
                'value': self.resource_stats['memory_percent'],
                'threshold': self.thresholds.MAX_MEMORY_USAGE
            })
        
        cpu_avg = statistics.mean(self.resource_stats.get('cpu_percent', [0]))
        if cpu_avg > self.thresholds.MAX_CPU_USAGE:
            alerts.append({
                'type': 'high_cpu_usage',
                'severity': 'warning',
                'message': f"CPU usage is {cpu_avg:.1f}%",
                'value': cpu_avg,
                'threshold': self.thresholds.MAX_CPU_USAGE
            })
        
        # Check endpoint performance
        for key, metrics in self.metrics.items():
            if metrics.p95_response_time > self.thresholds.RESPONSE_TIME_P95:
                alerts.append({
                    'type': 'slow_response_time',
                    'severity': 'warning',
                    'message': f"P95 response time for {key} is {metrics.p95_response_time:.3f}s",
                    'endpoint': key,
                    'value': metrics.p95_response_time,
                    'threshold': self.thresholds.RESPONSE_TIME_P95
                })
            
            if metrics.error_rate > self.thresholds.MAX_ERROR_RATE:
                alerts.append({
                    'type': 'high_error_rate',
                    'severity': 'critical' if metrics.error_rate > self.thresholds.CRITICAL_ERROR_RATE else 'warning',
                    'message': f"Error rate for {key} is {metrics.error_rate:.1f}%",
                    'endpoint': key,
                    'value': metrics.error_rate,
                    'threshold': self.thresholds.MAX_ERROR_RATE
                })
        
        # Store new alerts
        self.alerts.extend(alerts)
        
        # Log alerts
        for alert in alerts:
            if alert['severity'] == 'critical':
                logger.critical("Performance alert", **alert)
            else:
                logger.warning("Performance alert", **alert)
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary."""
        with self._lock:
            summary = {
                'endpoints': {},
                'resource_stats': self.resource_stats,
                'alerts': self.alerts[-10:],  # Last 10 alerts
                'timestamp': datetime.now(timezone.utc)
            }
            
            for key, metrics in self.metrics.items():
                summary['endpoints'][key] = {
                    'avg_response_time': metrics.avg_response_time,
                    'p50_response_time': metrics.p50_response_time,
                    'p95_response_time': metrics.p95_response_time,
                    'p99_response_time': metrics.p99_response_time,
                    'throughput': metrics.throughput,
                    'error_rate': metrics.error_rate,
                    'total_requests': metrics.total_count,
                    'error_count': metrics.error_count,
                    'last_updated': metrics.last_updated.isoformat()
                }
            
            return summary
    
    def get_recommendations(self) -> List[Dict[str, Any]]:
        """Get performance optimization recommendations."""
        recommendations = []
        
        # Analyze response times
        for key, metrics in self.metrics.items():
            if metrics.p95_response_time > self.thresholds.RESPONSE_TIME_P95:
                recommendations.append({
                    'type': 'optimize_response_time',
                    'endpoint': key,
                    'current_value': metrics.p95_response_time,
                    'target_value': self.thresholds.RESPONSE_TIME_P95,
                    'suggestions': [
                        'Add caching for frequently accessed data',
                        'Optimize database queries',
                        'Consider using async operations',
                        'Implement request batching'
                    ]
                })
            
            if metrics.error_rate > self.thresholds.MAX_ERROR_RATE:
                recommendations.append({
                    'type': 'reduce_error_rate',
                    'endpoint': key,
                    'current_value': metrics.error_rate,
                    'target_value': self.thresholds.MAX_ERROR_RATE,
                    'suggestions': [
                        'Add better error handling',
                        'Implement retry logic',
                        'Add input validation',
                        'Monitor external service health'
                    ]
                })
        
        # Resource recommendations
        if self.resource_stats.get('memory_percent', 0) > 70:
            recommendations.append({
                'type': 'memory_optimization',
                'current_value': self.resource_stats['memory_percent'],
                'target_value': 70,
                'suggestions': [
                    'Implement memory pooling',
                    'Add garbage collection tuning',
                    'Consider using streaming responses',
                    'Optimize data structures'
                ]
            })
        
        return recommendations

# Performance decorators
def monitor_performance(endpoint: str, method: str = "GET"):
    """Decorator to monitor endpoint performance."""
    def decorator(func: F) -> F:
        @wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            start_time = time.time()
            status_code = 200
            
            try:
                result = await func(*args, **kwargs)
                return result
            except Exception as e:
                status_code = 500
                raise
            finally:
                response_time = time.time() - start_time
                get_performance_monitor().record_request(
                    endpoint, method, response_time, status_code
                )
        
        @wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            start_time = time.time()
            status_code = 200
            
            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                status_code = 500
                raise
            finally:
                response_time = time.time() - start_time
                get_performance_monitor().record_request(
                    endpoint, method, response_time, status_code
                )
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    
    return decorator

@contextmanager
def profile_operation(operation_name: str):
    """Context manager for profiling operations."""
    start_time = time.time()
    start_memory = tracemalloc.get_traced_memory()[0]
    
    try:
        yield
    finally:
        duration = time.time() - start_time
        current_memory = tracemalloc.get_traced_memory()[0]
        memory_delta = current_memory - start_memory
        
        logger.info(
            "Operation profiled",
            operation=operation_name,
            duration=duration,
            memory_delta=memory_delta
        )

@asynccontextmanager
async def profile_async_operation(operation_name: str):
    """Async context manager for profiling operations."""
    start_time = time.time()
    start_memory = tracemalloc.get_traced_memory()[0]
    
    try:
        yield
    finally:
        duration = time.time() - start_time
        current_memory = tracemalloc.get_traced_memory()[0]
        memory_delta = current_memory - start_memory
        
        logger.info(
            "Async operation profiled",
            operation=operation_name,
            duration=duration,
            memory_delta=memory_delta
        )

# Database query optimization
class QueryOptimizer:
    """Database query optimization utilities."""
    
    def __init__(self):
        """Initialize query optimizer."""
        self.slow_queries: List[Dict[str, Any]] = []
        self.query_stats: Dict[str, Dict[str, Any]] = defaultdict(
            lambda: {'count': 0, 'total_time': 0.0, 'avg_time': 0.0}
        )
    
    def optimize_query(self, query: str) -> str:
        """Optimize SQL query."""
        # Basic query optimization
        optimized = query.strip()
        
        # Remove unnecessary whitespace
        optimized = ' '.join(optimized.split())
        
        # Add LIMIT if missing for SELECT queries
        if optimized.upper().startswith('SELECT') and 'LIMIT' not in optimized.upper():
            optimized += ' LIMIT 1000'
        
        return optimized
    
    def record_query_performance(
        self,
        query_type: str,
        table: str,
        duration: float,
        query: str
    ) -> None:
        """Record query performance for optimization."""
        key = f"{query_type}:{table}"
        
        # Update stats
        self.query_stats[key]['count'] += 1
        self.query_stats[key]['total_time'] += duration
        self.query_stats[key]['avg_time'] = (
            self.query_stats[key]['total_time'] / self.query_stats[key]['count']
        )
        
        # Record slow queries
        if duration > 0.1:  # 100ms threshold
            self.slow_queries.append({
                'query_type': query_type,
                'table': table,
                'duration': duration,
                'query': query,
                'timestamp': datetime.now(timezone.utc)
            })
        
        # Record metrics
        get_performance_monitor().record_database_query(
            query_type, table, duration
        )
    
    def get_optimization_recommendations(self) -> List[Dict[str, Any]]:
        """Get query optimization recommendations."""
        recommendations = []
        
        # Analyze slow queries
        for query_info in self.slow_queries[-10:]:  # Last 10 slow queries
            recommendations.append({
                'type': 'slow_query',
                'query_type': query_info['query_type'],
                'table': query_info['table'],
                'duration': query_info['duration'],
                'suggestions': [
                    'Add database indexes',
                    'Optimize WHERE clauses',
                    'Use query hints',
                    'Consider query caching'
                ]
            })
        
        # Analyze query patterns
        for key, stats in self.query_stats.items():
            if stats['avg_time'] > 0.05:  # 50ms average threshold
                query_type, table = key.split(':', 1)
                recommendations.append({
                    'type': 'frequent_slow_queries',
                    'query_type': query_type,
                    'table': table,
                    'avg_time': stats['avg_time'],
                    'count': stats['count'],
                    'suggestions': [
                        'Add composite indexes',
                        'Implement query result caching',
                        'Consider denormalization',
                        'Use database views'
                    ]
                })
        
        return recommendations

# Memory optimization
class MemoryOptimizer:
    """Memory optimization utilities."""
    
    def __init__(self):
        """Initialize memory optimizer."""
        self.memory_snapshots: List[Dict[str, Any]] = []
        self.gc_stats: Dict[str, Any] = {}
    
    def take_memory_snapshot(self, label: str) -> None:
        """Take a memory snapshot."""
        snapshot = tracemalloc.take_snapshot()
        memory_info = psutil.virtual_memory()
        
        self.memory_snapshots.append({
            'label': label,
            'timestamp': datetime.now(timezone.utc),
            'memory_percent': memory_info.percent,
            'memory_used': memory_info.used,
            'memory_available': memory_info.available,
            'snapshot': snapshot
        })
    
    def compare_snapshots(self, label1: str, label2: str) -> Dict[str, Any]:
        """Compare two memory snapshots."""
        snapshots = [s for s in self.memory_snapshots if s['label'] in [label1, label2]]
        if len(snapshots) < 2:
            return {}
        
        snapshot1 = snapshots[0]
        snapshot2 = snapshots[1]
        
        # Compare snapshots
        comparison = snapshot1['snapshot'].compare_to(snapshot2['snapshot'], 'lineno')
        
        return {
            'memory_delta': snapshot2['memory_used'] - snapshot1['memory_used'],
            'memory_percent_delta': snapshot2['memory_percent'] - snapshot1['memory_percent'],
            'top_differences': comparison[:10]  # Top 10 differences
        }
    
    def optimize_memory(self) -> Dict[str, Any]:
        """Perform memory optimization."""
        # Force garbage collection
        collected = gc.collect()
        
        # Get memory info
        memory_info = psutil.virtual_memory()
        
        # Update GC stats
        self.gc_stats = {
            'collected_objects': collected,
            'memory_percent': memory_info.percent,
            'memory_used': memory_info.used,
            'memory_available': memory_info.available,
            'timestamp': datetime.now(timezone.utc)
        }
        
        return self.gc_stats

# Global instances
_performance_monitor: Optional[PerformanceMonitor] = None
_query_optimizer: Optional[QueryOptimizer] = None
_memory_optimizer: Optional[MemoryOptimizer] = None

def get_performance_monitor() -> PerformanceMonitor:
    """Get global performance monitor instance."""
    global _performance_monitor
    
    if _performance_monitor is None:
        _performance_monitor = PerformanceMonitor()
    
    return _performance_monitor

def get_query_optimizer() -> QueryOptimizer:
    """Get global query optimizer instance."""
    global _query_optimizer
    
    if _query_optimizer is None:
        _query_optimizer = QueryOptimizer()
    
    return _query_optimizer

def get_memory_optimizer() -> MemoryOptimizer:
    """Get global memory optimizer instance."""
    global _memory_optimizer
    
    if _memory_optimizer is None:
        _memory_optimizer = MemoryOptimizer()
    
    return _memory_optimizer

# Performance utilities
def start_performance_monitoring() -> None:
    """Start performance monitoring."""
    monitor = get_performance_monitor()
    monitor.start_monitoring()
    
    # Enable memory tracking
    tracemalloc.start()

def stop_performance_monitoring() -> None:
    """Stop performance monitoring."""
    monitor = get_performance_monitor()
    monitor.stop_monitoring()
    
    # Stop memory tracking
    tracemalloc.stop()

def get_performance_summary() -> Dict[str, Any]:
    """Get comprehensive performance summary."""
    monitor = get_performance_monitor()
    optimizer = get_query_optimizer()
    memory_optimizer = get_memory_optimizer()
    
    return {
        'performance_metrics': monitor.get_performance_summary(),
        'query_optimization': optimizer.get_optimization_recommendations(),
        'memory_optimization': memory_optimizer.gc_stats,
        'recommendations': monitor.get_recommendations()
    }

# Export public API
__all__ = [
    # Classes
    'PerformanceMonitor',
    'PerformanceMetrics',
    'QueryOptimizer',
    'MemoryOptimizer',
    'PerformanceThresholds',
    
    # Decorators
    'monitor_performance',
    
    # Context managers
    'profile_operation',
    'profile_async_operation',
    
    # Functions
    'get_performance_monitor',
    'get_query_optimizer',
    'get_memory_optimizer',
    'start_performance_monitoring',
    'stop_performance_monitoring',
    'get_performance_summary',
] 