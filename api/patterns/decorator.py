"""
Decorator Pattern implementation for middleware and cross-cutting concerns.

This module implements the Decorator Pattern following SOLID principles:
- Single Responsibility: Each decorator handles one concern
- Open/Closed: New decorators can be added without modifying existing code
- Liskov Substitution: Decorators can be used interchangeably
- Interface Segregation: Specific interfaces for different decorator types
- Dependency Inversion: Depend on abstractions, not concrete implementations
"""

from abc import ABC, abstractmethod
from typing import Any, Callable, Awaitable, Optional, Dict, List, Type
from dataclasses import dataclass
from datetime import datetime
from functools import wraps
import asyncio
import logging
import time
import json
from enum import Enum

from .interfaces import Middleware

logger = logging.getLogger(__name__)


# ============================================================================
# BASE DECORATOR - Template for all decorators
# ============================================================================


class BaseDecorator(ABC):
    """
    Base decorator class.
    Template Method Pattern for decorator functionality.
    """
    
    def __init__(self, component: Any):
        self._component = component
        self._metrics = {
            'calls': 0,
            'errors': 0,
            'total_time': 0.0
        }
    
    async def __call__(self, *args, **kwargs) -> Any:
        """
        Execute decorated function.
        Template method with hooks for customization.
        """
        start_time = time.time()
        
        try:
            # Pre-processing
            args, kwargs = await self._before_call(*args, **kwargs)
            
            # Call wrapped component
            result = await self._call_component(*args, **kwargs)
            
            # Post-processing
            result = await self._after_call(result, *args, **kwargs)
            
            # Update metrics
            self._metrics['calls'] += 1
            self._metrics['total_time'] += time.time() - start_time
            
            return result
            
        except Exception as e:
            self._metrics['errors'] += 1
            return await self._handle_error(e, *args, **kwargs)
    
    async def _call_component(self, *args, **kwargs) -> Any:
        """Call the wrapped component."""
        if asyncio.iscoroutinefunction(self._component):
            return await self._component(*args, **kwargs)
        else:
            return self._component(*args, **kwargs)
    
    @abstractmethod
    async def _before_call(self, *args, **kwargs) -> tuple:
        """Hook called before component execution."""
        return args, kwargs
    
    @abstractmethod
    async def _after_call(self, result: Any, *args, **kwargs) -> Any:
        """Hook called after component execution."""
        return result
    
    @abstractmethod
    async def _handle_error(self, error: Exception, *args, **kwargs) -> Any:
        """Hook called on error."""
        raise error
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get decorator metrics."""
        return self._metrics.copy()


# ============================================================================
# LOGGING DECORATOR - Add logging to any component
# ============================================================================


class LoggingDecorator(BaseDecorator):
    """
    Logging decorator.
    Single Responsibility: Add logging to function calls.
    """
    
    def __init__(
        self,
        component: Any,
        log_level: str = "INFO",
        include_args: bool = True,
        include_result: bool = False
    ):
        super().__init__(component)
        self.log_level = getattr(logging, log_level.upper())
        self.include_args = include_args
        self.include_result = include_result
        self.component_name = getattr(component, '__name__', str(component))
    
    async def _before_call(self, *args, **kwargs) -> tuple:
        """Log before execution."""
        message = f"Calling {self.component_name}"
        
        if self.include_args:
            message += f" with args={args}, kwargs={kwargs}"
        
        logger.log(self.log_level, message)
        return args, kwargs
    
    async def _after_call(self, result: Any, *args, **kwargs) -> Any:
        """Log after execution."""
        message = f"Completed {self.component_name}"
        
        if self.include_result:
            message += f" with result={result}"
        
        logger.log(self.log_level, message)
        return result
    
    async def _handle_error(self, error: Exception, *args, **kwargs) -> Any:
        """Log errors."""
        logger.error(
            f"Error in {self.component_name}: {error}",
            exc_info=True
        )
        raise error


# ============================================================================
# TIMING DECORATOR - Add performance timing
# ============================================================================


class TimingDecorator(BaseDecorator):
    """
    Timing decorator.
    Single Responsibility: Measure execution time.
    """
    
    def __init__(
        self,
        component: Any,
        threshold_ms: Optional[float] = None,
        alert_callback: Optional[Callable] = None
    ):
        super().__init__(component)
        self.threshold_ms = threshold_ms
        self.alert_callback = alert_callback
        self.component_name = getattr(component, '__name__', str(component))
        self.execution_times: List[float] = []
    
    async def _before_call(self, *args, **kwargs) -> tuple:
        """Record start time."""
        kwargs['_timing_start'] = time.time()
        return args, kwargs
    
    async def _after_call(self, result: Any, *args, **kwargs) -> Any:
        """Calculate and log execution time."""
        start_time = kwargs.pop('_timing_start', time.time())
        execution_time = (time.time() - start_time) * 1000  # Convert to ms
        
        self.execution_times.append(execution_time)
        
        logger.debug(
            f"{self.component_name} executed in {execution_time:.2f}ms"
        )
        
        # Check threshold
        if self.threshold_ms and execution_time > self.threshold_ms:
            message = (
                f"Performance warning: {self.component_name} "
                f"took {execution_time:.2f}ms (threshold: {self.threshold_ms}ms)"
            )
            logger.warning(message)
            
            if self.alert_callback:
                await self.alert_callback(self.component_name, execution_time)
        
        return result
    
    async def _handle_error(self, error: Exception, *args, **kwargs) -> Any:
        """Clean up timing data on error."""
        kwargs.pop('_timing_start', None)
        raise error
    
    def get_statistics(self) -> Dict[str, float]:
        """Get timing statistics."""
        if not self.execution_times:
            return {}
        
        return {
            'count': len(self.execution_times),
            'total': sum(self.execution_times),
            'average': sum(self.execution_times) / len(self.execution_times),
            'min': min(self.execution_times),
            'max': max(self.execution_times)
        }


# ============================================================================
# CACHING DECORATOR - Add caching to any function
# ============================================================================


class CachingDecorator(BaseDecorator):
    """
    Caching decorator.
    Single Responsibility: Cache function results.
    """
    
    def __init__(
        self,
        component: Any,
        ttl_seconds: int = 3600,
        max_size: int = 1000,
        key_func: Optional[Callable] = None
    ):
        super().__init__(component)
        self.ttl_seconds = ttl_seconds
        self.max_size = max_size
        self.key_func = key_func or self._default_key_func
        self._cache: Dict[str, tuple[Any, float]] = {}
        self._lock = asyncio.Lock()
        self.cache_hits = 0
        self.cache_misses = 0
    
    def _default_key_func(self, *args, **kwargs) -> str:
        """Generate cache key from arguments."""
        key_parts = [str(arg) for arg in args]
        key_parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
        return ":".join(key_parts)
    
    async def _before_call(self, *args, **kwargs) -> tuple:
        """Check cache before execution."""
        cache_key = self.key_func(*args, **kwargs)
        
        async with self._lock:
            if cache_key in self._cache:
                result, timestamp = self._cache[cache_key]
                
                # Check if cached value is still valid
                if time.time() - timestamp < self.ttl_seconds:
                    self.cache_hits += 1
                    logger.debug(f"Cache hit for key: {cache_key}")
                    # Return cached result by raising special exception
                    raise CacheHitException(result)
                else:
                    # Expired - remove from cache
                    del self._cache[cache_key]
        
        self.cache_misses += 1
        kwargs['_cache_key'] = cache_key
        return args, kwargs
    
    async def _after_call(self, result: Any, *args, **kwargs) -> Any:
        """Cache the result."""
        cache_key = kwargs.pop('_cache_key', None)
        
        if cache_key:
            async with self._lock:
                # Implement LRU eviction if cache is full
                if len(self._cache) >= self.max_size:
                    # Remove oldest entry
                    oldest_key = min(
                        self._cache.keys(),
                        key=lambda k: self._cache[k][1]
                    )
                    del self._cache[oldest_key]
                
                # Cache the result
                self._cache[cache_key] = (result, time.time())
                logger.debug(f"Cached result for key: {cache_key}")
        
        return result
    
    async def _handle_error(self, error: Exception, *args, **kwargs) -> Any:
        """Handle cache hit exception."""
        if isinstance(error, CacheHitException):
            return error.result
        
        # Clean up cache key
        kwargs.pop('_cache_key', None)
        raise error
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total_requests if total_requests > 0 else 0
        
        return {
            'size': len(self._cache),
            'max_size': self.max_size,
            'hits': self.cache_hits,
            'misses': self.cache_misses,
            'hit_rate': hit_rate
        }
    
    async def clear_cache(self) -> None:
        """Clear the cache."""
        async with self._lock:
            self._cache.clear()
            logger.info("Cache cleared")


class CacheHitException(Exception):
    """Special exception to return cached results."""
    def __init__(self, result: Any):
        self.result = result


# ============================================================================
# RETRY DECORATOR - Add retry logic to any function
# ============================================================================


class RetryDecorator(BaseDecorator):
    """
    Retry decorator.
    Single Responsibility: Add retry logic for failures.
    """
    
    def __init__(
        self,
        component: Any,
        max_retries: int = 3,
        delay_seconds: float = 1.0,
        exponential_backoff: bool = True,
        retry_exceptions: tuple = (Exception,)
    ):
        super().__init__(component)
        self.max_retries = max_retries
        self.delay_seconds = delay_seconds
        self.exponential_backoff = exponential_backoff
        self.retry_exceptions = retry_exceptions
        self.component_name = getattr(component, '__name__', str(component))
    
    async def _before_call(self, *args, **kwargs) -> tuple:
        """Initialize retry counter."""
        kwargs['_retry_count'] = 0
        return args, kwargs
    
    async def _after_call(self, result: Any, *args, **kwargs) -> Any:
        """Clean up retry counter."""
        kwargs.pop('_retry_count', None)
        return result
    
    async def _handle_error(self, error: Exception, *args, **kwargs) -> Any:
        """Handle errors with retry logic."""
        retry_count = kwargs.get('_retry_count', 0)
        
        # Check if we should retry
        if (retry_count < self.max_retries and 
            isinstance(error, self.retry_exceptions)):
            
            retry_count += 1
            kwargs['_retry_count'] = retry_count
            
            # Calculate delay
            delay = self.delay_seconds
            if self.exponential_backoff:
                delay *= (2 ** (retry_count - 1))
            
            logger.warning(
                f"Retry {retry_count}/{self.max_retries} for {self.component_name} "
                f"after {delay}s delay. Error: {error}"
            )
            
            # Wait before retry
            await asyncio.sleep(delay)
            
            # Retry the call
            return await self(*args, **kwargs)
        
        # Max retries exceeded or non-retryable error
        kwargs.pop('_retry_count', None)
        logger.error(
            f"Failed after {retry_count} retries: {self.component_name}"
        )
        raise error


# ============================================================================
# VALIDATION DECORATOR - Add input/output validation
# ============================================================================


class ValidationDecorator(BaseDecorator):
    """
    Validation decorator.
    Single Responsibility: Validate inputs and outputs.
    """
    
    def __init__(
        self,
        component: Any,
        input_validator: Optional[Callable] = None,
        output_validator: Optional[Callable] = None,
        raise_on_invalid: bool = True
    ):
        super().__init__(component)
        self.input_validator = input_validator
        self.output_validator = output_validator
        self.raise_on_invalid = raise_on_invalid
        self.component_name = getattr(component, '__name__', str(component))
    
    async def _before_call(self, *args, **kwargs) -> tuple:
        """Validate inputs."""
        if self.input_validator:
            try:
                is_valid = self.input_validator(*args, **kwargs)
                if not is_valid:
                    error_msg = f"Invalid input for {self.component_name}"
                    if self.raise_on_invalid:
                        raise ValueError(error_msg)
                    else:
                        logger.warning(error_msg)
            except Exception as e:
                if self.raise_on_invalid:
                    raise
                else:
                    logger.warning(f"Input validation error: {e}")
        
        return args, kwargs
    
    async def _after_call(self, result: Any, *args, **kwargs) -> Any:
        """Validate output."""
        if self.output_validator:
            try:
                is_valid = self.output_validator(result)
                if not is_valid:
                    error_msg = f"Invalid output from {self.component_name}"
                    if self.raise_on_invalid:
                        raise ValueError(error_msg)
                    else:
                        logger.warning(error_msg)
            except Exception as e:
                if self.raise_on_invalid:
                    raise
                else:
                    logger.warning(f"Output validation error: {e}")
        
        return result
    
    async def _handle_error(self, error: Exception, *args, **kwargs) -> Any:
        """Log validation errors."""
        if isinstance(error, ValueError):
            logger.error(f"Validation error in {self.component_name}: {error}")
        raise error


# ============================================================================
# RATE LIMITING DECORATOR - Add rate limiting
# ============================================================================


class RateLimitDecorator(BaseDecorator):
    """
    Rate limiting decorator.
    Single Responsibility: Enforce rate limits.
    """
    
    def __init__(
        self,
        component: Any,
        max_calls: int = 100,
        time_window: int = 60,
        identifier_func: Optional[Callable] = None
    ):
        super().__init__(component)
        self.max_calls = max_calls
        self.time_window = time_window
        self.identifier_func = identifier_func or (lambda *args, **kwargs: "default")
        self._call_times: Dict[str, List[float]] = {}
        self._lock = asyncio.Lock()
    
    async def _before_call(self, *args, **kwargs) -> tuple:
        """Check rate limit before execution."""
        identifier = self.identifier_func(*args, **kwargs)
        current_time = time.time()
        
        async with self._lock:
            # Initialize if needed
            if identifier not in self._call_times:
                self._call_times[identifier] = []
            
            # Remove old entries outside time window
            cutoff_time = current_time - self.time_window
            self._call_times[identifier] = [
                t for t in self._call_times[identifier]
                if t > cutoff_time
            ]
            
            # Check rate limit
            if len(self._call_times[identifier]) >= self.max_calls:
                raise RateLimitExceeded(
                    f"Rate limit exceeded: {self.max_calls} calls "
                    f"per {self.time_window} seconds"
                )
            
            # Record this call
            self._call_times[identifier].append(current_time)
        
        return args, kwargs
    
    async def _after_call(self, result: Any, *args, **kwargs) -> Any:
        """No post-processing needed."""
        return result
    
    async def _handle_error(self, error: Exception, *args, **kwargs) -> Any:
        """Handle rate limit errors."""
        if isinstance(error, RateLimitExceeded):
            logger.warning(f"Rate limit exceeded: {error}")
        raise error


class RateLimitExceeded(Exception):
    """Exception raised when rate limit is exceeded."""
    pass


# ============================================================================
# MIDDLEWARE CHAIN - Compose multiple decorators
# ============================================================================


class MiddlewareChain:
    """
    Middleware chain for composing decorators.
    Composite Pattern: Combine multiple decorators.
    """
    
    def __init__(self, component: Any):
        self.component = component
        self.decorators: List[Type[BaseDecorator]] = []
        self.decorator_configs: List[Dict[str, Any]] = []
    
    def add(self, decorator_class: Type[BaseDecorator], **config) -> 'MiddlewareChain':
        """Add decorator to chain."""
        self.decorators.append(decorator_class)
        self.decorator_configs.append(config)
        return self
    
    def build(self) -> Any:
        """Build the decorated component."""
        result = self.component
        
        # Apply decorators in reverse order (innermost first)
        for decorator_class, config in reversed(
            list(zip(self.decorators, self.decorator_configs))
        ):
            result = decorator_class(result, **config)
        
        return result


# ============================================================================
# DECORATOR FACTORY - Create configured decorators
# ============================================================================


def create_logged_component(
    component: Any,
    log_level: str = "INFO",
    **kwargs
) -> LoggingDecorator:
    """Factory function for logged components."""
    return LoggingDecorator(component, log_level=log_level, **kwargs)


def create_timed_component(
    component: Any,
    threshold_ms: Optional[float] = None,
    **kwargs
) -> TimingDecorator:
    """Factory function for timed components."""
    return TimingDecorator(component, threshold_ms=threshold_ms, **kwargs)


def create_cached_component(
    component: Any,
    ttl_seconds: int = 3600,
    **kwargs
) -> CachingDecorator:
    """Factory function for cached components."""
    return CachingDecorator(component, ttl_seconds=ttl_seconds, **kwargs)


def create_retryable_component(
    component: Any,
    max_retries: int = 3,
    **kwargs
) -> RetryDecorator:
    """Factory function for retryable components."""
    return RetryDecorator(component, max_retries=max_retries, **kwargs)


# ============================================================================
# FUNCTION DECORATORS - Pythonic decorator syntax
# ============================================================================


def logged(log_level: str = "INFO", **kwargs):
    """Function decorator for logging."""
    def decorator(func):
        return LoggingDecorator(func, log_level=log_level, **kwargs)
    return decorator


def timed(threshold_ms: Optional[float] = None, **kwargs):
    """Function decorator for timing."""
    def decorator(func):
        return TimingDecorator(func, threshold_ms=threshold_ms, **kwargs)
    return decorator


def cached(ttl_seconds: int = 3600, **kwargs):
    """Function decorator for caching."""
    def decorator(func):
        return CachingDecorator(func, ttl_seconds=ttl_seconds, **kwargs)
    return decorator


def retry(max_retries: int = 3, **kwargs):
    """Function decorator for retry logic."""
    def decorator(func):
        return RetryDecorator(func, max_retries=max_retries, **kwargs)
    return decorator


def rate_limited(max_calls: int = 100, time_window: int = 60, **kwargs):
    """Function decorator for rate limiting."""
    def decorator(func):
        return RateLimitDecorator(
            func,
            max_calls=max_calls,
            time_window=time_window,
            **kwargs
        )
    return decorator


# Export public API
__all__ = [
    # Base classes
    'BaseDecorator',
    
    # Concrete decorators
    'LoggingDecorator',
    'TimingDecorator',
    'CachingDecorator',
    'RetryDecorator',
    'ValidationDecorator',
    'RateLimitDecorator',
    
    # Composition
    'MiddlewareChain',
    
    # Factory functions
    'create_logged_component',
    'create_timed_component',
    'create_cached_component',
    'create_retryable_component',
    
    # Function decorators
    'logged',
    'timed',
    'cached',
    'retry',
    'rate_limited',
    
    # Exceptions
    'RateLimitExceeded',
] 