"""
Retry Logic Module for Universal Knowledge Platform
Implements exponential backoff, circuit breaker, and retry strategies.
"""

import asyncio
import logging
import time
import random
from typing import TypeVar, Callable, Optional, Dict, Any, Union
from functools import wraps
from datetime import datetime, timedelta
from enum import Enum
import aiohttp

logger = logging.getLogger(__name__)

T = TypeVar('T')


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"  # Normal operation
    OPEN = "open"      # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing if service recovered


class RetryableError(Exception):
    """Base exception for retryable errors."""
    pass


class CircuitOpenError(Exception):
    """Exception raised when circuit breaker is open."""
    pass


class CircuitBreaker:
    """
    Circuit breaker implementation to prevent cascading failures.
    """
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: int = 60,
        expected_exception: type = Exception
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        
        self.failure_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.state = CircuitState.CLOSED
        
    def call(self, func: Callable, *args, **kwargs):
        """Execute function with circuit breaker protection."""
        if self.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                self.state = CircuitState.HALF_OPEN
            else:
                raise CircuitOpenError(
                    f"Circuit breaker is OPEN. Service unavailable until "
                    f"{self.last_failure_time + timedelta(seconds=self.recovery_timeout)}"
                )
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except self.expected_exception as e:
            self._on_failure()
            raise e
    
    async def async_call(self, func: Callable, *args, **kwargs):
        """Execute async function with circuit breaker protection."""
        if self.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                self.state = CircuitState.HALF_OPEN
            else:
                raise CircuitOpenError(
                    f"Circuit breaker is OPEN. Service unavailable until "
                    f"{self.last_failure_time + timedelta(seconds=self.recovery_timeout)}"
                )
        
        try:
            result = await func(*args, **kwargs)
            self._on_success()
            return result
        except self.expected_exception as e:
            self._on_failure()
            raise e
    
    def _should_attempt_reset(self) -> bool:
        """Check if we should attempt to reset the circuit."""
        return (
            self.last_failure_time and
            datetime.now() >= self.last_failure_time + timedelta(seconds=self.recovery_timeout)
        )
    
    def _on_success(self):
        """Handle successful call."""
        self.failure_count = 0
        self.state = CircuitState.CLOSED
        
    def _on_failure(self):
        """Handle failed call."""
        self.failure_count += 1
        self.last_failure_time = datetime.now()
        
        if self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN
            logger.warning(
                f"Circuit breaker opened after {self.failure_count} failures"
            )


class RetryConfig:
    """Configuration for retry behavior."""
    
    def __init__(
        self,
        max_attempts: int = 3,
        initial_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        jitter: bool = True,
        retryable_exceptions: tuple = (RetryableError, aiohttp.ClientError, asyncio.TimeoutError),
        circuit_breaker: Optional[CircuitBreaker] = None
    ):
        self.max_attempts = max_attempts
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter
        self.retryable_exceptions = retryable_exceptions
        self.circuit_breaker = circuit_breaker


def calculate_backoff_delay(
    attempt: int,
    initial_delay: float,
    max_delay: float,
    exponential_base: float,
    jitter: bool
) -> float:
    """
    Calculate exponential backoff delay with optional jitter.
    
    Args:
        attempt: Current attempt number (0-based)
        initial_delay: Initial delay in seconds
        max_delay: Maximum delay in seconds
        exponential_base: Base for exponential calculation
        jitter: Whether to add random jitter
        
    Returns:
        Delay in seconds
    """
    delay = min(initial_delay * (exponential_base ** attempt), max_delay)
    
    if jitter:
        # Add random jitter (±25% of calculated delay)
        jitter_range = delay * 0.25
        delay += random.uniform(-jitter_range, jitter_range)
    
    return max(0, delay)


def retry_async(config: Optional[RetryConfig] = None):
    """
    Decorator for async functions with retry logic.
    
    Args:
        config: Retry configuration
        
    Returns:
        Decorated function
    """
    if config is None:
        config = RetryConfig()
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(config.max_attempts):
                try:
                    # Use circuit breaker if configured
                    if config.circuit_breaker:
                        return await config.circuit_breaker.async_call(func, *args, **kwargs)
                    else:
                        return await func(*args, **kwargs)
                        
                except config.retryable_exceptions as e:
                    last_exception = e
                    
                    if attempt == config.max_attempts - 1:
                        logger.error(
                            f"Failed after {config.max_attempts} attempts: {func.__name__}"
                        )
                        raise
                    
                    delay = calculate_backoff_delay(
                        attempt,
                        config.initial_delay,
                        config.max_delay,
                        config.exponential_base,
                        config.jitter
                    )
                    
                    logger.warning(
                        f"Attempt {attempt + 1}/{config.max_attempts} failed for {func.__name__}. "
                        f"Retrying in {delay:.2f}s. Error: {str(e)}"
                    )
                    
                    await asyncio.sleep(delay)
                    
                except CircuitOpenError:
                    # Don't retry if circuit is open
                    raise
                    
                except Exception as e:
                    # Non-retryable exception
                    logger.error(f"Non-retryable error in {func.__name__}: {str(e)}")
                    raise
            
            # Should never reach here, but just in case
            if last_exception:
                raise last_exception
                
        return wrapper
    return decorator


def retry_sync(config: Optional[RetryConfig] = None):
    """
    Decorator for sync functions with retry logic.
    
    Args:
        config: Retry configuration
        
    Returns:
        Decorated function
    """
    if config is None:
        config = RetryConfig()
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(config.max_attempts):
                try:
                    # Use circuit breaker if configured
                    if config.circuit_breaker:
                        return config.circuit_breaker.call(func, *args, **kwargs)
                    else:
                        return func(*args, **kwargs)
                        
                except config.retryable_exceptions as e:
                    last_exception = e
                    
                    if attempt == config.max_attempts - 1:
                        logger.error(
                            f"Failed after {config.max_attempts} attempts: {func.__name__}"
                        )
                        raise
                    
                    delay = calculate_backoff_delay(
                        attempt,
                        config.initial_delay,
                        config.max_delay,
                        config.exponential_base,
                        config.jitter
                    )
                    
                    logger.warning(
                        f"Attempt {attempt + 1}/{config.max_attempts} failed for {func.__name__}. "
                        f"Retrying in {delay:.2f}s. Error: {str(e)}"
                    )
                    
                    time.sleep(delay)
                    
                except CircuitOpenError:
                    # Don't retry if circuit is open
                    raise
                    
                except Exception as e:
                    # Non-retryable exception
                    logger.error(f"Non-retryable error in {func.__name__}: {str(e)}")
                    raise
            
            # Should never reach here, but just in case
            if last_exception:
                raise last_exception
                
        return wrapper
    return decorator


# Predefined retry configurations for common scenarios
FAST_RETRY = RetryConfig(
    max_attempts=3,
    initial_delay=0.1,
    max_delay=1.0,
    jitter=True
)

STANDARD_RETRY = RetryConfig(
    max_attempts=3,
    initial_delay=1.0,
    max_delay=10.0,
    jitter=True
)

AGGRESSIVE_RETRY = RetryConfig(
    max_attempts=5,
    initial_delay=0.5,
    max_delay=30.0,
    jitter=True
)

EXTERNAL_SERVICE_RETRY = RetryConfig(
    max_attempts=4,
    initial_delay=2.0,
    max_delay=60.0,
    exponential_base=2.0,
    jitter=True,
    circuit_breaker=CircuitBreaker(
        failure_threshold=5,
        recovery_timeout=120,
        expected_exception=(aiohttp.ClientError, asyncio.TimeoutError)
    )
)


class RetryableHTTPClient:
    """HTTP client with built-in retry logic."""
    
    def __init__(self, retry_config: Optional[RetryConfig] = None):
        self.retry_config = retry_config or EXTERNAL_SERVICE_RETRY
        
    @retry_async(EXTERNAL_SERVICE_RETRY)
    async def get(self, url: str, **kwargs) -> aiohttp.ClientResponse:
        """Make GET request with retry logic."""
        async with aiohttp.ClientSession() as session:
            async with session.get(url, **kwargs) as response:
                response.raise_for_status()
                return response
    
    @retry_async(EXTERNAL_SERVICE_RETRY)
    async def post(self, url: str, **kwargs) -> aiohttp.ClientResponse:
        """Make POST request with retry logic."""
        async with aiohttp.ClientSession() as session:
            async with session.post(url, **kwargs) as response:
                response.raise_for_status()
                return response
    
    @retry_async(EXTERNAL_SERVICE_RETRY)
    async def put(self, url: str, **kwargs) -> aiohttp.ClientResponse:
        """Make PUT request with retry logic."""
        async with aiohttp.ClientSession() as session:
            async with session.put(url, **kwargs) as response:
                response.raise_for_status()
                return response
    
    @retry_async(EXTERNAL_SERVICE_RETRY)
    async def delete(self, url: str, **kwargs) -> aiohttp.ClientResponse:
        """Make DELETE request with retry logic."""
        async with aiohttp.ClientSession() as session:
            async with session.delete(url, **kwargs) as response:
                response.raise_for_status()
                return response 