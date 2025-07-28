"""
Rate Limiting Module - MAANG Standards.

This module implements sophisticated rate limiting following MAANG
best practices for protecting APIs and ensuring fair resource usage.

Features:
    - Multiple rate limiting algorithms (Token Bucket, Sliding Window, Fixed Window)
    - Distributed rate limiting with Redis
    - Per-user, per-IP, and per-endpoint limits
    - Burst handling
    - Graceful degradation
    - Rate limit headers in responses
    - Customizable rate limit strategies
    - Whitelist/blacklist support

Algorithms:
    - Token Bucket: Smooth rate limiting with burst capacity
    - Sliding Window Log: Precise rate limiting
    - Fixed Window Counter: Simple and efficient
    - Leaky Bucket: Smooth output rate

Authors:
    - Universal Knowledge Platform Engineering Team
    
Version:
    2.0.0 (2024-12-28)
"""

import time
import asyncio
import hashlib
from typing import (
    Optional, Dict, Any, List, Tuple, Protocol,
    Union, Callable, TypeVar, Awaitable
)
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass, field
from enum import Enum
from functools import wraps
import structlog

import aioredis
from fastapi import Request, Response, HTTPException, status

from api.exceptions import RateLimitError
from api.monitoring import Gauge, Counter
from api.config import get_settings

logger = structlog.get_logger(__name__)

# Type definitions
T = TypeVar('T')
F = TypeVar('F', bound=Callable[..., Any])

# Metrics
rate_limit_hits = Counter(
    'rate_limit_hits_total',
    'Rate limit hits',
    ['limit_type', 'identifier_type']
)

rate_limit_exceeded = Counter(
    'rate_limit_exceeded_total',
    'Rate limit exceeded',
    ['limit_type', 'identifier_type']
)

active_rate_limits = Gauge(
    'active_rate_limits',
    'Active rate limits',
    ['limit_type']
)

# Rate limiting algorithms
class RateLimitAlgorithm(str, Enum):
    """Rate limiting algorithms."""
    TOKEN_BUCKET = "token_bucket"
    SLIDING_WINDOW_LOG = "sliding_window_log"
    FIXED_WINDOW = "fixed_window"
    LEAKY_BUCKET = "leaky_bucket"

# Identifier types
class IdentifierType(str, Enum):
    """Types of identifiers for rate limiting."""
    USER = "user"
    IP = "ip"
    API_KEY = "api_key"
    ENDPOINT = "endpoint"
    COMBINED = "combined"

# Rate limit configuration
@dataclass
class RateLimitConfig:
    """Rate limit configuration."""
    
    requests_per_minute: int = 60
    requests_per_hour: Optional[int] = None
    requests_per_day: Optional[int] = None
    burst_size: Optional[int] = None
    algorithm: RateLimitAlgorithm = RateLimitAlgorithm.TOKEN_BUCKET
    identifier_type: IdentifierType = IdentifierType.IP
    whitelist: List[str] = field(default_factory=list)
    blacklist: List[str] = field(default_factory=list)
    lockout_duration: int = 300  # 5 minutes
    
    def __post_init__(self) -> None:
        """Validate and set defaults."""
        if self.burst_size is None:
            self.burst_size = self.requests_per_minute // 10
        
        # Calculate requests per second for algorithms
        self.requests_per_second = self.requests_per_minute / 60.0

# Rate limiter protocol
class RateLimiterBackend(Protocol):
    """Protocol for rate limiter backends."""
    
    async def check_rate_limit(
        self,
        identifier: str,
        config: RateLimitConfig
    ) -> Tuple[bool, Dict[str, Any]]:
        """Check if request is allowed."""
        ...
    
    async def reset_rate_limit(self, identifier: str) -> None:
        """Reset rate limit for identifier."""
        ...
    
    async def get_usage(self, identifier: str) -> Dict[str, Any]:
        """Get current usage for identifier."""
        ...

# Redis rate limiter backend
class RedisRateLimiter:
    """Redis-based distributed rate limiter."""
    
    def __init__(
        self,
        redis_url: str,
        prefix: str = "rate_limit:",
        pool_size: int = 10
    ) -> None:
        """
        Initialize Redis rate limiter.
        
        Args:
            redis_url: Redis connection URL
            prefix: Key prefix
            pool_size: Connection pool size
        """
        self.redis_url = redis_url
        self.prefix = prefix
        self.pool_size = pool_size
        self._pool: Optional[aioredis.ConnectionPool] = None
        self._client: Optional[aioredis.Redis] = None
    
    async def connect(self) -> None:
        """Connect to Redis."""
        if not self._pool:
            self._pool = aioredis.ConnectionPool.from_url(
                self.redis_url,
                max_connections=self.pool_size
            )
            self._client = aioredis.Redis(connection_pool=self._pool)
            await self._client.ping()
            logger.info("Redis rate limiter connected")
    
    async def close(self) -> None:
        """Close Redis connections."""
        if self._client:
            await self._client.close()
        if self._pool:
            await self._pool.disconnect()
    
    def _make_key(self, identifier: str, window: Optional[str] = None) -> str:
        """Create Redis key."""
        key = f"{self.prefix}{identifier}"
        if window:
            key = f"{key}:{window}"
        return key
    
    async def check_rate_limit(
        self,
        identifier: str,
        config: RateLimitConfig
    ) -> Tuple[bool, Dict[str, Any]]:
        """Check rate limit using configured algorithm."""
        if not self._client:
            await self.connect()
        
        # Check whitelist/blacklist
        if identifier in config.whitelist:
            return True, {"whitelisted": True}
        
        if identifier in config.blacklist:
            return False, {"blacklisted": True}
        
        # Route to appropriate algorithm
        if config.algorithm == RateLimitAlgorithm.TOKEN_BUCKET:
            return await self._token_bucket(identifier, config)
        elif config.algorithm == RateLimitAlgorithm.SLIDING_WINDOW_LOG:
            return await self._sliding_window_log(identifier, config)
        elif config.algorithm == RateLimitAlgorithm.FIXED_WINDOW:
            return await self._fixed_window(identifier, config)
        else:  # LEAKY_BUCKET
            return await self._leaky_bucket(identifier, config)
    
    async def _token_bucket(
        self,
        identifier: str,
        config: RateLimitConfig
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Token bucket algorithm implementation.
        
        Allows burst traffic while maintaining average rate.
        """
        key = self._make_key(identifier, "token_bucket")
        now = time.time()
        
        # Lua script for atomic token bucket
        lua_script = """
        local key = KEYS[1]
        local rate = tonumber(ARGV[1])
        local burst = tonumber(ARGV[2])
        local now = tonumber(ARGV[3])
        local requested = tonumber(ARGV[4])
        
        local bucket = redis.call('HMGET', key, 'tokens', 'last_refill')
        local tokens = tonumber(bucket[1]) or burst
        local last_refill = tonumber(bucket[2]) or now
        
        -- Calculate tokens to add
        local elapsed = now - last_refill
        local new_tokens = elapsed * rate
        tokens = math.min(burst, tokens + new_tokens)
        
        -- Check if we have enough tokens
        if tokens >= requested then
            tokens = tokens - requested
            redis.call('HMSET', key, 'tokens', tokens, 'last_refill', now)
            redis.call('EXPIRE', key, 3600)
            return {1, tokens, burst}
        else
            redis.call('HMSET', key, 'tokens', tokens, 'last_refill', now)
            redis.call('EXPIRE', key, 3600)
            return {0, tokens, burst}
        end
        """
        
        result = await self._client.eval(
            lua_script,
            keys=[key],
            args=[
                config.requests_per_second,
                config.burst_size,
                now,
                1  # Requested tokens
            ]
        )
        
        allowed = bool(result[0])
        tokens_remaining = float(result[1])
        burst_size = int(result[2])
        
        # Calculate when tokens will be available
        if not allowed:
            tokens_needed = 1 - tokens_remaining
            wait_time = tokens_needed / config.requests_per_second
        else:
            wait_time = 0
        
        info = {
            "algorithm": "token_bucket",
            "tokens_remaining": tokens_remaining,
            "burst_size": burst_size,
            "retry_after": int(wait_time) if not allowed else None
        }
        
        # Update metrics
        if allowed:
            rate_limit_hits.labels(
                limit_type=config.algorithm.value,
                identifier_type=config.identifier_type.value
            ).inc()
        else:
            rate_limit_exceeded.labels(
                limit_type=config.algorithm.value,
                identifier_type=config.identifier_type.value
            ).inc()
        
        return allowed, info
    
    async def _sliding_window_log(
        self,
        identifier: str,
        config: RateLimitConfig
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Sliding window log algorithm implementation.
        
        Most accurate but more memory intensive.
        """
        key = self._make_key(identifier, "sliding_log")
        now = time.time()
        window_start = now - 60  # 1 minute window
        
        # Remove old entries
        await self._client.zremrangebyscore(key, 0, window_start)
        
        # Count requests in window
        count = await self._client.zcard(key)
        
        if count < config.requests_per_minute:
            # Add current request
            await self._client.zadd(key, {str(now): now})
            await self._client.expire(key, 120)  # 2 minutes
            allowed = True
            remaining = config.requests_per_minute - count - 1
        else:
            allowed = False
            remaining = 0
            
            # Get oldest entry to calculate retry time
            oldest = await self._client.zrange(key, 0, 0, withscores=True)
            if oldest:
                oldest_time = oldest[0][1]
                retry_after = int(oldest_time + 60 - now + 1)
            else:
                retry_after = 60
        
        info = {
            "algorithm": "sliding_window_log",
            "requests_in_window": count,
            "remaining": remaining,
            "window_size": 60,
            "retry_after": retry_after if not allowed else None
        }
        
        return allowed, info
    
    async def _fixed_window(
        self,
        identifier: str,
        config: RateLimitConfig
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Fixed window counter algorithm implementation.
        
        Simple and efficient but can allow burst at window boundaries.
        """
        # Get current window
        now = int(time.time())
        window = now // 60  # 1-minute windows
        key = self._make_key(identifier, f"fixed_{window}")
        
        # Increment counter
        count = await self._client.incr(key)
        
        # Set expiry on first request
        if count == 1:
            await self._client.expire(key, 120)  # 2 minutes
        
        allowed = count <= config.requests_per_minute
        remaining = max(0, config.requests_per_minute - count)
        
        # Calculate retry time
        if not allowed:
            window_end = (window + 1) * 60
            retry_after = window_end - now
        else:
            retry_after = None
        
        info = {
            "algorithm": "fixed_window",
            "requests_in_window": count,
            "remaining": remaining,
            "window_start": window * 60,
            "window_end": (window + 1) * 60,
            "retry_after": retry_after
        }
        
        return allowed, info
    
    async def _leaky_bucket(
        self,
        identifier: str,
        config: RateLimitConfig
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Leaky bucket algorithm implementation.
        
        Ensures smooth output rate.
        """
        key = self._make_key(identifier, "leaky_bucket")
        now = time.time()
        
        # Lua script for atomic leaky bucket
        lua_script = """
        local key = KEYS[1]
        local rate = tonumber(ARGV[1])
        local capacity = tonumber(ARGV[2])
        local now = tonumber(ARGV[3])
        
        local bucket = redis.call('HMGET', key, 'volume', 'last_leak')
        local volume = tonumber(bucket[1]) or 0
        local last_leak = tonumber(bucket[2]) or now
        
        -- Calculate leaked amount
        local elapsed = now - last_leak
        local leaked = elapsed * rate
        volume = math.max(0, volume - leaked)
        
        -- Check if we can add
        if volume < capacity then
            volume = volume + 1
            redis.call('HMSET', key, 'volume', volume, 'last_leak', now)
            redis.call('EXPIRE', key, 3600)
            return {1, volume, capacity}
        else
            redis.call('HMSET', key, 'volume', volume, 'last_leak', now)
            redis.call('EXPIRE', key, 3600)
            return {0, volume, capacity}
        end
        """
        
        result = await self._client.eval(
            lua_script,
            keys=[key],
            args=[
                config.requests_per_second,
                config.burst_size,
                now
            ]
        )
        
        allowed = bool(result[0])
        volume = float(result[1])
        capacity = int(result[2])
        
        # Calculate when space will be available
        if not allowed:
            wait_time = (volume - capacity + 1) / config.requests_per_second
        else:
            wait_time = 0
        
        info = {
            "algorithm": "leaky_bucket",
            "current_volume": volume,
            "capacity": capacity,
            "leak_rate": config.requests_per_second,
            "retry_after": int(wait_time) if not allowed else None
        }
        
        return allowed, info
    
    async def reset_rate_limit(self, identifier: str) -> None:
        """Reset all rate limits for identifier."""
        pattern = f"{self.prefix}{identifier}:*"
        
        # Find and delete all keys
        cursor = 0
        while True:
            cursor, keys = await self._client.scan(
                cursor,
                match=pattern,
                count=100
            )
            
            if keys:
                await self._client.delete(*keys)
            
            if cursor == 0:
                break
        
        logger.info("Rate limit reset", identifier=identifier)
    
    async def get_usage(self, identifier: str) -> Dict[str, Any]:
        """Get current usage statistics."""
        usage = {
            "identifier": identifier,
            "algorithms": {}
        }
        
        # Check each algorithm's data
        for algo in RateLimitAlgorithm:
            if algo == RateLimitAlgorithm.TOKEN_BUCKET:
                key = self._make_key(identifier, "token_bucket")
                data = await self._client.hgetall(key)
                if data:
                    usage["algorithms"]["token_bucket"] = {
                        "tokens": float(data.get(b'tokens', 0)),
                        "last_refill": float(data.get(b'last_refill', 0))
                    }
            
            elif algo == RateLimitAlgorithm.SLIDING_WINDOW_LOG:
                key = self._make_key(identifier, "sliding_log")
                count = await self._client.zcard(key)
                if count > 0:
                    usage["algorithms"]["sliding_window_log"] = {
                        "requests_in_window": count
                    }
        
        return usage

# In-memory rate limiter (for testing/development)
class InMemoryRateLimiter:
    """In-memory rate limiter for single-instance deployments."""
    
    def __init__(self) -> None:
        """Initialize in-memory rate limiter."""
        self._buckets: Dict[str, Dict[str, Any]] = {}
        self._lock = asyncio.Lock()
    
    async def check_rate_limit(
        self,
        identifier: str,
        config: RateLimitConfig
    ) -> Tuple[bool, Dict[str, Any]]:
        """Check rate limit in memory."""
        async with self._lock:
            # Simple token bucket implementation
            now = time.time()
            
            if identifier not in self._buckets:
                self._buckets[identifier] = {
                    "tokens": float(config.burst_size),
                    "last_refill": now
                }
            
            bucket = self._buckets[identifier]
            
            # Refill tokens
            elapsed = now - bucket["last_refill"]
            tokens_to_add = elapsed * config.requests_per_second
            bucket["tokens"] = min(
                config.burst_size,
                bucket["tokens"] + tokens_to_add
            )
            bucket["last_refill"] = now
            
            # Check if we have tokens
            if bucket["tokens"] >= 1:
                bucket["tokens"] -= 1
                allowed = True
            else:
                allowed = False
            
            info = {
                "algorithm": "token_bucket",
                "tokens_remaining": bucket["tokens"],
                "burst_size": config.burst_size
            }
            
            return allowed, info
    
    async def reset_rate_limit(self, identifier: str) -> None:
        """Reset rate limit."""
        async with self._lock:
            self._buckets.pop(identifier, None)
    
    async def get_usage(self, identifier: str) -> Dict[str, Any]:
        """Get usage statistics."""
        async with self._lock:
            if identifier in self._buckets:
                return {
                    "identifier": identifier,
                    "bucket": self._buckets[identifier].copy()
                }
            return {"identifier": identifier, "bucket": None}
    
    async def close(self) -> None:
        """Close (no-op for in-memory)."""
        pass

# Main rate limiter class
class RateLimiter:
    """
    Main rate limiter with multiple backends and strategies.
    
    Features:
    - Multiple identifier strategies
    - Configurable algorithms
    - Distributed rate limiting
    - Graceful degradation
    """
    
    def __init__(
        self,
        backend: Optional[RateLimiterBackend] = None,
        default_config: Optional[RateLimitConfig] = None
    ) -> None:
        """
        Initialize rate limiter.
        
        Args:
            backend: Rate limiter backend
            default_config: Default configuration
        """
        self.backend = backend
        self.default_config = default_config or RateLimitConfig()
        self._endpoint_configs: Dict[str, RateLimitConfig] = {}
        self._initialized = False
    
    async def initialize(self) -> None:
        """Initialize rate limiter."""
        if self._initialized:
            return
        
        # Initialize default backend if none provided
        if not self.backend:
            settings = get_settings()
            
            if settings.redis_url and settings.rate_limit_enabled:
                self.backend = RedisRateLimiter(
                    redis_url=str(settings.redis_url),
                    prefix="rate_limit:"
                )
                await self.backend.connect()
            else:
                self.backend = InMemoryRateLimiter()
        
        self._initialized = True
        logger.info("Rate limiter initialized")
    
    async def shutdown(self) -> None:
        """Shutdown rate limiter."""
        if self.backend and hasattr(self.backend, 'close'):
            await self.backend.close()
        self._initialized = False
    
    def configure_endpoint(
        self,
        endpoint: str,
        config: RateLimitConfig
    ) -> None:
        """Configure rate limit for specific endpoint."""
        self._endpoint_configs[endpoint] = config
    
    def _get_identifier(
        self,
        request: Request,
        config: RateLimitConfig
    ) -> str:
        """Extract identifier from request based on config."""
        if config.identifier_type == IdentifierType.USER:
            # Get user ID from request state
            user_id = getattr(request.state, 'user_id', None)
            if user_id:
                return f"user:{user_id}"
            # Fall back to IP
            return f"ip:{request.client.host}"
        
        elif config.identifier_type == IdentifierType.IP:
            # Get real IP considering proxies
            forwarded = request.headers.get('X-Forwarded-For')
            if forwarded:
                ip = forwarded.split(',')[0].strip()
            else:
                ip = request.client.host
            return f"ip:{ip}"
        
        elif config.identifier_type == IdentifierType.API_KEY:
            # Get API key from headers
            api_key = request.headers.get('X-API-Key')
            if api_key:
                # Hash API key for privacy
                key_hash = hashlib.sha256(api_key.encode()).hexdigest()[:16]
                return f"api_key:{key_hash}"
            # Fall back to IP
            return f"ip:{request.client.host}"
        
        elif config.identifier_type == IdentifierType.ENDPOINT:
            # Rate limit by endpoint
            return f"endpoint:{request.url.path}"
        
        else:  # COMBINED
            # Combine user/IP and endpoint
            user_id = getattr(request.state, 'user_id', None)
            if user_id:
                base = f"user:{user_id}"
            else:
                base = f"ip:{request.client.host}"
            return f"{base}:{request.url.path}"
    
    async def check_rate_limit(
        self,
        identifier: str,
        endpoint: str,
        method: str
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Check if request is allowed.
        
        Args:
            identifier: Request identifier
            endpoint: API endpoint
            method: HTTP method
            
        Returns:
            Tuple of (allowed, info)
        """
        if not self._initialized:
            await self.initialize()
        
        # Get config for endpoint
        config = self._endpoint_configs.get(
            endpoint,
            self.default_config
        )
        
        # Check rate limit
        allowed, info = await self.backend.check_rate_limit(
            identifier,
            config
        )
        
        # Add additional info
        info.update({
            "identifier": identifier,
            "endpoint": endpoint,
            "method": method,
            "limit": config.requests_per_minute
        })
        
        return allowed, info
    
    async def get_limit_info(
        self,
        identifier: str
    ) -> Dict[str, Any]:
        """Get rate limit info for identifier."""
        if not self._initialized:
            await self.initialize()
        
        return await self.backend.get_usage(identifier)

# Middleware
async def rate_limit_middleware(
    request: Request,
    call_next: Callable
) -> Response:
    """
    Rate limiting middleware for FastAPI.
    
    Adds rate limit headers to responses.
    """
    # Skip rate limiting for health checks
    if request.url.path in ["/health", "/metrics"]:
        return await call_next(request)
    
    # Get rate limiter
    rate_limiter = get_rate_limiter()
    
    # Get identifier
    config = rate_limiter._endpoint_configs.get(
        request.url.path,
        rate_limiter.default_config
    )
    identifier = rate_limiter._get_identifier(request, config)
    
    # Check rate limit
    allowed, info = await rate_limiter.check_rate_limit(
        identifier,
        request.url.path,
        request.method
    )
    
    if not allowed:
        # Return 429 with headers
        response = Response(
            content="Rate limit exceeded",
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            headers={
                "X-RateLimit-Limit": str(info.get("limit", 60)),
                "X-RateLimit-Remaining": "0",
                "X-RateLimit-Reset": str(int(time.time()) + info.get("retry_after", 60)),
                "Retry-After": str(info.get("retry_after", 60))
            }
        )
        return response
    
    # Process request
    response = await call_next(request)
    
    # Add rate limit headers
    response.headers["X-RateLimit-Limit"] = str(info.get("limit", 60))
    response.headers["X-RateLimit-Remaining"] = str(info.get("remaining", 0))
    response.headers["X-RateLimit-Reset"] = str(
        int(time.time()) + 60  # Next window
    )
    
    return response

# Decorator
def rate_limit(
    config: Optional[RateLimitConfig] = None
) -> Callable:
    """
    Decorator to apply rate limiting to endpoints.
    
    Args:
        config: Rate limit configuration
        
    Example:
        @app.post("/api/query")
        @rate_limit(RateLimitConfig(requests_per_minute=10))
        async def query_endpoint():
            ...
    """
    def decorator(func: F) -> F:
        # Store config on function
        func._rate_limit_config = config or RateLimitConfig()
        
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Get request from args/kwargs
            request = None
            for arg in args:
                if isinstance(arg, Request):
                    request = arg
                    break
            if not request:
                request = kwargs.get('request')
            
            if request:
                # Check rate limit
                rate_limiter = get_rate_limiter()
                
                # Configure endpoint if needed
                endpoint = request.url.path
                if endpoint not in rate_limiter._endpoint_configs:
                    rate_limiter.configure_endpoint(
                        endpoint,
                        func._rate_limit_config
                    )
                
                # Get identifier
                identifier = rate_limiter._get_identifier(
                    request,
                    func._rate_limit_config
                )
                
                # Check limit
                allowed, info = await rate_limiter.check_rate_limit(
                    identifier,
                    endpoint,
                    request.method
                )
                
                if not allowed:
                    raise RateLimitError(
                        limit=info.get("limit", 60),
                        window="minute",
                        retry_after=info.get("retry_after", 60)
                    )
            
            # Execute function
            return await func(*args, **kwargs)
        
        return wrapper
    
    return decorator

# Global instance
_rate_limiter: Optional[RateLimiter] = None

def get_rate_limiter() -> RateLimiter:
    """Get global rate limiter instance."""
    global _rate_limiter
    
    if _rate_limiter is None:
        _rate_limiter = RateLimiter()
    
    return _rate_limiter

# Export public API
__all__ = [
    # Classes
    'RateLimiter',
    'RateLimitConfig',
    'RedisRateLimiter',
    'InMemoryRateLimiter',
    
    # Enums
    'RateLimitAlgorithm',
    'IdentifierType',
    
    # Functions
    'get_rate_limiter',
    'rate_limit',
    'rate_limit_middleware',
] 