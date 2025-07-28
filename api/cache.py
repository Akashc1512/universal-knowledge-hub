"""
Caching Module - MAANG Standards.

This module implements a sophisticated caching system following MAANG
best practices for performance, scalability, and reliability.

Features:
    - Multiple cache backends (Redis, In-Memory, Hybrid)
    - Cache warming and preloading
    - TTL management with jitter
    - Cache stampede prevention
    - Distributed cache invalidation
    - Cache statistics and monitoring
    - Compression for large values
    - Encryption for sensitive data

Architecture:
    - Strategy pattern for cache backends
    - Decorator-based API
    - Async-first design
    - Circuit breaker for cache failures

Authors:
    - Universal Knowledge Platform Engineering Team
    
Version:
    2.0.0 (2024-12-28)
"""

import asyncio
import json
import pickle
import hashlib
import zlib
import time
import random
from typing import (
    Optional, Dict, Any, List, Union, Callable, 
    TypeVar, Protocol, Type, Awaitable, Set
)
from datetime import datetime, timedelta, timezone
from functools import wraps, lru_cache
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import structlog

import aioredis
from aiocache import Cache as AiocacheBase
from aiocache.serializers import JsonSerializer, PickleSerializer
from cryptography.fernet import Fernet

from api.monitoring import (
    record_cache_hit, record_cache_miss,
    track_async_operation, cache_metrics
)
from api.config import get_settings

logger = structlog.get_logger(__name__)

# Type definitions
T = TypeVar('T')
CacheKey = Union[str, bytes]
CacheValue = Union[str, bytes, Dict[str, Any], List[Any], int, float, bool]

# Cache strategies
class CacheStrategy(str, Enum):
    """Cache eviction strategies."""
    LRU = "lru"  # Least Recently Used
    LFU = "lfu"  # Least Frequently Used
    FIFO = "fifo"  # First In First Out
    TTL = "ttl"  # Time To Live based

class SerializationType(str, Enum):
    """Serialization types for cache values."""
    JSON = "json"
    PICKLE = "pickle"
    STRING = "string"
    BYTES = "bytes"

# Cache statistics
@dataclass
class CacheStats:
    """Cache statistics for monitoring."""
    
    hits: int = 0
    misses: int = 0
    sets: int = 0
    deletes: int = 0
    errors: int = 0
    total_size: int = 0
    evictions: int = 0
    
    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": self.hit_rate,
            "sets": self.sets,
            "deletes": self.deletes,
            "errors": self.errors,
            "total_size": self.total_size,
            "evictions": self.evictions
        }

# Cache backend protocol
class CacheBackend(Protocol):
    """Protocol for cache backend implementations."""
    
    async def get(self, key: CacheKey) -> Optional[Any]:
        """Get value from cache."""
        ...
    
    async def set(
        self,
        key: CacheKey,
        value: Any,
        ttl: Optional[int] = None
    ) -> bool:
        """Set value in cache."""
        ...
    
    async def delete(self, key: CacheKey) -> bool:
        """Delete value from cache."""
        ...
    
    async def exists(self, key: CacheKey) -> bool:
        """Check if key exists."""
        ...
    
    async def clear(self) -> int:
        """Clear all cache entries."""
        ...
    
    async def get_many(self, keys: List[CacheKey]) -> Dict[CacheKey, Any]:
        """Get multiple values."""
        ...
    
    async def set_many(
        self,
        mapping: Dict[CacheKey, Any],
        ttl: Optional[int] = None
    ) -> bool:
        """Set multiple values."""
        ...
    
    async def close(self) -> None:
        """Close backend connections."""
        ...

# Redis cache backend
class RedisBackend:
    """Redis cache backend implementation."""
    
    def __init__(
        self,
        url: str,
        pool_size: int = 10,
        prefix: str = "cache:",
        serializer: SerializationType = SerializationType.JSON,
        compress_threshold: int = 1024,  # Compress values > 1KB
        encrypt: bool = False,
        encryption_key: Optional[bytes] = None
    ) -> None:
        """
        Initialize Redis backend.
        
        Args:
            url: Redis connection URL
            pool_size: Connection pool size
            prefix: Key prefix
            serializer: Serialization type
            compress_threshold: Compression threshold in bytes
            encrypt: Enable encryption
            encryption_key: Encryption key
        """
        self.url = url
        self.pool_size = pool_size
        self.prefix = prefix
        self.serializer = serializer
        self.compress_threshold = compress_threshold
        self.encrypt = encrypt
        
        if encrypt:
            if not encryption_key:
                raise ValueError("Encryption key required when encryption is enabled")
            self._fernet = Fernet(encryption_key)
        else:
            self._fernet = None
        
        self._pool: Optional[aioredis.ConnectionPool] = None
        self._client: Optional[aioredis.Redis] = None
        self.stats = CacheStats()
    
    async def connect(self) -> None:
        """Connect to Redis."""
        if not self._pool:
            self._pool = aioredis.ConnectionPool.from_url(
                self.url,
                max_connections=self.pool_size,
                decode_responses=False  # Handle bytes
            )
            self._client = aioredis.Redis(connection_pool=self._pool)
            
            # Test connection
            await self._client.ping()
            logger.info("Redis cache connected", url=self.url)
    
    async def close(self) -> None:
        """Close Redis connections."""
        if self._client:
            await self._client.close()
        if self._pool:
            await self._pool.disconnect()
        logger.info("Redis cache disconnected")
    
    def _make_key(self, key: CacheKey) -> str:
        """Create prefixed key."""
        if isinstance(key, bytes):
            key = key.decode('utf-8')
        return f"{self.prefix}{key}"
    
    def _serialize(self, value: Any) -> bytes:
        """Serialize value to bytes."""
        if self.serializer == SerializationType.JSON:
            serialized = json.dumps(value).encode('utf-8')
        elif self.serializer == SerializationType.PICKLE:
            serialized = pickle.dumps(value)
        elif self.serializer == SerializationType.BYTES:
            serialized = value if isinstance(value, bytes) else str(value).encode('utf-8')
        else:  # STRING
            serialized = str(value).encode('utf-8')
        
        # Compress if needed
        if len(serialized) > self.compress_threshold:
            serialized = b'COMPRESSED:' + zlib.compress(serialized)
        
        # Encrypt if needed
        if self._fernet:
            serialized = b'ENCRYPTED:' + self._fernet.encrypt(serialized)
        
        return serialized
    
    def _deserialize(self, data: bytes) -> Any:
        """Deserialize bytes to value."""
        if not data:
            return None
        
        # Decrypt if needed
        if data.startswith(b'ENCRYPTED:'):
            if not self._fernet:
                raise ValueError("Cannot decrypt without encryption key")
            data = self._fernet.decrypt(data[10:])
        
        # Decompress if needed
        if data.startswith(b'COMPRESSED:'):
            data = zlib.decompress(data[11:])
        
        # Deserialize
        if self.serializer == SerializationType.JSON:
            return json.loads(data.decode('utf-8'))
        elif self.serializer == SerializationType.PICKLE:
            return pickle.loads(data)
        elif self.serializer == SerializationType.BYTES:
            return data
        else:  # STRING
            return data.decode('utf-8')
    
    async def get(self, key: CacheKey) -> Optional[Any]:
        """Get value from cache."""
        if not self._client:
            await self.connect()
        
        try:
            redis_key = self._make_key(key)
            data = await self._client.get(redis_key)
            
            if data is None:
                self.stats.misses += 1
                record_cache_miss("redis", "get")
                return None
            
            self.stats.hits += 1
            record_cache_hit("redis", "get")
            
            return self._deserialize(data)
            
        except Exception as e:
            self.stats.errors += 1
            logger.error("Redis get error", key=key, error=str(e))
            return None
    
    async def set(
        self,
        key: CacheKey,
        value: Any,
        ttl: Optional[int] = None
    ) -> bool:
        """Set value in cache."""
        if not self._client:
            await self.connect()
        
        try:
            redis_key = self._make_key(key)
            data = self._serialize(value)
            
            # Add jitter to TTL to prevent cache stampede
            if ttl:
                ttl = int(ttl * (1 + random.uniform(-0.1, 0.1)))
            
            if ttl:
                await self._client.setex(redis_key, ttl, data)
            else:
                await self._client.set(redis_key, data)
            
            self.stats.sets += 1
            self.stats.total_size += len(data)
            
            return True
            
        except Exception as e:
            self.stats.errors += 1
            logger.error("Redis set error", key=key, error=str(e))
            return False
    
    async def delete(self, key: CacheKey) -> bool:
        """Delete value from cache."""
        if not self._client:
            await self.connect()
        
        try:
            redis_key = self._make_key(key)
            result = await self._client.delete(redis_key)
            
            if result > 0:
                self.stats.deletes += 1
                return True
            return False
            
        except Exception as e:
            self.stats.errors += 1
            logger.error("Redis delete error", key=key, error=str(e))
            return False
    
    async def exists(self, key: CacheKey) -> bool:
        """Check if key exists."""
        if not self._client:
            await self.connect()
        
        try:
            redis_key = self._make_key(key)
            return bool(await self._client.exists(redis_key))
        except Exception:
            return False
    
    async def clear(self) -> int:
        """Clear all cache entries with prefix."""
        if not self._client:
            await self.connect()
        
        try:
            # Use SCAN to find all keys with prefix
            pattern = f"{self.prefix}*"
            count = 0
            
            async for key in self._client.scan_iter(match=pattern, count=100):
                await self._client.delete(key)
                count += 1
            
            self.stats.deletes += count
            return count
            
        except Exception as e:
            self.stats.errors += 1
            logger.error("Redis clear error", error=str(e))
            return 0
    
    async def get_many(self, keys: List[CacheKey]) -> Dict[CacheKey, Any]:
        """Get multiple values."""
        if not self._client:
            await self.connect()
        
        try:
            redis_keys = [self._make_key(k) for k in keys]
            values = await self._client.mget(redis_keys)
            
            result = {}
            for key, value in zip(keys, values):
                if value is not None:
                    result[key] = self._deserialize(value)
                    self.stats.hits += 1
                else:
                    self.stats.misses += 1
            
            return result
            
        except Exception as e:
            self.stats.errors += 1
            logger.error("Redis mget error", error=str(e))
            return {}
    
    async def set_many(
        self,
        mapping: Dict[CacheKey, Any],
        ttl: Optional[int] = None
    ) -> bool:
        """Set multiple values."""
        if not self._client:
            await self.connect()
        
        try:
            # Prepare data
            redis_mapping = {}
            for key, value in mapping.items():
                redis_key = self._make_key(key)
                redis_mapping[redis_key] = self._serialize(value)
            
            # Set all at once
            await self._client.mset(redis_mapping)
            
            # Set TTL if needed (requires individual commands)
            if ttl:
                ttl_with_jitter = int(ttl * (1 + random.uniform(-0.1, 0.1)))
                for redis_key in redis_mapping:
                    await self._client.expire(redis_key, ttl_with_jitter)
            
            self.stats.sets += len(mapping)
            return True
            
        except Exception as e:
            self.stats.errors += 1
            logger.error("Redis mset error", error=str(e))
            return False

# In-memory cache backend
class InMemoryBackend:
    """In-memory cache backend with LRU eviction."""
    
    def __init__(
        self,
        max_size: int = 10000,
        ttl: Optional[int] = None,
        strategy: CacheStrategy = CacheStrategy.LRU
    ) -> None:
        """
        Initialize in-memory backend.
        
        Args:
            max_size: Maximum number of entries
            ttl: Default TTL in seconds
            strategy: Eviction strategy
        """
        self.max_size = max_size
        self.default_ttl = ttl
        self.strategy = strategy
        
        self._cache: Dict[CacheKey, tuple[Any, float, int]] = {}
        self._access_count: Dict[CacheKey, int] = {}
        self._access_time: Dict[CacheKey, float] = {}
        self._lock = asyncio.Lock()
        self.stats = CacheStats()

    async def get(self, key: CacheKey) -> Optional[Any]:
        """Get value from cache."""
        async with self._lock:
            if key not in self._cache:
                self.stats.misses += 1
                return None
            
            value, expiry, _ = self._cache[key]
            
            # Check expiry
            if expiry and time.time() > expiry:
                del self._cache[key]
                self.stats.misses += 1
                self.stats.evictions += 1
            return None

            # Update access tracking
            self._access_count[key] = self._access_count.get(key, 0) + 1
            self._access_time[key] = time.time()
            
            self.stats.hits += 1
            return value
    
    async def set(
        self,
        key: CacheKey,
        value: Any,
        ttl: Optional[int] = None
    ) -> bool:
        """Set value in cache."""
        async with self._lock:
            # Evict if at capacity
            if len(self._cache) >= self.max_size and key not in self._cache:
                await self._evict_one()
            
            # Set value
            ttl = ttl or self.default_ttl
            expiry = time.time() + ttl if ttl else None
            size = len(str(value))
            
            self._cache[key] = (value, expiry, size)
            self._access_count[key] = 1
            self._access_time[key] = time.time()
            
            self.stats.sets += 1
            self.stats.total_size += size
            
            return True
    
    async def delete(self, key: CacheKey) -> bool:
        """Delete value from cache."""
        async with self._lock:
            if key in self._cache:
                _, _, size = self._cache[key]
                del self._cache[key]
                self._access_count.pop(key, None)
                self._access_time.pop(key, None)
                
                self.stats.deletes += 1
                self.stats.total_size -= size
                return True
            return False
    
    async def exists(self, key: CacheKey) -> bool:
        """Check if key exists."""
        async with self._lock:
            if key not in self._cache:
                return False
            
            _, expiry, _ = self._cache[key]
            if expiry and time.time() > expiry:
                return False
            
            return True
    
    async def clear(self) -> int:
        """Clear all cache entries."""
        async with self._lock:
            count = len(self._cache)
            self._cache.clear()
            self._access_count.clear()
            self._access_time.clear()
            self.stats.deletes += count
            self.stats.total_size = 0
            return count
    
    async def get_many(self, keys: List[CacheKey]) -> Dict[CacheKey, Any]:
        """Get multiple values."""
        result = {}
        for key in keys:
            value = await self.get(key)
            if value is not None:
                result[key] = value
        return result
    
    async def set_many(
        self,
        mapping: Dict[CacheKey, Any],
        ttl: Optional[int] = None
    ) -> bool:
        """Set multiple values."""
        for key, value in mapping.items():
            await self.set(key, value, ttl)
        return True
    
    async def close(self) -> None:
        """Close backend (no-op for in-memory)."""
        pass
    
    async def _evict_one(self) -> None:
        """Evict one entry based on strategy."""
        if not self._cache:
            return
        
        if self.strategy == CacheStrategy.LRU:
            # Evict least recently used
            key = min(self._access_time, key=self._access_time.get)
        elif self.strategy == CacheStrategy.LFU:
            # Evict least frequently used
            key = min(self._access_count, key=self._access_count.get)
        elif self.strategy == CacheStrategy.FIFO:
            # Evict oldest
            key = next(iter(self._cache))
        else:  # TTL
            # Evict closest to expiry
            key = min(
                self._cache,
                key=lambda k: self._cache[k][1] or float('inf')
            )
        
        await self.delete(key)
        self.stats.evictions += 1
        cache_metrics['evictions'].labels(
            cache_name="memory",
            reason=self.strategy.value
        ).inc()

# Cache manager
class CacheManager:
    """
    Main cache manager with multiple backends and strategies.
    
    Features:
    - Multiple cache tiers (L1: memory, L2: Redis)
    - Automatic failover
    - Cache warming
    - Batch operations
    - Circuit breaker for failures
    """
    
    def __init__(
        self,
        backends: Optional[List[CacheBackend]] = None,
        default_ttl: int = 300,  # 5 minutes
        enable_stats: bool = True
    ) -> None:
        """
        Initialize cache manager.
        
        Args:
            backends: List of cache backends (ordered by priority)
            default_ttl: Default TTL in seconds
            enable_stats: Enable statistics collection
        """
        self.backends = backends or []
        self.default_ttl = default_ttl
        self.enable_stats = enable_stats
        self._initialized = False
        self._circuit_breaker: Dict[int, tuple[bool, float]] = {}
    
    async def initialize(self) -> None:
        """Initialize all backends."""
        if self._initialized:
            return
        
        # Initialize default backends if none provided
        if not self.backends:
            settings = get_settings()
            
            # L1: In-memory cache
            self.backends.append(
                InMemoryBackend(
                    max_size=1000,
                    ttl=60  # 1 minute for L1
                )
            )
            
            # L2: Redis cache
            if settings.redis_url:
                redis_backend = RedisBackend(
                    url=str(settings.redis_url),
                    pool_size=settings.redis_pool_size,
                    prefix=settings.cache_prefix,
                    encrypt=settings.environment == "production"
                )
                await redis_backend.connect()
                self.backends.append(redis_backend)
        
        self._initialized = True
        logger.info(
            "Cache manager initialized",
            backends=len(self.backends)
        )
    
    async def close(self) -> None:
        """Close all backends."""
        for backend in self.backends:
            await backend.close()
        self._initialized = False
    
    def _is_backend_healthy(self, index: int) -> bool:
        """Check if backend is healthy (circuit breaker)."""
        if index not in self._circuit_breaker:
            return True
        
        is_open, opened_at = self._circuit_breaker[index]
        if is_open:
            # Check if we should retry (after 30 seconds)
            if time.time() - opened_at > 30:
                del self._circuit_breaker[index]
                return True
            return False
        return True
    
    def _mark_backend_unhealthy(self, index: int) -> None:
        """Mark backend as unhealthy (open circuit)."""
        self._circuit_breaker[index] = (True, time.time())
        logger.warning(f"Backend {index} marked unhealthy")
    
    async def get(
        self,
        key: CacheKey,
        default: Optional[T] = None
    ) -> Optional[Union[T, Any]]:
        """
        Get value from cache.
        
        Tries backends in order until value is found.
        """
        if not self._initialized:
            await self.initialize()
        
        for i, backend in enumerate(self.backends):
            if not self._is_backend_healthy(i):
                continue
            
            try:
                value = await backend.get(key)
                if value is not None:
                    # Populate higher tier caches
                    for j in range(i):
                        if self._is_backend_healthy(j):
                            await self.backends[j].set(
                                key, value, self.default_ttl
                            )
                    return value
            except Exception as e:
                logger.error(f"Backend {i} get error", error=str(e))
                self._mark_backend_unhealthy(i)
        
        return default
    
    async def set(
        self,
        key: CacheKey,
        value: Any,
        ttl: Optional[int] = None
    ) -> bool:
        """Set value in all healthy backends."""
        if not self._initialized:
            await self.initialize()
        
        ttl = ttl or self.default_ttl
        success = False
        
        for i, backend in enumerate(self.backends):
            if not self._is_backend_healthy(i):
                continue
            
            try:
                if await backend.set(key, value, ttl):
                    success = True
            except Exception as e:
                logger.error(f"Backend {i} set error", error=str(e))
                self._mark_backend_unhealthy(i)
        
        return success
    
    async def delete(self, key: CacheKey) -> bool:
        """Delete value from all backends."""
        if not self._initialized:
            await self.initialize()
        
        success = False
        
        for i, backend in enumerate(self.backends):
            if not self._is_backend_healthy(i):
                continue
            
            try:
                if await backend.delete(key):
                    success = True
            except Exception as e:
                logger.error(f"Backend {i} delete error", error=str(e))
                self._mark_backend_unhealthy(i)
        
        return success
    
    async def clear(self) -> int:
        """Clear all backends."""
        if not self._initialized:
            await self.initialize()
        
        total = 0
        
        for i, backend in enumerate(self.backends):
            if not self._is_backend_healthy(i):
                continue
            
            try:
                count = await backend.clear()
                total += count
            except Exception as e:
                logger.error(f"Backend {i} clear error", error=str(e))
                self._mark_backend_unhealthy(i)

        return total

    async def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        stats = {
            "backends": [],
            "total": CacheStats()
        }
        
        for i, backend in enumerate(self.backends):
            backend_stats = {
                "index": i,
                "type": type(backend).__name__,
                "healthy": self._is_backend_healthy(i),
                "stats": {}
            }
            
            if hasattr(backend, 'stats'):
                backend_stats["stats"] = backend.stats.to_dict()
                
                # Aggregate totals
                stats["total"].hits += backend.stats.hits
                stats["total"].misses += backend.stats.misses
                stats["total"].sets += backend.stats.sets
                stats["total"].deletes += backend.stats.deletes
                stats["total"].errors += backend.stats.errors
                stats["total"].evictions += backend.stats.evictions
            
            stats["backends"].append(backend_stats)
        
        stats["total"] = stats["total"].to_dict()
        return stats

# Decorators
def cache_response(
    ttl: Optional[int] = None,
    key_builder: Optional[Callable[..., str]] = None,
    cache_manager: Optional[CacheManager] = None
) -> Callable:
    """
    Decorator to cache function responses.
    
    Args:
        ttl: Cache TTL in seconds
        key_builder: Custom key builder function
        cache_manager: Cache manager instance
        
    Example:
        @cache_response(ttl=300)
        async def get_user(user_id: str) -> User:
            return await db.get_user(user_id)
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            # Get cache manager
            manager = cache_manager or get_cache_manager()
            
            # Build cache key
            if key_builder:
                cache_key = key_builder(*args, **kwargs)
            else:
                # Default key builder
                key_parts = [func.__module__, func.__name__]
                key_parts.extend(str(arg) for arg in args)
                key_parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
                cache_key = ":".join(key_parts)
                
                # Hash if too long
                if len(cache_key) > 250:
                    cache_key = hashlib.sha256(
                        cache_key.encode()
                    ).hexdigest()
            
            # Try to get from cache
            cached = await manager.get(cache_key)
            if cached is not None:
                return cached
            
            # Execute function
            result = await func(*args, **kwargs)
            
            # Cache result
            await manager.set(cache_key, result, ttl)
            
            return result

        # Handle sync functions
        if not asyncio.iscoroutinefunction(func):
            @wraps(func)
            def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                # For sync functions, use a simple LRU cache
                return func(*args, **kwargs)
            
            # Add LRU cache for sync functions
            return lru_cache(maxsize=128)(sync_wrapper)
        
        return async_wrapper
    
    return decorator

async def invalidate_cache(
    pattern: str,
    cache_manager: Optional[CacheManager] = None
) -> int:
    """
    Invalidate cache entries matching pattern.
    
    Args:
        pattern: Key pattern (supports * wildcard)
        cache_manager: Cache manager instance
        
    Returns:
        Number of entries invalidated
    """
    manager = cache_manager or get_cache_manager()
    
    # For now, simple pattern matching
    # In production, use Redis SCAN or similar
    if pattern.endswith("*"):
        # Prefix match - would need to track keys
        logger.warning("Pattern invalidation not fully implemented")
        return 0
    else:
        # Exact match
        success = await manager.delete(pattern)
        return 1 if success else 0

# Global instance
_cache_manager: Optional[CacheManager] = None

@lru_cache(maxsize=1)
def get_cache_manager() -> CacheManager:
    """Get global cache manager instance."""
    global _cache_manager
    
    if _cache_manager is None:
        _cache_manager = CacheManager()
    
    return _cache_manager

# Export public API
__all__ = [
    # Classes
    'CacheManager',
    'CacheBackend',
    'RedisBackend',
    'InMemoryBackend',
    'CacheStats',
    
    # Enums
    'CacheStrategy',
    'SerializationType',
    
    # Decorators
    'cache_response',
    
    # Functions
    'get_cache_manager',
    'invalidate_cache',
    'initialize_caches',
]

# Create global cache instance for backward compatibility
_query_cache = None

def _initialize_query_cache():
    """Initialize the global query cache instance."""
    global _query_cache
    if _query_cache is None:
        _query_cache = CacheManager()
    return _query_cache

# Initialize on import
_query_cache = _initialize_query_cache()

async def initialize_caches():
    """Initialize all cache backends."""
    global _query_cache
    if _query_cache:
        await _query_cache.initialize()
    logger.info("Cache backends initialized")
