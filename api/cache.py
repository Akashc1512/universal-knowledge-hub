"""
Caching System for Universal Knowledge Platform
Provides intelligent caching for queries, results, and metadata.
"""

import hashlib
import json
import time
import os
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import logging
from collections import OrderedDict
import asyncio
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

# Global thread pool for CPU-intensive operations
_cache_thread_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="cache")


@dataclass
class CacheEntry:
    """Cache entry with metadata."""
    key: str
    value: Any
    timestamp: float
    ttl: int  # Time to live in seconds
    access_count: int = 0
    last_accessed: float = 0.0
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        self.last_accessed = self.timestamp
    
    def is_expired(self) -> bool:
        """Check if cache entry has expired."""
        return time.time() - self.timestamp > self.ttl
    
    def access(self):
        """Record access to cache entry."""
        self.access_count += 1
        self.last_accessed = time.time()


class LRUCache:
    """Least Recently Used cache implementation with async operations."""
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.cache: OrderedDict = OrderedDict()
        self.stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'size': 0
        }
        self._lock = asyncio.Lock()
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache (async)."""
        async with self._lock:
            if key in self.cache:
                entry = self.cache[key]
                if not entry.is_expired():
                    # Move to end (most recently used)
                    self.cache.move_to_end(key)
                    entry.access()
                    self.stats['hits'] += 1
                    return entry.value
                else:
                    # Remove expired entry
                    del self.cache[key]
                    self.stats['size'] -= 1
            
            self.stats['misses'] += 1
            return None
    
    async def put(self, key: str, value: Any, ttl: int = 3600, metadata: Dict[str, Any] = None) -> None:
        """Put value in cache (async)."""
        async with self._lock:
            # Remove if exists
            if key in self.cache:
                del self.cache[key]
                self.stats['size'] -= 1
            
            # Evict oldest if cache is full
            while len(self.cache) >= self.max_size:
                oldest_key = next(iter(self.cache))
                del self.cache[oldest_key]
                self.stats['evictions'] += 1
                self.stats['size'] -= 1
            
            # Add new entry
            entry = CacheEntry(
                key=key,
                value=value,
                timestamp=time.time(),
                ttl=ttl,
                metadata=metadata or {}
            )
            
            self.cache[key] = entry
            self.stats['size'] += 1
    
    async def clear(self):
        """Clear all cache entries."""
        async with self._lock:
            self.cache.clear()
            self.stats['size'] = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            **self.stats,
            'max_size': self.max_size,
            'hit_rate': self.stats['hits'] / (self.stats['hits'] + self.stats['misses']) if (self.stats['hits'] + self.stats['misses']) > 0 else 0
        }


class QueryCache:
    """Optimized query cache with reduced memory usage."""
    
    def __init__(self, max_size: int = 500):
        self.cache = LRUCache(max_size)
    
    def _generate_key_sync(self, query: str, user_context: Dict[str, Any] = None) -> str:
        """Generate cache key (synchronous version for thread pool)."""
        # Create a simplified context for hashing
        context_str = ""
        if user_context:
            # Only include essential context fields to reduce key size
            essential_context = {
                'user_id': user_context.get('user_id'),
                'session_id': user_context.get('session_id'),
                'preferences': user_context.get('preferences', {})
            }
            context_str = json.dumps(essential_context, sort_keys=True)
        
        # Create hash of query + context
        key_data = f"{query}:{context_str}".encode('utf-8')
        return hashlib.md5(key_data).hexdigest()
    
    async def _generate_key(self, query: str, user_context: Dict[str, Any] = None) -> str:
        """Generate cache key (async)."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            _cache_thread_pool,
            self._generate_key_sync,
            query,
            user_context
        )
    
    def _determine_ttl(self, query: str, result: Dict[str, Any]) -> int:
        """Determine TTL based on query and result characteristics."""
        # Base TTL
        ttl = 3600  # 1 hour default
        
        # Adjust based on query type
        query_lower = query.lower()
        if any(word in query_lower for word in ['news', 'current', 'recent', 'latest']):
            ttl = 1800  # 30 minutes for time-sensitive queries
        elif any(word in query_lower for word in ['definition', 'what is', 'meaning']):
            ttl = 7200  # 2 hours for definition queries
        elif any(word in query_lower for word in ['research', 'study', 'analysis']):
            ttl = 14400  # 4 hours for research queries
        
        # Adjust based on result confidence
        confidence = result.get('confidence', 0.0)
        if confidence > 0.9:
            ttl = int(ttl * 1.5)  # Longer TTL for high-confidence results
        elif confidence < 0.5:
            ttl = int(ttl * 0.5)  # Shorter TTL for low-confidence results
        
        return ttl
    
    def _optimize_result_for_cache(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize result for caching by removing unnecessary data."""
        # Create a minimal cached version
        cached_result = {
            'answer': result.get('answer', ''),
            'confidence': result.get('confidence', 0.0),
            'citations': result.get('citations', []),
            'metadata': {
                'agents_used': result.get('metadata', {}).get('agents_used', []),
                'synthesis_method': result.get('metadata', {}).get('synthesis_method', 'unknown'),
                'fact_count': result.get('metadata', {}).get('fact_count', 0)
            }
        }
        
        # Limit citation data to essential fields
        optimized_citations = []
        for citation in cached_result['citations']:
            optimized_citation = {
                'title': citation.get('title', ''),
                'url': citation.get('url', ''),
                'author': citation.get('author', ''),
                'year': citation.get('year', '')
            }
            optimized_citations.append(optimized_citation)
        
        cached_result['citations'] = optimized_citations
        return cached_result
    
    async def get(self, query: str, user_context: Dict[str, Any] = None) -> Optional[Dict[str, Any]]:
        """Get cached result (async)."""
        key = await self._generate_key(query, user_context)
        return await self.cache.get(key)
    
    async def put(self, query: str, result: Dict[str, Any], user_context: Dict[str, Any] = None) -> None:
        """Put result in cache (async)."""
        key = await self._generate_key(query, user_context)
        optimized_result = self._optimize_result_for_cache(result)
        ttl = self._determine_ttl(query, result)
        
        await self.cache.put(key, optimized_result, ttl, {
            'original_query': query,
            'user_context_keys': list(user_context.keys()) if user_context else []
        })
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return self.cache.get_stats()


class SemanticCache:
    """Semantic cache for similar queries."""
    
    def __init__(self, max_size: int = 200):
        self.cache = LRUCache(max_size)
        self.similarity_threshold = 0.85
    
    def _calculate_similarity_sync(self, query1: str, query2: str) -> float:
        """Calculate similarity between queries (synchronous version)."""
        # Simple word overlap similarity
        words1 = set(query1.lower().split())
        words2 = set(query2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union)
    
    async def _calculate_similarity(self, query1: str, query2: str) -> float:
        """Calculate similarity between queries (async)."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            _cache_thread_pool,
            self._calculate_similarity_sync,
            query1,
            query2
        )
    
    async def find_similar(self, query: str) -> Optional[Dict[str, Any]]:
        """Find similar cached query (async)."""
        # Check all cached entries for similarity
        async with self.cache._lock:
            for key, entry in self.cache.cache.items():
                if not entry.is_expired():
                    cached_query = entry.metadata.get('original_query', '')
                    if cached_query:
                        similarity = await self._calculate_similarity(query, cached_query)
                        if similarity >= self.similarity_threshold:
                            return entry.value
        
        return None
    
    async def put(self, query: str, result: Dict[str, Any], user_context: Dict[str, Any] = None) -> None:
        """Put result in semantic cache (async)."""
        # Use a hash of the query as key
        key = hashlib.md5(query.encode('utf-8')).hexdigest()
        
        await self.cache.put(key, result, 3600, {
            'original_query': query,
            'user_context_keys': list(user_context.keys()) if user_context else []
        })


# Global cache instances
_query_cache = QueryCache()
_semantic_cache = SemanticCache()


async def get_cached_result(query: str, user_context: Dict[str, Any] = None) -> Optional[Dict[str, Any]]:
    """Get cached result for query (async)."""
    # First check exact match
    result = await _query_cache.get(query, user_context)
    if result:
        return result
    
    # Then check semantic similarity
    result = await _semantic_cache.find_similar(query)
    return result


async def cache_result(query: str, result: Dict[str, Any], user_context: Dict[str, Any] = None) -> None:
    """Cache result for query (async)."""
    # Cache in both exact and semantic caches
    await _query_cache.put(query, result, user_context)
    await _semantic_cache.put(query, result, user_context)


async def get_cache_stats() -> Dict[str, Any]:
    """Get comprehensive cache statistics."""
    query_stats = await _query_cache.get_stats()
    semantic_stats = await _semantic_cache.cache.get_stats()
    
    return {
        'query_cache': query_stats,
        'semantic_cache': semantic_stats,
        'total_memory_usage': query_stats['size'] + semantic_stats['size']
    } 