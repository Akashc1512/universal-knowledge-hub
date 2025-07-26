"""
Semantic Cache Manager - Handles caching of query results with semantic similarity.
"""

import asyncio
import time
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import hashlib
import json

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Represents a cached query result."""
    query: str
    response: Dict[str, Any]
    timestamp: float
    ttl: int  # Time to live in seconds
    similarity_score: float = 0.0


class SemanticCacheManager:
    """
    Manages caching of query results with semantic similarity for query reuse.
    """
    
    def __init__(self, similarity_threshold: float = 0.95, max_cache_size: int = 1000):
        self.similarity_threshold = similarity_threshold
        self.max_cache_size = max_cache_size
        self.cache: Dict[str, CacheEntry] = {}
        self.query_embeddings: Dict[str, List[float]] = {}  # TODO: Store actual embeddings
        
    async def get_cached_response(self, query: str) -> Optional[Dict[str, Any]]:
        """
        Get a cached response for a query, using semantic similarity.
        
        Args:
            query: The query to look up
            
        Returns:
            Cached response if found and not expired, None otherwise
        """
        # First, try exact match
        cache_key = self._generate_cache_key(query)
        if cache_key in self.cache:
            entry = self.cache[cache_key]
            if not self._is_expired(entry):
                logger.info(f"Cache hit (exact match) for query: {query[:50]}...")
                return entry.response
        
        # Then, try semantic similarity
        similar_entry = await self._find_similar_query(query)
        if similar_entry and not self._is_expired(similar_entry):
            logger.info(f"Cache hit (semantic similarity) for query: {query[:50]}...")
            return similar_entry.response
        
        logger.debug(f"Cache miss for query: {query[:50]}...")
        return None
    
    async def cache_response(self, query: str, response: Dict[str, Any], ttl: int = 3600):
        """
        Cache a query response.
        
        Args:
            query: The query that was processed
            response: The response to cache
            ttl: Time to live in seconds
        """
        cache_key = self._generate_cache_key(query)
        
        # Remove oldest entries if cache is full
        if len(self.cache) >= self.max_cache_size:
            self._evict_oldest_entries()
        
        entry = CacheEntry(
            query=query,
            response=response,
            timestamp=time.time(),
            ttl=ttl
        )
        
        self.cache[cache_key] = entry
        logger.debug(f"Cached response for query: {query[:50]}...")
    
    def invalidate_cache(self, pattern: Optional[str] = None):
        """
        Invalidate cache entries.
        
        Args:
            pattern: Optional pattern to match against queries. If None, clears all cache.
        """
        if pattern is None:
            self.cache.clear()
            self.query_embeddings.clear()
            logger.info("Cache cleared")
        else:
            keys_to_remove = [
                key for key, entry in self.cache.items()
                if pattern.lower() in entry.query.lower()
            ]
            for key in keys_to_remove:
                del self.cache[key]
            logger.info(f"Invalidated {len(keys_to_remove)} cache entries matching pattern: {pattern}")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_entries = len(self.cache)
        expired_entries = sum(1 for entry in self.cache.values() if self._is_expired(entry))
        active_entries = total_entries - expired_entries
        
        return {
            'total_entries': total_entries,
            'active_entries': active_entries,
            'expired_entries': expired_entries,
            'cache_size_mb': self._estimate_cache_size_mb(),
            'similarity_threshold': self.similarity_threshold
        }
    
    def _generate_cache_key(self, query: str) -> str:
        """Generate a cache key for a query."""
        return hashlib.md5(query.encode()).hexdigest()
    
    def _is_expired(self, entry: CacheEntry) -> bool:
        """Check if a cache entry is expired."""
        return time.time() - entry.timestamp > entry.ttl
    
    async def _find_similar_query(self, query: str) -> Optional[CacheEntry]:
        """
        Find a similar query in the cache using semantic similarity.
        
        TODO: Implement actual embedding-based similarity search.
        For now, uses simple keyword overlap as a placeholder.
        """
        query_words = set(query.lower().split())
        
        best_match = None
        best_score = 0.0
        
        for entry in self.cache.values():
            if self._is_expired(entry):
                continue
                
            entry_words = set(entry.query.lower().split())
            overlap = len(query_words.intersection(entry_words))
            total_words = len(query_words.union(entry_words))
            
            if total_words > 0:
                similarity = overlap / total_words
                if similarity > best_score and similarity >= self.similarity_threshold:
                    best_score = similarity
                    best_match = entry
        
        return best_match
    
    def _evict_oldest_entries(self, num_to_evict: int = 10):
        """Evict the oldest cache entries."""
        sorted_entries = sorted(
            self.cache.items(),
            key=lambda x: x[1].timestamp
        )
        
        for i in range(min(num_to_evict, len(sorted_entries))):
            key, _ = sorted_entries[i]
            del self.cache[key]
        
        logger.debug(f"Evicted {min(num_to_evict, len(sorted_entries))} oldest cache entries")
    
    def _estimate_cache_size_mb(self) -> float:
        """Estimate cache size in MB."""
        total_size = 0
        for entry in self.cache.values():
            # Rough estimate: query + response as JSON string
            entry_size = len(entry.query) + len(json.dumps(entry.response))
            total_size += entry_size
        
        return total_size / (1024 * 1024)  # Convert to MB 