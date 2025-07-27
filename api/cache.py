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
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


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
    """Least Recently Used cache implementation."""
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.cache: OrderedDict = OrderedDict()
        self.stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'size': 0
        }
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
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
    
    def put(self, key: str, value: Any, ttl: int = 3600, metadata: Dict[str, Any] = None) -> None:
        """Put value in cache."""
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
    
    def clear(self):
        """Clear all cache entries."""
        self.cache.clear()
        self.stats['size'] = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        hit_rate = self.stats['hits'] / (self.stats['hits'] + self.stats['misses']) if (self.stats['hits'] + self.stats['misses']) > 0 else 0
        return {
            **self.stats,
            'hit_rate': hit_rate,
            'current_size': len(self.cache)
        }


class QueryCache:
    """Specialized cache for query results."""
    
    def __init__(self, max_size: int = 500):
        self.cache = LRUCache(max_size)
        self.query_patterns = {
            'factual': int(os.getenv('FACTUAL_QUERY_TTL', '1800')),  # 30 minutes
            'analytical': int(os.getenv('ANALYTICAL_QUERY_TTL', '3600')),  # 1 hour
            'creative': int(os.getenv('CREATIVE_QUERY_TTL', '7200')),  # 2 hours
            'default': int(os.getenv('DEFAULT_QUERY_TTL', '3600'))  # 1 hour
        }
    
    def _generate_key(self, query: str, user_context: Dict[str, Any] = None) -> str:
        """Generate cache key from query and context."""
        # Normalize query
        normalized_query = query.strip().lower()
        
        # Create context hash
        context_hash = ""
        if user_context:
            # Sort keys for consistent hashing
            sorted_context = json.dumps(user_context, sort_keys=True)
            context_hash = hashlib.md5(sorted_context.encode()).hexdigest()
        
        # Combine query and context
        combined = f"{normalized_query}:{context_hash}"
        return hashlib.md5(combined.encode()).hexdigest()
    
    def _determine_ttl(self, query: str, result: Dict[str, Any]) -> int:
        """Determine TTL based on query type and result confidence."""
        query_lower = query.lower()
        
        # Check for factual queries
        if any(word in query_lower for word in ['what is', 'who is', 'when', 'where', 'how many']):
            return self.query_patterns['factual']
        
        # Check for analytical queries
        if any(word in query_lower for word in ['compare', 'analyze', 'explain', 'why']):
            return self.query_patterns['analytical']
        
        # Check for creative queries
        if any(word in query_lower for word in ['imagine', 'create', 'design', 'suggest']):
            return self.query_patterns['creative']
        
        # Default TTL
        return self.query_patterns['default']
    
    def get(self, query: str, user_context: Dict[str, Any] = None) -> Optional[Dict[str, Any]]:
        """Get cached query result."""
        key = self._generate_key(query, user_context)
        return self.cache.get(key)
    
    def put(self, query: str, result: Dict[str, Any], user_context: Dict[str, Any] = None) -> None:
        """Cache query result."""
        key = self._generate_key(query, user_context)
        ttl = self._determine_ttl(query, result)
        
        # Add cache metadata
        metadata = {
            'query_type': 'cached',
            'original_query': query,
            'user_context': user_context,
            'confidence': result.get('confidence', 0.0),
            'cached_at': datetime.now().isoformat()
        }
        
        self.cache.put(key, result, ttl, metadata)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return self.cache.get_stats()


class SemanticCache:
    """Semantic similarity-based cache."""
    
    def __init__(self, max_size: int = 200):
        self.cache = LRUCache(max_size)
        self.similarity_threshold = 0.85
    
    def _calculate_similarity(self, query1: str, query2: str) -> float:
        """Calculate semantic similarity between queries."""
        # Simple implementation - can be enhanced with embeddings
        words1 = set(query1.lower().split())
        words2 = set(query2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union)
    
    def find_similar(self, query: str) -> Optional[Dict[str, Any]]:
        """Find semantically similar cached query."""
        best_match = None
        best_similarity = 0.0
        
        for key, entry in self.cache.cache.items():
            if entry.is_expired():
                continue
            
            cached_query = entry.metadata.get('original_query', '')
            similarity = self._calculate_similarity(query, cached_query)
            
            if similarity > self.similarity_threshold and similarity > best_similarity:
                best_similarity = similarity
                best_match = entry.value
        
        return best_match
    
    def put(self, query: str, result: Dict[str, Any], user_context: Dict[str, Any] = None) -> None:
        """Cache query with semantic metadata."""
        key = hashlib.md5(query.encode()).hexdigest()
        ttl = int(os.getenv('SEMANTIC_QUERY_TTL', '3600'))  # 1 hour default
        
        metadata = {
            'original_query': query,
            'user_context': user_context,
            'semantic_key': key
        }
        
        self.cache.put(key, result, ttl, metadata)


# Global cache instances
query_cache = QueryCache()
semantic_cache = SemanticCache()


async def get_cached_result(query: str, user_context: Dict[str, Any] = None) -> Optional[Dict[str, Any]]:
    """Get cached result for query."""
    # Try exact match first
    result = query_cache.get(query, user_context)
    if result:
        logger.info(f"Cache HIT (exact): {query[:50]}...")
        return result
    
    # Try semantic similarity
    result = semantic_cache.find_similar(query)
    if result:
        logger.info(f"Cache HIT (semantic): {query[:50]}...")
        return result
    
    logger.info(f"Cache MISS: {query[:50]}...")
    return None


async def cache_result(query: str, result: Dict[str, Any], user_context: Dict[str, Any] = None) -> None:
    """Cache query result."""
    # Cache in both exact and semantic caches
    query_cache.put(query, result, user_context)
    semantic_cache.put(query, result, user_context)
    logger.info(f"Cached result for: {query[:50]}...")


def get_cache_stats() -> Dict[str, Any]:
    """Get comprehensive cache statistics."""
    return {
        'query_cache': query_cache.get_stats(),
        'semantic_cache': semantic_cache.cache.get_stats(),
        'total_entries': query_cache.cache.stats['size'] + semantic_cache.cache.stats['size']
    } 