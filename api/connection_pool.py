"""
Connection Pooling Module for Universal Knowledge Platform
Manages connection pools for all external services to improve performance.
"""

import aiohttp
import asyncio
import logging
import os
from typing import Optional, Dict, Any
import redis.asyncio as aioredis
from elasticsearch import AsyncElasticsearch
from contextlib import asynccontextmanager
import time

logger = logging.getLogger(__name__)

# Connection pool configuration
POOL_SIZE = int(os.getenv("CONNECTION_POOL_SIZE", "10"))
POOL_TIMEOUT = float(os.getenv("CONNECTION_POOL_TIMEOUT", "30.0"))
MAX_KEEPALIVE_TIME = int(os.getenv("MAX_KEEPALIVE_TIME", "300"))  # 5 minutes

# Service URLs
VECTOR_DB_URL = os.getenv("VECTOR_DB_URL", "http://localhost:6333")
ELASTICSEARCH_URL = os.getenv("ELASTICSEARCH_URL", "http://localhost:9200")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
SPARQL_ENDPOINT = os.getenv("SPARQL_ENDPOINT", "http://localhost:7200/repositories/knowledge")


class ConnectionPoolManager:
    """Manages connection pools for all external services."""
    
    def __init__(self):
        self._http_session: Optional[aiohttp.ClientSession] = None
        self._redis_pool: Optional[aioredis.Redis] = None
        self._elasticsearch_client: Optional[AsyncElasticsearch] = None
        self._initialized = False
        self._lock = asyncio.Lock()
        
    async def initialize(self):
        """Initialize all connection pools."""
        async with self._lock:
            if self._initialized:
                return
                
            logger.info("Initializing connection pools...")
            
            # Initialize HTTP session with connection pooling
            connector = aiohttp.TCPConnector(
                limit=POOL_SIZE,
                limit_per_host=POOL_SIZE,
                ttl_dns_cache=300,
                keepalive_timeout=MAX_KEEPALIVE_TIME,
                enable_cleanup_closed=True
            )
            
            timeout = aiohttp.ClientTimeout(
                total=POOL_TIMEOUT,
                connect=5.0,
                sock_read=POOL_TIMEOUT
            )
            
            self._http_session = aiohttp.ClientSession(
                connector=connector,
                timeout=timeout,
                headers={
                    "User-Agent": "UniversalKnowledgePlatform/1.0"
                }
            )
            
            # Initialize Redis connection pool
            try:
                self._redis_pool = await aioredis.from_url(
                    REDIS_URL,
                    max_connections=POOL_SIZE,
                    socket_connect_timeout=5.0,
                    socket_keepalive=True,
                    health_check_interval=30
                )
                await self._redis_pool.ping()
                logger.info("Redis connection pool initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize Redis pool: {e}")
                self._redis_pool = None
            
            # Initialize Elasticsearch client with connection pooling
            try:
                self._elasticsearch_client = AsyncElasticsearch(
                    [ELASTICSEARCH_URL],
                    maxsize=POOL_SIZE,
                    sniff_on_start=False,
                    sniff_on_connection_fail=True,
                    sniffer_timeout=60,
                    retry_on_timeout=True,
                    max_retries=3
                )
                await self._elasticsearch_client.ping()
                logger.info("Elasticsearch connection pool initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize Elasticsearch pool: {e}")
                self._elasticsearch_client = None
            
            self._initialized = True
            logger.info("All connection pools initialized successfully")
    
    async def shutdown(self):
        """Shutdown all connection pools gracefully."""
        async with self._lock:
            if not self._initialized:
                return
                
            logger.info("Shutting down connection pools...")
            
            # Close HTTP session
            if self._http_session:
                await self._http_session.close()
                # Wait for connector cleanup
                await asyncio.sleep(0.25)
            
            # Close Redis pool
            if self._redis_pool:
                await self._redis_pool.close()
                await self._redis_pool.wait_closed()
            
            # Close Elasticsearch client
            if self._elasticsearch_client:
                await self._elasticsearch_client.close()
            
            self._initialized = False
            logger.info("All connection pools shut down successfully")
    
    @asynccontextmanager
    async def get_http_session(self):
        """Get HTTP session from pool."""
        if not self._initialized:
            await self.initialize()
        
        if not self._http_session:
            raise RuntimeError("HTTP session not initialized")
            
        yield self._http_session
    
    @asynccontextmanager
    async def get_redis_connection(self):
        """Get Redis connection from pool."""
        if not self._initialized:
            await self.initialize()
        
        if not self._redis_pool:
            raise RuntimeError("Redis pool not initialized")
            
        yield self._redis_pool
    
    @asynccontextmanager
    async def get_elasticsearch_client(self):
        """Get Elasticsearch client from pool."""
        if not self._initialized:
            await self.initialize()
        
        if not self._elasticsearch_client:
            raise RuntimeError("Elasticsearch client not initialized")
            
        yield self._elasticsearch_client
    
    async def make_http_request(
        self,
        method: str,
        url: str,
        **kwargs
    ) -> aiohttp.ClientResponse:
        """
        Make HTTP request using pooled connection.
        
        Args:
            method: HTTP method (GET, POST, etc.)
            url: Target URL
            **kwargs: Additional arguments for the request
            
        Returns:
            Response object
        """
        async with self.get_http_session() as session:
            async with session.request(method, url, **kwargs) as response:
                return response
    
    async def vector_db_request(
        self,
        endpoint: str,
        method: str = "GET",
        **kwargs
    ) -> Dict[str, Any]:
        """
        Make request to vector database using pooled connection.
        
        Args:
            endpoint: API endpoint
            method: HTTP method
            **kwargs: Request parameters
            
        Returns:
            JSON response
        """
        url = f"{VECTOR_DB_URL}{endpoint}"
        
        async with self.get_http_session() as session:
            async with session.request(method, url, **kwargs) as response:
                response.raise_for_status()
                return await response.json()
    
    async def sparql_query(
        self,
        query: str,
        timeout: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Execute SPARQL query using pooled connection.
        
        Args:
            query: SPARQL query string
            timeout: Query timeout
            
        Returns:
            Query results
        """
        async with self.get_http_session() as session:
            async with session.post(
                SPARQL_ENDPOINT,
                data={"query": query},
                headers={"Accept": "application/json"},
                timeout=timeout or POOL_TIMEOUT
            ) as response:
                response.raise_for_status()
                return await response.json()
    
    def get_pool_stats(self) -> Dict[str, Any]:
        """Get statistics about connection pools."""
        stats = {
            "initialized": self._initialized,
            "pools": {}
        }
        
        if self._http_session and self._http_session.connector:
            connector = self._http_session.connector
            stats["pools"]["http"] = {
                "limit": connector.limit,
                "limit_per_host": connector.limit_per_host,
                "connections": len(connector._conns) if hasattr(connector, '_conns') else 0
            }
        
        if self._redis_pool:
            stats["pools"]["redis"] = {
                "max_connections": POOL_SIZE,
                "initialized": True
            }
        
        if self._elasticsearch_client:
            stats["pools"]["elasticsearch"] = {
                "max_connections": POOL_SIZE,
                "initialized": True
            }
        
        return stats


# Global connection pool manager instance
_pool_manager: Optional[ConnectionPoolManager] = None
_pool_lock = asyncio.Lock()


async def get_pool_manager() -> ConnectionPoolManager:
    """Get or create the global connection pool manager."""
    global _pool_manager
    
    if _pool_manager is None:
        async with _pool_lock:
            if _pool_manager is None:
                _pool_manager = ConnectionPoolManager()
                await _pool_manager.initialize()
    
    return _pool_manager


async def shutdown_pools():
    """Shutdown all connection pools."""
    global _pool_manager
    
    if _pool_manager:
        await _pool_manager.shutdown()
        _pool_manager = None


# Convenience functions for common operations
async def make_pooled_request(method: str, url: str, **kwargs) -> aiohttp.ClientResponse:
    """Make HTTP request using pooled connection."""
    manager = await get_pool_manager()
    return await manager.make_http_request(method, url, **kwargs)


async def get_redis_connection():
    """Get Redis connection from pool."""
    manager = await get_pool_manager()
    async with manager.get_redis_connection() as redis:
        yield redis


async def get_elasticsearch_client():
    """Get Elasticsearch client from pool."""
    manager = await get_pool_manager()
    async with manager.get_elasticsearch_client() as client:
        yield client 