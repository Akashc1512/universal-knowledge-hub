"""
Health Checks Module for Universal Knowledge Platform
Provides actual connectivity checks for all external services.
"""

import asyncio
import aiohttp
import logging
import os
from typing import Dict, Any, Optional, List
from datetime import datetime
import redis.asyncio as aioredis
from elasticsearch import AsyncElasticsearch
import time

logger = logging.getLogger(__name__)

# Service URLs from environment
VECTOR_DB_URL = os.getenv("VECTOR_DB_URL", "http://localhost:6333")
ELASTICSEARCH_URL = os.getenv("ELASTICSEARCH_URL", "http://localhost:9200")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
SPARQL_ENDPOINT = os.getenv("SPARQL_ENDPOINT", "http://localhost:7200/repositories/knowledge")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

# Timeout for health checks
HEALTH_CHECK_TIMEOUT = 5.0


async def check_vector_db() -> Dict[str, Any]:
    """Check Pinecone/Qdrant vector database connectivity."""
    start_time = time.time()
    try:
        async with aiohttp.ClientSession() as session:
            # Check Qdrant health endpoint
            async with session.get(
                f"{VECTOR_DB_URL}/healthz",
                timeout=aiohttp.ClientTimeout(total=HEALTH_CHECK_TIMEOUT)
            ) as response:
                if response.status == 200:
                    return {
                        "healthy": True,
                        "latency_ms": int((time.time() - start_time) * 1000),
                        "status_code": response.status
                    }
                else:
                    return {
                        "healthy": False,
                        "error": f"HTTP {response.status}",
                        "latency_ms": int((time.time() - start_time) * 1000)
                    }
    except asyncio.TimeoutError:
        return {
            "healthy": False,
            "error": "Timeout",
            "latency_ms": int(HEALTH_CHECK_TIMEOUT * 1000)
        }
    except Exception as e:
        logger.error(f"Vector DB health check failed: {e}")
        return {
            "healthy": False,
            "error": str(e),
            "latency_ms": int((time.time() - start_time) * 1000)
        }


async def check_elasticsearch() -> Dict[str, Any]:
    """Check Elasticsearch connectivity."""
    start_time = time.time()
    try:
        es = AsyncElasticsearch([ELASTICSEARCH_URL])
        health = await es.cluster.health()
        await es.close()
        
        return {
            "healthy": health["status"] in ["green", "yellow"],
            "status": health["status"],
            "cluster_name": health.get("cluster_name"),
            "latency_ms": int((time.time() - start_time) * 1000)
        }
    except asyncio.TimeoutError:
        return {
            "healthy": False,
            "error": "Timeout",
            "latency_ms": int(HEALTH_CHECK_TIMEOUT * 1000)
        }
    except Exception as e:
        logger.error(f"Elasticsearch health check failed: {e}")
        return {
            "healthy": False,
            "error": str(e),
            "latency_ms": int((time.time() - start_time) * 1000)
        }


async def check_redis() -> Dict[str, Any]:
    """Check Redis connectivity."""
    start_time = time.time()
    try:
        redis = await aioredis.from_url(
            REDIS_URL,
            socket_connect_timeout=HEALTH_CHECK_TIMEOUT
        )
        
        # Ping Redis
        await redis.ping()
        
        # Get basic info
        info = await redis.info()
        
        await redis.close()
        
        return {
            "healthy": True,
            "version": info.get("redis_version", "unknown"),
            "connected_clients": info.get("connected_clients", 0),
            "latency_ms": int((time.time() - start_time) * 1000)
        }
    except asyncio.TimeoutError:
        return {
            "healthy": False,
            "error": "Timeout",
            "latency_ms": int(HEALTH_CHECK_TIMEOUT * 1000)
        }
    except Exception as e:
        logger.error(f"Redis health check failed: {e}")
        return {
            "healthy": False,
            "error": str(e),
            "latency_ms": int((time.time() - start_time) * 1000)
        }


async def check_knowledge_graph() -> Dict[str, Any]:
    """Check SPARQL endpoint connectivity."""
    start_time = time.time()
    try:
        async with aiohttp.ClientSession() as session:
            # Simple SPARQL query to check connectivity
            query = "SELECT ?s WHERE { ?s ?p ?o } LIMIT 1"
            
            async with session.post(
                SPARQL_ENDPOINT,
                data={"query": query},
                headers={"Accept": "application/json"},
                timeout=aiohttp.ClientTimeout(total=HEALTH_CHECK_TIMEOUT)
            ) as response:
                if response.status == 200:
                    return {
                        "healthy": True,
                        "latency_ms": int((time.time() - start_time) * 1000),
                        "status_code": response.status
                    }
                else:
                    return {
                        "healthy": False,
                        "error": f"HTTP {response.status}",
                        "latency_ms": int((time.time() - start_time) * 1000)
                    }
    except asyncio.TimeoutError:
        return {
            "healthy": False,
            "error": "Timeout",
            "latency_ms": int(HEALTH_CHECK_TIMEOUT * 1000)
        }
    except Exception as e:
        logger.error(f"Knowledge graph health check failed: {e}")
        return {
            "healthy": False,
            "error": str(e),
            "latency_ms": int((time.time() - start_time) * 1000)
        }


async def check_openai_api() -> Dict[str, Any]:
    """Check OpenAI API connectivity."""
    start_time = time.time()
    
    if not OPENAI_API_KEY:
        return {
            "healthy": False,
            "error": "API key not configured",
            "latency_ms": 0
        }
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                "https://api.openai.com/v1/models",
                headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
                timeout=aiohttp.ClientTimeout(total=HEALTH_CHECK_TIMEOUT)
            ) as response:
                if response.status == 200:
                    return {
                        "healthy": True,
                        "latency_ms": int((time.time() - start_time) * 1000),
                        "status_code": response.status
                    }
                else:
                    return {
                        "healthy": False,
                        "error": f"HTTP {response.status}",
                        "latency_ms": int((time.time() - start_time) * 1000)
                    }
    except asyncio.TimeoutError:
        return {
            "healthy": False,
            "error": "Timeout",
            "latency_ms": int(HEALTH_CHECK_TIMEOUT * 1000)
        }
    except Exception as e:
        logger.error(f"OpenAI API health check failed: {e}")
        return {
            "healthy": False,
            "error": str(e),
            "latency_ms": int((time.time() - start_time) * 1000)
        }


async def check_anthropic_api() -> Dict[str, Any]:
    """Check Anthropic API connectivity."""
    start_time = time.time()
    
    if not ANTHROPIC_API_KEY:
        return {
            "healthy": False,
            "error": "API key not configured",
            "latency_ms": 0
        }
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": ANTHROPIC_API_KEY,
                    "anthropic-version": "2023-06-01"
                },
                timeout=aiohttp.ClientTimeout(total=HEALTH_CHECK_TIMEOUT)
            ) as response:
                # Anthropic returns 401 for GET requests, but that's expected
                if response.status in [401, 405]:
                    return {
                        "healthy": True,
                        "latency_ms": int((time.time() - start_time) * 1000),
                        "status_code": response.status,
                        "note": "API accessible"
                    }
                else:
                    return {
                        "healthy": False,
                        "error": f"Unexpected HTTP {response.status}",
                        "latency_ms": int((time.time() - start_time) * 1000)
                    }
    except asyncio.TimeoutError:
        return {
            "healthy": False,
            "error": "Timeout",
            "latency_ms": int(HEALTH_CHECK_TIMEOUT * 1000)
        }
    except Exception as e:
        logger.error(f"Anthropic API health check failed: {e}")
        return {
            "healthy": False,
            "error": str(e),
            "latency_ms": int((time.time() - start_time) * 1000)
        }


async def check_all_services() -> Dict[str, Any]:
    """
    Check all external services in parallel.
    
    Returns:
        Dictionary with health status of all services
    """
    start_time = time.time()
    
    # Run all health checks in parallel
    results = await asyncio.gather(
        check_vector_db(),
        check_elasticsearch(),
        check_redis(),
        check_knowledge_graph(),
        check_openai_api(),
        check_anthropic_api(),
        return_exceptions=True
    )
    
    # Map results to service names
    service_names = [
        "vector_database",
        "elasticsearch",
        "redis",
        "knowledge_graph",
        "openai_api",
        "anthropic_api"
    ]
    
    health_status = {}
    all_healthy = True
    
    for name, result in zip(service_names, results):
        if isinstance(result, Exception):
            health_status[name] = {
                "healthy": False,
                "error": str(result)
            }
            all_healthy = False
        else:
            health_status[name] = result
            if not result.get("healthy", False):
                all_healthy = False
    
    return {
        "all_healthy": all_healthy,
        "services": health_status,
        "total_latency_ms": int((time.time() - start_time) * 1000),
        "timestamp": datetime.utcnow().isoformat()
    }


async def get_service_dependencies() -> Dict[str, List[str]]:
    """
    Get service dependency information.
    
    Returns:
        Dictionary mapping services to their dependencies
    """
    return {
        "api": ["redis", "elasticsearch", "vector_database"],
        "retrieval_agent": ["vector_database", "elasticsearch", "knowledge_graph"],
        "fact_check_agent": ["openai_api", "anthropic_api"],
        "synthesis_agent": ["openai_api", "anthropic_api"],
        "citation_agent": [],
        "cache": ["redis"],
        "analytics": ["redis"]
    } 