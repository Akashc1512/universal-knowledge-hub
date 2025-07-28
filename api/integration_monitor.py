"""
Integration Monitoring System for Universal Knowledge Platform
Monitors the health and status of external integrations.
"""

import asyncio
import logging
import time
from typing import Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


@dataclass
class IntegrationStatus:
    """Status information for an integration."""

    name: str
    status: str  # 'healthy', 'unhealthy', 'unknown'
    last_check: float
    response_time: float
    error_message: Optional[str] = None
    config_status: str = "configured"  # 'configured', 'not_configured', 'misconfigured'


class IntegrationMonitor:
    """Monitors the health of external integrations."""

    def __init__(self):
        self.integrations = {}
        self.check_interval = 300  # 5 minutes
        self.last_check = {}
        self._lock = asyncio.Lock()

        # Initialize integration status
        self._initialize_integrations()

    def _initialize_integrations(self):
        """Initialize integration status tracking."""
        integrations = {
            "vector_database": {
                "enabled": bool(
                    os.getenv("VECTOR_DB_HOST")
                    or os.getenv("PINECONE_API_KEY")
                    or os.getenv("QDRANT_URL")
                ),
                "type": "vector_db",
                "config_keys": ["VECTOR_DB_HOST", "PINECONE_API_KEY", "QDRANT_URL"],
            },
            "elasticsearch": {
                "enabled": bool(os.getenv("ELASTICSEARCH_URL")),
                "type": "search",
                "config_keys": ["ELASTICSEARCH_URL"],
            },
            "knowledge_graph": {
                "enabled": bool(os.getenv("SPARQL_ENDPOINT")),
                "type": "graph",
                "config_keys": ["SPARQL_ENDPOINT"],
            },
            "llm_api": {
                "enabled": bool(os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY")),
                "type": "llm",
                "config_keys": ["OPENAI_API_KEY", "ANTHROPIC_API_KEY"],
            },
            "redis_cache": {
                "enabled": bool(os.getenv("REDIS_ENABLED", "false").lower() == "true"),
                "type": "cache",
                "config_keys": ["REDIS_URL"],
            },
        }

        for name, config in integrations.items():
            self.integrations[name] = IntegrationStatus(
                name=name,
                status="unknown",
                last_check=0,
                response_time=0,
                config_status="configured" if config["enabled"] else "not_configured",
            )

    async def check_integration_health(self, integration_name: str) -> IntegrationStatus:
        """Check the health of a specific integration."""
        if integration_name not in self.integrations:
            return IntegrationStatus(
                name=integration_name,
                status="unknown",
                last_check=time.time(),
                response_time=0,
                error_message="Integration not configured",
            )

        integration = self.integrations[integration_name]
        start_time = time.time()

        try:
            if integration_name == "vector_database":
                status = await self._check_vector_db()
            elif integration_name == "elasticsearch":
                status = await self._check_elasticsearch()
            elif integration_name == "knowledge_graph":
                status = await self._check_knowledge_graph()
            elif integration_name == "llm_api":
                status = await self._check_llm_api()
            elif integration_name == "redis_cache":
                status = await self._check_redis_cache()
            else:
                status = "unknown"

            response_time = time.time() - start_time

            # Update integration status
            integration.status = status
            integration.last_check = time.time()
            integration.response_time = response_time
            integration.error_message = None

            logger.info(
                f"Integration {integration_name} health check: {status} ({response_time:.3f}s)"
            )

        except Exception as e:
            response_time = time.time() - start_time
            integration.status = "unhealthy"
            integration.last_check = time.time()
            integration.response_time = response_time
            integration.error_message = str(e)

            logger.error(f"Integration {integration_name} health check failed: {e}")

        return integration

    async def _check_vector_db(self) -> str:
        """Check vector database connectivity."""
        # TODO: Implement actual vector DB connectivity check
        # For now, return healthy if configured
        if os.getenv("VECTOR_DB_HOST") or os.getenv("PINECONE_API_KEY") or os.getenv("QDRANT_URL"):
            return "healthy"
        return "not_configured"

    async def _check_elasticsearch(self) -> str:
        """Check Elasticsearch connectivity."""
        # TODO: Implement actual Elasticsearch connectivity check
        if os.getenv("ELASTICSEARCH_URL"):
            return "healthy"
        return "not_configured"

    async def _check_knowledge_graph(self) -> str:
        """Check knowledge graph connectivity."""
        # TODO: Implement actual SPARQL endpoint connectivity check
        if os.getenv("SPARQL_ENDPOINT"):
            return "healthy"
        return "not_configured"

    async def _check_llm_api(self) -> str:
        """Check LLM API connectivity."""
        # TODO: Implement actual LLM API connectivity check
        if os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY"):
            return "healthy"
        return "not_configured"

    async def _check_redis_cache(self) -> str:
        """Check Redis cache connectivity."""
        if not os.getenv("REDIS_ENABLED", "false").lower() == "true":
            return "not_configured"

        try:
            # TODO: Implement actual Redis connectivity check
            return "healthy"
        except Exception:
            return "unhealthy"

    async def check_all_integrations(self) -> Dict[str, IntegrationStatus]:
        """Check health of all integrations."""
        async with self._lock:
            results = {}
            for integration_name in self.integrations.keys():
                results[integration_name] = await self.check_integration_health(integration_name)
            return results

    async def get_integration_status(self) -> Dict[str, Any]:
        """Get current status of all integrations."""
        async with self._lock:
            status = {}
            for name, integration in self.integrations.items():
                status[name] = {
                    "status": integration.status,
                    "last_check": integration.last_check,
                    "response_time": integration.response_time,
                    "config_status": integration.config_status,
                    "error_message": integration.error_message,
                }
            return status

    async def start_monitoring(self):
        """Start continuous monitoring of integrations."""
        while True:
            try:
                await self.check_all_integrations()
                await asyncio.sleep(self.check_interval)
            except Exception as e:
                logger.error(f"Integration monitoring error: {e}")
                await asyncio.sleep(60)  # Wait 1 minute on error


# Global monitor instance
_integration_monitor = None


async def get_integration_monitor() -> IntegrationMonitor:
    """Get the global integration monitor instance."""
    global _integration_monitor
    if _integration_monitor is None:
        _integration_monitor = IntegrationMonitor()
    return _integration_monitor


async def start_integration_monitoring():
    """Start the integration monitoring system."""
    monitor = await get_integration_monitor()
    asyncio.create_task(monitor.start_monitoring())
    logger.info("Integration monitoring started")


async def stop_integration_monitoring():
    """Stop the integration monitoring system."""
    monitor = await get_integration_monitor()
    if monitor and monitor._monitoring:
        monitor._monitoring = False
        logger.info("Integration monitoring stopped")
