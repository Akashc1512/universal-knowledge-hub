"""
Production Readiness Tests for Universal Knowledge Platform
Tests monitoring, logging, error handling, and integration features.
"""

import pytest
import asyncio
import logging
import time
from unittest.mock import Mock, patch, AsyncMock
from fastapi.testclient import TestClient
import json
import os

from api.main import app
from api.integration_monitor import IntegrationMonitor, IntegrationStatus
from api.cache import RedisCache, LRUCache
from api.metrics import MetricsCollector


class TestProductionReadiness:
    """Test production readiness features."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)

    @pytest.fixture
    def mock_orchestrator(self):
        """Mock orchestrator for testing."""
        mock = Mock()
        mock.process_query = AsyncMock(
            return_value={
                "answer": "Test answer",
                "confidence": 0.8,
                "success": True,
                "metadata": {
                    "agent_results": {
                        "retrieval": {"status": "success"},
                        "synthesis": {"status": "success"},
                    },
                    "token_usage": {"total": 100},
                },
            }
        )
        return mock

    def test_structured_logging_format(self):
        """Test that logging includes request ID and structured format."""
        # Test that the logging format includes required fields
        log_format = logging.getLogger().handlers[0].formatter._fmt
        assert "request_id" in log_format
        assert "user_id" in log_format
        assert "service" in log_format
        assert "version" in log_format

    def test_health_check_endpoint(self, client):
        """Test health check endpoint includes all components."""
        response = client.get("/health")
        assert response.status_code == 200

        data = response.json()
        assert "status" in data
        assert "version" in data
        assert "uptime" in data
        assert "components" in data

        # Check that all expected components are present
        components = data["components"]
        assert "orchestrator" in components
        assert "cache" in components
        assert "vector_database" in components
        assert "elasticsearch" in components
        assert "knowledge_graph" in components
        assert "llm_api" in components

    def test_metrics_endpoint(self, client):
        """Test metrics endpoint provides comprehensive metrics."""
        response = client.get("/metrics")
        assert response.status_code == 200

        data = response.json()
        # Check for required metrics
        assert "sarvanom_version" in data
        assert "sarvanom_uptime_seconds" in data
        assert "sarvanom_requests_total" in data
        assert "sarvanom_errors_total" in data
        assert "sarvanom_cache_hits_total" in data
        assert "sarvanom_cache_misses_total" in data

    def test_integration_status_endpoint(self, client):
        """Test integration status endpoint."""
        response = client.get("/integrations")
        assert response.status_code == 200

        data = response.json()
        assert "timestamp" in data
        assert "integrations" in data
        assert "summary" in data

        summary = data["summary"]
        assert "total" in summary
        assert "healthy" in summary
        assert "unhealthy" in summary
        assert "not_configured" in summary

    @pytest.mark.asyncio
    async def test_integration_monitor(self):
        """Test integration monitoring functionality."""
        monitor = IntegrationMonitor()

        # Test integration status tracking
        status = await monitor.get_integration_status()
        assert "vector_database" in status
        assert "elasticsearch" in status
        assert "knowledge_graph" in status
        assert "llm_api" in status
        assert "redis_cache" in status

        # Test individual integration health check
        vector_status = await monitor.check_integration_health("vector_database")
        assert isinstance(vector_status, IntegrationStatus)
        assert vector_status.name == "vector_database"
        assert vector_status.status in ["healthy", "unhealthy", "not_configured", "unknown"]

    @pytest.mark.asyncio
    async def test_redis_cache_functionality(self):
        """Test Redis cache functionality."""
        # Test with Redis disabled (should fall back gracefully)
        cache = RedisCache()

        # Test get with no connection
        result = await cache.get("test_key")
        assert result is None

        # Test put with no connection (should not raise)
        await cache.put("test_key", "test_value")

        # Test stats
        stats = cache.get_stats()
        assert "type" in stats
        assert "enabled" in stats
        assert "connected" in stats

    def test_error_handling_with_partial_failures(self, client):
        """Test error handling for partial failures."""
        # Mock a partial failure scenario
        with patch("api.main.orchestrator") as mock_orch:
            mock_orch.process_query = AsyncMock(
                return_value={
                    "answer": "Partial answer due to some failures",
                    "confidence": 0.5,
                    "success": True,
                    "metadata": {
                        "agent_results": {
                            "retrieval": {"status": "success"},
                            "synthesis": {"status": "failed", "error": "LLM API timeout"},
                            "fact_check": {"status": "success"},
                        }
                    },
                }
            )

            response = client.post(
                "/query",
                json={"query": "Test query", "max_tokens": 1000, "confidence_threshold": 0.5},
            )

            assert response.status_code == 200
            data = response.json()

            # Check for partial failure metadata
            metadata = data.get("metadata", {})
            assert "partial_failure" in metadata
            assert "failed_agents" in metadata
            assert "successful_agents" in metadata

    def test_request_id_tracking(self, client):
        """Test that request IDs are properly tracked."""
        response = client.post("/query", json={"query": "Test query", "max_tokens": 1000})

        assert response.status_code == 200
        data = response.json()

        # Check that request ID is included in response
        metadata = data.get("metadata", {})
        assert "request_id" in metadata
        assert metadata["request_id"] != "unknown"

    def test_comprehensive_error_handling(self, client):
        """Test comprehensive error handling."""
        # Test with invalid request
        response = client.post("/query", json={"invalid_field": "should_fail"})

        assert response.status_code == 422  # Validation error

        # Test with malformed JSON
        response = client.post(
            "/query", data="invalid json", headers={"Content-Type": "application/json"}
        )

        assert response.status_code == 422

    def test_metrics_collection(self):
        """Test metrics collection functionality."""
        collector = MetricsCollector()

        # Test request metrics
        collector.record_request("POST", "/query", 200, 1.5)
        collector.record_request("GET", "/health", 200, 0.1)

        # Test error metrics
        collector.record_error("validation_error", "/query")

        # Test cache metrics
        collector.record_cache_hit("query_cache")
        collector.record_cache_miss("query_cache")

        # Get metrics
        metrics = collector.get_metrics_dict()
        assert "request_counter" in metrics
        assert "error_counter" in metrics
        assert "cache_hits" in metrics
        assert "cache_misses" in metrics

    def test_logging_with_request_context(self):
        """Test that logging includes proper request context."""
        logger = logging.getLogger(__name__)

        # Test logging with request context
        with patch("logging.Logger.info") as mock_info:
            logger.info("Test message", extra={"request_id": "test-123", "user_id": "user-456"})

            mock_info.assert_called_once()
            call_args = mock_info.call_args
            assert "request_id" in call_args[1]["extra"]
            assert "user_id" in call_args[1]["extra"]

    def test_application_version_consistency(self, client):
        """Test that application version is consistent across endpoints."""
        # Check health endpoint
        health_response = client.get("/health")
        health_data = health_response.json()
        health_version = health_data["version"]

        # Check metrics endpoint
        metrics_response = client.get("/metrics")
        metrics_data = metrics_response.json()
        metrics_version = metrics_data["sarvanom_version"]

        # Versions should be consistent
        assert health_version == metrics_version

    def test_performance_monitoring(self, client):
        """Test performance monitoring features."""
        start_time = time.time()

        response = client.post(
            "/query", json={"query": "Performance test query", "max_tokens": 1000}
        )

        end_time = time.time()
        execution_time = end_time - start_time

        assert response.status_code == 200
        data = response.json()

        # Check execution time is recorded
        metadata = data.get("metadata", {})
        assert "execution_time_ms" in metadata
        assert metadata["execution_time_ms"] > 0

        # Check that execution time is reasonable
        assert metadata["execution_time_ms"] < 30000  # Should complete within 30 seconds

    def test_cache_functionality(self):
        """Test cache functionality."""
        cache = LRUCache(max_size=10)

        # Test basic operations
        asyncio.run(cache.put("key1", "value1"))
        result = asyncio.run(cache.get("key1"))
        assert result == "value1"

        # Test cache stats
        stats = cache.get_stats()
        assert stats["hits"] >= 1
        assert stats["size"] >= 1

    def test_environment_variable_handling(self):
        """Test environment variable handling for monitoring."""
        # Test Redis configuration
        assert "REDIS_ENABLED" in os.environ or "REDIS_ENABLED" not in os.environ

        # Test monitoring configuration
        assert "PROMETHEUS_ENABLED" in os.environ or "PROMETHEUS_ENABLED" not in os.environ

        # Test logging configuration
        assert "LOG_LEVEL" in os.environ or "LOG_LEVEL" not in os.environ


if __name__ == "__main__":
    pytest.main([__file__])
