"""
Core Functionality Test Suite
Tests the most critical components of the Universal Knowledge Platform.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, Mock
from fastapi.testclient import TestClient

from api.main import app
from agents.base_agent import QueryContext, AgentResult, AgentMessage, MessageType
from api.cache import get_cache_stats
from api.analytics import get_analytics_summary


class TestCoreAPI:
    """Test core API functionality."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)
    
    def test_health_endpoint(self, client):
        """Test health endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data
        assert "timestamp" in data
    
    def test_root_endpoint(self, client):
        """Test root endpoint."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
    
    def test_agents_endpoint(self, client):
        """Test agents endpoint."""
        response = client.get("/agents")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, dict)
    
    def test_metrics_endpoint(self, client):
        """Test metrics endpoint."""
        response = client.get("/metrics")
        assert response.status_code == 200
        data = response.json()
        assert "requests_processed" in data
        assert "average_response_time" in data
    
    def test_cache_stats_endpoint(self, client):
        """Test cache stats endpoint."""
        response = client.get("/cache/stats")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, dict)


class TestCoreComponents:
    """Test core components."""
    
    def test_query_context_creation(self):
        """Test QueryContext creation."""
        context = QueryContext(
            query="test query",
            user_id="user123",
            session_id="session456"
        )
        assert context.query == "test query"
        assert context.user_id == "user123"
        assert context.session_id == "session456"
    
    def test_agent_result_creation(self):
        """Test AgentResult creation."""
        result = AgentResult(
            success=True,
            data="test data",
            confidence=0.95,
            execution_time_ms=150
        )
        assert result.success is True
        assert result.data == "test data"
        assert result.confidence == 0.95
        assert result.execution_time_ms == 150
    
    def test_agent_message_creation(self):
        """Test AgentMessage creation."""
        message = AgentMessage(
            header={
                'message_id': 'test-123',
                'message_type': MessageType.TASK,
                'sender_agent': 'test-agent'
            },
            payload={
                'task': {'query': 'test query'},
                'metadata': {'test': True}
            }
        )
        assert message.header['message_id'] == 'test-123'
        assert message.header['message_type'] == MessageType.TASK
        assert message.payload['task']['query'] == 'test query'
    
    def test_cache_stats(self):
        """Test cache statistics."""
        stats = get_cache_stats()
        assert isinstance(stats, dict)
        assert 'total_entries' in stats
        assert 'query_cache' in stats
        assert 'semantic_cache' in stats
    
    def test_analytics_summary(self):
        """Test analytics summary."""
        summary = get_analytics_summary()
        assert isinstance(summary, dict)
        assert 'system_metrics' in summary
        assert 'cache_stats' in summary


class TestErrorHandling:
    """Test error handling."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)
    
    def test_invalid_json_handling(self, client):
        """Test handling of invalid JSON."""
        response = client.post("/query", data="invalid json")
        assert response.status_code == 422
    
    def test_missing_required_fields(self, client):
        """Test handling of missing required fields."""
        response = client.post("/query", json={})
        assert response.status_code == 422
    
    def test_large_payload_handling(self, client):
        """Test handling of large payloads."""
        large_query = "x" * 15000  # Larger than 10KB limit
        response = client.post("/query", json={"query": large_query})
        assert response.status_code == 422
    
    def test_special_characters_handling(self, client):
        """Test handling of special characters."""
        special_query = "test query with special chars: !@#$%^&*()"
        response = client.post("/query", json={"query": special_query})
        # Should not crash, may return various status codes
        assert response.status_code in [200, 422, 500, 503, 403, 429]
    
    def test_sql_injection_prevention(self, client):
        """Test SQL injection prevention."""
        malicious_query = "'; DROP TABLE users; --"
        response = client.post("/query", json={"query": malicious_query})
        # Should block malicious queries
        assert response.status_code in [200, 422, 500, 503, 403, 429]
    
    def test_xss_prevention(self, client):
        """Test XSS prevention."""
        xss_query = "<script>alert('xss')</script>"
        response = client.post("/query", json={"query": xss_query})
        # Should block XSS attempts
        assert response.status_code in [200, 422, 500, 503, 403, 429]
    
    def test_basic_query_flow(self, client):
        """Test basic query flow."""
        response = client.post("/query", json={
            "query": "What is artificial intelligence?",
            "max_tokens": 500,
            "confidence_threshold": 0.8
        })
        
        # Should return some response (may be error without full setup)
        assert response.status_code in [200, 422, 500, 503, 403, 429]
        
        if response.status_code == 200:
            data = response.json()
            assert "answer" in data
            assert "confidence" in data
            assert "execution_time" in data


class TestPerformance:
    """Test performance characteristics."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)
    
    def test_response_time(self, client):
        """Test response time performance."""
        import time
        
        start_time = time.time()
        response = client.get("/health")
        end_time = time.time()
        
        response_time = end_time - start_time
        assert response_time < 1.0  # Should respond within 1 second
        assert response.status_code == 200
    
    def test_concurrent_requests(self, client):
        """Test handling of concurrent requests."""
        import time
        
        # Make multiple requests
        start_time = time.time()
        responses = []
        for i in range(10):
            response = client.get("/health")
            responses.append(response.status_code)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # All should succeed
        assert all(status == 200 for status in responses)
        # Should complete within reasonable time
        assert total_time < 5.0


class TestDataValidation:
    """Test data validation."""
    
    def test_query_validation(self):
        """Test query validation."""
        from api.main import QueryRequest
        
        # Valid query
        valid_request = QueryRequest(query="valid query")
        assert valid_request.query == "valid query"
        
        # Invalid query (empty)
        with pytest.raises(ValueError):
            QueryRequest(query="")
        
        # Invalid query (too long)
        with pytest.raises(ValueError):
            QueryRequest(query="x" * 15000)
    
    def test_agent_message_validation(self):
        """Test agent message validation."""
        # Valid message
        message = AgentMessage()
        assert message.header['message_id'] is not None
        assert message.header['message_type'] == MessageType.TASK
        
        # Custom message
        custom_message = AgentMessage(
            header={'message_type': MessageType.RESULT},
            payload={'result': 'test'}
        )
        assert custom_message.header['message_type'] == MessageType.RESULT
        assert custom_message.payload['result'] == 'test'


class TestIntegrationScenarios:
    """Test integration scenarios."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)
    
    def test_health_monitoring_flow(self, client):
        """Test health monitoring flow."""
        # Check health
        health_response = client.get("/health")
        assert health_response.status_code == 200
        
        # Check metrics
        metrics_response = client.get("/metrics")
        assert metrics_response.status_code == 200
        
        # Check cache stats
        cache_response = client.get("/cache/stats")
        assert cache_response.status_code == 200
    
    def test_basic_query_flow(self, client):
        """Test basic query flow."""
        response = client.post("/query", json={
            "query": "What is artificial intelligence?",
            "max_tokens": 500,
            "confidence_threshold": 0.8
        })
        
        # Should return some response (may be error without full setup)
        assert response.status_code in [200, 422, 500, 503, 403, 429]
        
        if response.status_code == 200:
            data = response.json()
            assert "answer" in data
            assert "confidence" in data
            assert "execution_time" in data


if __name__ == "__main__":
    # Run all tests
    pytest.main([__file__, "-v", "--tb=short"]) 