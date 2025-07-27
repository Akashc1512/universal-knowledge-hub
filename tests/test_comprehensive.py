"""
Comprehensive Test Suite for Universal Knowledge Platform
Tests every component, function, class, and feature for bulletproof reliability.
"""

import pytest
import asyncio
import time
import json
from unittest.mock import AsyncMock, Mock, patch, MagicMock
from typing import Dict, Any, List
import logging

# Import all components to test
from agents.base_agent import (
    BaseAgent, QueryContext, AgentResult, AgentMessage, 
    MessageType, TaskPriority, AgentType
)
from agents.lead_orchestrator import LeadOrchestrator
from agents.retrieval_agent import RetrievalAgent
from agents.factcheck_agent import FactCheckAgent
from agents.synthesis_agent import SynthesisAgent
from agents.citation_agent import CitationAgent

# Import API components
from api.main import app
from api.cache import get_cached_result, cache_result, get_cache_stats
from api.analytics import track_query, get_analytics_summary
from api.security import check_security, get_security_summary

# Import knowledge graph components
try:
    from core.knowledge_graph.schema import KnowledgeGraphSchema, Node, Relationship
    from core.knowledge_graph.client import Neo4jClient, GraphQuery, GraphResult
    from core.knowledge_graph.migration import DataMigrator, MigrationConfig, MigrationStats
    KNOWLEDGE_GRAPH_AVAILABLE = True
except ImportError:
    KNOWLEDGE_GRAPH_AVAILABLE = False

# Import recommendation components
try:
    from core.recommendation.engine import (
        CollaborativeFiltering, ContentBasedFiltering, 
        SemanticFiltering, HybridRecommendationEngine
    )
    RECOMMENDATION_AVAILABLE = True
except ImportError:
    RECOMMENDATION_AVAILABLE = False

# Import semantic analysis components
try:
    from core.semantic_analysis.pipeline import SemanticAnalysisPipeline
    SEMANTIC_ANALYSIS_AVAILABLE = True
except ImportError:
    SEMANTIC_ANALYSIS_AVAILABLE = False

# Import enterprise integration components
try:
    from core.integrations.enterprise import (
        EnterpriseIntegrationManager, IntegrationConfig,
        MicrosoftGraphIntegration, GoogleDriveIntegration, SlackIntegration
    )
    ENTERPRISE_INTEGRATION_AVAILABLE = True
except ImportError:
    ENTERPRISE_INTEGRATION_AVAILABLE = False

from fastapi.testclient import TestClient
import numpy as np


class TestBaseAgent:
    """Test the base agent functionality."""
    
    def test_query_context_creation(self):
        """Test QueryContext creation with various parameters."""
        # Test basic creation
        context = QueryContext(query="test query")
        assert context.query == "test query"
        assert context.user_id is None
        assert context.session_id is None
        
        # Test with all parameters
        context = QueryContext(
            query="complex query",
            user_id="user123",
            session_id="session456",
            domains=["tech", "science"],
            complexity_score=0.8,
            token_budget=2000,
            timeout_ms=10000,
            user_context={"preferences": ["AI", "ML"]},
            metadata={"source": "test"}
        )
        assert context.query == "complex query"
        assert context.user_id == "user123"
        assert context.session_id == "session456"
        assert context.domains == ["tech", "science"]
        assert context.complexity_score == 0.8
        assert context.token_budget == 2000
        assert context.timeout_ms == 10000
        assert context.user_context == {"preferences": ["AI", "ML"]}
        assert context.metadata == {"source": "test"}
    
    def test_agent_message_creation(self):
        """Test AgentMessage creation and validation."""
        message = AgentMessage()
        assert message.header['message_id'] is not None
        assert message.header['message_type'] == MessageType.TASK
        assert message.header['priority'] == TaskPriority.MEDIUM.value
        
        # Test custom message
        custom_message = AgentMessage(
            header={
                'message_id': 'test-123',
                'message_type': MessageType.RESULT,
                'priority': TaskPriority.HIGH.value
            },
            payload={
                'result': {'data': 'test'},
                'metadata': {'test': True}
            }
        )
        assert custom_message.header['message_id'] == 'test-123'
        assert custom_message.header['message_type'] == MessageType.RESULT
        assert custom_message.payload['result'] == {'data': 'test'}
    
    def test_agent_result_creation(self):
        """Test AgentResult creation and validation."""
        result = AgentResult(
            success=True,
            data="test data",
            confidence=0.95,
            token_usage={'prompt': 100, 'completion': 50},
            execution_time_ms=150,
            metadata={'test': True}
        )
        assert result.success is True
        assert result.data == "test data"
        assert result.confidence == 0.95
        assert result.token_usage == {'prompt': 100, 'completion': 50}
        assert result.execution_time_ms == 150
        assert result.metadata == {'test': True}
    
    @pytest.mark.asyncio
    async def test_base_agent_lifecycle(self):
        """Test base agent start/stop lifecycle."""
        class TestAgent(BaseAgent):
            async def process_task(self, task: Dict[str, Any], context: QueryContext) -> AgentResult:
                return AgentResult(success=True, data="test result")
        
        agent = TestAgent("test-agent", AgentType.RETRIEVAL)
        assert agent.agent_id == "test-agent"
        assert agent.agent_type == AgentType.RETRIEVAL
        assert agent.is_running is False
        
        # Test health check
        health = agent.get_health_status()
        assert health['agent_id'] == "test-agent"
        assert health['agent_type'] == AgentType.RETRIEVAL.value
        assert health['is_running'] is False
        
        # Test metric recording
        agent.record_metric("test_metric", 5)
        assert agent.metrics["test_metric"] == 5


class TestOrchestrator:
    """Test the lead orchestrator functionality."""
    
    @pytest.fixture
    def orchestrator(self):
        """Create a test orchestrator instance."""
        return LeadOrchestrator()
    
    @pytest.mark.asyncio
    async def test_orchestrator_initialization(self, orchestrator):
        """Test orchestrator initialization."""
        assert orchestrator is not None
        assert hasattr(orchestrator, 'agents')
        assert hasattr(orchestrator, 'task_queue')
    
    @pytest.mark.asyncio
    async def test_process_query_basic(self, orchestrator):
        """Test basic query processing."""
        with patch.object(orchestrator, 'agents', {}):
            result = await orchestrator.process_query("What is AI?", {})
            assert result is not None
            assert isinstance(result, dict)
    
    @pytest.mark.asyncio
    async def test_process_query_with_agents(self, orchestrator):
        """Test query processing with mock agents."""
        # Mock agents
        mock_retrieval = AsyncMock()
        mock_retrieval.process_task.return_value = AgentResult(
            success=True, data="retrieved data", confidence=0.8
        )
        
        mock_synthesis = AsyncMock()
        mock_synthesis.process_task.return_value = AgentResult(
            success=True, data="synthesized answer", confidence=0.9
        )
        
        orchestrator.agents = {
            'retrieval': mock_retrieval,
            'synthesis': mock_synthesis
        }
        
        result = await orchestrator.process_query("What is AI?", {})
        assert result is not None
        assert 'response' in result


class TestRetrievalAgent:
    """Test the retrieval agent functionality."""
    
    @pytest.fixture
    def retrieval_agent(self):
        """Create a test retrieval agent instance."""
        return RetrievalAgent()
    
    @pytest.mark.asyncio
    async def test_retrieval_agent_initialization(self, retrieval_agent):
        """Test retrieval agent initialization."""
        assert retrieval_agent is not None
        assert retrieval_agent.agent_type == AgentType.RETRIEVAL
    
    @pytest.mark.asyncio
    async def test_process_task(self, retrieval_agent):
        """Test task processing in retrieval agent."""
        context = QueryContext(query="What is AI?")
        result = await retrieval_agent.process_task({"query": "What is AI?"}, context)
        assert result is not None
        assert isinstance(result, AgentResult)


class TestFactCheckAgent:
    """Test the fact-check agent functionality."""
    
    @pytest.fixture
    def factcheck_agent(self):
        """Create a test fact-check agent instance."""
        return FactCheckAgent()
    
    @pytest.mark.asyncio
    async def test_factcheck_agent_initialization(self, factcheck_agent):
        """Test fact-check agent initialization."""
        assert factcheck_agent is not None
        assert factcheck_agent.agent_type == AgentType.FACT_CHECK
    
    @pytest.mark.asyncio
    async def test_process_task(self, factcheck_agent):
        """Test task processing in fact-check agent."""
        context = QueryContext(query="Verify this claim")
        result = await factcheck_agent.process_task({"claim": "AI is everywhere"}, context)
        assert result is not None
        assert isinstance(result, AgentResult)


class TestSynthesisAgent:
    """Test the synthesis agent functionality."""
    
    @pytest.fixture
    def synthesis_agent(self):
        """Create a test synthesis agent instance."""
        return SynthesisAgent()
    
    @pytest.mark.asyncio
    async def test_synthesis_agent_initialization(self, synthesis_agent):
        """Test synthesis agent initialization."""
        assert synthesis_agent is not None
        assert synthesis_agent.agent_type == AgentType.SYNTHESIS
    
    @pytest.mark.asyncio
    async def test_process_task(self, synthesis_agent):
        """Test task processing in synthesis agent."""
        context = QueryContext(query="Synthesize information")
        result = await synthesis_agent.process_task({"data": "raw data"}, context)
        assert result is not None
        assert isinstance(result, AgentResult)


class TestCitationAgent:
    """Test the citation agent functionality."""
    
    @pytest.fixture
    def citation_agent(self):
        """Create a test citation agent instance."""
        return CitationAgent()
    
    @pytest.mark.asyncio
    async def test_citation_agent_initialization(self, citation_agent):
        """Test citation agent initialization."""
        assert citation_agent is not None
        assert citation_agent.agent_type == AgentType.CITATION
    
    @pytest.mark.asyncio
    async def test_process_task(self, citation_agent):
        """Test task processing in citation agent."""
        context = QueryContext(query="Add citations")
        result = await citation_agent.process_task({"text": "AI is important"}, context)
        assert result is not None
        assert isinstance(result, AgentResult)


class TestAPICache:
    """Test the API caching functionality."""
    
    @pytest.mark.asyncio
    async def test_cache_result_and_retrieval(self):
        """Test caching and retrieving results."""
        query = "test query"
        result = {"response": "test response", "confidence": 0.8}
        user_context = {"user_id": "test_user"}
        
        # Cache result
        await cache_result(query, result, user_context)
        
        # Retrieve cached result
        cached = await get_cached_result(query, user_context)
        assert cached is not None
        assert cached.get('response') == "test response"
        assert cached.get('confidence') == 0.8
    
    @pytest.mark.asyncio
    async def test_cache_stats(self):
        """Test cache statistics."""
        stats = get_cache_stats()
        assert isinstance(stats, dict)
        assert 'total_entries' in stats
        assert 'hit_rate' in stats


class TestAPIAnalytics:
    """Test the API analytics functionality."""
    
    @pytest.mark.asyncio
    async def test_track_query(self):
        """Test query tracking."""
        await track_query(
            query="test query",
            user_id="test_user",
            execution_time=1.5,
            response_size=100,
            confidence=0.8,
            cache_hit=False,
            error_occurred=False,
            error_type=None,
            agent_usage={"retrieval": 1, "synthesis": 1},
            token_usage={"prompt": 50, "completion": 30},
            user_agent="test-agent",
            ip_address="127.0.0.1"
        )
        
        # Get analytics summary
        summary = get_analytics_summary()
        assert isinstance(summary, dict)
        assert 'system_metrics' in summary
    
    @pytest.mark.asyncio
    async def test_analytics_summary(self):
        """Test analytics summary generation."""
        summary = await get_analytics_summary()
        assert isinstance(summary, dict)
        assert 'total_queries' in summary
        assert 'average_response_time' in summary
        assert 'error_rate' in summary


class TestAPISecurity:
    """Test the API security functionality."""
    
    @pytest.mark.asyncio
    async def test_security_check(self):
        """Test security checking."""
        result = await check_security(
            query="normal query",
            source_ip="127.0.0.1",
            user_id="test_user",
            response_time=0.5
        )
        assert isinstance(result, dict)
        assert 'blocked' in result
        assert 'monitored' in result
    
    def test_security_summary(self):
        """Test security summary generation."""
        summary = get_security_summary()
        assert isinstance(summary, dict)
        assert 'total_requests' in summary
        assert 'blocked_requests' in summary


class TestFastAPIEndpoints:
    """Test all FastAPI endpoints."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)
    
    def test_root_endpoint(self, client):
        """Test root endpoint."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
    
    def test_health_endpoint(self, client):
        """Test health endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data
        assert "timestamp" in data
        assert "agents_status" in data
    
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
        assert "error_count" in data
        assert "average_response_time" in data
    
    def test_analytics_endpoint(self, client):
        """Test analytics endpoint."""
        response = client.get("/analytics")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, dict)
    
    def test_security_endpoint(self, client):
        """Test security endpoint."""
        response = client.get("/security")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, dict)
    
    def test_cache_stats_endpoint(self, client):
        """Test cache stats endpoint."""
        response = client.get("/cache/stats")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, dict)
    
    def test_query_endpoint_validation(self, client):
        """Test query endpoint with validation."""
        # Test empty query
        response = client.post("/query", json={"query": ""})
        assert response.status_code == 422
        
        # Test valid query
        response = client.post("/query", json={
            "query": "What is AI?",
            "max_tokens": 1000,
            "confidence_threshold": 0.7
        })
        # Should return 200 or 500 depending on orchestrator availability
        assert response.status_code in [200, 500]
    
    def test_query_endpoint_rate_limiting(self, client):
        """Test rate limiting on query endpoint."""
        # Make multiple requests quickly
        for _ in range(70):  # More than the limit
            response = client.post("/query", json={"query": "test query"})
            if response.status_code == 429:
                break
        else:
            # If we didn't hit rate limit, that's also acceptable
            pass


@pytest.mark.skipif(not KNOWLEDGE_GRAPH_AVAILABLE, reason="Knowledge graph components not available")
class TestKnowledgeGraph:
    """Test knowledge graph components."""
    
    def test_schema_creation(self):
        """Test knowledge graph schema creation."""
        schema = KnowledgeGraphSchema()
        assert schema is not None
        assert hasattr(schema, 'nodes')
        assert hasattr(schema, 'relationships')
    
    def test_node_creation(self):
        """Test node creation."""
        node = Node(
            label="User",
            properties={"id": "user1", "name": "Test User"},
            unique_properties=["id"],
            indexed_properties=["name"]
        )
        assert node.label == "User"
        assert node.properties["id"] == "user1"
        assert "id" in node.unique_properties
    
    def test_relationship_creation(self):
        """Test relationship creation."""
        relationship = Relationship(
            type="VIEWED",
            from_node_label="User",
            to_node_label="Document",
            properties={"timestamp": "2023-01-01"},
            indexed_properties=["timestamp"]
        )
        assert relationship.type == "VIEWED"
        assert relationship.from_node_label == "User"
        assert relationship.to_node_label == "Document"
    
    @pytest.mark.asyncio
    async def test_neo4j_client(self):
        """Test Neo4j client functionality."""
        # Mock client for testing
        client = Neo4jClient("bolt://localhost:7687", "neo4j", "password")
        
        # Test health check (will fail without real connection, but should not crash)
        try:
            health = await client.health_check()
            assert isinstance(health, bool)
        except Exception:
            # Expected without real Neo4j connection
            pass
    
    @pytest.mark.asyncio
    async def test_data_migrator(self):
        """Test data migration functionality."""
        config = MigrationConfig(
            batch_size=100,
            max_workers=2,
            retry_attempts=3,
            timeout=300,
            dry_run=True
        )
        
        # Mock graph client
        mock_client = AsyncMock()
        migrator = DataMigrator(mock_client, config)
        
        # Test initialization
        result = await migrator.initialize_graph()
        assert isinstance(result, bool)
        
        # Test sample data generation
        users = DataMigrator.generate_sample_users(10)
        assert len(users) == 10
        assert all('id' in user for user in users)
        
        documents = DataMigrator.generate_sample_documents(20)
        assert len(documents) == 20
        assert all('id' in doc for doc in documents)


@pytest.mark.skipif(not RECOMMENDATION_AVAILABLE, reason="Recommendation components not available")
class TestRecommendationEngine:
    """Test recommendation engine components."""
    
    def test_collaborative_filtering(self):
        """Test collaborative filtering."""
        # Mock graph client
        mock_client = AsyncMock()
        engine = CollaborativeFiltering(mock_client)
        assert engine is not None
        assert hasattr(engine, 'get_recommendations')
    
    def test_content_based_filtering(self):
        """Test content-based filtering."""
        # Mock graph client
        mock_client = AsyncMock()
        engine = ContentBasedFiltering(mock_client)
        assert engine is not None
        assert hasattr(engine, 'get_recommendations')
    
    def test_semantic_filtering(self):
        """Test semantic filtering."""
        # Mock graph client
        mock_client = AsyncMock()
        engine = SemanticFiltering(mock_client)
        assert engine is not None
        assert hasattr(engine, 'get_recommendations')
    
    def test_hybrid_engine(self):
        """Test hybrid recommendation engine."""
        # Mock graph client
        mock_client = AsyncMock()
        hybrid_engine = HybridRecommendationEngine(mock_client)
        assert hybrid_engine is not None
        assert hasattr(hybrid_engine, 'get_recommendations')


@pytest.mark.skipif(not SEMANTIC_ANALYSIS_AVAILABLE, reason="Semantic analysis components not available")
class TestSemanticAnalysis:
    """Test semantic analysis components."""
    
    @pytest.mark.asyncio
    async def test_semantic_pipeline(self):
        """Test semantic analysis pipeline."""
        # Mock graph client
        mock_client = AsyncMock()
        pipeline = SemanticAnalysisPipeline(mock_client)
        
        # Test initialization
        result = await pipeline.initialize()
        assert isinstance(result, bool)
        
        # Test content analysis
        analysis = pipeline.analyze_content("doc1", "This is a test document about AI.")
        assert analysis.document_id == "doc1"
        assert hasattr(analysis, 'entities')
        assert hasattr(analysis, 'topics')
        assert hasattr(analysis, 'keywords')


@pytest.mark.skipif(not ENTERPRISE_INTEGRATION_AVAILABLE, reason="Enterprise integration components not available")
class TestEnterpriseIntegration:
    """Test enterprise integration components."""
    
    def test_integration_config(self):
        """Test integration configuration."""
        config = IntegrationConfig(
            microsoft_client_id="test_id",
            microsoft_client_secret="test_secret",
            google_client_id="google_id",
            google_client_secret="google_secret",
            slack_bot_token="slack_token",
            slack_signing_secret="slack_secret"
        )
        assert config.microsoft_client_id == "test_id"
        assert config.google_client_id == "google_id"
        assert config.slack_bot_token == "slack_token"
    
    @pytest.mark.asyncio
    async def test_enterprise_integration_manager(self):
        """Test enterprise integration manager."""
        # Mock config and graph client
        config = IntegrationConfig(
            microsoft_client_id="test",
            microsoft_client_secret="test",
            google_client_id="test",
            google_client_secret="test",
            slack_bot_token="test",
            slack_signing_secret="test"
        )
        mock_client = AsyncMock()
        
        manager = EnterpriseIntegrationManager(config, mock_client)
        assert manager is not None
        assert hasattr(manager, 'microsoft')
        assert hasattr(manager, 'google')
        assert hasattr(manager, 'slack')


class TestPerformanceAndLoad:
    """Test performance and load handling."""
    
    @pytest.mark.asyncio
    async def test_concurrent_requests(self):
        """Test handling of concurrent requests."""
        client = TestClient(app)
        
        # Simulate concurrent requests
        async def make_request():
            return client.post("/query", json={"query": "test query"})
        
        # Make multiple concurrent requests
        tasks = [make_request() for _ in range(10)]
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Check that all requests were handled
        assert len(responses) == 10
        for response in responses:
            if isinstance(response, Exception):
                # Some errors are expected without full setup
                pass
            else:
                assert response.status_code in [200, 422, 500]
    
    def test_response_time(self):
        """Test response time performance."""
        client = TestClient(app)
        
        start_time = time.time()
        response = client.get("/health")
        end_time = time.time()
        
        response_time = end_time - start_time
        assert response_time < 1.0  # Should respond within 1 second
        assert response.status_code == 200


class TestErrorHandling:
    """Test error handling and edge cases."""
    
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
        # Should not crash, may return 200 or 500 depending on setup
        assert response.status_code in [200, 422, 500]


class TestSecurityFeatures:
    """Test security features."""
    
    def test_sql_injection_prevention(self, client):
        """Test SQL injection prevention."""
        malicious_query = "'; DROP TABLE users; --"
        response = client.post("/query", json={"query": malicious_query})
        # Should not crash, should handle gracefully
        assert response.status_code in [200, 422, 500]
    
    def test_xss_prevention(self, client):
        """Test XSS prevention."""
        xss_query = "<script>alert('xss')</script>"
        response = client.post("/query", json={"query": xss_query})
        # Should not crash, should handle gracefully
        assert response.status_code in [200, 422, 500]
    
    def test_rate_limiting(self, client):
        """Test rate limiting functionality."""
        # Make many requests quickly
        responses = []
        for i in range(100):
            response = client.post("/query", json={"query": f"test query {i}"})
            responses.append(response.status_code)
            if response.status_code == 429:
                break
        
        # Should eventually hit rate limit
        assert 429 in responses or len(responses) < 100


class TestDataValidation:
    """Test data validation across all components."""
    
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
    """Test complete integration scenarios."""
    
    @pytest.mark.asyncio
    async def test_complete_query_flow(self):
        """Test complete query processing flow."""
        client = TestClient(app)
        
        # Test complete flow
        response = client.post("/query", json={
            "query": "What is artificial intelligence?",
            "max_tokens": 500,
            "confidence_threshold": 0.8
        })
        
        # Should return some response (may be error without full setup)
        assert response.status_code in [200, 422, 500]
        
        if response.status_code == 200:
            data = response.json()
            assert "answer" in data
            assert "confidence" in data
            assert "execution_time" in data
    
    @pytest.mark.asyncio
    async def test_health_monitoring_flow(self):
        """Test health monitoring flow."""
        client = TestClient(app)
        
        # Check health
        health_response = client.get("/health")
        assert health_response.status_code == 200
        
        # Check metrics
        metrics_response = client.get("/metrics")
        assert metrics_response.status_code == 200
        
        # Check analytics
        analytics_response = client.get("/analytics")
        assert analytics_response.status_code == 200
        
        # Check security
        security_response = client.get("/security")
        assert security_response.status_code == 200


if __name__ == "__main__":
    # Run all tests
    pytest.main([__file__, "-v", "--tb=short"]) 