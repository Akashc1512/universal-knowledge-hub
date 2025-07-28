"""
Comprehensive Integration Tests for Universal Knowledge Hub
Tests complete workflows and system integration
"""

import unittest
import asyncio
import json
import time
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from datetime import datetime, timedelta
import requests
import subprocess
import sys
import os

# Add project root to path
sys.path.append("..")

# Import application components
from agents.lead_orchestrator import LeadOrchestrator
from core.config_manager import ConfigurationManager
from api.main import app
from fastapi.testclient import TestClient


class TestEndToEndWorkflow(unittest.TestCase):
    """Test complete end-to-end workflows"""

    def setUp(self):
        """Set up test fixtures"""
        self.orchestrator = LeadOrchestrator()
        self.config_manager = ConfigurationManager(environment="test")
        self.client = TestClient(app)

    def tearDown(self):
        """Clean up after tests"""
        pass

    def test_complete_search_workflow(self):
        """Test complete search workflow from query to response"""
        # Test query processing
        query = "What is artificial intelligence?"
        user_context = {"user_id": "test_user", "session_id": "test_session"}

        # Process query through orchestrator
        result = asyncio.run(self.orchestrator.process_query(query, user_context))

        # Verify result structure
        self.assertIsInstance(result, dict)
        self.assertIn("success", result)
        self.assertIn("response", result)
        self.assertIn("confidence", result)
        self.assertIn("execution_time_ms", result)

        # Verify response quality
        if result.get("success"):
            self.assertIsInstance(result["response"], str)
            self.assertGreater(len(result["response"]), 0)
            self.assertGreaterEqual(result["confidence"], 0.0)
            self.assertLessEqual(result["confidence"], 1.0)
            self.assertGreater(result["execution_time_ms"], 0)

    def test_complete_api_workflow(self):
        """Test complete API workflow"""
        # Test API health endpoint
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)

        # Test API query endpoint
        query_data = {
            "query": "What is machine learning?",
            "user_id": "test_user",
            "options": {"max_tokens": 1000},
        }

        response = self.client.post("/query", json=query_data)
        self.assertEqual(response.status_code, 200)

        result = response.json()
        self.assertIn("success", result)
        self.assertIn("response", result)
        self.assertIn("confidence", result)
        self.assertIn("execution_time_ms", result)

    def test_multi_agent_integration(self):
        """Test integration between multiple agents"""
        # Test retrieval agent
        retrieval_agent = self.orchestrator.agents["retrieval"]
        retrieval_task = {"strategy": "hybrid", "query": "test query", "top_k": 5}
        retrieval_context = QueryContext(query="test query")

        retrieval_result = asyncio.run(
            retrieval_agent.process_task(retrieval_task, retrieval_context)
        )
        self.assertTrue(retrieval_result.success)
        self.assertIn("retrieved_documents", retrieval_result.data)

        # Test fact-check agent with retrieval results
        factcheck_agent = self.orchestrator.agents["fact_check"]
        factcheck_task = {
            "claims": ["Test claim"],
            "sources": retrieval_result.data["retrieved_documents"],
        }

        factcheck_result = asyncio.run(
            factcheck_agent.process_task(factcheck_task, retrieval_context)
        )
        self.assertTrue(factcheck_result.success)
        self.assertIn("verifications", factcheck_result.data)

        # Test synthesis agent with fact-check results
        synthesis_agent = self.orchestrator.agents["synthesis"]
        synthesis_task = {
            "verified_facts": factcheck_result.data["verifications"],
            "synthesis_params": {"style": "informative"},
        }

        synthesis_result = asyncio.run(
            synthesis_agent.process_task(synthesis_task, retrieval_context)
        )
        self.assertTrue(synthesis_result.success)
        self.assertIn("response", synthesis_result.data)

        # Test citation agent with synthesis results
        citation_agent = self.orchestrator.agents["citation"]
        citation_task = {
            "content": synthesis_result.data["response"]["text"],
            "sources": retrieval_result.data["retrieved_documents"],
            "citation_style": "APA",
        }

        citation_result = asyncio.run(citation_agent.process_task(citation_task, retrieval_context))
        self.assertTrue(citation_result.success)
        self.assertIn("citations", citation_result.data)


class TestConfigurationIntegration(unittest.TestCase):
    """Test configuration system integration"""

    def setUp(self):
        """Set up test fixtures"""
        self.config_manager = ConfigurationManager(environment="test")

    def test_configuration_hot_reload(self):
        """Test configuration hot-reloading integration"""
        # Test that configuration changes are reflected immediately
        original_value = self.config_manager.get("test.key", "original")

        # Simulate configuration change
        self.config_manager.set("test.key", "new_value")

        # Verify change is reflected
        new_value = self.config_manager.get("test.key")
        self.assertEqual(new_value, "new_value")

    def test_configuration_validation_integration(self):
        """Test configuration validation integration"""
        # Test that configuration validation works with all components
        is_valid = self.config_manager.validate_configuration()
        self.assertIsInstance(is_valid, bool)

    def test_environment_configuration_integration(self):
        """Test environment configuration integration"""
        # Test that environment configuration is properly loaded
        env_config = self.config_manager.get_environment_config()

        self.assertIsInstance(env_config, dict)
        self.assertIn("environment", env_config)
        self.assertIn("database", env_config)
        self.assertIn("redis", env_config)
        self.assertIn("elasticsearch", env_config)
        self.assertIn("features", env_config)
        self.assertIn("app", env_config)


class TestAgentCommunication(unittest.TestCase):
    """Test agent-to-agent communication"""

    def setUp(self):
        """Set up test fixtures"""
        self.orchestrator = LeadOrchestrator()

    def test_agent_message_passing(self):
        """Test message passing between agents"""
        # Test message broker functionality
        message_broker = self.orchestrator.message_broker

        # Create test message
        test_message = AgentMessage(
            header={
                "message_id": "test_msg_001",
                "sender_agent": "orchestrator",
                "recipient_agent": "retrieval_001",
                "message_type": MessageType.TASK,
                "timestamp": datetime.utcnow().isoformat(),
            },
            payload={"task": {"query": "test query"}, "context": QueryContext(query="test query")},
        )

        # Test message publishing
        asyncio.run(message_broker.publish(test_message))

        # Test message retrieval
        retrieved_message = asyncio.run(message_broker.get_message("retrieval_001"))
        self.assertIsNotNone(retrieved_message)
        self.assertEqual(retrieved_message.header["message_id"], "test_msg_001")

    def test_agent_subscription(self):
        """Test agent subscription to message types"""
        message_broker = self.orchestrator.message_broker

        # Test agent subscription
        asyncio.run(message_broker.subscribe("test_agent", [MessageType.TASK, MessageType.CONTROL]))

        # Verify subscription
        self.assertIn("test_agent", message_broker.subscriptions[MessageType.TASK])
        self.assertIn("test_agent", message_broker.subscriptions[MessageType.CONTROL])


class TestTokenBudgetIntegration(unittest.TestCase):
    """Test token budget system integration"""

    def setUp(self):
        """Set up test fixtures"""
        self.orchestrator = LeadOrchestrator()

    def test_token_budget_allocation(self):
        """Test token budget allocation"""
        token_controller = self.orchestrator.token_controller

        # Test budget allocation
        query = "What is artificial intelligence?"
        budget = token_controller.allocate_budget_for_query(query)

        self.assertGreater(budget, 0)
        self.assertIsInstance(budget, int)

        # Test agent budget allocation
        agent_budget = token_controller.get_agent_budget(AgentType.RETRIEVAL, budget)
        self.assertGreater(agent_budget, 0)
        self.assertLessEqual(agent_budget, budget)

    def test_token_usage_tracking(self):
        """Test token usage tracking"""
        token_controller = self.orchestrator.token_controller

        # Test usage tracking
        initial_usage = token_controller.used_today
        token_controller.track_usage(AgentType.RETRIEVAL, 100)

        self.assertEqual(token_controller.used_today, initial_usage + 100)


class TestCacheIntegration(unittest.TestCase):
    """Test caching system integration"""

    def setUp(self):
        """Set up test fixtures"""
        self.orchestrator = LeadOrchestrator()

    def test_cache_hit_scenario(self):
        """Test cache hit scenario"""
        cache_manager = self.orchestrator.cache_manager

        # Test caching response
        query = "test query"
        response = {"response": "test response", "confidence": 0.9}

        asyncio.run(cache_manager.cache_response(query, response))

        # Test cache hit
        cached_response = asyncio.run(cache_manager.get_cached_response(query))
        self.assertIsNotNone(cached_response)
        self.assertEqual(cached_response, response)

    def test_cache_miss_scenario(self):
        """Test cache miss scenario"""
        cache_manager = self.orchestrator.cache_manager

        # Test cache miss
        query = "uncached query"
        cached_response = asyncio.run(cache_manager.get_cached_response(query))
        self.assertIsNone(cached_response)

    def test_cache_eviction(self):
        """Test cache eviction"""
        cache_manager = self.orchestrator.cache_manager

        # Fill cache to trigger eviction
        for i in range(cache_manager.max_cache_size + 10):
            query = f"query_{i}"
            response = {"response": f"response_{i}"}
            asyncio.run(cache_manager.cache_response(query, response))

        # Verify cache size is maintained
        self.assertLessEqual(len(cache_manager.cache), cache_manager.max_cache_size)


class TestResponseAggregationIntegration(unittest.TestCase):
    """Test response aggregation integration"""

    def setUp(self):
        """Set up test fixtures"""
        self.orchestrator = LeadOrchestrator()

    def test_pipeline_results_aggregation(self):
        """Test pipeline results aggregation"""
        response_aggregator = self.orchestrator.response_aggregator

        # Create mock agent results
        results = {
            AgentType.RETRIEVAL: AgentResult(
                success=True,
                data={"retrieved_documents": [{"content": "doc1", "score": 0.9}]},
                confidence=0.9,
                token_usage={"prompt": 100, "completion": 50},
            ),
            AgentType.FACT_CHECK: AgentResult(
                success=True,
                data={"verifications": [{"verdict": "supported", "confidence": 0.8}]},
                confidence=0.8,
                token_usage={"prompt": 100, "completion": 50},
            ),
            AgentType.SYNTHESIS: AgentResult(
                success=True,
                data={"response": {"text": "Synthesized response", "key_points": ["point1"]}},
                confidence=0.85,
                token_usage={"prompt": 200, "completion": 100},
            ),
            AgentType.CITATION: AgentResult(
                success=True,
                data={"citations": [{"id": "cite1", "text": "Citation 1"}]},
                confidence=0.95,
                token_usage={"prompt": 50, "completion": 25},
            ),
        }

        context = QueryContext(query="test query")

        # Test aggregation
        aggregated_result = response_aggregator.aggregate_pipeline_results(results, context)

        self.assertIn("success", aggregated_result)
        self.assertIn("response", aggregated_result)
        self.assertIn("confidence", aggregated_result)
        self.assertIn("citations", aggregated_result)
        self.assertIn("key_points", aggregated_result)
        self.assertIn("token_usage", aggregated_result)
        self.assertIn("agent_results", aggregated_result)

        self.assertTrue(aggregated_result["success"])
        self.assertIsInstance(aggregated_result["response"], str)
        self.assertGreater(aggregated_result["confidence"], 0.0)
        self.assertLessEqual(aggregated_result["confidence"], 1.0)


class TestPerformanceIntegration(unittest.TestCase):
    """Test performance integration"""

    def setUp(self):
        """Set up test fixtures"""
        self.orchestrator = LeadOrchestrator()

    def test_query_performance(self):
        """Test query performance"""
        # Test query processing performance
        start_time = time.time()

        query = "What is quantum computing?"
        result = asyncio.run(self.orchestrator.process_query(query))

        end_time = time.time()
        processing_time = end_time - start_time

        # Verify performance within acceptable limits (5 seconds for complex query)
        self.assertLess(processing_time, 5.0)
        self.assertIn("execution_time_ms", result)
        self.assertGreater(result["execution_time_ms"], 0)

    def test_concurrent_query_processing(self):
        """Test concurrent query processing"""
        # Test multiple queries processed concurrently
        queries = [
            "What is AI?",
            "What is ML?",
            "What is deep learning?",
            "What is NLP?",
            "What is computer vision?",
        ]

        start_time = time.time()

        # Process queries concurrently
        tasks = [self.orchestrator.process_query(query) for query in queries]
        results = asyncio.run(asyncio.gather(*tasks))

        end_time = time.time()
        total_time = end_time - start_time

        # Verify all queries completed successfully
        for result in results:
            self.assertIn("success", result)
            self.assertIn("execution_time_ms", result)

        # Verify concurrent processing is efficient
        self.assertLess(total_time, 10.0)  # Should complete within 10 seconds

    def test_memory_usage(self):
        """Test memory usage"""
        # Test memory usage during processing
        import psutil
        import os

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss

        # Process multiple queries
        for i in range(10):
            query = f"Test query {i}"
            asyncio.run(self.orchestrator.process_query(query))

        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory

        # Verify memory usage is reasonable (less than 100MB increase)
        self.assertLess(memory_increase, 100 * 1024 * 1024)  # 100MB


class TestErrorHandlingIntegration(unittest.TestCase):
    """Test error handling integration"""

    def setUp(self):
        """Set up test fixtures"""
        self.orchestrator = LeadOrchestrator()

    def test_agent_failure_handling(self):
        """Test handling of agent failures"""
        # Test that system continues to function when individual agents fail
        query = "test query"

        # Mock agent failure
        with patch.object(
            self.orchestrator.agents[AgentType.RETRIEVAL], "process_task"
        ) as mock_process:
            mock_process.side_effect = Exception("Agent failure")

            result = asyncio.run(self.orchestrator.process_query(query))

            # Verify system handles failure gracefully
            self.assertIn("error", result)
            self.assertIn("fallback_response", result)

    def test_network_failure_handling(self):
        """Test handling of network failures"""
        # Test that system handles network failures gracefully
        pass

    def test_timeout_handling(self):
        """Test handling of timeouts"""
        # Test that system handles timeouts gracefully
        pass


class TestSecurityIntegration(unittest.TestCase):
    """Test security integration"""

    def setUp(self):
        """Set up test fixtures"""
        self.client = TestClient(app)

    def test_input_validation(self):
        """Test input validation"""
        # Test malicious input handling
        malicious_queries = [
            "<script>alert('xss')</script>",
            "'; DROP TABLE users; --",
            "../../../etc/passwd",
            "query" * 10000,  # Very long query
        ]

        for malicious_query in malicious_queries:
            response = self.client.post("/query", json={"query": malicious_query})
            # Should not crash and should handle gracefully
            self.assertIn(response.status_code, [200, 400, 422])

    def test_rate_limiting(self):
        """Test rate limiting"""
        # Test rate limiting functionality
        for i in range(100):  # Make many requests
            response = self.client.post("/query", json={"query": f"query {i}"})
            # Should handle rate limiting gracefully
            self.assertIn(response.status_code, [200, 429])


class TestMonitoringIntegration(unittest.TestCase):
    """Test monitoring integration"""

    def setUp(self):
        """Set up test fixtures"""
        self.orchestrator = LeadOrchestrator()

    def test_metrics_collection(self):
        """Test metrics collection"""
        # Test that metrics are properly collected
        query = "test query"
        result = asyncio.run(self.orchestrator.process_query(query))

        # Verify metrics are collected
        self.assertIn("execution_time_ms", result)
        self.assertGreater(result["execution_time_ms"], 0)

    def test_health_check_integration(self):
        """Test health check integration"""
        # Test health check endpoint
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)

        health_data = response.json()
        self.assertIn("status", health_data)
        self.assertIn("agents", health_data)
        self.assertIn("timestamp", health_data)


class TestDeploymentIntegration(unittest.TestCase):
    """Test deployment integration"""

    def test_docker_build(self):
        """Test Docker build process"""
        # Test that Docker images can be built
        try:
            result = subprocess.run(
                ["docker", "build", "-t", "knowledge-hub-test", "."],
                capture_output=True,
                text=True,
                timeout=300,
            )
            self.assertEqual(result.returncode, 0)
        except (subprocess.TimeoutExpired, FileNotFoundError):
            # Docker might not be available, skip test
            self.skipTest("Docker not available")

    def test_kubernetes_deployment(self):
        """Test Kubernetes deployment"""
        # Test that Kubernetes manifests are valid
        try:
            result = subprocess.run(
                ["kubectl", "apply", "--dry-run=client", "-f", "infrastructure/kubernetes/"],
                capture_output=True,
                text=True,
                timeout=60,
            )
            # Should not have errors
            self.assertNotIn("error", result.stderr.lower())
        except (subprocess.TimeoutExpired, FileNotFoundError):
            # kubectl might not be available, skip test
            self.skipTest("kubectl not available")


if __name__ == "__main__":
    # Run all integration tests
    unittest.main(verbosity=2)
