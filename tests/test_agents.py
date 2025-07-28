"""
Comprehensive Unit Tests for All Agent Components
Tests all agents: Retrieval, FactCheck, Synthesis, Citation, and LeadOrchestrator
"""

import unittest
import asyncio
import json
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from datetime import datetime, timedelta
from typing import Dict, List, Any

# Import agent components
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.base_agent import BaseAgent, QueryContext, AgentResult, AgentMessage, AgentType
from agents.lead_orchestrator import LeadOrchestrator
from agents.retrieval_agent import RetrievalAgent
from agents.factcheck_agent import FactCheckAgent
from agents.synthesis_agent import SynthesisAgent
from agents.citation_agent import CitationAgent


class TestBaseAgent(unittest.TestCase):
    """Test BaseAgent abstract class"""

    def setUp(self):
        """Set up test fixtures"""
        self.agent_id = "test_agent_001"
        self.agent_type = AgentType.RETRIEVAL

    def test_base_agent_initialization(self):
        """Test BaseAgent initialization"""

        # Create a concrete implementation for testing
        class TestAgent(BaseAgent):
            async def process_task(
                self, task: Dict[str, Any], context: QueryContext
            ) -> AgentResult:
                return AgentResult(
                    success=True,
                    data={"test": "data"},
                    confidence=0.9,
                    token_usage={"prompt": 100, "completion": 50},
                    execution_time_ms=100,
                )

        agent = TestAgent(self.agent_id, self.agent_type)

        self.assertEqual(agent.agent_id, self.agent_id)
        self.assertEqual(agent.agent_type, self.agent_type)
        self.assertIsInstance(agent.message_queue, asyncio.Queue)
        self.assertIsInstance(agent.metrics, dict)
        self.assertFalse(agent.is_running)
        self.assertEqual(agent.health_status, "healthy")

    def test_base_agent_message_handling(self):
        """Test BaseAgent message handling"""

        class TestAgent(BaseAgent):
            async def process_task(
                self, task: Dict[str, Any], context: QueryContext
            ) -> AgentResult:
                return AgentResult(success=True, data={"processed": task}, confidence=0.9)

        agent = TestAgent(self.agent_id, self.agent_type)

        # Test task message handling
        task_message = AgentMessage(
            content="test query",
            sender_id="orchestrator",
            recipient_id=self.agent_id,
            message_type="task",
        )

        # Test message processing
        result = asyncio.run(agent.process_task(task_message))
        self.assertIsNotNone(result)
        self.assertTrue(result.success)

    def test_base_agent_error_handling(self):
        """Test BaseAgent error handling"""

        class TestAgent(BaseAgent):
            async def process_task(
                self, task: Dict[str, Any], context: QueryContext
            ) -> AgentResult:
                raise Exception("Test error")

        agent = TestAgent(self.agent_id, self.agent_type)

        task_message = AgentMessage(
            content="test query",
            sender_id="orchestrator",
            recipient_id=self.agent_id,
            message_type="task",
        )

        # Test error message handling
        result = asyncio.run(agent.handle_message(task_message))
        self.assertEqual(result.header["message_type"], MessageType.ERROR)
        self.assertIn("Test error", result.payload["error"])

    def test_base_agent_heartbeat(self):
        """Test BaseAgent heartbeat handling"""

        class TestAgent(BaseAgent):
            async def process_task(
                self, task: Dict[str, Any], context: QueryContext
            ) -> AgentResult:
                return AgentResult(success=True, data={})

        agent = TestAgent(self.agent_id, self.agent_type)

        heartbeat_message = AgentMessage(
            header={
                "message_id": "heartbeat_001",
                "sender_agent": "monitor",
                "recipient_agent": self.agent_id,
                "message_type": MessageType.HEARTBEAT,
                "timestamp": datetime.utcnow().isoformat(),
            },
            payload={},
        )

        # Test heartbeat response
        result = asyncio.run(agent.handle_message(heartbeat_message))
        self.assertIsNotNone(result)
        self.assertIn("health_status", result.payload)
        self.assertIn("metrics", result.payload)


class TestRetrievalAgent(unittest.TestCase):
    """Test RetrievalAgent functionality"""

    def setUp(self):
        """Set up test fixtures"""
        self.agent = RetrievalAgent()  # Use default config
        self.query_context = QueryContext(query="test query")

    def test_retrieval_agent_initialization(self):
        """Test RetrievalAgent initialization"""
        self.assertEqual(self.agent.agent_id, "retrieval_001")
        self.assertEqual(self.agent.agent_type.value, AgentType.RETRIEVAL.value)
        self.assertIsNone(self.agent.vector_db_client)
        self.assertIsNone(self.agent.search_client)
        self.assertIsNone(self.agent.graph_client)

    @patch("agents.retrieval_agent.asyncio.sleep")
    def test_vector_search(self, mock_sleep):
        """Test vector search functionality"""
        mock_sleep.return_value = None

        # Test vector search
        results = asyncio.run(self.agent.vector_search("test query", 5))

        self.assertIsInstance(results, list)
        self.assertLessEqual(len(results), 5)

        for result in results:
            self.assertIn("content", result)
            self.assertIn("score", result)
            self.assertIn("source", result)
            self.assertIn("metadata", result)

    @patch("agents.retrieval_agent.asyncio.sleep")
    def test_keyword_search(self, mock_sleep):
        """Test keyword search functionality"""
        mock_sleep.return_value = None

        filters = {"category": "test", "date_range": "last_week"}
        results = asyncio.run(self.agent.keyword_search("test query", filters))

        self.assertIsInstance(results, list)
        self.assertGreater(len(results), 0)

        for result in results:
            self.assertIn("content", result)
            self.assertIn("score", result)
            self.assertIn("source", result)
            self.assertIn("metadata", result)

    @patch("agents.retrieval_agent.asyncio.sleep")
    def test_graph_query(self, mock_sleep):
        """Test graph query functionality"""
        mock_sleep.return_value = None

        entities = ["entity1", "entity2"]
        results = asyncio.run(self.agent.graph_query(entities, 2))

        self.assertIsInstance(results, list)
        self.assertEqual(len(results), len(entities))

        for result in results:
            self.assertIn("entity", result)
            self.assertIn("facts", result)
            self.assertIn("relationships", result)
            self.assertIn("confidence", result)

    def test_hybrid_retrieval(self):
        """Test hybrid retrieval functionality"""
        task = {
            "strategy": "hybrid",
            "top_k": 10,
            "filters": {"category": "test"},
            "entities": ["entity1"],
        }

        results = asyncio.run(self.agent._hybrid_retrieval("test query", task))

        self.assertIsInstance(results, list)
        self.assertGreater(len(results), 0)

    def test_retrieval_confidence_calculation(self):
        """Test retrieval confidence calculation"""
        # Test with empty results
        confidence = self.agent._calculate_retrieval_confidence([])
        self.assertEqual(confidence, 0.0)

        # Test with results
        results = [
            {"score": 0.9, "content": "test1"},
            {"score": 0.8, "content": "test2"},
            {"score": 0.7, "content": "test3"},
        ]

        confidence = self.agent._calculate_retrieval_confidence(results)
        self.assertGreater(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)

    def test_process_task_success(self):
        """Test successful task processing"""
        task = {"strategy": "hybrid", "top_k": 5, "filters": {}, "entities": []}

        result = asyncio.run(self.agent.process_task(task, self.query_context))

        self.assertTrue(result.success)
        self.assertIn("retrieved_documents", result.data)
        self.assertGreater(result.confidence, 0.0)
        self.assertIn("prompt", result.token_usage)
        self.assertIn("completion", result.token_usage)
        self.assertGreater(result.execution_time_ms, 0)

    def test_process_task_error(self):
        """Test task processing with error"""
        task = {"strategy": "invalid_strategy", "top_k": 5}

        result = asyncio.run(self.agent.process_task(task, self.query_context))

        self.assertFalse(result.success)
        self.assertIsNotNone(result.error)
        self.assertGreater(result.execution_time_ms, 0)


class TestFactCheckAgent(unittest.TestCase):
    """Test FactCheckAgent functionality"""

    def setUp(self):
        """Set up test fixtures"""
        self.agent = FactCheckAgent("factcheck_001")
        self.query_context = QueryContext(query="test query")

    def test_factcheck_agent_initialization(self):
        """Test FactCheckAgent initialization"""
        self.assertEqual(self.agent.agent_id, "factcheck_001")
        self.assertEqual(self.agent.agent_type, AgentType.FACT_CHECK)
        self.assertIsNone(self.agent.knowledge_base)
        self.assertIsNone(self.agent.fact_check_model)

    @patch("agents.factcheck_agent.asyncio.sleep")
    def test_verify_claim(self, mock_sleep):
        """Test claim verification"""
        mock_sleep.return_value = None

        claim = "The Earth is round"
        sources = [{"id": "source1", "content": "Earth is spherical"}]

        result = asyncio.run(self.agent.verify_claim(claim, sources))

        self.assertIn("claim", result)
        self.assertIn("verdict", result)
        self.assertIn("confidence", result)
        self.assertIn("evidence", result)
        self.assertIn("reasoning", result)

        self.assertEqual(result["claim"], claim)
        self.assertIn(result["verdict"], ["supported", "refuted", "unverifiable"])
        self.assertGreaterEqual(result["confidence"], 0.0)
        self.assertLessEqual(result["confidence"], 1.0)

    def test_verify_claims(self):
        """Test multiple claims verification"""
        claims = ["The Earth is round", "Water boils at 100°C", "The sky is blue"]
        sources = [{"id": "source1", "content": "Test source content"}]

        results = asyncio.run(self.agent.verify_claims(claims, sources))

        self.assertIsInstance(results, list)
        self.assertEqual(len(results), len(claims))

        for result in results:
            self.assertIn("claim", result)
            self.assertIn("verdict", result)
            self.assertIn("confidence", result)

    def test_verification_confidence_calculation(self):
        """Test verification confidence calculation"""
        # Test with empty results
        confidence = self.agent._calculate_verification_confidence([])
        self.assertEqual(confidence, 0.0)

        # Test with supported claims
        results = [
            {"verdict": "supported", "confidence": 0.9},
            {"verdict": "supported", "confidence": 0.8},
            {"verdict": "supported", "confidence": 0.7},
        ]

        confidence = self.agent._calculate_verification_confidence(results)
        self.assertGreater(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)

        # Test with unverifiable claims (should reduce confidence)
        results_with_unverifiable = [
            {"verdict": "supported", "confidence": 0.9},
            {"verdict": "unverifiable", "confidence": 0.0},
            {"verdict": "supported", "confidence": 0.8},
        ]

        confidence_with_penalty = self.agent._calculate_verification_confidence(
            results_with_unverifiable
        )
        self.assertLess(confidence_with_penalty, confidence)

    def test_process_task_success(self):
        """Test successful fact-checking task"""
        task = {
            "claims": ["The Earth is round", "Water boils at 100°C"],
            "sources": [{"id": "source1", "content": "Test source"}],
            "verification_depth": "thorough",
        }

        result = asyncio.run(self.agent.process_task(task, self.query_context))

        self.assertTrue(result.success)
        self.assertIn("verifications", result.data)
        self.assertGreater(result.confidence, 0.0)
        self.assertIn("prompt", result.token_usage)
        self.assertIn("completion", result.token_usage)
        self.assertGreater(result.execution_time_ms, 0)

    def test_process_task_error(self):
        """Test fact-checking task with error"""
        task = {"claims": None, "sources": []}  # Invalid input

        result = asyncio.run(self.agent.process_task(task, self.query_context))

        self.assertFalse(result.success)
        self.assertIsNotNone(result.error)
        self.assertGreater(result.execution_time_ms, 0)


class TestSynthesisAgent(unittest.TestCase):
    """Test SynthesisAgent functionality"""

    def setUp(self):
        """Set up test fixtures"""
        self.agent = SynthesisAgent("synthesis_001")
        self.query_context = QueryContext(query="test query")

    def test_synthesis_agent_initialization(self):
        """Test SynthesisAgent initialization"""
        self.assertEqual(self.agent.agent_id, "synthesis_001")
        self.assertEqual(self.agent.agent_type, AgentType.SYNTHESIS)
        self.assertIsNone(self.agent.synthesis_model)
        self.assertIsNone(self.agent.coherence_scorer)

    @patch("agents.synthesis_agent.asyncio.sleep")
    def test_synthesize(self, mock_sleep):
        """Test synthesis functionality"""
        mock_sleep.return_value = None

        verified_facts = [
            {"claim": "Fact 1", "confidence": 0.9},
            {"claim": "Fact 2", "confidence": 0.8},
            {"claim": "Fact 3", "confidence": 0.7},
        ]

        params = {"style": "informative", "length": "medium", "include_uncertainty": True}

        result = asyncio.run(self.agent.synthesize(verified_facts, self.query_context, params))

        self.assertIn("text", result)
        self.assertIn("key_points", result)
        self.assertIn("confidence", result)
        self.assertIn("reasoning_trace", result)
        self.assertIn("uncertainty_areas", result)
        self.assertIn("coherence_score", result)

        self.assertIsInstance(result["text"], str)
        self.assertIsInstance(result["key_points"], list)
        self.assertGreaterEqual(result["confidence"], 0.0)
        self.assertLessEqual(result["confidence"], 1.0)
        self.assertIsInstance(result["reasoning_trace"], list)
        self.assertIsInstance(result["uncertainty_areas"], list)
        self.assertGreaterEqual(result["coherence_score"], 0.0)
        self.assertLessEqual(result["coherence_score"], 1.0)

    def test_process_task_success(self):
        """Test successful synthesis task"""
        task = {
            "verified_facts": [
                {"claim": "Fact 1", "confidence": 0.9},
                {"claim": "Fact 2", "confidence": 0.8},
            ],
            "synthesis_params": {
                "style": "informative",
                "length": "medium",
                "include_uncertainty": True,
            },
        }

        result = asyncio.run(self.agent.process_task(task, self.query_context))

        self.assertTrue(result.success)
        self.assertIn("response", result.data)
        self.assertGreater(result.confidence, 0.0)
        self.assertIn("prompt", result.token_usage)
        self.assertIn("completion", result.token_usage)
        self.assertGreater(result.execution_time_ms, 0)

    def test_process_task_error(self):
        """Test synthesis task with error"""
        task = {"verified_facts": None, "synthesis_params": {}}  # Invalid input

        result = asyncio.run(self.agent.process_task(task, self.query_context))

        self.assertFalse(result.success)
        self.assertIsNotNone(result.error)
        self.assertGreater(result.execution_time_ms, 0)


class TestCitationAgent(unittest.TestCase):
    """Test CitationAgent functionality"""

    def setUp(self):
        """Set up test fixtures"""
        self.agent = CitationAgent("citation_001")
        self.query_context = QueryContext(query="test query")

    def test_citation_agent_initialization(self):
        """Test CitationAgent initialization"""
        self.assertEqual(self.agent.agent_id, "citation_001")
        self.assertEqual(self.agent.agent_type, AgentType.CITATION)
        self.assertIsNone(self.agent.citation_formatter)
        self.assertIsNone(self.agent.reliability_scorer)

    @patch("agents.citation_agent.asyncio.sleep")
    def test_generate_citations(self, mock_sleep):
        """Test citation generation"""
        mock_sleep.return_value = None

        content = "This is a test content that needs citations."
        sources = [
            {"id": "source1", "title": "Test Source 1", "url": "http://example1.com"},
            {"id": "source2", "title": "Test Source 2", "url": "http://example2.com"},
        ]
        style = "APA"

        result = asyncio.run(self.agent.generate_citations(content, sources, style))

        self.assertIn("cited_content", result)
        self.assertIn("inline_citations", result)
        self.assertIn("bibliography", result)
        self.assertIn("citation_style", result)
        self.assertIn("total_sources", result)

        self.assertEqual(result["citation_style"], style)
        self.assertEqual(result["total_sources"], len(sources))
        self.assertIsInstance(result["inline_citations"], list)
        self.assertIsInstance(result["bibliography"], list)

    def test_process_task_success(self):
        """Test successful citation task"""
        task = {
            "content": "This is test content.",
            "sources": [{"id": "source1", "title": "Test Source", "url": "http://example.com"}],
            "citation_style": "APA",
        }

        result = asyncio.run(self.agent.process_task(task, self.query_context))

        self.assertTrue(result.success)
        self.assertIn("citations", result.data)
        self.assertGreater(result.confidence, 0.0)
        self.assertIn("prompt", result.token_usage)
        self.assertIn("completion", result.token_usage)
        self.assertGreater(result.execution_time_ms, 0)

    def test_process_task_error(self):
        """Test citation task with error"""
        task = {"content": None, "sources": [], "citation_style": "APA"}  # Invalid input

        result = asyncio.run(self.agent.process_task(task, self.query_context))

        self.assertFalse(result.success)
        self.assertIsNotNone(result.error)
        self.assertGreater(result.execution_time_ms, 0)


class TestLeadOrchestrator(unittest.TestCase):
    """Test LeadOrchestrator functionality"""

    def setUp(self):
        """Set up test fixtures"""
        self.orchestrator = LeadOrchestrator()

    def test_orchestrator_initialization(self):
        """Test LeadOrchestrator initialization"""
        self.assertIsInstance(self.orchestrator.agents, dict)
        self.assertIn(AgentType.RETRIEVAL, self.orchestrator.agents)
        self.assertIn(AgentType.FACT_CHECK, self.orchestrator.agents)
        self.assertIn(AgentType.SYNTHESIS, self.orchestrator.agents)
        self.assertIn(AgentType.CITATION, self.orchestrator.agents)

        self.assertIsNotNone(self.orchestrator.message_broker)
        self.assertIsNotNone(self.orchestrator.token_controller)
        self.assertIsNotNone(self.orchestrator.cache_manager)
        self.assertIsNotNone(self.orchestrator.response_aggregator)

    def test_analyze_and_plan(self):
        """Test query analysis and planning"""
        context = QueryContext(query="What is quantum computing?")

        plan = asyncio.run(self.orchestrator.analyze_and_plan(context))

        self.assertIn("pattern", plan)
        self.assertIn("complexity_score", plan)
        self.assertIn("required_agents", plan)
        self.assertIn("estimated_tokens", plan)
        self.assertIn("timeout_ms", plan)

        self.assertIn(plan["pattern"], ["simple", "fork_join", "scatter_gather"])
        self.assertGreaterEqual(plan["complexity_score"], 0.0)
        self.assertIsInstance(plan["required_agents"], list)
        self.assertGreater(plan["estimated_tokens"], 0)
        self.assertGreater(plan["timeout_ms"], 0)

    def test_execute_pipeline(self):
        """Test pipeline execution"""
        context = QueryContext(query="test query")
        plan = {
            "pattern": "simple",
            "required_agents": [
                AgentType.RETRIEVAL,
                AgentType.FACT_CHECK,
                AgentType.SYNTHESIS,
                AgentType.CITATION,
            ],
            "timeout_ms": 5000,
        }

        result = asyncio.run(self.orchestrator.execute_pipeline(context, plan))

        self.assertIsInstance(result, dict)
        # Note: In a real test, we would mock the agent responses

    def test_process_query_success(self):
        """Test successful query processing"""
        query = "What is artificial intelligence?"
        user_context = {"user_id": "test_user"}

        result = asyncio.run(self.orchestrator.process_query(query, user_context))

        self.assertIsInstance(result, dict)
        # Note: In a real test, we would mock all dependencies

    def test_process_query_error(self):
        """Test query processing with error"""
        query = None  # Invalid query
        user_context = {}

        result = asyncio.run(self.orchestrator.process_query(query, user_context))

        self.assertIn("error", result)
        self.assertIn("execution_time_ms", result)
        self.assertIn("fallback_response", result)

    def test_cache_hit(self):
        """Test cache hit scenario"""
        query = "cached query"

        # Mock cache hit
        with patch.object(self.orchestrator.cache_manager, "get_cached_response") as mock_cache:
            mock_cache.return_value = {"response": "cached response"}

            result = asyncio.run(self.orchestrator.process_query(query))

            self.assertEqual(result["response"], "cached response")
            mock_cache.assert_called_once_with(query)

    def test_merge_retrieval_results(self):
        """Test merging retrieval results"""
        results = [
            AgentResult(
                success=True,
                data={"retrieved_documents": [{"content": "doc1", "score": 0.9}]},
                confidence=0.9,
                token_usage={"prompt": 100, "completion": 50},
            ),
            AgentResult(
                success=True,
                data={"retrieved_documents": [{"content": "doc2", "score": 0.8}]},
                confidence=0.8,
                token_usage={"prompt": 100, "completion": 50},
            ),
        ]

        merged = self.orchestrator._merge_retrieval_results(results)

        self.assertTrue(merged.success)
        self.assertIn("retrieved_documents", merged.data)
        self.assertGreater(merged.confidence, 0.0)
        self.assertIn("prompt", merged.token_usage)
        self.assertIn("completion", merged.token_usage)

    def test_assess_result_quality(self):
        """Test result quality assessment"""
        # Test retrieval agent quality assessment
        retrieval_result = AgentResult(
            success=True,
            data={"retrieved_documents": [{"content": "doc1", "score": 0.9}] * 10},
            confidence=0.9,
        )

        quality = self.orchestrator._assess_result_quality(AgentType.RETRIEVAL, retrieval_result)
        self.assertGreater(quality, 0.0)
        self.assertLessEqual(quality, 1.0)

        # Test fact-check agent quality assessment
        factcheck_result = AgentResult(
            success=True,
            data={
                "verifications": [
                    {"verdict": "supported", "confidence": 0.9},
                    {"verdict": "supported", "confidence": 0.8},
                ]
            },
            confidence=0.85,
        )

        quality = self.orchestrator._assess_result_quality(AgentType.FACT_CHECK, factcheck_result)
        self.assertGreater(quality, 0.0)
        self.assertLessEqual(quality, 1.0)

    def test_generate_improvement_feedback(self):
        """Test feedback generation"""
        result = AgentResult(success=True, data={"test": "data"}, confidence=0.5)  # Low confidence

        feedback = self.orchestrator._generate_improvement_feedback(AgentType.RETRIEVAL, result)

        self.assertIn("quality_issues", feedback)
        self.assertIn("suggestions", feedback)
        self.assertIn("priority", feedback)
        self.assertIsInstance(feedback["suggestions"], list)


class TestAgentIntegration(unittest.TestCase):
    """Test integration between agents"""

    def setUp(self):
        """Set up test fixtures"""
        self.orchestrator = LeadOrchestrator()
        self.query_context = QueryContext(query="test query")

    def test_agent_communication(self):
        """Test agent-to-agent communication"""
        # Test message passing between agents
        retrieval_agent = self.orchestrator.agents[AgentType.RETRIEVAL]
        synthesis_agent = self.orchestrator.agents[AgentType.SYNTHESIS]

        # Create a task message
        task_message = AgentMessage(
            header={
                "message_id": "test_msg_001",
                "sender_agent": "orchestrator",
                "recipient_agent": retrieval_agent.agent_id,
                "message_type": MessageType.TASK,
                "timestamp": datetime.utcnow().isoformat(),
            },
            payload={"task": {"strategy": "hybrid", "top_k": 5}, "context": self.query_context},
        )

        # Process the message
        result_message = asyncio.run(retrieval_agent.handle_message(task_message))

        self.assertIsNotNone(result_message)
        self.assertEqual(result_message.header["message_type"], MessageType.RESULT)

    def test_end_to_end_workflow(self):
        """Test complete end-to-end workflow"""
        # This would test the complete workflow from query to response
        # In a real implementation, we would mock all external dependencies
        pass


if __name__ == "__main__":
    # Run all tests
    unittest.main(verbosity=2)
