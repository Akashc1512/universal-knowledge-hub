"""
Unit tests for the refactored LeadOrchestrator.
Tests the new modular pipeline execution methods.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime

from agents.lead_orchestrator import LeadOrchestrator
from agents.base_agent import QueryContext, AgentResult, AgentType


class TestLeadOrchestratorRefactored:
    """Test the refactored LeadOrchestrator with modular pipeline methods."""
    
    @pytest.fixture
    def orchestrator(self):
        """Create a LeadOrchestrator instance for testing."""
        return LeadOrchestrator()
    
    @pytest.fixture
    def query_context(self):
        """Create a test QueryContext."""
        return QueryContext(
            query="What is machine learning?",
            user_id="test_user",
            user_context={"max_tokens": 1000},
            token_budget=1000,
            trace_id="test_trace_123"
        )
    
    @pytest.fixture
    def mock_agents(self, orchestrator):
        """Mock all agents in the orchestrator."""
        mock_retrieval = AsyncMock()
        mock_fact_check = AsyncMock()
        mock_synthesis = AsyncMock()
        mock_citation = AsyncMock()
        
        orchestrator.agents = {
            AgentType.RETRIEVAL: mock_retrieval,
            AgentType.FACT_CHECK: mock_fact_check,
            AgentType.SYNTHESIS: mock_synthesis,
            AgentType.CITATION: mock_citation,
        }
        
        return {
            "retrieval": mock_retrieval,
            "fact_check": mock_fact_check,
            "synthesis": mock_synthesis,
            "citation": mock_citation,
        }
    
    def create_success_result(self, data=None):
        """Helper to create a successful AgentResult."""
        return AgentResult(
            success=True,
            data=data or {},
            confidence=0.8,
            execution_time_ms=100
        )
    
    def create_failure_result(self, error="Test error"):
        """Helper to create a failed AgentResult."""
        return AgentResult(
            success=False,
            error=error,
            confidence=0.0,
            execution_time_ms=0
        )

    @pytest.mark.asyncio
    async def test_execute_retrieval_phase_success(self, orchestrator, query_context, mock_agents):
        """Test successful retrieval phase execution."""
        # Mock entity extraction
        with patch.object(orchestrator, '_extract_entities_parallel') as mock_entities:
            mock_entities.return_value = [{"text": "machine learning", "type": "CONCEPT"}]
            
            # Mock successful retrieval
            mock_agents["retrieval"].process_task.return_value = self.create_success_result({
                "documents": [{"content": "ML is a subset of AI", "score": 0.9}]
            })
            
            # Execute retrieval phase
            results = await orchestrator._execute_retrieval_phase(query_context, {})
            
            # Verify results
            assert AgentType.RETRIEVAL in results
            assert results[AgentType.RETRIEVAL].success
            assert "documents" in results[AgentType.RETRIEVAL].data
            
            # Verify entity extraction was called
            mock_entities.assert_called_once_with(query_context.query)
            
            # Verify retrieval agent was called with correct task
            mock_agents["retrieval"].process_task.assert_called_once()
            call_args = mock_agents["retrieval"].process_task.call_args[0]
            assert call_args[0]["query"] == query_context.query
            assert "entities" in call_args[0]

    @pytest.mark.asyncio
    async def test_execute_retrieval_phase_timeout(self, orchestrator, query_context, mock_agents):
        """Test retrieval phase timeout handling."""
        # Mock entity extraction timeout
        with patch.object(orchestrator, '_extract_entities_parallel') as mock_entities:
            mock_entities.side_effect = asyncio.TimeoutError()
            
            # Execute retrieval phase
            results = await orchestrator._execute_retrieval_phase(query_context, {})
            
            # Verify timeout result
            assert AgentType.RETRIEVAL in results
            assert not results[AgentType.RETRIEVAL].success
            assert "timed out" in results[AgentType.RETRIEVAL].error.lower()
            assert results[AgentType.RETRIEVAL].execution_time_ms == 15000

    @pytest.mark.asyncio
    async def test_execute_fact_checking_phase_success(self, orchestrator, query_context, mock_agents):
        """Test successful fact checking phase execution."""
        # Setup retrieval results
        retrieval_result = self.create_success_result({
            "documents": [{"content": "ML is a subset of AI", "score": 0.9}]
        })
        initial_results = {AgentType.RETRIEVAL: retrieval_result}
        
        # Mock successful fact checking
        mock_agents["fact_check"].process_task.return_value = self.create_success_result({
            "verified_facts": [{"claim": "ML is AI subset", "confidence": 0.9}]
        })
        
        # Execute fact checking phase
        results = await orchestrator._execute_fact_checking_phase(query_context, initial_results)
        
        # Verify results
        assert AgentType.FACT_CHECK in results
        assert results[AgentType.FACT_CHECK].success
        assert "verified_facts" in results[AgentType.FACT_CHECK].data

    @pytest.mark.asyncio
    async def test_execute_fact_checking_phase_skip_on_retrieval_failure(self, orchestrator, query_context, mock_agents):
        """Test fact checking phase skips when retrieval fails."""
        # Setup failed retrieval results
        retrieval_result = self.create_failure_result("Retrieval failed")
        initial_results = {AgentType.RETRIEVAL: retrieval_result}
        
        # Execute fact checking phase
        results = await orchestrator._execute_fact_checking_phase(query_context, initial_results)
        
        # Verify skipping behavior
        assert AgentType.FACT_CHECK in results
        assert not results[AgentType.FACT_CHECK].success
        assert "skipped" in results[AgentType.FACT_CHECK].error.lower()
        
        # Verify fact check agent was not called
        mock_agents["fact_check"].process_task.assert_not_called()

    @pytest.mark.asyncio
    async def test_execute_synthesis_phase_success(self, orchestrator, query_context, mock_agents):
        """Test successful synthesis phase execution."""
        # Mock synthesis input preparation
        with patch.object(orchestrator, '_prepare_synthesis_input') as mock_prepare:
            mock_prepare.return_value = {"query": query_context.query, "facts": []}
            
            # Mock successful synthesis
            mock_agents["synthesis"].process_task.return_value = self.create_success_result({
                "response": "Machine learning is a subset of artificial intelligence."
            })
            
            # Execute synthesis phase
            results = await orchestrator._execute_synthesis_phase(query_context, {})
            
            # Verify results
            assert AgentType.SYNTHESIS in results
            assert results[AgentType.SYNTHESIS].success
            assert "response" in results[AgentType.SYNTHESIS].data

    @pytest.mark.asyncio
    async def test_execute_citation_phase_success(self, orchestrator, query_context, mock_agents):
        """Test successful citation phase execution."""
        # Setup previous results
        synthesis_result = self.create_success_result({
            "response": "Machine learning is a subset of AI."
        })
        retrieval_result = self.create_success_result({
            "documents": [{"content": "ML info", "source": "test.com"}]
        })
        initial_results = {
            AgentType.SYNTHESIS: synthesis_result,
            AgentType.RETRIEVAL: retrieval_result
        }
        
        # Mock successful citation
        mock_agents["citation"].process_task.return_value = self.create_success_result({
            "citations": ["[1] Test source"]
        })
        
        # Execute citation phase
        results = await orchestrator._execute_citation_phase(query_context, initial_results)
        
        # Verify results
        assert AgentType.CITATION in results
        assert results[AgentType.CITATION].success
        assert "citations" in results[AgentType.CITATION].data

    @pytest.mark.asyncio
    async def test_execute_pipeline_full_success(self, orchestrator, query_context, mock_agents):
        """Test complete pipeline execution with all phases succeeding."""
        # Mock entity extraction
        with patch.object(orchestrator, '_extract_entities_parallel') as mock_entities:
            mock_entities.return_value = [{"text": "machine learning", "type": "CONCEPT"}]
            
            # Mock synthesis input preparation
            with patch.object(orchestrator, '_prepare_synthesis_input') as mock_prepare:
                mock_prepare.return_value = {"query": query_context.query, "facts": []}
                
                # Setup all agents to succeed
                mock_agents["retrieval"].process_task.return_value = self.create_success_result({
                    "documents": [{"content": "ML info", "score": 0.9}]
                })
                mock_agents["fact_check"].process_task.return_value = self.create_success_result({
                    "verified_facts": [{"claim": "ML is AI", "confidence": 0.9}]
                })
                mock_agents["synthesis"].process_task.return_value = self.create_success_result({
                    "response": "ML is a subset of AI."
                })
                mock_agents["citation"].process_task.return_value = self.create_success_result({
                    "citations": ["[1] Source"]
                })
                
                # Execute full pipeline
                results = await orchestrator.execute_pipeline(query_context, {}, 1000)
                
                # Verify all phases completed
                assert len(results) == 4
                assert all(agent_type in results for agent_type in [
                    AgentType.RETRIEVAL, AgentType.FACT_CHECK, 
                    AgentType.SYNTHESIS, AgentType.CITATION
                ])
                assert all(result.success for result in results.values())

    @pytest.mark.asyncio
    async def test_execute_pipeline_graceful_degradation(self, orchestrator, query_context, mock_agents):
        """Test pipeline graceful degradation when some phases fail."""
        # Mock entity extraction
        with patch.object(orchestrator, '_extract_entities_parallel') as mock_entities:
            mock_entities.return_value = []
            
            # Mock synthesis input preparation
            with patch.object(orchestrator, '_prepare_synthesis_input') as mock_prepare:
                mock_prepare.return_value = {"query": query_context.query, "facts": []}
                
                # Setup mixed success/failure
                mock_agents["retrieval"].process_task.return_value = self.create_success_result({
                    "documents": [{"content": "ML info", "score": 0.9}]
                })
                mock_agents["fact_check"].process_task.side_effect = Exception("Fact check failed")
                mock_agents["synthesis"].process_task.return_value = self.create_success_result({
                    "response": "ML is a subset of AI."
                })
                mock_agents["citation"].process_task.return_value = self.create_success_result({
                    "citations": ["[1] Source"]
                })
                
                # Execute pipeline
                results = await orchestrator.execute_pipeline(query_context, {}, 1000)
                
                # Verify partial success
                assert len(results) == 4
                assert results[AgentType.RETRIEVAL].success
                assert not results[AgentType.FACT_CHECK].success
                assert results[AgentType.SYNTHESIS].success
                assert results[AgentType.CITATION].success

    def test_create_timeout_result(self, orchestrator):
        """Test timeout result creation."""
        result = orchestrator._create_timeout_result("Test timeout", 5000)
        
        assert not result.success
        assert "Test timeout" in result.error
        assert result.confidence == 0.0
        assert result.execution_time_ms == 5000

    def test_create_error_result(self, orchestrator):
        """Test error result creation."""
        result = orchestrator._create_error_result("Test error")
        
        assert not result.success
        assert "Test error" in result.error
        assert result.confidence == 0.0
        assert result.execution_time_ms == 0

    def test_handle_pipeline_failure(self, orchestrator):
        """Test pipeline failure handling."""
        partial_results = {
            AgentType.RETRIEVAL: self.create_success_result()
        }
        
        results = orchestrator._handle_pipeline_failure(partial_results, "Pipeline error")
        
        # Verify all agent types have results
        assert len(results) == 4
        assert results[AgentType.RETRIEVAL].success  # Existing result preserved
        assert not results[AgentType.FACT_CHECK].success  # Error result added
        assert not results[AgentType.SYNTHESIS].success  # Error result added
        assert not results[AgentType.CITATION].success  # Error result added
        
        # Verify error messages
        for agent_type in [AgentType.FACT_CHECK, AgentType.SYNTHESIS, AgentType.CITATION]:
            assert "Pipeline error" in results[agent_type].error

    @pytest.mark.asyncio
    async def test_pipeline_timeout_resilience(self, orchestrator, query_context, mock_agents):
        """Test pipeline resilience to individual phase timeouts."""
        # Mock entity extraction
        with patch.object(orchestrator, '_extract_entities_parallel') as mock_entities:
            mock_entities.return_value = []
            
            # Mock synthesis input preparation
            with patch.object(orchestrator, '_prepare_synthesis_input') as mock_prepare:
                mock_prepare.return_value = {"query": query_context.query, "facts": []}
                
                # Setup timeouts in different phases
                mock_agents["retrieval"].process_task.side_effect = asyncio.TimeoutError()
                mock_agents["fact_check"].process_task.return_value = self.create_success_result()
                mock_agents["synthesis"].process_task.side_effect = asyncio.TimeoutError()
                mock_agents["citation"].process_task.return_value = self.create_success_result()
                
                # Execute pipeline
                results = await orchestrator.execute_pipeline(query_context, {}, 1000)
                
                # Verify timeout handling
                assert len(results) == 4
                assert not results[AgentType.RETRIEVAL].success
                assert "timed out" in results[AgentType.RETRIEVAL].error.lower()
                assert not results[AgentType.SYNTHESIS].success
                assert "timed out" in results[AgentType.SYNTHESIS].error.lower()

    @pytest.mark.asyncio
    async def test_pipeline_logging(self, orchestrator, query_context, mock_agents):
        """Test that pipeline execution includes proper logging."""
        # Mock entity extraction
        with patch.object(orchestrator, '_extract_entities_parallel') as mock_entities:
            mock_entities.return_value = []
            
            # Mock synthesis input preparation
            with patch.object(orchestrator, '_prepare_synthesis_input') as mock_prepare:
                mock_prepare.return_value = {"query": query_context.query, "facts": []}
                
                # Setup successful agents
                for agent in mock_agents.values():
                    agent.process_task.return_value = self.create_success_result()
                
                # Mock logger to verify logging calls
                with patch('agents.lead_orchestrator.logger') as mock_logger:
                    # Execute pipeline
                    await orchestrator.execute_pipeline(query_context, {}, 1000)
                    
                    # Verify logging calls were made
                    assert mock_logger.info.call_count >= 4  # At least one per phase
                    
                    # Check for completion log
                    completion_calls = [call for call in mock_logger.info.call_args_list 
                                     if "Pipeline completed" in str(call)]
                    assert len(completion_calls) > 0 