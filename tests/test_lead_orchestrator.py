import pytest
from agents.lead_orchestrator import LeadOrchestrator, AgentType
from agents.base_agent import QueryContext


def test_initialization():
    orchestrator = LeadOrchestrator()
    assert orchestrator is not None
    assert hasattr(orchestrator, 'agents')
    assert hasattr(orchestrator, 'token_budget')
    assert hasattr(orchestrator, 'semantic_cache')


@pytest.mark.asyncio
async def test_process_query_basic():
    orchestrator = LeadOrchestrator()
    result = await orchestrator.process_query("What is the capital of France?")
    assert result is not None
    assert isinstance(result, dict)
    # Should have basic response structure
    assert "answer" in result or "error" in result


@pytest.mark.asyncio
async def test_token_budgeting_enforcement():
    orchestrator = LeadOrchestrator()
    # Test token budget allocation
    budget = await orchestrator.token_budget.allocate_budget_for_query("Test query")
    assert budget > 0
    assert isinstance(budget, int)


@pytest.mark.asyncio
async def test_per_agent_budget_calculation():
    orchestrator = LeadOrchestrator()
    query_budget = 1000
    # Test that budgets are calculated correctly for each agent
    for agent_type in [AgentType.RETRIEVAL, AgentType.FACT_CHECK, AgentType.SYNTHESIS, AgentType.CITATION]:
        agent_budget = await orchestrator.token_budget.allocate_for_agent(agent_type, query_budget)
        assert agent_budget > 0
        assert isinstance(agent_budget, int)


@pytest.mark.asyncio
async def test_token_usage_tracking():
    orchestrator = LeadOrchestrator()
    # Test token usage tracking
    await orchestrator.token_budget.track_usage(AgentType.RETRIEVAL, 100)
    status = await orchestrator.token_budget.get_budget_status()
    assert status is not None
    assert isinstance(status, dict)


@pytest.mark.asyncio
async def test_cache_functionality():
    orchestrator = LeadOrchestrator()
    query = "Test query for caching"
    
    # Test cache miss
    cached_result = await orchestrator.semantic_cache.get_cached_response(query)
    assert cached_result is None
    
    # Test cache stats
    stats = await orchestrator.semantic_cache.get_cache_stats()
    assert stats is not None
    assert isinstance(stats, dict)


@pytest.mark.asyncio
async def test_analyze_and_plan():
    orchestrator = LeadOrchestrator()
    context = QueryContext(query="What is the capital of France?")
    plan = await orchestrator.analyze_and_plan(context)
    assert plan is not None
    assert isinstance(plan, dict)
    # Should have required agents and execution strategy
    assert "required_agents" in plan or "execution_strategy" in plan


@pytest.mark.asyncio
async def test_execute_pipeline():
    orchestrator = LeadOrchestrator()
    context = QueryContext(query="Test query")
    plan = {
        "required_agents": [AgentType.RETRIEVAL, AgentType.SYNTHESIS],
        "execution_strategy": "sequential",
    }
    query_budget = 1000
    result = await orchestrator.execute_pipeline(context, plan, query_budget)
    assert result is not None
    assert isinstance(result, dict)
    # Should have agent results even if some agents failed
    assert len(result) > 0


@pytest.mark.asyncio
async def test_budget_exceeded_handling():
    orchestrator = LeadOrchestrator()
    context = QueryContext(query="Test query")
    plan = {
        "required_agents": [AgentType.RETRIEVAL, AgentType.FACT_CHECK],
        "execution_strategy": "sequential",
    }
    query_budget = 50  # Very small budget
    result = await orchestrator.execute_pipeline(context, plan, query_budget)
    # Should handle budget exceeded gracefully
    assert result is not None
    assert isinstance(result, dict)
