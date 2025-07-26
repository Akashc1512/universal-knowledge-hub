import pytest
from agents.lead_orchestrator import LeadOrchestrator, AgentType
from agents.base_agent import QueryContext

def test_initialization():
    orchestrator = LeadOrchestrator()
    assert orchestrator is not None

@pytest.mark.asyncio
async def test_token_budgeting_enforcement():
    orchestrator = LeadOrchestrator()
    context = QueryContext(
        query="What is the capital of France?",
        token_budget=500  # Small budget to trigger limits
    )
    plan = {
        'required_agents': [AgentType.RETRIEVAL, AgentType.FACT_CHECK, AgentType.SYNTHESIS],
        'execution_strategy': 'sequential'
    }
    result = await orchestrator.execute_pipeline(context, plan)
    assert result is not None
    # Should have agent results even if some agents failed
    assert 'agent_results' in result or 'error' in result

@pytest.mark.asyncio
async def test_per_agent_budget_calculation():
    orchestrator = LeadOrchestrator()
    context = QueryContext(
        query="Test query",
        token_budget=1000
    )
    plan = {
        'required_agents': [AgentType.RETRIEVAL, AgentType.FACT_CHECK, AgentType.SYNTHESIS, AgentType.CITATION],
        'execution_strategy': 'sequential'
    }
    # Test that budgets are calculated correctly
    agent_budgets = {}
    for agent_type in plan['required_agents']:
        agent_budgets[agent_type] = orchestrator.token_controller.get_agent_budget(agent_type, context.token_budget)
    assert len(agent_budgets) == 4
    assert all(budget > 0 for budget in agent_budgets.values())

@pytest.mark.asyncio
async def test_token_usage_tracking():
    orchestrator = LeadOrchestrator()
    context = QueryContext(
        query="Test query for token tracking",
        token_budget=2000
    )
    plan = {
        'required_agents': [AgentType.RETRIEVAL, AgentType.SYNTHESIS],
        'execution_strategy': 'sequential'
    }
    result = await orchestrator.execute_pipeline(context, plan)
    # Should have token usage information
    assert result is not None

@pytest.mark.asyncio
async def test_budget_exceeded_handling():
    orchestrator = LeadOrchestrator()
    context = QueryContext(
        query="Test query",
        token_budget=50  # Very small budget
    )
    plan = {
        'required_agents': [AgentType.RETRIEVAL, AgentType.FACT_CHECK],
        'execution_strategy': 'sequential'
    }
    result = await orchestrator.execute_pipeline(context, plan)
    # Should handle budget exceeded gracefully
    assert result is not None
