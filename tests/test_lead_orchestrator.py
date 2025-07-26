from agents.lead_orchestrator import LeadOrchestrator

def test_initialization():
    orchestrator = LeadOrchestrator(config={})
    assert orchestrator is not None
