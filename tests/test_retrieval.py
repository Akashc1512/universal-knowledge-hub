from agents.lead_orchestrator import RetrievalAgent

def test_retrieval_init():
    agent = RetrievalAgent()
    assert agent is not None
