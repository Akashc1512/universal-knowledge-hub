from agents.retrieval_agent import RetrievalAgent

def test_retrieval_init():
    agent = RetrievalAgent()
    assert agent is not None
