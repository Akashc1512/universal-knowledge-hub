import pytest
from agents.retrieval_agent import RetrievalAgent, SemanticCache, SearchResult
from core.types import Document

def test_retrieval_init():
    agent = RetrievalAgent()
    assert agent is not None

@pytest.mark.asyncio
async def test_semantic_cache_exact_match():
    cache = SemanticCache()
    query = "What is the capital of France?"
    result = SearchResult(
        documents=[Document(content="Paris is the capital", metadata={})],
        search_type="hybrid",
        query_time_ms=100,
        total_hits=1
    )
    await cache.set(query, result)
    cached = await cache.get(query)
    assert cached is not None
    assert cached.documents[0].content == "Paris is the capital"

@pytest.mark.asyncio
async def test_semantic_cache_similarity():
    cache = SemanticCache()
    query1 = "What is the capital of France?"
    query2 = "What is the capital of France and Germany?"
    # Debug: check word overlap
    words1 = set(query1.lower().split())
    words2 = set(query2.lower().split())
    overlap = words1 & words2
    print(f"Query1 words: {words1}")
    print(f"Query2 words: {words2}")
    print(f"Overlap: {overlap}, count: {len(overlap)}")
    result = SearchResult(
        documents=[Document(content="Paris is the capital", metadata={})],
        search_type="hybrid",
        query_time_ms=100,
        total_hits=1
    )
    await cache.set(query1, result)
    cached = await cache.get(query2)
    assert cached is not None
    assert cached.documents[0].content == "Paris is the capital"

@pytest.mark.asyncio
async def test_semantic_cache_miss():
    cache = SemanticCache()
    query = "What is the capital of France?"
    cached = await cache.get(query)
    assert cached is None

@pytest.mark.asyncio
async def test_cache_ttl_expiration():
    cache = SemanticCache()  # Use default TTL
    cache.ttl_seconds = 0  # Override for immediate expiration
    query = "What is the capital of France?"
    result = SearchResult(
        documents=[Document(content="Paris", metadata={})],
        search_type="hybrid",
        query_time_ms=100,
        total_hits=1
    )
    await cache.set(query, result)
    cached = await cache.get(query)
    assert cached is None  # Should be expired

@pytest.mark.asyncio
async def test_retrieval_agent_with_cache():
    agent = RetrievalAgent()
    query = "What is the capital of France?"
    context = type('Context', (), {'query': query})()
    result = await agent.hybrid_retrieve(query)
    assert result is not None
    assert hasattr(result, 'documents')
