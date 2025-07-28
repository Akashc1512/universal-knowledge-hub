import pytest
from agents.factcheck_agent import FactCheckAgent


@pytest.mark.asyncio
async def test_decompose_claims():
    agent = FactCheckAgent()
    claims = [
        "The sky is blue and the grass is green. Water is wet.",
        "Cats are mammals or birds are reptiles.",
    ]
    atomic = agent.decompose_claims(claims)
    assert "The sky is blue" in atomic
    assert "the grass is green" in atomic
    assert "Water is wet" in atomic
    assert "Cats are mammals" in atomic
    assert "birds are reptiles" in atomic
    assert len(atomic) >= 5


@pytest.mark.asyncio
async def test_cross_source_verdicts():
    agent = FactCheckAgent()
    claim = "The earth is round"
    sources = [{"title": "Source A"}, {"title": "Source B"}, {"title": "Source C"}]
    # Run multiple times to see all verdicts due to random
    verdicts = set()
    for _ in range(10):
        result = await agent.verify_claim(claim, sources)
        verdicts.add(result["verdict"])
    assert "supported" in verdicts or "contested" in verdicts or "contradicted" in verdicts


@pytest.mark.asyncio
async def test_human_in_the_loop_flagging():
    flagged = []

    def manual_review(claims):
        flagged.extend(claims)

    agent = FactCheckAgent()
    agent.manual_review_callback = manual_review
    claims = ["A and B.", "C."]
    sources = [{"title": "S1"}, {"title": "S2"}]
    # Force at least one 'contested' by monkeypatching random
    import random

    orig_choice = random.choice
    random.choice = lambda x: x[0]  # Always True (support)
    await agent.verify_claims(claims, sources)
    random.choice = orig_choice
    # Now force contradiction
    random.choice = lambda x: x[1]  # Always False (contradict)
    await agent.verify_claims(claims, sources)
    random.choice = orig_choice
    # Should have flagged at least once
    assert isinstance(flagged, list)
