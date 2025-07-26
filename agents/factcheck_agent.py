from agents.lead_orchestrator import BaseAgent, AgentMessage, AgentResult, QueryContext, AgentType
import asyncio
import time
from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)

class FactCheckAgent(BaseAgent):
    """
    Agent responsible for verifying claims and fact-checking retrieved information.
    Implements claim decomposition, cross-source verification, explainability, and human-in-the-loop hooks.
    """
    def __init__(self, agent_id: str = "fact_check_001"):
        super().__init__(agent_id, AgentType.FACT_CHECK)
        self.knowledge_base = None  # TODO: Initialize knowledge base connection
        self.fact_check_model = None  # TODO: Initialize fact-checking model
        self.manual_review_callback = None  # Optional: Set to a function for human-in-the-loop

    async def process_task(self, task: Dict[str, Any], context: QueryContext) -> AgentResult:
        """Process fact-checking tasks with decomposition and cross-source verification."""
        start_time = time.time()
        try:
            claims = task.get('claims', [])
            sources = task.get('sources', [])
            logger.info(f"FactCheckAgent: Decomposing and verifying {len(claims)} claims.")
            atomic_claims = self.decompose_claims(claims)
            verification_results = await self.verify_claims(atomic_claims, sources)
            # Human-in-the-loop: flag low-confidence claims
            flagged = [r for r in verification_results if r['confidence'] < 0.5]
            if flagged and self.manual_review_callback:
                logger.info(f"Flagging {len(flagged)} claims for manual review.")
                self.manual_review_callback(flagged)
            return AgentResult(
                success=True,
                data={'verifications': verification_results},
                confidence=self._calculate_verification_confidence(verification_results),
                token_usage={'prompt': 100, 'completion': 50},  # TODO: Track actual usage
                execution_time_ms=int((time.time() - start_time) * 1000)
            )
        except Exception as e:
            logger.error(f"Fact-check error: {str(e)}")
            return AgentResult(
                success=False,
                data=None,
                error=str(e),
                execution_time_ms=int((time.time() - start_time) * 1000)
            )

    def decompose_claims(self, claims: List[str]) -> List[str]:
        """
        Decompose complex statements into atomic claims for independent verification.
        Simulate LLM/NLP-based decomposition: split by periods and conjunctions (and, or, but).
        """
        atomic_claims = []
        for claim in claims:
            # Split by period
            sentences = [c.strip() for c in claim.split('.') if c.strip()]
            for sentence in sentences:
                # Further split by conjunctions (simulate LLM/NLP)
                for conj in [' and ', ' or ', ' but ']:
                    if conj in sentence:
                        parts = [p.strip() for p in sentence.split(conj) if p.strip()]
                        atomic_claims.extend(parts)
                        break
                else:
                    atomic_claims.append(sentence)
        logger.info(f"Decomposed claims: {atomic_claims}")
        return atomic_claims

    async def verify_claim(self, claim: str, sources: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Verify a single claim against multiple sources.
        Simulate cross-source validation: if some sources support and some contradict, mark as 'contested'.
        """
        await asyncio.sleep(0.05)  # Simulate processing
        evidence = []
        verdict = 'unverifiable'
        confidence = 0.0
        justification = []
        support_count = 0
        contradict_count = 0
        for source in sources:
            # Simulate: alternate support/contradict for demo
            import random
            support = random.choice([True, False])
            if support:
                evidence.append({
                    'source': source,
                    'supporting_text': f'Evidence supporting: {claim}',
                    'relevance_score': 0.9
                })
                support_count += 1
                justification.append(f"Supported by '{source.get('title', 'unknown')}'")
            else:
                contradict_count += 1
                justification.append(f"Contradicted by '{source.get('title', 'unknown')}'")
        if support_count and contradict_count:
            verdict = 'contested'
            confidence = 0.5
            justification.append("Conflicting evidence found across sources.")
        elif support_count:
            verdict = 'supported'
            confidence = min(1.0, 0.7 + 0.1 * support_count)
        elif contradict_count:
            verdict = 'contradicted'
            confidence = 0.3
        else:
            justification.append("No supporting or contradicting evidence found.")
        return {
            'claim': claim,
            'verdict': verdict,
            'confidence': confidence,
            'evidence': evidence,
            'reasoning': ' | '.join(justification)
        }

    async def verify_claims(self, claims: List[str], sources: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Verify multiple claims in parallel, consulting all sources for each claim.
        Flag 'contested' or 'unverifiable' claims for human review if callback is set.
        """
        verification_tasks = [self.verify_claim(claim, sources) for claim in claims]
        results = await asyncio.gather(*verification_tasks)
        # Human-in-the-loop: flag contested/unverifiable
        flagged = [r for r in results if r['verdict'] in ('contested', 'unverifiable')]
        if flagged and self.manual_review_callback:
            logger.info(f"Flagging {len(flagged)} claims for manual review (contested/unverifiable).")
            self.manual_review_callback(flagged)
        # TODO: Implement cross-claim consistency checking
        return results

    def _calculate_verification_confidence(self, results: List[Dict[str, Any]]) -> float:
        """Calculate overall confidence in verification results."""
        if not results:
            return 0.0
        total_confidence = sum(r.get('confidence', 0) for r in results)
        unverifiable_count = sum(1 for r in results if r.get('verdict') == 'unverifiable')
        penalty = unverifiable_count * 0.1
        return max(0, min(1, (total_confidence / len(results)) - penalty))
