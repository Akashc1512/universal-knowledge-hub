"""
Advanced synthesis agent that creates comprehensive responses.
"""

import asyncio
import logging
import time
import os
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from agents.base_agent import BaseAgent, AgentType, AgentMessage, AgentResult, QueryContext
from agents.data_models import SynthesisResult, VerifiedFactModel, convert_to_standard_factcheck

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class VerifiedFact:
    """Represents a verified fact with confidence and source."""

    claim: str
    confidence: float
    source: str
    evidence: List[str] = None
    metadata: Dict[str, Any] = None


# --- SynthesisAgent: OpenAI/Anthropic LLM Integration ---
# Required environment variables:
#   LLM_PROVIDER=openai|anthropic
#   OPENAI_API_KEY, OPENAI_LLM_MODEL
#   ANTHROPIC_API_KEY, ANTHROPIC_MODEL

from agents.llm_client import LLMClient


class SynthesisAgent(BaseAgent):
    """
    SynthesisAgent that combines verified facts into coherent answers.
    """

    def __init__(self):
        """Initialize the synthesis agent."""
        super().__init__(agent_id="synthesis_agent", agent_type=AgentType.SYNTHESIS)
        logger.info("✅ SynthesisAgent initialized successfully")

    async def process_task(self, task: Dict[str, Any], context: QueryContext) -> AgentResult:
        """
        Process synthesis task by combining verified facts into a coherent answer.

        Args:
            task: Task data containing verified facts and query
            context: Query context

        Returns:
            AgentResult with synthesized answer
        """
        start_time = time.time()

        try:
            # Extract task data
            verified_facts = task.get("verified_facts", [])
            query = task.get("query", "")
            synthesis_params = task.get("synthesis_params", {})

            logger.info(f"Synthesizing answer for query: {query[:50]}...")
            logger.info(f"Number of verified facts: {len(verified_facts)}")

            # Validate input
            if not verified_facts:
                return AgentResult(
                    success=False, error="No verified facts provided for synthesis", confidence=0.0
                )

            # Synthesize answer
            synthesis_result = await self._synthesize_answer(
                verified_facts, query, synthesis_params
            )

            # Calculate confidence based on fact quality
            confidence = self._calculate_synthesis_confidence(verified_facts)

            processing_time = time.time() - start_time

            # Create standardized synthesis result
            synthesis_data = SynthesisResult(
                answer=synthesis_result,
                synthesis_method="rule_based",
                fact_count=len(verified_facts),
                processing_time_ms=int(processing_time * 1000),
                metadata={"agent_id": self.agent_id},
            )

            return AgentResult(
                success=True,
                data=synthesis_data.dict(),
                confidence=confidence,
                execution_time_ms=int(processing_time * 1000),
            )

        except Exception as e:
            logger.error(f"Synthesis failed: {str(e)}")
            return AgentResult(success=False, error=f"Synthesis failed: {str(e)}", confidence=0.0)

    async def _synthesize_answer(
        self, verified_facts: List[Dict], query: str, params: Dict[str, Any]
    ) -> str:
        """
        Synthesize answer from verified facts using LLMClient (OpenAI or Anthropic).
        """
        if not verified_facts:
            return "I don't have enough verified information to provide a comprehensive answer."

        # Build prompt
        facts_text = "\n".join(
            f"- {f.get('claim', '')} (confidence: {f.get('confidence', 0):.2f})"
            for f in verified_facts
        )
        prompt = (
            f"You are an expert assistant. Given the following user question and a set of verified facts, synthesize a clear, concise, and accurate answer.\n"
            f"User question: {query}\n"
            f"Verified facts:\n{facts_text}\n"
            f"Answer:"
        )
        try:
            llm = LLMClient()
            return llm.synthesize(prompt, max_tokens=params.get("max_length", 500), temperature=0.2)
        except Exception as e:
            logger.error(f"LLM synthesis error: {e}")
            # Fallback to rule-based synthesis
            answer_parts = ["Based on verified information:"]
            for fact in verified_facts[:3]:
                answer_parts.append(f"• {fact.get('claim', '')}")
            return "\n".join(answer_parts)

    def _calculate_synthesis_confidence(self, verified_facts: List[Dict]) -> float:
        """
        Calculate confidence based on fact quality and quantity.

        Args:
            verified_facts: List of verified facts

        Returns:
            Confidence score between 0 and 1
        """
        if not verified_facts:
            return 0.0

        # Calculate average confidence
        avg_confidence = sum(f.get("confidence", 0) for f in verified_facts) / len(verified_facts)

        # Boost confidence based on number of facts
        fact_count_boost = min(0.2, len(verified_facts) * 0.05)

        # Boost confidence based on high-confidence facts
        high_conf_facts = [f for f in verified_facts if f.get("confidence", 0) >= 0.8]
        high_conf_boost = min(0.1, len(high_conf_facts) * 0.02)

        final_confidence = min(1.0, avg_confidence + fact_count_boost + high_conf_boost)

        return final_confidence


# Example usage
async def main():
    """Example usage of SynthesisAgent."""
    agent = SynthesisAgent()

    # Example verified facts
    verified_facts = [
        {
            "claim": "The Earth orbits around the Sun",
            "confidence": 0.95,
            "source": "astronomical_database",
        },
        {"claim": "The Sun is a star", "confidence": 0.98, "source": "scientific_literature"},
    ]

    task = {
        "verified_facts": verified_facts,
        "query": "What is the relationship between Earth and the Sun?",
        "synthesis_params": {"style": "concise", "max_length": 500},
    }

    context = QueryContext(query="What is the relationship between Earth and the Sun?")

    result = await agent.process_task(task, context)
    print(f"Success: {result.success}")
    print(f"Answer: {result.data.get('response', '')}")
    print(f"Confidence: {result.confidence}")


if __name__ == "__main__":
    asyncio.run(main())
