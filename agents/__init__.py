"""
Agents Package
==============

Multi-agent system for knowledge processing and orchestration.
"""

from .base_agent import BaseAgent, AgentType, AgentMessage, AgentResult, QueryContext
from .lead_orchestrator import LeadOrchestrator
from .retrieval_agent import RetrievalAgent
from .factcheck_agent import FactCheckAgent
from .synthesis_agent import SynthesisAgent
from .citation_agent import CitationAgent

__all__ = [
    "BaseAgent",
    "AgentType",
    "AgentMessage",
    "AgentResult",
    "QueryContext",
    "LeadOrchestrator",
    "RetrievalAgent",
    "FactCheckAgent",
    "SynthesisAgent",
    "CitationAgent",
]
