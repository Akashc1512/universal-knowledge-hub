"""
Universal Knowledge Platform
===========================

AI-driven knowledge hub with multi-agent systems, RAG pipeline, and enterprise-grade architecture.

This package provides:
- FastAPI-based REST API
- Multi-agent orchestration system
- Retrieval-Augmented Generation (RAG) pipeline
- Vector database integration
- Knowledge graph capabilities
- Enterprise-grade security and monitoring
"""

__version__ = "1.0.0"
__author__ = "Universal Knowledge Platform Team"
__description__ = "AI-driven knowledge hub with multi-agent systems"

# Import main components
try:
    from api.main import app
    from agents.lead_orchestrator import LeadOrchestrator
    from agents.base_agent import BaseAgent, AgentType, AgentMessage, AgentResult, QueryContext

    # Export main components
    __all__ = [
        'app',
        'LeadOrchestrator',
        'BaseAgent',
        'AgentType',
        'AgentMessage',
        'AgentResult',
        'QueryContext',
        '__version__',
        '__author__',
        '__description__'
    ]
except ImportError:
    # If imports fail, still provide basic package info
    __all__ = [
        '__version__',
        '__author__',
        '__description__'
    ] 