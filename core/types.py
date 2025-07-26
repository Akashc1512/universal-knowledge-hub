"""
Common data types and enums for the Universal Knowledge Platform.
"""

from agents.base_agent import (
    AgentType,
    MessageType,
    TaskPriority,
    AgentMessage,
    QueryContext,
    AgentResult,
    BaseAgent
)

from dataclasses import dataclass, field
from typing import Dict, Any, Optional

@dataclass
class Document:
    content: str
    metadata: Optional[Dict[str, Any]] = field(default_factory=dict)

__all__ = [
    'AgentType',
    'MessageType', 
    'TaskPriority',
    'AgentMessage',
    'QueryContext',
    'AgentResult',
    'BaseAgent'
] 