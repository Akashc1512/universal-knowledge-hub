"""
Factory Pattern implementation for agent creation.

This module implements the Factory Pattern following SOLID principles:
- Single Responsibility: Each factory creates one type of object
- Open/Closed: New agent types can be added without modifying existing code
- Dependency Inversion: Depend on agent interfaces, not concrete implementations
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Type, Protocol, runtime_checkable
from enum import Enum
import logging

from agents.base_agent import BaseAgent, AgentType
from agents.retrieval_agent import RetrievalAgent
from agents.factcheck_agent import FactCheckAgent
from agents.synthesis_agent import SynthesisAgent
from agents.citation_agent import CitationAgent
from agents.lead_orchestrator import LeadOrchestrator

from .interfaces import Factory, AsyncFactory

logger = logging.getLogger(__name__)


# ============================================================================
# ABSTRACT FACTORY PATTERN
# ============================================================================


class AgentFactory(ABC):
    """
    Abstract Factory for creating agents.
    Follows Open/Closed Principle - new factories can extend without modification.
    """
    
    @abstractmethod
    def create_agent(self, config: Dict[str, Any]) -> BaseAgent:
        """Create an agent with given configuration."""
        pass
    
    @abstractmethod
    def get_supported_types(self) -> list[AgentType]:
        """Get list of agent types this factory can create."""
        pass
    
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """
        Validate configuration before creating agent.
        Can be overridden by subclasses for specific validation.
        """
        required_fields = ['agent_id', 'agent_type']
        return all(field in config for field in required_fields)


# ============================================================================
# CONCRETE FACTORIES - Single Responsibility for each agent type
# ============================================================================


class RetrievalAgentFactory(AgentFactory):
    """Factory for creating Retrieval agents."""
    
    def create_agent(self, config: Dict[str, Any]) -> RetrievalAgent:
        """Create a Retrieval agent."""
        if not self.validate_config(config):
            raise ValueError("Invalid configuration for RetrievalAgent")
        
        # Extract retrieval-specific configuration
        vector_db_config = config.get('vector_db', {})
        elasticsearch_config = config.get('elasticsearch', {})
        cache_config = config.get('cache', {})
        
        agent = RetrievalAgent(
            agent_id=config['agent_id'],
            vector_db_config=vector_db_config,
            elasticsearch_config=elasticsearch_config,
            cache_config=cache_config
        )
        
        logger.info(f"Created RetrievalAgent: {agent.agent_id}")
        return agent
    
    def get_supported_types(self) -> list[AgentType]:
        """Get supported agent types."""
        return [AgentType.RETRIEVAL]
    
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate retrieval-specific configuration."""
        if not super().validate_config(config):
            return False
        
        # Validate at least one search backend is configured
        has_vector_db = bool(config.get('vector_db'))
        has_elasticsearch = bool(config.get('elasticsearch'))
        has_knowledge_graph = bool(config.get('knowledge_graph'))
        
        return has_vector_db or has_elasticsearch or has_knowledge_graph


class FactCheckAgentFactory(AgentFactory):
    """Factory for creating FactCheck agents."""
    
    def create_agent(self, config: Dict[str, Any]) -> FactCheckAgent:
        """Create a FactCheck agent."""
        if not self.validate_config(config):
            raise ValueError("Invalid configuration for FactCheckAgent")
        
        # Extract fact-checking specific configuration
        verification_threshold = config.get('verification_threshold', 0.7)
        fact_sources = config.get('fact_sources', [])
        
        agent = FactCheckAgent(
            agent_id=config['agent_id'],
            verification_threshold=verification_threshold,
            fact_sources=fact_sources
        )
        
        logger.info(f"Created FactCheckAgent: {agent.agent_id}")
        return agent
    
    def get_supported_types(self) -> list[AgentType]:
        """Get supported agent types."""
        return [AgentType.FACT_CHECK]


class SynthesisAgentFactory(AgentFactory):
    """Factory for creating Synthesis agents."""
    
    def create_agent(self, config: Dict[str, Any]) -> SynthesisAgent:
        """Create a Synthesis agent."""
        if not self.validate_config(config):
            raise ValueError("Invalid configuration for SynthesisAgent")
        
        # Extract synthesis-specific configuration
        synthesis_model = config.get('synthesis_model', 'gpt-4')
        max_tokens = config.get('max_tokens', 1000)
        temperature = config.get('temperature', 0.7)
        
        agent = SynthesisAgent(
            agent_id=config['agent_id'],
            model=synthesis_model,
            max_tokens=max_tokens,
            temperature=temperature
        )
        
        logger.info(f"Created SynthesisAgent: {agent.agent_id}")
        return agent
    
    def get_supported_types(self) -> list[AgentType]:
        """Get supported agent types."""
        return [AgentType.SYNTHESIS]


class CitationAgentFactory(AgentFactory):
    """Factory for creating Citation agents."""
    
    def create_agent(self, config: Dict[str, Any]) -> CitationAgent:
        """Create a Citation agent."""
        if not self.validate_config(config):
            raise ValueError("Invalid configuration for CitationAgent")
        
        # Extract citation-specific configuration
        citation_style = config.get('citation_style', 'APA')
        include_urls = config.get('include_urls', True)
        
        agent = CitationAgent(
            agent_id=config['agent_id'],
            citation_style=citation_style,
            include_urls=include_urls
        )
        
        logger.info(f"Created CitationAgent: {agent.agent_id}")
        return agent
    
    def get_supported_types(self) -> list[AgentType]:
        """Get supported agent types."""
        return [AgentType.CITATION]


class LeadOrchestratorFactory(AgentFactory):
    """Factory for creating Lead Orchestrator agents."""
    
    def create_agent(self, config: Dict[str, Any]) -> LeadOrchestrator:
        """Create a Lead Orchestrator agent."""
        if not self.validate_config(config):
            raise ValueError("Invalid configuration for LeadOrchestrator")
        
        # Extract orchestrator-specific configuration
        pipeline_config = config.get('pipeline', {})
        timeout_config = config.get('timeouts', {})
        retry_config = config.get('retries', {})
        
        agent = LeadOrchestrator(
            agent_id=config['agent_id'],
            pipeline_config=pipeline_config,
            timeout_config=timeout_config,
            retry_config=retry_config
        )
        
        logger.info(f"Created LeadOrchestrator: {agent.agent_id}")
        return agent
    
    def get_supported_types(self) -> list[AgentType]:
        """Get supported agent types."""
        return [AgentType.LEAD_ORCHESTRATOR]


# ============================================================================
# FACTORY REGISTRY - Dependency Injection Container
# ============================================================================


class AgentFactoryRegistry:
    """
    Registry for agent factories.
    Implements Dependency Inversion - high-level code depends on factory abstraction.
    """
    
    def __init__(self):
        self._factories: Dict[AgentType, AgentFactory] = {}
        self._register_default_factories()
    
    def _register_default_factories(self):
        """Register default factories for all agent types."""
        self.register(AgentType.RETRIEVAL, RetrievalAgentFactory())
        self.register(AgentType.FACT_CHECK, FactCheckAgentFactory())
        self.register(AgentType.SYNTHESIS, SynthesisAgentFactory())
        self.register(AgentType.CITATION, CitationAgentFactory())
        self.register(AgentType.LEAD_ORCHESTRATOR, LeadOrchestratorFactory())
    
    def register(self, agent_type: AgentType, factory: AgentFactory):
        """
        Register a factory for an agent type.
        Follows Open/Closed - new factories can be registered without modification.
        """
        self._factories[agent_type] = factory
        logger.info(f"Registered factory for {agent_type.value}")
    
    def create_agent(self, config: Dict[str, Any]) -> BaseAgent:
        """Create an agent using the appropriate factory."""
        agent_type = AgentType(config.get('agent_type'))
        
        if agent_type not in self._factories:
            raise ValueError(f"No factory registered for agent type: {agent_type}")
        
        factory = self._factories[agent_type]
        return factory.create_agent(config)
    
    def get_factory(self, agent_type: AgentType) -> Optional[AgentFactory]:
        """Get factory for specific agent type."""
        return self._factories.get(agent_type)
    
    def list_supported_types(self) -> list[AgentType]:
        """List all supported agent types."""
        return list(self._factories.keys())


# ============================================================================
# BUILDER PATTERN - Complex agent configuration
# ============================================================================


class AgentBuilder:
    """
    Builder for complex agent configurations.
    Single Responsibility: Build agent configurations step by step.
    """
    
    def __init__(self):
        self._config: Dict[str, Any] = {}
        self._reset()
    
    def _reset(self):
        """Reset builder to initial state."""
        self._config = {
            'agent_id': None,
            'agent_type': None,
            'metadata': {},
            'capabilities': [],
            'dependencies': []
        }
    
    def with_id(self, agent_id: str) -> 'AgentBuilder':
        """Set agent ID."""
        self._config['agent_id'] = agent_id
        return self
    
    def with_type(self, agent_type: AgentType) -> 'AgentBuilder':
        """Set agent type."""
        self._config['agent_type'] = agent_type.value
        return self
    
    def with_metadata(self, key: str, value: Any) -> 'AgentBuilder':
        """Add metadata."""
        self._config['metadata'][key] = value
        return self
    
    def with_capability(self, capability: str) -> 'AgentBuilder':
        """Add capability."""
        self._config['capabilities'].append(capability)
        return self
    
    def with_dependency(self, dependency: str) -> 'AgentBuilder':
        """Add dependency."""
        self._config['dependencies'].append(dependency)
        return self
    
    def with_vector_db(self, config: Dict[str, Any]) -> 'AgentBuilder':
        """Configure vector database."""
        self._config['vector_db'] = config
        return self
    
    def with_elasticsearch(self, config: Dict[str, Any]) -> 'AgentBuilder':
        """Configure Elasticsearch."""
        self._config['elasticsearch'] = config
        return self
    
    def with_cache(self, config: Dict[str, Any]) -> 'AgentBuilder':
        """Configure cache."""
        self._config['cache'] = config
        return self
    
    def build(self) -> Dict[str, Any]:
        """Build and return configuration."""
        if not self._config['agent_id'] or not self._config['agent_type']:
            raise ValueError("Agent ID and type are required")
        
        config = self._config.copy()
        self._reset()
        return config


# ============================================================================
# SINGLETON PATTERN - Global factory instance
# ============================================================================


class AgentFactorySingleton:
    """
    Singleton pattern for global agent factory access.
    Ensures single instance of factory registry across application.
    """
    
    _instance: Optional[AgentFactoryRegistry] = None
    
    @classmethod
    def get_instance(cls) -> AgentFactoryRegistry:
        """Get singleton instance of factory registry."""
        if cls._instance is None:
            cls._instance = AgentFactoryRegistry()
        return cls._instance
    
    @classmethod
    def reset(cls):
        """Reset singleton instance (mainly for testing)."""
        cls._instance = None


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================


def create_agent(config: Dict[str, Any]) -> BaseAgent:
    """
    Convenience function to create agent using global factory.
    Hides complexity of factory pattern from users.
    """
    registry = AgentFactorySingleton.get_instance()
    return registry.create_agent(config)


def create_agent_with_builder(
    agent_id: str,
    agent_type: AgentType,
    **kwargs
) -> BaseAgent:
    """
    Create agent using builder pattern for configuration.
    Provides fluent interface for agent creation.
    """
    builder = AgentBuilder()
    builder.with_id(agent_id).with_type(agent_type)
    
    # Add optional configurations
    for key, value in kwargs.items():
        if hasattr(builder, f'with_{key}'):
            getattr(builder, f'with_{key}')(value)
        else:
            builder.with_metadata(key, value)
    
    config = builder.build()
    return create_agent(config)


# Export public API
__all__ = [
    # Abstract factory
    'AgentFactory',
    
    # Concrete factories
    'RetrievalAgentFactory',
    'FactCheckAgentFactory',
    'SynthesisAgentFactory',
    'CitationAgentFactory',
    'LeadOrchestratorFactory',
    
    # Registry and builder
    'AgentFactoryRegistry',
    'AgentBuilder',
    'AgentFactorySingleton',
    
    # Convenience functions
    'create_agent',
    'create_agent_with_builder',
] 