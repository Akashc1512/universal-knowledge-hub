"""
SOLID Principles and Design Patterns Implementation.

This module provides enterprise-grade design patterns and SOLID principle
implementations for the Universal Knowledge Platform.

Patterns Implemented:
- Factory Pattern: Agent creation and configuration
- Strategy Pattern: Search algorithms and recommendation strategies
- Observer Pattern: Event-driven architecture and notifications
- Repository Pattern: Data access abstraction
- Decorator Pattern: Middleware and cross-cutting concerns
- Singleton Pattern: Resource management
- Builder Pattern: Complex object construction
- Adapter Pattern: External service integration
- Command Pattern: Request handling and queuing
- Chain of Responsibility: Request processing pipeline

SOLID Principles:
- Single Responsibility: Each class has one reason to change
- Open/Closed: Open for extension, closed for modification
- Liskov Substitution: Derived classes substitute base classes
- Interface Segregation: Specific interfaces over general ones
- Dependency Inversion: Depend on abstractions, not concretions
"""

from typing import Protocol, runtime_checkable

__all__ = [
    # Protocols and Interfaces
    'SearchStrategy',
    'Repository',
    'Observer',
    'Subject',
    'Factory',
    'Builder',
    'Command',
    'Handler',
    
    # Implementations
    'AgentFactory',
    'SearchStrategyFactory',
    'EventBus',
    'RepositoryRegistry',
    'MiddlewareDecorator',
] 