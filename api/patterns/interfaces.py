"""
Interface definitions following SOLID principles.

This module defines protocols and abstract base classes that follow
the Interface Segregation Principle (ISP) and Dependency Inversion
Principle (DIP).
"""

from abc import ABC, abstractmethod
from typing import (
    Any, Dict, List, Optional, Protocol, TypeVar, Generic,
    runtime_checkable, Union, Callable, Awaitable
)
from datetime import datetime
from enum import Enum

# Type variables for generic interfaces
T = TypeVar('T')
K = TypeVar('K')
V = TypeVar('V')
TResult = TypeVar('TResult')
TQuery = TypeVar('TQuery')
TEntity = TypeVar('TEntity')


# ============================================================================
# SINGLE RESPONSIBILITY: Each interface has one clear purpose
# ============================================================================


@runtime_checkable
class Identifiable(Protocol):
    """Entity that can be uniquely identified."""
    
    @property
    def id(self) -> str:
        """Get unique identifier."""
        ...


@runtime_checkable
class Timestamped(Protocol):
    """Entity with timestamp information."""
    
    @property
    def created_at(self) -> datetime:
        """Get creation timestamp."""
        ...
    
    @property
    def updated_at(self) -> datetime:
        """Get last update timestamp."""
        ...


@runtime_checkable
class Versioned(Protocol):
    """Entity with version control."""
    
    @property
    def version(self) -> int:
        """Get current version number."""
        ...
    
    def increment_version(self) -> None:
        """Increment version number."""
        ...


# ============================================================================
# REPOSITORY PATTERN: Data access abstraction
# ============================================================================


@runtime_checkable
class Repository(Protocol[TEntity]):
    """
    Generic repository interface for data access.
    Follows Interface Segregation and Dependency Inversion principles.
    """
    
    async def find_by_id(self, entity_id: str) -> Optional[TEntity]:
        """Find entity by ID."""
        ...
    
    async def find_all(self, limit: int = 100, offset: int = 0) -> List[TEntity]:
        """Find all entities with pagination."""
        ...
    
    async def save(self, entity: TEntity) -> TEntity:
        """Save or update entity."""
        ...
    
    async def delete(self, entity_id: str) -> bool:
        """Delete entity by ID."""
        ...


@runtime_checkable
class QueryRepository(Repository[TEntity], Protocol[TEntity, TQuery]):
    """Extended repository with query capabilities."""
    
    async def find_by_query(self, query: TQuery) -> List[TEntity]:
        """Find entities matching query."""
        ...
    
    async def count_by_query(self, query: TQuery) -> int:
        """Count entities matching query."""
        ...


@runtime_checkable
class CacheableRepository(Repository[TEntity], Protocol[TEntity]):
    """Repository with caching capabilities."""
    
    async def invalidate_cache(self, entity_id: str) -> None:
        """Invalidate cache for specific entity."""
        ...
    
    async def warm_cache(self, entity_ids: List[str]) -> None:
        """Pre-load entities into cache."""
        ...


# ============================================================================
# STRATEGY PATTERN: Algorithm abstraction
# ============================================================================


@runtime_checkable
class SearchStrategy(Protocol[TQuery, TResult]):
    """
    Search strategy interface for different search algorithms.
    Follows Open/Closed Principle - new strategies can be added without modification.
    """
    
    @property
    def name(self) -> str:
        """Get strategy name."""
        ...
    
    async def search(self, query: TQuery, **kwargs) -> TResult:
        """Execute search with this strategy."""
        ...
    
    def supports_query(self, query: TQuery) -> bool:
        """Check if strategy supports given query type."""
        ...


@runtime_checkable
class RankingStrategy(Protocol[TResult]):
    """Strategy for ranking search results."""
    
    async def rank(self, results: List[TResult], context: Dict[str, Any]) -> List[TResult]:
        """Rank results based on strategy."""
        ...


@runtime_checkable
class FilterStrategy(Protocol[TResult]):
    """Strategy for filtering results."""
    
    async def filter(self, results: List[TResult], criteria: Dict[str, Any]) -> List[TResult]:
        """Filter results based on criteria."""
        ...


# ============================================================================
# OBSERVER PATTERN: Event-driven architecture
# ============================================================================


class EventType(Enum):
    """Standard event types in the system."""
    CREATED = "created"
    UPDATED = "updated"
    DELETED = "deleted"
    SEARCHED = "searched"
    VIEWED = "viewed"
    ERROR = "error"
    PERFORMANCE = "performance"


@runtime_checkable
class Event(Protocol):
    """Event interface for observer pattern."""
    
    @property
    def event_type(self) -> EventType:
        """Get event type."""
        ...
    
    @property
    def timestamp(self) -> datetime:
        """Get event timestamp."""
        ...
    
    @property
    def data(self) -> Dict[str, Any]:
        """Get event data."""
        ...
    
    @property
    def source(self) -> str:
        """Get event source identifier."""
        ...


@runtime_checkable
class Observer(Protocol):
    """
    Observer interface for event handling.
    Single Responsibility: Handle specific events.
    """
    
    async def update(self, event: Event) -> None:
        """Handle event notification."""
        ...
    
    def can_handle(self, event: Event) -> bool:
        """Check if observer can handle this event type."""
        ...


@runtime_checkable
class Subject(Protocol):
    """Subject interface for observer pattern."""
    
    def attach(self, observer: Observer) -> None:
        """Attach observer to subject."""
        ...
    
    def detach(self, observer: Observer) -> None:
        """Detach observer from subject."""
        ...
    
    async def notify(self, event: Event) -> None:
        """Notify all observers of event."""
        ...


# ============================================================================
# FACTORY PATTERN: Object creation abstraction
# ============================================================================


@runtime_checkable
class Factory(Protocol[T]):
    """
    Generic factory interface.
    Follows Dependency Inversion - depend on factory abstraction.
    """
    
    def create(self, config: Dict[str, Any]) -> T:
        """Create instance with configuration."""
        ...
    
    def can_create(self, type_name: str) -> bool:
        """Check if factory can create given type."""
        ...


@runtime_checkable
class AsyncFactory(Protocol[T]):
    """Asynchronous factory interface."""
    
    async def create(self, config: Dict[str, Any]) -> T:
        """Create instance asynchronously."""
        ...


# ============================================================================
# BUILDER PATTERN: Complex object construction
# ============================================================================


@runtime_checkable
class Builder(Protocol[T]):
    """
    Builder interface for complex object construction.
    Single Responsibility: Build one type of object.
    """
    
    def reset(self) -> None:
        """Reset builder to initial state."""
        ...
    
    def build(self) -> T:
        """Build and return the constructed object."""
        ...


# ============================================================================
# COMMAND PATTERN: Request encapsulation
# ============================================================================


@runtime_checkable
class Command(Protocol):
    """
    Command interface for request encapsulation.
    Single Responsibility: Encapsulate one action.
    """
    
    async def execute(self) -> Any:
        """Execute the command."""
        ...
    
    async def undo(self) -> None:
        """Undo the command if possible."""
        ...
    
    def can_execute(self) -> bool:
        """Check if command can be executed."""
        ...


# ============================================================================
# HANDLER PATTERN: Chain of Responsibility
# ============================================================================


@runtime_checkable
class Handler(Protocol[T]):
    """
    Handler interface for chain of responsibility.
    Open/Closed: New handlers can be added without modifying existing ones.
    """
    
    async def handle(self, request: T) -> Optional[Any]:
        """Handle request or pass to next handler."""
        ...
    
    def set_next(self, handler: 'Handler[T]') -> 'Handler[T]':
        """Set next handler in chain."""
        ...


# ============================================================================
# ADAPTER PATTERN: External service integration
# ============================================================================


@runtime_checkable
class ExternalServiceAdapter(Protocol):
    """
    Adapter interface for external services.
    Dependency Inversion: Depend on adapter interface, not concrete services.
    """
    
    async def connect(self) -> bool:
        """Connect to external service."""
        ...
    
    async def disconnect(self) -> None:
        """Disconnect from external service."""
        ...
    
    async def health_check(self) -> bool:
        """Check service health."""
        ...


# ============================================================================
# MIDDLEWARE PATTERN: Cross-cutting concerns
# ============================================================================


@runtime_checkable
class Middleware(Protocol):
    """
    Middleware interface for cross-cutting concerns.
    Single Responsibility: Handle one aspect (auth, logging, etc).
    """
    
    async def process(
        self,
        request: Any,
        call_next: Callable[[Any], Awaitable[Any]]
    ) -> Any:
        """Process request and call next middleware."""
        ...


# ============================================================================
# CACHE PATTERN: Caching abstraction
# ============================================================================


@runtime_checkable
class Cache(Protocol[K, V]):
    """
    Cache interface for various caching strategies.
    Interface Segregation: Separate interfaces for different cache operations.
    """
    
    async def get(self, key: K) -> Optional[V]:
        """Get value from cache."""
        ...
    
    async def set(self, key: K, value: V, ttl: Optional[int] = None) -> None:
        """Set value in cache with optional TTL."""
        ...
    
    async def delete(self, key: K) -> bool:
        """Delete value from cache."""
        ...
    
    async def exists(self, key: K) -> bool:
        """Check if key exists in cache."""
        ...


@runtime_checkable
class DistributedCache(Cache[K, V], Protocol[K, V]):
    """Extended cache interface for distributed caching."""
    
    async def get_many(self, keys: List[K]) -> Dict[K, V]:
        """Get multiple values from cache."""
        ...
    
    async def set_many(self, items: Dict[K, V], ttl: Optional[int] = None) -> None:
        """Set multiple values in cache."""
        ...
    
    async def clear(self) -> None:
        """Clear entire cache."""
        ...


# ============================================================================
# METRICS PATTERN: Observability abstraction
# ============================================================================


@runtime_checkable
class MetricsCollector(Protocol):
    """
    Metrics collector interface.
    Single Responsibility: Collect specific metrics.
    """
    
    def increment(self, metric: str, value: float = 1.0, tags: Optional[Dict[str, str]] = None) -> None:
        """Increment counter metric."""
        ...
    
    def gauge(self, metric: str, value: float, tags: Optional[Dict[str, str]] = None) -> None:
        """Set gauge metric."""
        ...
    
    def histogram(self, metric: str, value: float, tags: Optional[Dict[str, str]] = None) -> None:
        """Record histogram metric."""
        ...
    
    def timing(self, metric: str, duration: float, tags: Optional[Dict[str, str]] = None) -> None:
        """Record timing metric."""
        ...


# Export all interfaces
__all__ = [
    # Basic interfaces
    'Identifiable',
    'Timestamped',
    'Versioned',
    
    # Repository pattern
    'Repository',
    'QueryRepository',
    'CacheableRepository',
    
    # Strategy pattern
    'SearchStrategy',
    'RankingStrategy',
    'FilterStrategy',
    
    # Observer pattern
    'Event',
    'EventType',
    'Observer',
    'Subject',
    
    # Factory pattern
    'Factory',
    'AsyncFactory',
    
    # Builder pattern
    'Builder',
    
    # Command pattern
    'Command',
    
    # Handler pattern
    'Handler',
    
    # Adapter pattern
    'ExternalServiceAdapter',
    
    # Middleware pattern
    'Middleware',
    
    # Cache pattern
    'Cache',
    'DistributedCache',
    
    # Metrics pattern
    'MetricsCollector',
] 