"""
Repository Pattern implementation for data access.

This module implements the Repository Pattern following SOLID principles:
- Single Responsibility: Each repository handles one entity type
- Open/Closed: New repositories can be added without modifying existing code
- Liskov Substitution: All repositories can be used interchangeably
- Interface Segregation: Specific interfaces for different repository capabilities
- Dependency Inversion: Depend on repository interfaces, not implementations
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, TypeVar, Generic, Type, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import asyncio
import logging
import json
from pathlib import Path

from .interfaces import (
    Repository, QueryRepository, CacheableRepository,
    Identifiable, Timestamped, Versioned
)

logger = logging.getLogger(__name__)

# Type variables
TEntity = TypeVar('TEntity', bound=Identifiable)
TQuery = TypeVar('TQuery')


# ============================================================================
# BASE REPOSITORY - Template for common operations
# ============================================================================


class BaseRepository(Repository[TEntity], Generic[TEntity]):
    """
    Base repository with common functionality.
    Template Method Pattern for data access operations.
    """
    
    def __init__(self, entity_type: Type[TEntity]):
        self.entity_type = entity_type
        self.metrics = {
            'finds': 0,
            'saves': 0,
            'deletes': 0,
            'errors': 0
        }
    
    async def find_by_id(self, entity_id: str) -> Optional[TEntity]:
        """Find entity by ID with metrics tracking."""
        try:
            self.metrics['finds'] += 1
            entity = await self._do_find_by_id(entity_id)
            
            if entity:
                await self._after_find(entity)
            
            return entity
            
        except Exception as e:
            self.metrics['errors'] += 1
            logger.error(f"Error finding entity {entity_id}: {e}")
            raise
    
    async def find_all(self, limit: int = 100, offset: int = 0) -> List[TEntity]:
        """Find all entities with pagination."""
        try:
            self.metrics['finds'] += 1
            entities = await self._do_find_all(limit, offset)
            
            for entity in entities:
                await self._after_find(entity)
            
            return entities
            
        except Exception as e:
            self.metrics['errors'] += 1
            logger.error(f"Error finding entities: {e}")
            raise
    
    async def save(self, entity: TEntity) -> TEntity:
        """Save or update entity."""
        try:
            self.metrics['saves'] += 1
            
            # Pre-save processing
            await self._before_save(entity)
            
            # Perform save
            saved_entity = await self._do_save(entity)
            
            # Post-save processing
            await self._after_save(saved_entity)
            
            return saved_entity
            
        except Exception as e:
            self.metrics['errors'] += 1
            logger.error(f"Error saving entity: {e}")
            raise
    
    async def delete(self, entity_id: str) -> bool:
        """Delete entity by ID."""
        try:
            self.metrics['deletes'] += 1
            
            # Pre-delete processing
            await self._before_delete(entity_id)
            
            # Perform delete
            result = await self._do_delete(entity_id)
            
            # Post-delete processing
            if result:
                await self._after_delete(entity_id)
            
            return result
            
        except Exception as e:
            self.metrics['errors'] += 1
            logger.error(f"Error deleting entity {entity_id}: {e}")
            raise
    
    # Abstract methods to be implemented by subclasses
    @abstractmethod
    async def _do_find_by_id(self, entity_id: str) -> Optional[TEntity]:
        """Actual find implementation."""
        pass
    
    @abstractmethod
    async def _do_find_all(self, limit: int, offset: int) -> List[TEntity]:
        """Actual find all implementation."""
        pass
    
    @abstractmethod
    async def _do_save(self, entity: TEntity) -> TEntity:
        """Actual save implementation."""
        pass
    
    @abstractmethod
    async def _do_delete(self, entity_id: str) -> bool:
        """Actual delete implementation."""
        pass
    
    # Hook methods for customization
    async def _before_save(self, entity: TEntity) -> None:
        """Hook called before saving."""
        pass
    
    async def _after_save(self, entity: TEntity) -> None:
        """Hook called after saving."""
        pass
    
    async def _after_find(self, entity: TEntity) -> None:
        """Hook called after finding."""
        pass
    
    async def _before_delete(self, entity_id: str) -> None:
        """Hook called before deleting."""
        pass
    
    async def _after_delete(self, entity_id: str) -> None:
        """Hook called after deleting."""
        pass
    
    def get_metrics(self) -> Dict[str, int]:
        """Get repository metrics."""
        return self.metrics.copy()


# ============================================================================
# IN-MEMORY REPOSITORY - For testing and caching
# ============================================================================


class InMemoryRepository(BaseRepository[TEntity], Generic[TEntity]):
    """
    In-memory repository implementation.
    Single Responsibility: Manage entities in memory.
    """
    
    def __init__(self, entity_type: Type[TEntity]):
        super().__init__(entity_type)
        self._storage: Dict[str, TEntity] = {}
        self._lock = asyncio.Lock()
    
    async def _do_find_by_id(self, entity_id: str) -> Optional[TEntity]:
        """Find entity in memory."""
        async with self._lock:
            return self._storage.get(entity_id)
    
    async def _do_find_all(self, limit: int, offset: int) -> List[TEntity]:
        """Find all entities in memory."""
        async with self._lock:
            entities = list(self._storage.values())
            return entities[offset:offset + limit]
    
    async def _do_save(self, entity: TEntity) -> TEntity:
        """Save entity to memory."""
        async with self._lock:
            self._storage[entity.id] = entity
            return entity
    
    async def _do_delete(self, entity_id: str) -> bool:
        """Delete entity from memory."""
        async with self._lock:
            if entity_id in self._storage:
                del self._storage[entity_id]
                return True
            return False
    
    async def clear(self) -> None:
        """Clear all entities from memory."""
        async with self._lock:
            self._storage.clear()


# ============================================================================
# FILE-BASED REPOSITORY - JSON file persistence
# ============================================================================


class JsonFileRepository(BaseRepository[TEntity], Generic[TEntity]):
    """
    JSON file-based repository implementation.
    Single Responsibility: Persist entities to JSON files.
    """
    
    def __init__(self, entity_type: Type[TEntity], file_path: Path):
        super().__init__(entity_type)
        self.file_path = file_path
        self._lock = asyncio.Lock()
        self._ensure_file_exists()
    
    def _ensure_file_exists(self):
        """Ensure storage file exists."""
        if not self.file_path.exists():
            self.file_path.parent.mkdir(parents=True, exist_ok=True)
            self.file_path.write_text('{}')
    
    async def _load_data(self) -> Dict[str, Dict[str, Any]]:
        """Load data from JSON file."""
        async with self._lock:
            try:
                content = self.file_path.read_text()
                return json.loads(content) if content else {}
            except Exception as e:
                logger.error(f"Error loading data: {e}")
                return {}
    
    async def _save_data(self, data: Dict[str, Dict[str, Any]]) -> None:
        """Save data to JSON file."""
        async with self._lock:
            try:
                content = json.dumps(data, indent=2, default=str)
                self.file_path.write_text(content)
            except Exception as e:
                logger.error(f"Error saving data: {e}")
                raise
    
    async def _do_find_by_id(self, entity_id: str) -> Optional[TEntity]:
        """Find entity in JSON file."""
        data = await self._load_data()
        entity_data = data.get(entity_id)
        
        if entity_data:
            return self._deserialize(entity_data)
        return None
    
    async def _do_find_all(self, limit: int, offset: int) -> List[TEntity]:
        """Find all entities in JSON file."""
        data = await self._load_data()
        entities = []
        
        for entity_data in list(data.values())[offset:offset + limit]:
            entity = self._deserialize(entity_data)
            if entity:
                entities.append(entity)
        
        return entities
    
    async def _do_save(self, entity: TEntity) -> TEntity:
        """Save entity to JSON file."""
        data = await self._load_data()
        entity_data = self._serialize(entity)
        data[entity.id] = entity_data
        await self._save_data(data)
        return entity
    
    async def _do_delete(self, entity_id: str) -> bool:
        """Delete entity from JSON file."""
        data = await self._load_data()
        if entity_id in data:
            del data[entity_id]
            await self._save_data(data)
            return True
        return False
    
    def _serialize(self, entity: TEntity) -> Dict[str, Any]:
        """Serialize entity to dictionary."""
        if hasattr(entity, 'to_dict'):
            return entity.to_dict()
        elif hasattr(entity, '__dict__'):
            return entity.__dict__
        else:
            return {'id': entity.id}
    
    def _deserialize(self, data: Dict[str, Any]) -> Optional[TEntity]:
        """Deserialize dictionary to entity."""
        try:
            if hasattr(self.entity_type, 'from_dict'):
                return self.entity_type.from_dict(data)
            else:
                return self.entity_type(**data)
        except Exception as e:
            logger.error(f"Error deserializing entity: {e}")
            return None


# ============================================================================
# CACHED REPOSITORY - Decorator pattern for caching
# ============================================================================


class CachedRepository(CacheableRepository[TEntity], Generic[TEntity]):
    """
    Cached repository decorator.
    Decorator Pattern: Adds caching to any repository.
    """
    
    def __init__(
        self,
        base_repository: Repository[TEntity],
        cache_ttl: int = 3600
    ):
        self.base_repository = base_repository
        self.cache_ttl = cache_ttl
        self._cache: Dict[str, TEntity] = {}
        self._cache_times: Dict[str, datetime] = {}
        self._lock = asyncio.Lock()
    
    async def find_by_id(self, entity_id: str) -> Optional[TEntity]:
        """Find with caching."""
        # Check cache first
        cached = await self._get_from_cache(entity_id)
        if cached:
            return cached
        
        # Load from base repository
        entity = await self.base_repository.find_by_id(entity_id)
        
        # Cache the result
        if entity:
            await self._put_in_cache(entity_id, entity)
        
        return entity
    
    async def find_all(self, limit: int = 100, offset: int = 0) -> List[TEntity]:
        """Find all - delegates to base repository."""
        return await self.base_repository.find_all(limit, offset)
    
    async def save(self, entity: TEntity) -> TEntity:
        """Save and update cache."""
        saved = await self.base_repository.save(entity)
        await self._put_in_cache(entity.id, saved)
        return saved
    
    async def delete(self, entity_id: str) -> bool:
        """Delete and invalidate cache."""
        result = await self.base_repository.delete(entity_id)
        if result:
            await self.invalidate_cache(entity_id)
        return result
    
    async def invalidate_cache(self, entity_id: str) -> None:
        """Invalidate cache for specific entity."""
        async with self._lock:
            self._cache.pop(entity_id, None)
            self._cache_times.pop(entity_id, None)
    
    async def warm_cache(self, entity_ids: List[str]) -> None:
        """Pre-load entities into cache."""
        for entity_id in entity_ids:
            entity = await self.base_repository.find_by_id(entity_id)
            if entity:
                await self._put_in_cache(entity_id, entity)
    
    async def _get_from_cache(self, entity_id: str) -> Optional[TEntity]:
        """Get entity from cache if not expired."""
        async with self._lock:
            if entity_id in self._cache:
                cache_time = self._cache_times.get(entity_id)
                if cache_time:
                    age = (datetime.now() - cache_time).total_seconds()
                    if age < self.cache_ttl:
                        return self._cache[entity_id]
                    else:
                        # Expired - remove from cache
                        del self._cache[entity_id]
                        del self._cache_times[entity_id]
        return None
    
    async def _put_in_cache(self, entity_id: str, entity: TEntity) -> None:
        """Put entity in cache."""
        async with self._lock:
            self._cache[entity_id] = entity
            self._cache_times[entity_id] = datetime.now()


# ============================================================================
# QUERY REPOSITORY - Advanced querying capabilities
# ============================================================================


@dataclass
class Query:
    """Base query class."""
    filters: Dict[str, Any] = field(default_factory=dict)
    sort_by: Optional[str] = None
    sort_desc: bool = False
    limit: int = 100
    offset: int = 0


class QueryableRepository(QueryRepository[TEntity, Query], BaseRepository[TEntity], Generic[TEntity]):
    """
    Repository with advanced querying capabilities.
    Single Responsibility: Handle complex queries.
    """
    
    def __init__(self, entity_type: Type[TEntity]):
        super().__init__(entity_type)
    
    async def find_by_query(self, query: Query) -> List[TEntity]:
        """Find entities matching query."""
        # Get all entities (in production, this would be optimized)
        all_entities = await self.find_all(limit=10000, offset=0)
        
        # Apply filters
        filtered = self._apply_filters(all_entities, query.filters)
        
        # Apply sorting
        if query.sort_by:
            filtered = self._apply_sorting(filtered, query.sort_by, query.sort_desc)
        
        # Apply pagination
        start = query.offset
        end = query.offset + query.limit
        
        return filtered[start:end]
    
    async def count_by_query(self, query: Query) -> int:
        """Count entities matching query."""
        all_entities = await self.find_all(limit=10000, offset=0)
        filtered = self._apply_filters(all_entities, query.filters)
        return len(filtered)
    
    def _apply_filters(self, entities: List[TEntity], filters: Dict[str, Any]) -> List[TEntity]:
        """Apply filters to entities."""
        if not filters:
            return entities
        
        filtered = []
        for entity in entities:
            matches = True
            
            for key, value in filters.items():
                entity_value = getattr(entity, key, None)
                
                if isinstance(value, dict):
                    # Handle operators
                    if '$eq' in value and entity_value != value['$eq']:
                        matches = False
                    elif '$ne' in value and entity_value == value['$ne']:
                        matches = False
                    elif '$gt' in value and entity_value <= value['$gt']:
                        matches = False
                    elif '$gte' in value and entity_value < value['$gte']:
                        matches = False
                    elif '$lt' in value and entity_value >= value['$lt']:
                        matches = False
                    elif '$lte' in value and entity_value > value['$lte']:
                        matches = False
                    elif '$in' in value and entity_value not in value['$in']:
                        matches = False
                    elif '$nin' in value and entity_value in value['$nin']:
                        matches = False
                elif entity_value != value:
                    matches = False
                
                if not matches:
                    break
            
            if matches:
                filtered.append(entity)
        
        return filtered
    
    def _apply_sorting(
        self,
        entities: List[TEntity],
        sort_by: str,
        desc: bool = False
    ) -> List[TEntity]:
        """Apply sorting to entities."""
        try:
            return sorted(
                entities,
                key=lambda e: getattr(e, sort_by, ''),
                reverse=desc
            )
        except Exception as e:
            logger.error(f"Error sorting entities: {e}")
            return entities


# ============================================================================
# UNIT OF WORK PATTERN - Transaction management
# ============================================================================


class UnitOfWork:
    """
    Unit of Work pattern for transaction management.
    Single Responsibility: Coordinate transactions across repositories.
    """
    
    def __init__(self):
        self._repositories: Dict[str, Repository] = {}
        self._new_entities: List[Identifiable] = []
        self._dirty_entities: List[Identifiable] = []
        self._removed_entities: List[Identifiable] = []
        self._committed = False
    
    def register_repository(self, name: str, repository: Repository) -> None:
        """Register a repository with the unit of work."""
        self._repositories[name] = repository
    
    def get_repository(self, name: str) -> Optional[Repository]:
        """Get registered repository by name."""
        return self._repositories.get(name)
    
    def register_new(self, entity: Identifiable) -> None:
        """Register new entity for insertion."""
        self._new_entities.append(entity)
    
    def register_dirty(self, entity: Identifiable) -> None:
        """Register modified entity for update."""
        if entity not in self._dirty_entities:
            self._dirty_entities.append(entity)
    
    def register_removed(self, entity: Identifiable) -> None:
        """Register entity for deletion."""
        self._removed_entities.append(entity)
    
    async def commit(self) -> None:
        """Commit all changes."""
        if self._committed:
            raise RuntimeError("Unit of work already committed")
        
        try:
            # Save new entities
            for entity in self._new_entities:
                repo = self._find_repository_for_entity(entity)
                if repo:
                    await repo.save(entity)
            
            # Update dirty entities
            for entity in self._dirty_entities:
                repo = self._find_repository_for_entity(entity)
                if repo:
                    await repo.save(entity)
            
            # Delete removed entities
            for entity in self._removed_entities:
                repo = self._find_repository_for_entity(entity)
                if repo:
                    await repo.delete(entity.id)
            
            self._committed = True
            
        except Exception as e:
            logger.error(f"Error committing unit of work: {e}")
            await self.rollback()
            raise
    
    async def rollback(self) -> None:
        """Rollback all changes."""
        # Clear all pending changes
        self._new_entities.clear()
        self._dirty_entities.clear()
        self._removed_entities.clear()
    
    def _find_repository_for_entity(self, entity: Identifiable) -> Optional[Repository]:
        """Find appropriate repository for entity type."""
        entity_type = type(entity).__name__.lower()
        
        # Try exact match
        repo = self._repositories.get(f"{entity_type}_repository")
        if repo:
            return repo
        
        # Try partial match
        for name, repo in self._repositories.items():
            if entity_type in name:
                return repo
        
        return None


# ============================================================================
# REPOSITORY FACTORY - Create configured repositories
# ============================================================================


class RepositoryFactory:
    """
    Factory for creating repositories.
    Factory Pattern: Centralized repository creation.
    """
    
    @staticmethod
    def create_in_memory_repository(entity_type: Type[TEntity]) -> Repository[TEntity]:
        """Create in-memory repository."""
        return InMemoryRepository(entity_type)
    
    @staticmethod
    def create_file_repository(
        entity_type: Type[TEntity],
        file_path: Path
    ) -> Repository[TEntity]:
        """Create file-based repository."""
        return JsonFileRepository(entity_type, file_path)
    
    @staticmethod
    def create_cached_repository(
        base_repository: Repository[TEntity],
        cache_ttl: int = 3600
    ) -> CacheableRepository[TEntity]:
        """Create cached repository decorator."""
        return CachedRepository(base_repository, cache_ttl)
    
    @staticmethod
    def create_queryable_repository(entity_type: Type[TEntity]) -> QueryRepository[TEntity, Query]:
        """Create repository with query capabilities."""
        return QueryableRepository(entity_type)


# Export public API
__all__ = [
    # Base classes
    'BaseRepository',
    
    # Concrete implementations
    'InMemoryRepository',
    'JsonFileRepository',
    'CachedRepository',
    'QueryableRepository',
    
    # Query support
    'Query',
    
    # Unit of Work
    'UnitOfWork',
    
    # Factory
    'RepositoryFactory',
] 