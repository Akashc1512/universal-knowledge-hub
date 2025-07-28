"""
Strategy Pattern implementation for search algorithms.

This module implements the Strategy Pattern following SOLID principles:
- Single Responsibility: Each strategy handles one search algorithm
- Open/Closed: New strategies can be added without modifying existing code
- Liskov Substitution: All strategies can be used interchangeably
- Interface Segregation: Specific interfaces for different strategy types
- Dependency Inversion: Depend on strategy interfaces, not implementations
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Protocol, runtime_checkable
from dataclasses import dataclass
from enum import Enum
import logging
import asyncio
from datetime import datetime

from .interfaces import SearchStrategy, RankingStrategy, FilterStrategy

logger = logging.getLogger(__name__)


# ============================================================================
# DATA MODELS
# ============================================================================


@dataclass
class SearchQuery:
    """Search query with metadata."""
    text: str
    filters: Dict[str, Any]
    limit: int = 10
    offset: int = 0
    include_metadata: bool = True
    user_context: Optional[Dict[str, Any]] = None


@dataclass
class SearchResult:
    """Search result with scoring and metadata."""
    id: str
    content: str
    score: float
    source: str
    metadata: Dict[str, Any]
    timestamp: datetime


class SearchAlgorithm(Enum):
    """Available search algorithms."""
    VECTOR = "vector"
    KEYWORD = "keyword"
    SEMANTIC = "semantic"
    HYBRID = "hybrid"
    GRAPH = "graph"
    ML_ENHANCED = "ml_enhanced"


# ============================================================================
# SEARCH STRATEGIES - Single Responsibility for each algorithm
# ============================================================================


class BaseSearchStrategy(ABC):
    """
    Base search strategy with common functionality.
    Template Method Pattern for common operations.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.metrics = {
            'searches': 0,
            'total_time': 0.0,
            'errors': 0
        }
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Get strategy name."""
        pass
    
    async def search(self, query: SearchQuery, **kwargs) -> List[SearchResult]:
        """
        Execute search with metrics tracking.
        Template method with hooks for customization.
        """
        start_time = datetime.now()
        try:
            # Pre-processing hook
            processed_query = await self._preprocess_query(query)
            
            # Core search implementation
            results = await self._execute_search(processed_query, **kwargs)
            
            # Post-processing hook
            results = await self._postprocess_results(results, query)
            
            # Update metrics
            self.metrics['searches'] += 1
            self.metrics['total_time'] += (datetime.now() - start_time).total_seconds()
            
            return results
            
        except Exception as e:
            self.metrics['errors'] += 1
            logger.error(f"Search error in {self.name}: {e}")
            raise
    
    @abstractmethod
    async def _execute_search(self, query: SearchQuery, **kwargs) -> List[SearchResult]:
        """Execute the actual search - must be implemented by subclasses."""
        pass
    
    async def _preprocess_query(self, query: SearchQuery) -> SearchQuery:
        """Preprocess query - can be overridden by subclasses."""
        return query
    
    async def _postprocess_results(
        self,
        results: List[SearchResult],
        query: SearchQuery
    ) -> List[SearchResult]:
        """Postprocess results - can be overridden by subclasses."""
        return results
    
    def supports_query(self, query: SearchQuery) -> bool:
        """Check if strategy supports given query."""
        return True
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get strategy metrics."""
        return self.metrics.copy()


class VectorSearchStrategy(BaseSearchStrategy):
    """
    Vector similarity search strategy.
    Single Responsibility: Handle vector-based searches.
    """
    
    @property
    def name(self) -> str:
        return "vector_search"
    
    async def _execute_search(self, query: SearchQuery, **kwargs) -> List[SearchResult]:
        """Execute vector similarity search."""
        # Import here to avoid circular dependencies
        from api.llm_client import LLMClient
        
        # Get query embedding
        llm_client = LLMClient()
        query_embedding = await llm_client.get_embedding(query.text)
        
        # Search in vector database
        results = []
        
        # Simulate vector search (in production, use actual vector DB)
        for i in range(min(query.limit, 5)):
            results.append(SearchResult(
                id=f"vec_{i}",
                content=f"Vector result {i} for: {query.text}",
                score=0.9 - i * 0.1,
                source="vector_db",
                metadata={"embedding_model": "text-embedding-ada-002"},
                timestamp=datetime.now()
            ))
        
        return results
    
    async def _preprocess_query(self, query: SearchQuery) -> SearchQuery:
        """Preprocess query for vector search."""
        # Clean and normalize query text
        query.text = query.text.strip().lower()
        return query
    
    def supports_query(self, query: SearchQuery) -> bool:
        """Vector search supports all text queries."""
        return bool(query.text)


class KeywordSearchStrategy(BaseSearchStrategy):
    """
    Keyword-based search strategy.
    Single Responsibility: Handle keyword/BM25 searches.
    """
    
    @property
    def name(self) -> str:
        return "keyword_search"
    
    async def _execute_search(self, query: SearchQuery, **kwargs) -> List[SearchResult]:
        """Execute keyword search."""
        # In production, use Elasticsearch or similar
        keywords = query.text.split()
        
        results = []
        for i in range(min(query.limit, 5)):
            results.append(SearchResult(
                id=f"kw_{i}",
                content=f"Keyword match {i} for: {' '.join(keywords)}",
                score=0.85 - i * 0.1,
                source="elasticsearch",
                metadata={"match_type": "BM25", "keywords": keywords},
                timestamp=datetime.now()
            ))
        
        return results
    
    async def _postprocess_results(
        self,
        results: List[SearchResult],
        query: SearchQuery
    ) -> List[SearchResult]:
        """Apply keyword highlighting to results."""
        keywords = query.text.lower().split()
        
        for result in results:
            # Highlight keywords in content
            content = result.content
            for keyword in keywords:
                content = content.replace(
                    keyword,
                    f"**{keyword}**"
                )
            result.content = content
        
        return results


class SemanticSearchStrategy(BaseSearchStrategy):
    """
    Semantic understanding search strategy.
    Uses NLP for query understanding.
    """
    
    @property
    def name(self) -> str:
        return "semantic_search"
    
    async def _execute_search(self, query: SearchQuery, **kwargs) -> List[SearchResult]:
        """Execute semantic search with query understanding."""
        # Extract entities and intent
        entities = await self._extract_entities(query.text)
        intent = await self._classify_intent(query.text)
        
        # Search based on semantic understanding
        results = []
        for i in range(min(query.limit, 5)):
            results.append(SearchResult(
                id=f"sem_{i}",
                content=f"Semantic result {i} - Intent: {intent}, Entities: {entities}",
                score=0.88 - i * 0.1,
                source="semantic_engine",
                metadata={"intent": intent, "entities": entities},
                timestamp=datetime.now()
            ))
        
        return results
    
    async def _extract_entities(self, text: str) -> List[str]:
        """Extract named entities from query."""
        # In production, use spaCy or similar
        return ["entity1", "entity2"]
    
    async def _classify_intent(self, text: str) -> str:
        """Classify query intent."""
        # In production, use intent classification model
        if "how" in text.lower():
            return "explanation"
        elif "what" in text.lower():
            return "definition"
        elif "why" in text.lower():
            return "reasoning"
        else:
            return "general"


class HybridSearchStrategy(BaseSearchStrategy):
    """
    Hybrid search combining multiple strategies.
    Composite Pattern: Combines other strategies.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.strategies = [
            VectorSearchStrategy(config),
            KeywordSearchStrategy(config),
            SemanticSearchStrategy(config)
        ]
        self.weights = config.get('weights', {
            'vector': 0.4,
            'keyword': 0.3,
            'semantic': 0.3
        })
    
    @property
    def name(self) -> str:
        return "hybrid_search"
    
    async def _execute_search(self, query: SearchQuery, **kwargs) -> List[SearchResult]:
        """Execute hybrid search combining multiple strategies."""
        # Run all strategies in parallel
        tasks = [
            strategy.search(query, **kwargs)
            for strategy in self.strategies
        ]
        
        all_results = await asyncio.gather(*tasks)
        
        # Merge and re-rank results
        merged_results = self._merge_results(all_results)
        ranked_results = self._rerank_results(merged_results)
        
        return ranked_results[:query.limit]
    
    def _merge_results(self, results_lists: List[List[SearchResult]]) -> List[SearchResult]:
        """Merge results from multiple strategies."""
        # Deduplicate by ID
        seen_ids = set()
        merged = []
        
        for results in results_lists:
            for result in results:
                if result.id not in seen_ids:
                    seen_ids.add(result.id)
                    merged.append(result)
        
        return merged
    
    def _rerank_results(self, results: List[SearchResult]) -> List[SearchResult]:
        """Re-rank results based on weighted scores."""
        # Apply strategy weights
        for result in results:
            if result.source == "vector_db":
                result.score *= self.weights['vector']
            elif result.source == "elasticsearch":
                result.score *= self.weights['keyword']
            elif result.source == "semantic_engine":
                result.score *= self.weights['semantic']
        
        # Sort by weighted score
        return sorted(results, key=lambda r: r.score, reverse=True)


# ============================================================================
# RANKING STRATEGIES - Post-processing result ranking
# ============================================================================


class PersonalizedRankingStrategy:
    """
    Personalized ranking based on user preferences.
    Single Responsibility: Personalize rankings.
    """
    
    async def rank(
        self,
        results: List[SearchResult],
        context: Dict[str, Any]
    ) -> List[SearchResult]:
        """Rank results based on user preferences."""
        user_preferences = context.get('user_preferences', {})
        
        # Apply personalization scoring
        for result in results:
            # Boost score based on user preferences
            if any(pref in result.content.lower() for pref in user_preferences.get('topics', [])):
                result.score *= 1.2
            
            # Recency preference
            if user_preferences.get('prefer_recent', False):
                age_hours = (datetime.now() - result.timestamp).total_seconds() / 3600
                recency_boost = 1.0 / (1.0 + age_hours / 24)  # Decay over days
                result.score *= recency_boost
        
        return sorted(results, key=lambda r: r.score, reverse=True)


class DiversityRankingStrategy:
    """
    Diversity-aware ranking to avoid redundancy.
    Single Responsibility: Ensure result diversity.
    """
    
    async def rank(
        self,
        results: List[SearchResult],
        context: Dict[str, Any]
    ) -> List[SearchResult]:
        """Rank results with diversity consideration."""
        diversity_threshold = context.get('diversity_threshold', 0.7)
        
        ranked = []
        for result in sorted(results, key=lambda r: r.score, reverse=True):
            # Check similarity with already ranked results
            is_diverse = True
            for ranked_result in ranked:
                similarity = self._calculate_similarity(result, ranked_result)
                if similarity > diversity_threshold:
                    is_diverse = False
                    break
            
            if is_diverse:
                ranked.append(result)
        
        return ranked
    
    def _calculate_similarity(self, result1: SearchResult, result2: SearchResult) -> float:
        """Calculate similarity between two results."""
        # Simple word overlap similarity
        words1 = set(result1.content.lower().split())
        words2 = set(result2.content.lower().split())
        
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        return intersection / union if union > 0 else 0.0


# ============================================================================
# FILTER STRATEGIES - Result filtering
# ============================================================================


class MetadataFilterStrategy:
    """
    Filter results based on metadata criteria.
    Single Responsibility: Metadata-based filtering.
    """
    
    async def filter(
        self,
        results: List[SearchResult],
        criteria: Dict[str, Any]
    ) -> List[SearchResult]:
        """Filter results based on metadata criteria."""
        filtered = []
        
        for result in results:
            matches = True
            
            for key, value in criteria.items():
                if key not in result.metadata:
                    matches = False
                    break
                
                # Handle different comparison types
                if isinstance(value, dict):
                    # Range queries
                    if 'gte' in value and result.metadata[key] < value['gte']:
                        matches = False
                    if 'lte' in value and result.metadata[key] > value['lte']:
                        matches = False
                elif isinstance(value, list):
                    # In queries
                    if result.metadata[key] not in value:
                        matches = False
                else:
                    # Exact match
                    if result.metadata[key] != value:
                        matches = False
            
            if matches:
                filtered.append(result)
        
        return filtered


class QualityFilterStrategy:
    """
    Filter results based on quality metrics.
    Single Responsibility: Quality-based filtering.
    """
    
    async def filter(
        self,
        results: List[SearchResult],
        criteria: Dict[str, Any]
    ) -> List[SearchResult]:
        """Filter results based on quality criteria."""
        min_score = criteria.get('min_score', 0.5)
        min_confidence = criteria.get('min_confidence', 0.6)
        
        filtered = []
        for result in results:
            # Check score threshold
            if result.score < min_score:
                continue
            
            # Check confidence if available
            confidence = result.metadata.get('confidence', 1.0)
            if confidence < min_confidence:
                continue
            
            filtered.append(result)
        
        return filtered


# ============================================================================
# STRATEGY CONTEXT - Manages strategy selection and execution
# ============================================================================


class SearchStrategyContext:
    """
    Context for managing search strategies.
    Follows Dependency Inversion - depends on strategy interface.
    """
    
    def __init__(self):
        self._strategies: Dict[str, BaseSearchStrategy] = {}
        self._ranking_strategies: List[RankingStrategy] = []
        self._filter_strategies: List[FilterStrategy] = []
        self._default_strategy = "hybrid"
    
    def register_search_strategy(self, name: str, strategy: BaseSearchStrategy):
        """Register a search strategy."""
        self._strategies[name] = strategy
        logger.info(f"Registered search strategy: {name}")
    
    def register_ranking_strategy(self, strategy: RankingStrategy):
        """Register a ranking strategy."""
        self._ranking_strategies.append(strategy)
    
    def register_filter_strategy(self, strategy: FilterStrategy):
        """Register a filter strategy."""
        self._filter_strategies.append(strategy)
    
    async def search(
        self,
        query: SearchQuery,
        strategy_name: Optional[str] = None,
        **kwargs
    ) -> List[SearchResult]:
        """
        Execute search with specified or default strategy.
        Applies all registered ranking and filter strategies.
        """
        # Select strategy
        strategy_name = strategy_name or self._default_strategy
        strategy = self._strategies.get(strategy_name)
        
        if not strategy:
            raise ValueError(f"Unknown search strategy: {strategy_name}")
        
        # Execute search
        results = await strategy.search(query, **kwargs)
        
        # Apply filters
        for filter_strategy in self._filter_strategies:
            if hasattr(filter_strategy, 'filter'):
                results = await filter_strategy.filter(results, query.filters)
        
        # Apply ranking
        context = {
            'query': query,
            'user_context': query.user_context
        }
        
        for ranking_strategy in self._ranking_strategies:
            if hasattr(ranking_strategy, 'rank'):
                results = await ranking_strategy.rank(results, context)
        
        return results
    
    def set_default_strategy(self, strategy_name: str):
        """Set default search strategy."""
        if strategy_name not in self._strategies:
            raise ValueError(f"Unknown strategy: {strategy_name}")
        self._default_strategy = strategy_name
    
    def get_available_strategies(self) -> List[str]:
        """Get list of available search strategies."""
        return list(self._strategies.keys())


# ============================================================================
# FACTORY FUNCTION - Create configured search context
# ============================================================================


def create_search_context(config: Dict[str, Any]) -> SearchStrategyContext:
    """
    Factory function to create configured search context.
    Hides complexity of strategy setup.
    """
    context = SearchStrategyContext()
    
    # Register search strategies
    context.register_search_strategy("vector", VectorSearchStrategy(config))
    context.register_search_strategy("keyword", KeywordSearchStrategy(config))
    context.register_search_strategy("semantic", SemanticSearchStrategy(config))
    context.register_search_strategy("hybrid", HybridSearchStrategy(config))
    
    # Register ranking strategies
    if config.get('enable_personalization', True):
        context.register_ranking_strategy(PersonalizedRankingStrategy())
    
    if config.get('enable_diversity', True):
        context.register_ranking_strategy(DiversityRankingStrategy())
    
    # Register filter strategies
    context.register_filter_strategy(MetadataFilterStrategy())
    context.register_filter_strategy(QualityFilterStrategy())
    
    # Set default strategy
    default = config.get('default_strategy', 'hybrid')
    context.set_default_strategy(default)
    
    return context


# Export public API
__all__ = [
    # Data models
    'SearchQuery',
    'SearchResult',
    'SearchAlgorithm',
    
    # Base classes
    'BaseSearchStrategy',
    
    # Search strategies
    'VectorSearchStrategy',
    'KeywordSearchStrategy',
    'SemanticSearchStrategy',
    'HybridSearchStrategy',
    
    # Ranking strategies
    'PersonalizedRankingStrategy',
    'DiversityRankingStrategy',
    
    # Filter strategies
    'MetadataFilterStrategy',
    'QualityFilterStrategy',
    
    # Context and factory
    'SearchStrategyContext',
    'create_search_context',
] 