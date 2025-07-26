
"""
RetrievalAgent Implementation for Multi-Agent Knowledge Platform
This module implements a sophisticated retrieval agent that combines vector search,
keyword search, and knowledge graph queries with caching and error handling.
"""

import asyncio
import hashlib
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import numpy as np
from functools import lru_cache
import aiohttp
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from core.types import Document
from agents.base_agent import AgentResult  # Add import for AgentResult

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# Data Models
# ============================================================================

@dataclass
class Document:
    """Represents a retrieved document/chunk"""
    content: str
    score: float
    source: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    doc_id: str = ""
    chunk_id: Optional[str] = None
    timestamp: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'content': self.content,
            'score': self.score,
            'source': self.source,
            'metadata': self.metadata,
            'doc_id': self.doc_id,
            'chunk_id': self.chunk_id,
            'timestamp': self.timestamp
        }


@dataclass
class KnowledgeTriple:
    """Represents a fact from knowledge graph"""
    subject: str
    predicate: str
    object: str
    confidence: float = 1.0
    source: str = "knowledge_graph"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SearchResult:
    """Unified search result format"""
    documents: List[Document]
    search_type: str
    query_time_ms: int
    total_hits: int
    metadata: Dict[str, Any] = field(default_factory=dict)


# ============================================================================
# Database Client Interfaces
# ============================================================================

class VectorDBClient:
    """
    Interface for vector database operations.
    TODO: Replace with actual Qdrant/Weaviate/Pinecone client
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.embedding_model = None  # TODO: Initialize embedding model
        
    async def search(self, query_embedding: List[float], top_k: int, filters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Perform vector similarity search.
        TODO: Implement actual vector DB search
        """
        # Placeholder implementation
        await asyncio.sleep(0.1)  # Simulate network latency
        
        results = []
        for i in range(min(top_k, 5)):
            results.append({
                'id': f'vec_{i}',
                'score': 0.95 - (i * 0.05),
                'payload': {
                    'content': f'Vector search result {i} (placeholder)',
                    'metadata': {'source': 'vector_db', 'index': i}
                }
            })
        return results
    
    async def get_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for text.
        TODO: Integrate with actual embedding model (e.g., sentence-transformers)
        """
        # Placeholder: return random embedding
        return np.random.rand(384).tolist()
    
    async def search_similar(self, query: str, top_k: int = 20) -> List[Dict[str, Any]]:
        """
        Search for similar vectors using text query.
        This is a convenience method that combines embedding and search.
        """
        # Get embedding for the query
        query_embedding = await self.get_embedding(query)
        
        # Search using the embedding
        results = await self.search(query_embedding, top_k)
        
        # Convert results to the expected format
        formatted_results = []
        for result in results:
            formatted_results.append({
                'id': result.get('id', ''),
                'content': result.get('payload', {}).get('content', ''),
                'score': result.get('score', 0.0),
                'metadata': result.get('payload', {}).get('metadata', {})
            })
        
        return formatted_results


class ElasticsearchClient:
    """
    Interface for Elasticsearch operations.
    TODO: Replace with actual Elasticsearch client
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.index_name = config.get('index_name', 'knowledge_base')
        
    async def search(self, query: str, top_k: int, filters: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Perform BM25 keyword search.
        TODO: Implement actual Elasticsearch query
        """
        # Placeholder implementation
        await asyncio.sleep(0.08)  # Simulate network latency
        
        return {
            'hits': {
                'total': {'value': top_k * 2},
                'hits': [
                    {
                        '_id': f'es_{i}',
                        '_score': 10.5 - (i * 0.5),
                        '_source': {
                            'content': f'Keyword search result {i} for query: {query}',
                            'title': f'Document {i}',
                            'metadata': {'source': 'elasticsearch'}
                        }
                    }
                    for i in range(min(top_k, 5))
                ]
            }
        }
    
    async def search_documents(self, query: str, top_k: int = 20) -> List[Dict[str, Any]]:
        """
        Search documents using Elasticsearch.
        This is a convenience method that returns documents in a simplified format.
        """
        # Perform the search
        search_result = await self.search(query, top_k)
        
        # Extract and format documents
        documents = []
        for hit in search_result.get('hits', {}).get('hits', []):
            source = hit.get('_source', {})
            documents.append({
                'id': hit.get('_id', ''),
                'content': source.get('content', ''),
                'score': hit.get('_score', 0.0),
                'source': source.get('source', 'elasticsearch')
            })
        
        return documents


class KnowledgeGraphClient:
    """
    Interface for knowledge graph operations.
    TODO: Replace with actual SPARQL endpoint client
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.endpoint = config.get('sparql_endpoint', 'http://localhost:8890/sparql')
        
    async def query_sparql(self, query: str) -> List[Dict[str, Any]]:
        """
        Execute SPARQL query.
        TODO: Implement actual SPARQL query execution
        """
        # Placeholder implementation
        await asyncio.sleep(0.12)  # Simulate network latency
        
        return [
            {
                'subject': 'Entity1',
                'predicate': 'relatedTo',
                'object': 'Entity2',
                'confidence': 0.9
            }
        ]


# ============================================================================
# Cache Implementations
# ============================================================================

class SemanticCache:
    """
    Semantic cache for query results.
    TODO: Integrate with Redis and implement actual semantic similarity
    """
    
    def __init__(self, similarity_threshold: float = 0.92):
        self.cache = {}  # TODO: Replace with Redis client
        self.embeddings = {}  # TODO: Store in vector DB for similarity search
        self.similarity_threshold = similarity_threshold
        self.ttl_seconds = 3600  # 1 hour default TTL
        self.logger = logging.getLogger("SemanticCache")
        
    async def get(self, query: str, query_embedding: List[float] = None) -> Optional[SearchResult]:
        """
        Retrieve cached result for semantically similar query.
        - First, try exact match
        - Then, try stubbed semantic similarity (returns first cache entry if query is similar)
        """
        cache_key = self._generate_cache_key(query)
        if cache_key in self.cache:
            cached_data = self.cache[cache_key]
            if self._is_valid(cached_data):
                self.logger.info(f"[CACHE] Exact match hit for query: {query[:50]}...")
                return cached_data['result']
            else:
                self.logger.info(f"[CACHE] Exact match expired for query: {query[:50]}...")
        # Stub: simulate semantic similarity by returning any valid cache entry if query is similar
        for k, v in self.cache.items():
            if self._is_valid(v):
                # Get original query from cache data
                original_query = v.get('original_query')
                if original_query and self._is_semantically_similar(query, original_query):
                    self.logger.info(f"[CACHE] Semantic similarity hit for query: {query[:50]} ~ {original_query[:50]}")
                    return v['result']
        self.logger.info(f"[CACHE] Miss for query: {query[:50]}...")
        return None
    
    async def set(self, query: str, result: SearchResult, query_embedding: List[float] = None):
        """
        Cache query result with embedding.
        """
        cache_key = self._generate_cache_key(query)
        self.cache[cache_key] = {
            'result': result,
            'timestamp': datetime.utcnow(),
            'embedding': query_embedding,
            'original_query': query  # Store original query for semantic similarity
        }
        self.logger.info(f"[CACHE] Set for query: {query[:50]}...")
    
    def _generate_cache_key(self, query: str) -> str:
        """Generate cache key from query"""
        return hashlib.md5(query.encode()).hexdigest()
    

    
    def _is_valid(self, cached_data: Dict[str, Any]) -> bool:
        """Check if cached data is still valid"""
        age = (datetime.utcnow() - cached_data['timestamp']).total_seconds()
        return age < self.ttl_seconds
    
    def _is_semantically_similar(self, query1: str, query2: str) -> bool:
        """
        Stub: consider queries similar if they share at least 3 words (case-insensitive).
        Replace with real embedding similarity in production.
        """
        words1 = set(query1.lower().split())
        words2 = set(query2.lower().split())
        return len(words1 & words2) >= 3


# ============================================================================
# RetrievalAgent Implementation
# ============================================================================

class RetrievalAgent:
    """
    Advanced retrieval agent that combines vector search, keyword search, and knowledge graph queries.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or self._default_config()
        self.max_retries = self.config.get('max_retries', 3)
        self.timeout = self.config.get('timeout', 30)
        
        # Initialize clients
        self.vector_db = VectorDBClient(self.config.get('vector_db', {}))
        self.elasticsearch_client = ElasticsearchClient(self.config.get('elasticsearch', {}))
        self.knowledge_graph_client = KnowledgeGraphClient(self.config.get('knowledge_graph', {}))
        
        # Initialize semantic cache
        self.semantic_cache = SemanticCache(
            similarity_threshold=self.config.get('cache_similarity_threshold', 0.92)
        )
        
        # Performance tracking
        self.stats = {
            'total_queries': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'avg_response_time': 0.0
        }
        
        logger.info("RetrievalAgent initialized successfully")
    
    def _default_config(self) -> Dict[str, Any]:
        """Default configuration for the retrieval agent."""
        return {
            'max_retries': 3,
            'timeout': 30,
            'cache_similarity_threshold': 0.92,
            'vector_db': {
                'host': 'localhost',
                'port': 6333,
                'collection': 'knowledge_base'
            },
            'elasticsearch': {
                'host': 'localhost',
                'port': 9200,
                'index': 'knowledge_base'
            },
            'knowledge_graph': {
                'endpoint': 'http://localhost:7200/repositories/knowledge_base',
                'username': '',
                'password': ''
            }
        }
    
    async def vector_search(self, query: str, top_k: int = 20) -> SearchResult:
        """Perform vector similarity search."""
        start_time = time.time()
        
        try:
            # Get query embedding
            query_embedding = await self.vector_db.get_embedding(query)
            
            # Search vector database
            results = await self.vector_db.search(query_embedding, top_k)
            
            # Convert to Document objects
            documents = []
            for result in results:
                doc = Document(
                    content=result.get('content', ''),
                    score=result.get('score', 0.0),
                    source='vector_search',
                    metadata=result.get('metadata', {}),
                    doc_id=result.get('id', ''),
                    chunk_id=result.get('chunk_id'),
                    timestamp=result.get('timestamp')
                )
                documents.append(doc)
            
            query_time = int((time.time() - start_time) * 1000)
            
            return SearchResult(
                documents=documents,
                search_type='vector',
                query_time_ms=query_time,
                total_hits=len(documents)
            )
            
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return SearchResult(
                documents=[],
                search_type='vector',
                query_time_ms=int((time.time() - start_time) * 1000),
                total_hits=0
            )
    
    async def keyword_search(self, query: str, top_k: int = 20) -> SearchResult:
        """Perform keyword-based search using Elasticsearch."""
        start_time = time.time()
        
        try:
            # Search Elasticsearch
            results = await self.elasticsearch_client.search_documents(query, top_k)
            
            # Convert to Document objects
            documents = []
            for result in results:
                doc = Document(
                    content=result.get('content', ''),
                    score=result.get('score', 0.0),
                    source='keyword_search',
                    metadata=result.get('metadata', {}),
                    doc_id=result.get('id', ''),
                    chunk_id=None, # Elasticsearch doesn't have chunk_id in this simplified format
                    timestamp=None # Elasticsearch doesn't have timestamp in this simplified format
                )
                documents.append(doc)
            
            query_time = int((time.time() - start_time) * 1000)
            
            return SearchResult(
                documents=documents,
                search_type='keyword',
                query_time_ms=query_time,
                total_hits=len(documents) # Elasticsearch doesn't return total_hits in this simplified format
            )
            
        except Exception as e:
            logger.error(f"Keyword search failed: {e}")
            return SearchResult(
                documents=[],
                search_type='keyword',
                query_time_ms=int((time.time() - start_time) * 1000),
                total_hits=0
            )
    
    async def graph_search(self, entities: List[str], top_k: int = 20) -> SearchResult:
        """Perform knowledge graph search using SPARQL."""
        start_time = time.time()
        
        try:
            # Build SPARQL query from entities
            sparql_query = self._build_sparql_query(entities)
            
            # Query knowledge graph
            results = await self.knowledge_graph_client.query_sparql(sparql_query)
            
            # Convert to Document objects
            documents = []
            for result in results:
                # Convert SPARQL result to document format
                content = f"{result.get('subject', '')} {result.get('predicate', '')} {result.get('object', '')}"
                doc = Document(
                    content=content,
                    score=result.get('confidence', 1.0),
                    source='knowledge_graph',
                    metadata={
                        'subject': result.get('subject', ''),
                        'predicate': result.get('predicate', ''),
                        'object': result.get('object', ''),
                        'confidence': result.get('confidence', 1.0)
                    },
                    doc_id=f"graph_{hash(content)}",
                    timestamp=datetime.now().isoformat()
                )
                documents.append(doc)
            
            query_time = int((time.time() - start_time) * 1000)
            
            return SearchResult(
                documents=documents,
                search_type='graph',
                query_time_ms=query_time,
                total_hits=len(documents)
            )
            
        except Exception as e:
            logger.error(f"Graph search failed: {e}")
            return SearchResult(
                documents=[],
                search_type='graph',
                query_time_ms=int((time.time() - start_time) * 1000),
                total_hits=0
            )
    
    def _build_sparql_query(self, entities: List[str]) -> str:
        """Build SPARQL query from entities."""
        if not entities:
            return ""
        
        entity_filters = " || ".join([f'?s = <{entity}>' for entity in entities])
        
        query = f"""
        SELECT ?s ?p ?o ?confidence
        WHERE {{
            ?s ?p ?o .
            FILTER({entity_filters})
            OPTIONAL {{ ?s <http://example.org/confidence> ?confidence }}
        }}
        LIMIT 20
        """
        return query
    
    async def hybrid_retrieve(self, query: str, entities: List[str] = None) -> SearchResult:
        """
        Execute all search strategies in parallel and merge results intelligently.
        
        TODO Integration Points:
        - Implement sophisticated result merging algorithm
        - Add learning-based weight optimization
        - Implement diversity-aware selection
        - Add result explanation generation
        """
        start_time = time.time()
        
        try:
            # Extract entities from query if not provided
            if not entities:
                entities = await self._extract_entities(query)
                logger.info(f"Extracted entities: {entities}")
            
            # Check cache for hybrid results
            cache_key = f"hybrid:{query}:{','.join(sorted(entities or []))}"
            cached_result = await self.semantic_cache.get(cache_key)
            if cached_result:
                return cached_result
            
            # Execute all search strategies in parallel
            search_tasks = [
                self.vector_search(query, top_k=30),
                self.keyword_search(query, top_k=30)
            ]
            
            # Add graph search if entities are available
            if entities:
                search_tasks.append(self.graph_search(entities, top_k=20))
            
            # Execute with timeout
            results = await asyncio.wait_for(
                asyncio.gather(*search_tasks, return_exceptions=True),
                timeout=self.timeout * 1.5  # Allow more time for parallel execution
            )
            
            # Process results and handle failures
            valid_results = []
            search_types = ['vector', 'keyword', 'graph']
            
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"{search_types[i]} search failed: {result}")
                elif isinstance(result, SearchResult) and result.documents:
                    valid_results.append(result)
            
            # Merge and deduplicate results
            merged_documents = await self._merge_and_deduplicate(valid_results)
            
            # TODO: LLM-based result fusion and ranking
            """
            # Result Fusion Template (TODO: Implement)
            fusion_prompt = f'''
            Query: "{query}"
            
            I have retrieved documents from multiple sources:
            - Vector search: {len(vector_docs)} documents
            - Keyword search: {len(keyword_docs)} documents  
            - Knowledge graph: {len(graph_docs)} facts
            
            Documents:
            {self._format_merged_results_for_llm(merged_documents[:30])}
            
            Please:
            1. Identify the most relevant documents for answering the query
            2. Detect and merge duplicate information
            3. Rank by relevance considering all search signals
            4. Explain the ranking rationale
            
            Return as JSON with document IDs and relevance scores.
            '''
            # final_ranking = await self._llm_fuse_results(fusion_prompt)
            """
            
            # Apply diversity constraints
            final_documents = self._apply_diversity_constraints(merged_documents)
            
            result = SearchResult(
                documents=final_documents[:20],  # Return top 20
                search_type='hybrid',
                query_time_ms=int((time.time() - start_time) * 1000),
                total_hits=len(merged_documents),
                metadata={
                    'search_strategies': [r.search_type for r in valid_results],
                    'entities': entities,
                    'merge_stats': {
                        'total_before_merge': sum(r.total_hits for r in valid_results),
                        'after_dedup': len(merged_documents),
                        'final_count': len(final_documents)
                    }
                }
            )
            
            # Cache the result
            await self.semantic_cache.set(cache_key, result)
            
            return result
            
        except asyncio.TimeoutError:
            logger.error("Hybrid retrieval timeout")
            # Fallback to whatever results we have
            return await self._emergency_fallback(query)
            
        except Exception as e:
            logger.error(f"Hybrid retrieval error: {str(e)}")
            return await self._emergency_fallback(query)
    
    async def _merge_and_deduplicate(self, results: List[SearchResult]) -> List[Document]:
        """
        Merge results from multiple sources and remove duplicates.
        
        TODO:
        - Implement semantic deduplication using embeddings
        - Preserve best metadata from duplicates
        - Implement source-based weighting
        """
        # Collect all documents with source weights
        source_weights = {
            'vector': 1.2,
            'keyword': 1.0,
            'graph': 1.3
        }
        
        all_docs = []
        seen_content = set()
        seen_hashes = set()
        
        for result in results:
            weight = source_weights.get(result.search_type, 1.0)
            
            for doc in result.documents:
                # Simple content-based deduplication
                content_hash = hashlib.md5(doc.content.encode()).hexdigest()
                
                if content_hash not in seen_hashes:
                    # Adjust score based on source weight
                    doc.score *= weight
                    all_docs.append(doc)
                    seen_hashes.add(content_hash)
                else:
                    # Merge metadata if duplicate
                    # TODO: Implement metadata merging logic
                    pass
        
        # Sort by adjusted score
        all_docs.sort(key=lambda x: x.score, reverse=True)
        
        return all_docs
    
    def _apply_diversity_constraints(self, documents: List[Document]) -> List[Document]:
        """
        Apply diversity constraints to ensure varied results.
        
        TODO:
        - Implement MMR (Maximal Marginal Relevance)
        - Add source diversity requirements
        - Implement topic clustering
        """
        diverse_docs = []
        source_counts = {'vector': 0, 'keyword': 0, 'graph': 0}
        max_per_source = 7
        
        for doc in documents:
            source = doc.source.replace('_search', '')
            
            # Enforce source diversity
            if source_counts.get(source, 0) < max_per_source:
                diverse_docs.append(doc)
                source_counts[source] = source_counts.get(source, 0) + 1
            
            if len(diverse_docs) >= 20:
                break
        
        return diverse_docs
    
    async def _extract_entities(self, query: str) -> List[str]:
        """
        Extract entities from query using NER.
        
        TODO:
        - Implement actual NER using SpaCy/Transformers
        - Add entity linking to knowledge base
        - Implement coreference resolution
        """
        # Placeholder implementation
        # TODO: Use SpaCy or similar for actual entity extraction
        entities = []
        
        # Simple heuristic: extract capitalized words
        words = query.split()
        for word in words:
            if word[0].isupper() and len(word) > 2:
                entities.append(word)
        
        return entities[:5]  # Limit to 5 entities
    
    async def _emergency_fallback(self, query: str) -> SearchResult:
        """Emergency fallback when primary retrieval fails"""
        try:
            # Try simple keyword search with minimal requirements
            return await self.keyword_search(query, top_k=10)
        except:
            # Return empty result as last resort
            return SearchResult(
                documents=[],
                search_type='fallback',
                query_time_ms=0,
                total_hits=0,
                metadata={'error': 'All retrieval methods failed'}
            )
    
    def _format_documents_for_llm(self, documents: List[Document]) -> str:
        """Format documents for LLM prompt"""
        formatted = []
        for i, doc in enumerate(documents):
            formatted.append(f"""
Document {i+1} (ID: {doc.doc_id}, Score: {doc.score:.3f}):
{doc.content[:500]}...
Source: {doc.source}
""")
        return "\n".join(formatted)
    
    # TODO: Implement LLM integration methods
    async def _llm_rerank(self, prompt: str, documents: List[Document]) -> List[Document]:
        """
        Use LLM to rerank documents.
        
        TODO:
        - Implement actual LLM API call
        - Parse LLM response
        - Handle errors gracefully
        """
        # Placeholder - return documents as-is
        return documents
    
    async def _llm_expand_query(self, prompt: str) -> List[str]:
        """
        Use LLM to expand query.
        
        TODO:
        - Implement actual LLM API call
        - Parse expanded queries
        - Validate expansions
        """
        # Placeholder
        return []
    
    async def _llm_synthesize_facts(self, prompt: str) -> Dict[str, Any]:
        """
        Use LLM to synthesize facts from knowledge graph.
        
        TODO:
        - Implement actual LLM API call
        - Structure synthesized facts
        - Add confidence scores
        """
        # Placeholder
        return {}
    
    async def _llm_fuse_results(self, prompt: str) -> List[Tuple[str, float]]:
        """
        Use LLM to fuse results from multiple sources.
        
        TODO:
        - Implement actual LLM API call
        - Parse ranking decisions
        - Extract rationale
        """
        # Placeholder
        return []

    async def process_task(self, task: Dict[str, Any], context) -> AgentResult:
        """Process a retrieval task."""
        try:
            query = task.get('query', context.query)
            entities = task.get('entities', [])
            
            # Perform hybrid retrieval
            result = await self.hybrid_retrieve(query, entities)
            
            # Convert to AgentResult format
            return AgentResult(
                success=True,
                data=result.documents,
                confidence=0.8,  # Placeholder confidence
                token_usage={'prompt': 0, 'completion': 0},
                execution_time_ms=result.query_time_ms,
                metadata={'search_type': result.search_type, 'total_hits': result.total_hits}
            )
        except Exception as e:
            logger.error(f"Retrieval task failed: {e}")
            return AgentResult(
                success=False,
                data=[],
                confidence=0.0,
                token_usage={'prompt': 0, 'completion': 0},
                execution_time_ms=0,
                error=str(e)
            )


# ============================================================================
# Usage Example and Testing
# ============================================================================

async def main():
    """Example usage of RetrievalAgent"""
    
    # Initialize agent with configuration
    config = {
        'vector_db': {
            'url': 'http://localhost:6333',
            'api_key': 'your-api-key',  # TODO: Add actual API key
            'collection': 'knowledge_base'
        },
        'elasticsearch': {
            'url': 'http://localhost:9200',
            'username': 'elastic',  # TODO: Add credentials
            'password': 'password',
            'index_name': 'knowledge_base'
        },
        'knowledge_graph': {
            'sparql_endpoint': 'http://localhost:8890/sparql',
            'default_graph': 'http://example.org/knowledge'
        },
        'llm_endpoint': 'https://api.openai.com/v1/completions',  # TODO: Configure
        'llm_api_key': 'your-llm-api-key',  # TODO: Add actual API key
        'timeout_seconds': 5,
        'max_retries': 3
    }
    
    agent = RetrievalAgent(config)
    
    # Example 1: Vector search
    print("=== Vector Search Example ===")
    vector_results = await agent.vector_search("quantum computing applications", top_k=10)
    print(f"Found {len(vector_results.documents)} documents in {vector_results.query_time_ms}ms")
    for doc in vector_results.documents[:3]:
        print(f"- Score: {doc.score:.3f}, Source: {doc.source}")
        print(f"  Content: {doc.content[:100]}...")
    
    # Example 2: Keyword search
    print("\n=== Keyword Search Example ===")
    keyword_results = await agent.keyword_search("machine learning healthcare", top_k=10)
    print(f"Found {keyword_results.total_hits} total hits, returned {len(keyword_results.documents)}")
    
    # Example 3: Knowledge graph query
    print("\n=== Knowledge Graph Example ===")
    entities = ["Albert_Einstein", "Theory_of_Relativity"]
    graph_results = await agent.graph_search(entities, top_k=15)
    print(f"Found {len(graph_results.documents)} facts about {entities}")
    
    # Example 4: Hybrid retrieval
    print("\n=== Hybrid Retrieval Example ===")
    hybrid_results = await agent.hybrid_retrieve(
        "What are the latest advances in quantum computing?",
        entities=["Quantum_Computing", "IBM", "Google"]
    )
    print(f"Hybrid search completed in {hybrid_results.query_time_ms}ms")
    print(f"Merged {hybrid_results.metadata['merge_stats']['total_before_merge']} documents")
    print(f"Final result count: {len(hybrid_results.documents)}")
    
    # Display search strategy distribution
    source_dist = {}
    for doc in hybrid_results.documents:
        source_dist[doc.source] = source_dist.get(doc.source, 0) + 1
    print(f"Source distribution: {source_dist}")


if __name__ == "__main__":
    agent = RetrievalAgent()
    results = agent.retrieve("What is the capital of France?")
    for doc in results:
        print(doc.content)
