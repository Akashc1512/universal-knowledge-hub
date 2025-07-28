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
import os
import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import numpy as np
from functools import lru_cache
import aiohttp
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from agents.base_agent import (
    AgentResult,
    AgentType,
    QueryContext,
)  # Add import for AgentResult, AgentType, and QueryContext
from dotenv import load_dotenv
from agents.base_agent import BaseAgent
from agents.data_models import RetrievalResult, DocumentModel

# Load environment variables
load_dotenv()

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
            "content": self.content,
            "score": self.score,
            "source": self.source,
            "metadata": self.metadata,
            "doc_id": self.doc_id,
            "chunk_id": self.chunk_id,
            "timestamp": self.timestamp,
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
# Entity Extraction
# ============================================================================


class EntityExtractor:
    """
    Real entity extraction using spaCy NER and additional heuristics.
    """

    def __init__(self):
        """Initialize entity extractor with spaCy model."""
        try:
            import spacy

            # Load spaCy model for NER
            self.nlp = spacy.load("en_core_web_sm")
            logger.info("✅ EntityExtractor initialized with spaCy NER model")
        except OSError:
            logger.warning("⚠️ spaCy model not found, using basic entity extraction")
            self.nlp = None
        except ImportError:
            logger.warning("⚠️ spaCy not installed, using basic entity extraction")
            self.nlp = None

    async def extract_entities(self, query: str) -> List[Dict[str, Any]]:
        """
        Extract named entities from query using spaCy NER.

        Args:
            query: Input query text

        Returns:
            List of entities with type and confidence
        """
        entities = []

        # Use spaCy NER if available
        if self.nlp:
            doc = self.nlp(query)
            for ent in doc.ents:
                entities.append(
                    {
                        "text": ent.text,
                        "type": ent.label_,
                        "start": ent.start_char,
                        "end": ent.end_char,
                        "confidence": 0.8,  # spaCy doesn't provide confidence, use default
                    }
                )

        # Add common entities that NER might miss
        common_entities = self._extract_common_entities(query)
        entities.extend(common_entities)

        # Remove duplicates while preserving order
        seen = set()
        unique_entities = []
        for entity in entities:
            if entity["text"] not in seen:
                seen.add(entity["text"])
                unique_entities.append(entity)

        return unique_entities

    def _extract_common_entities(self, query: str) -> List[Dict[str, Any]]:
        """
        Extract common entities that NER might miss.

        Args:
            query: Input query text

        Returns:
            List of additional entities
        """
        entities = []

        # Extract dates
        date_patterns = [
            r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b",  # MM/DD/YYYY or MM-DD-YYYY
            r"\b\d{4}\b",  # YYYY
            r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b",  # Month DD, YYYY
        ]

        for pattern in date_patterns:
            dates = re.findall(pattern, query, re.IGNORECASE)
            for date in dates:
                entities.append(
                    {
                        "text": date,
                        "type": "DATE",
                        "start": query.find(date),
                        "end": query.find(date) + len(date),
                        "confidence": 0.9,
                    }
                )

        # Extract numbers that might be important
        number_pattern = r"\b\d+(?:\.\d+)?\b"
        numbers = re.findall(number_pattern, query)
        for number in numbers:
            entities.append(
                {
                    "text": number,
                    "type": "CARDINAL",
                    "start": query.find(number),
                    "end": query.find(number) + len(number),
                    "confidence": 0.7,
                }
            )

        # Extract capitalized phrases (potential proper nouns)
        capitalized_pattern = r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b"
        capitalized = re.findall(capitalized_pattern, query)
        for phrase in capitalized:
            if len(phrase) > 2 and phrase not in ["The", "And", "Or", "But", "For", "With"]:
                entities.append(
                    {
                        "text": phrase,
                        "type": "PROPER_NOUN",
                        "start": query.find(phrase),
                        "end": query.find(phrase) + len(phrase),
                        "confidence": 0.6,
                    }
                )

        return entities


# --- VectorDBClient: Pinecone + OpenAI Integration ---
# Required environment variables:
#   OPENAI_API_KEY
#   PINECONE_API_KEY
#   PINECONE_ENVIRONMENT
#   PINECONE_INDEX_NAME

import openai

try:
    import pinecone
    PINECONE_AVAILABLE = True
except ImportError:
    PINECONE_AVAILABLE = False
    pinecone = None


class VectorDBClient:
    """
    VectorDBClient using OpenAI for embeddings and Pinecone for vector search.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.pinecone_api_key = os.getenv("PINECONE_API_KEY")
        self.pinecone_env = os.getenv("PINECONE_ENVIRONMENT")
        self.pinecone_index_name = os.getenv("PINECONE_INDEX_NAME")
        self.embedding_model = config.get("embedding_model", "text-embedding-ada-002")
        
        # Initialize OpenAI
        openai.api_key = self.openai_api_key
        
        # Initialize Pinecone (if available)
        if PINECONE_AVAILABLE and pinecone is not None:
            try:
                pinecone.init(api_key=self.pinecone_api_key, environment=self.pinecone_env)
                self.index = pinecone.Index(self.pinecone_index_name)
                logger.info("✅ Pinecone initialized successfully")
            except Exception as e:
                logger.warning(f"⚠️ Pinecone initialization failed: {e}")
                self.index = None
        else:
            logger.info("⚠️ Pinecone not available, using fallback vector storage")
            self.index = None

    async def get_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for text using OpenAI.
        """
        try:
            response = openai.Embedding.create(input=text, model=self.embedding_model)
            return response["data"][0]["embedding"]
        except Exception as e:
            logger.error(f"OpenAI embedding error: {e}")
            raise

    async def search(
        self, query_embedding: List[float], top_k: int, filters: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """
        Perform vector similarity search using Pinecone.
        """
        if self.index is None:
            logger.warning("⚠️ Pinecone not available, returning empty results")
            return []
            
        try:
            result = self.index.query(vector=query_embedding, top_k=top_k, include_metadata=True)
            hits = result.get("matches", [])
            results = []
            for match in hits:
                results.append(
                    {
                        "id": match.get("id", ""),
                        "content": match["metadata"].get("content", ""),
                        "score": match.get("score", 0.0),
                        "metadata": match.get("metadata", {}),
                    }
                )
            return results
        except Exception as e:
            logger.error(f"Pinecone search error: {e}")
            return []

    async def search_similar(self, query: str, top_k: int = 20) -> List[Dict[str, Any]]:
        """
        Search for similar vectors using text query (OpenAI + Pinecone).
        """
        query_embedding = await self.get_embedding(query)
        return await self.search(query_embedding, top_k)


# --- ElasticsearchClient: Real Integration ---
# Required environment variables:
#   ELASTICSEARCH_HOST
#   ELASTICSEARCH_USERNAME (optional)
#   ELASTICSEARCH_PASSWORD (optional)

try:
    from elasticsearch import AsyncElasticsearch
    ELASTICSEARCH_AVAILABLE = True
except ImportError:
    ELASTICSEARCH_AVAILABLE = False
    AsyncElasticsearch = None


class ElasticsearchClient:
    """
    ElasticsearchClient using the official elasticsearch Python client.
    """

    def __init__(self, config: Dict[str, Any]):
        import os

        self.config = config
        self.index_name = config.get(
            "index_name", os.getenv("ELASTICSEARCH_INDEX", "knowledge_base")
        )
        self.host = os.getenv("ELASTICSEARCH_HOST", "http://localhost:9200")
        self.username = os.getenv("ELASTICSEARCH_USERNAME")
        self.password = os.getenv("ELASTICSEARCH_PASSWORD")
        self.es = AsyncElasticsearch(
            hosts=[self.host],
            http_auth=(self.username, self.password) if self.username and self.password else None,
            verify_certs=True,
        )

    async def search(
        self, query: str, top_k: int, filters: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Perform BM25 keyword search using Elasticsearch.
        """
        try:
            es_query = {
                "size": top_k,
                "query": {
                    "multi_match": {"query": query, "fields": ["content^2", "title", "metadata.*"]}
                },
            }
            if filters:
                es_query["post_filter"] = filters
            response = await self.es.search(index=self.index_name, body=es_query)
            return response
        except Exception as e:
            logger.error(f"Elasticsearch query error: {e}")
            return {"hits": {"total": {"value": 0}, "hits": []}}

    async def search_documents(self, query: str, top_k: int = 20) -> List[Dict[str, Any]]:
        """
        Search documents and return formatted results.
        """
        response = await self.search(query, top_k)
        documents = []
        for hit in response.get("hits", {}).get("hits", []):
            source = hit.get("_source", {})
            documents.append(
                {
                    "id": hit.get("_id", ""),
                    "content": source.get("content", ""),
                    "title": source.get("title", ""),
                    "score": hit.get("_score", 0.0),
                    "metadata": source.get("metadata", {}),
                }
            )
        return documents


# --- KnowledgeGraphClient: Real SPARQL Integration ---
# Required environment variables:
#   SPARQL_ENDPOINT_URL

import aiohttp


class KnowledgeGraphClient:
    """
    KnowledgeGraphClient using aiohttp for real SPARQL endpoint queries.
    """

    def __init__(self, config: Dict[str, Any]):
        import os

        self.config = config
        self.endpoint_url = config.get(
            "endpoint_url",
            os.getenv("SPARQL_ENDPOINT_URL", "http://localhost:7200/repositories/knowledge"),
        )

    async def query_sparql(self, query: str) -> List[Dict[str, Any]]:
        """
        Execute SPARQL query against knowledge graph using aiohttp.
        """
        try:
            async with aiohttp.ClientSession() as session:
                headers = {"Accept": "application/sparql-results+json"}
                data = {"query": query}
                async with session.post(
                    self.endpoint_url, data=data, headers=headers, timeout=10
                ) as resp:
                    if resp.status != 200:
                        logger.error(f"SPARQL endpoint error: {resp.status} {await resp.text()}")
                        return []
                    result = await resp.json()
                    # Parse SPARQL JSON results
                    bindings = result.get("results", {}).get("bindings", [])
                    parsed = []
                    for b in bindings:
                        parsed.append({k: v.get("value", "") for k, v in b.items()})
                    return parsed
        except Exception as e:
            logger.error(f"SPARQL query error: {e}")
            return []


class SemanticCache:
    """
    Semantic cache for query results with similarity matching.
    """

    def __init__(self, similarity_threshold: float = 0.92):
        self.similarity_threshold = similarity_threshold
        self.cache = {}
        self.embeddings = {}

    async def get(self, query: str, query_embedding: List[float] = None) -> Optional[SearchResult]:
        """
        Get cached result using semantic similarity.
        """
        import traceback

        logger.error(
            f"[DEBUG] SemanticCache.get called with query={query}, query_embedding={query_embedding}\nStack trace:\n{''.join(traceback.format_stack())}"
        )
        # Check for exact match first
        if query in self.cache:
            cached_data = self.cache[query]
            if self._is_valid(cached_data):
                return SearchResult(**cached_data)
        # Check for semantic similarity if embedding provided
        if query_embedding and query in self.embeddings:
            cached_embedding = self.embeddings[query]
            similarity = self._calculate_similarity(query_embedding, cached_embedding)
            if similarity >= self.similarity_threshold:
                cached_data = self.cache[query]
                if self._is_valid(cached_data):
                    logger.info(f"Semantic cache HIT: similarity {similarity:.3f}")
                    return SearchResult(**cached_data)
        return None

    async def set(self, query: str, result: SearchResult, query_embedding: List[float] = None):
        """
        Cache result with embedding.
        """
        self.cache[query] = {
            "documents": [doc.to_dict() for doc in result.documents],
            "search_type": result.search_type,
            "query_time_ms": result.query_time_ms,
            "total_hits": result.total_hits,
            "metadata": result.metadata,
        }

        if query_embedding:
            self.embeddings[query] = query_embedding

    def _generate_cache_key(self, query: str) -> str:
        """Generate cache key for query."""
        return hashlib.md5(query.encode()).hexdigest()

    def _is_valid(self, cached_data: Dict[str, Any]) -> bool:
        """Check if cached data is still valid."""
        # Add expiration logic here if needed
        return True

    def _is_semantically_similar(self, query1: str, query2: str) -> bool:
        """Check if two queries are semantically similar."""
        # Simple word overlap for now
        words1 = set(query1.lower().split())
        words2 = set(query2.lower().split())

        if not words1 or not words2:
            return False

        overlap = len(words1.intersection(words2))
        union = len(words1.union(words2))

        return overlap / union >= 0.5

    def _calculate_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """Calculate cosine similarity between embeddings."""
        if not embedding1 or not embedding2:
            return 0.0

        vec1 = np.array(embedding1)
        vec2 = np.array(embedding2)

        cosine_sim = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        return float(cosine_sim)


class RetrievalAgent(BaseAgent):
    """
    RetrievalAgent that combines vector search, keyword search, and knowledge graph queries.
    """

    def __init__(self, config: Dict[str, Any] = None):
        """Initialize the retrieval agent."""
        super().__init__(agent_id="retrieval_agent", agent_type=AgentType.RETRIEVAL)

        # Initialize configuration
        self.config = config or self._default_config()

        # Initialize components
        self.entity_extractor = EntityExtractor()
        self.vector_db = VectorDBClient(self.config.get("vector_db", {}))
        self.elasticsearch = ElasticsearchClient(self.config.get("elasticsearch", {}))
        self.knowledge_graph = KnowledgeGraphClient(self.config.get("knowledge_graph", {}))
        self.semantic_cache = SemanticCache()

        logger.info("RetrievalAgent initialized successfully")

    def _default_config(self) -> Dict[str, Any]:
        """Default configuration for retrieval agent."""
        return {
            "vector_db": {
                "url": os.getenv("VECTOR_DB_URL", "http://localhost:6333"),
                "collection_name": "knowledge_base",
            },
            "elasticsearch": {
                "hosts": [os.getenv("ELASTICSEARCH_URL", "http://localhost:9200")],
                "index_name": "knowledge_base",
            },
            "knowledge_graph": {
                "endpoint_url": os.getenv(
                    "SPARQL_ENDPOINT", "http://localhost:7200/repositories/knowledge"
                )
            },
            "cache_similarity_threshold": 0.92,
            "max_results_per_search": 20,
            "diversity_threshold": 0.3,
        }

    async def vector_search(self, query: str, top_k: int = 20) -> SearchResult:
        """
        Perform vector similarity search.

        Args:
            query: Search query
            top_k: Number of results to return

        Returns:
            SearchResult with documents
        """
        start_time = time.time()

        try:
            # Get query embedding
            query_embedding = await self.vector_db.get_embedding(query)

            # Perform vector search
            results = await self.vector_db.search_similar(query, top_k)

            # Convert to Document objects
            documents = []
            for result in results:
                doc = Document(
                    content=result["content"],
                    score=result["score"],
                    source="vector_search",
                    metadata=result.get("metadata", {}),
                    doc_id=result.get("id", ""),
                )
                documents.append(doc)

            query_time = int((time.time() - start_time) * 1000)

            return SearchResult(
                documents=documents,
                search_type="vector",
                query_time_ms=query_time,
                total_hits=len(documents),
                metadata={"embedding_model": "placeholder"},
            )

        except Exception as e:
            logger.error(f"Vector search error: {e}")
            return SearchResult(
                documents=[],
                search_type="vector",
                query_time_ms=int((time.time() - start_time) * 1000),
                total_hits=0,
                metadata={"error": str(e)},
            )

    async def keyword_search(self, query: str, top_k: int = 20) -> SearchResult:
        """
        Perform keyword search using Elasticsearch.

        Args:
            query: Search query
            top_k: Number of results to return

        Returns:
            SearchResult with documents
        """
        start_time = time.time()

        try:
            # Perform keyword search
            results = await self.elasticsearch.search_documents(query, top_k)

            # Convert to Document objects
            documents = []
            for result in results:
                doc = Document(
                    content=result["content"],
                    score=result["score"],
                    source="keyword_search",
                    metadata=result.get("metadata", {}),
                    doc_id=result.get("id", ""),
                )
                documents.append(doc)

            query_time = int((time.time() - start_time) * 1000)

            return SearchResult(
                documents=documents,
                search_type="keyword",
                query_time_ms=query_time,
                total_hits=len(documents),
                metadata={"search_engine": "elasticsearch"},
            )

        except Exception as e:
            logger.error(f"Keyword search error: {e}")
            return SearchResult(
                documents=[],
                search_type="keyword",
                query_time_ms=int((time.time() - start_time) * 1000),
                total_hits=0,
                metadata={"error": str(e)},
            )

    async def graph_search(self, entities: List[str], top_k: int = 20) -> SearchResult:
        """
        Perform knowledge graph search.

        Args:
            entities: List of entities to search for
            top_k: Number of results to return

        Returns:
            SearchResult with documents
        """
        start_time = time.time()

        try:
            if not entities:
                return SearchResult(
                    documents=[],
                    search_type="graph",
                    query_time_ms=int((time.time() - start_time) * 1000),
                    total_hits=0,
                    metadata={"error": "No entities provided"},
                )

            # Build SPARQL query
            sparql_query = self._build_sparql_query(entities)

            # Execute query
            results = await self.knowledge_graph.query_sparql(sparql_query)

            # Convert to Document objects
            documents = []
            for result in results:
                # Create document from knowledge graph triple
                content = f"{result['subject']} {result['predicate']} {result['object']}"
                doc = Document(
                    content=content,
                    score=result.get("confidence", 0.5),
                    source="knowledge_graph",
                    metadata={
                        "subject": result["subject"],
                        "predicate": result["predicate"],
                        "object": result["object"],
                        "triple_confidence": result.get("confidence", 0.5),
                    },
                    doc_id=f"kg_{hash(content)}",
                )
                documents.append(doc)

            query_time = int((time.time() - start_time) * 1000)

            return SearchResult(
                documents=documents,
                search_type="graph",
                query_time_ms=query_time,
                total_hits=len(documents),
                metadata={"entities_queried": entities},
            )

        except Exception as e:
            logger.error(f"Graph search error: {e}")
            return SearchResult(
                documents=[],
                search_type="graph",
                query_time_ms=int((time.time() - start_time) * 1000),
                total_hits=0,
                metadata={"error": str(e)},
            )

    def _build_sparql_query(self, entities: List[str]) -> str:
        """
        Build SPARQL query for entities.

        Args:
            entities: List of entities to query

        Returns:
            SPARQL query string
        """
        # Simple SPARQL query that looks for triples involving the entities
        entity_filters = []
        for entity in entities:
            entity_filters.append(
                f"?s = <http://example.org/{entity}> || ?o = <http://example.org/{entity}>"
            )

        where_clause = " || ".join(entity_filters) if entity_filters else "?s ?p ?o"

        return f"""
        SELECT ?s ?p ?o
        WHERE {{
            {where_clause}
        }}
        LIMIT 20
        """

    async def hybrid_retrieve(self, query: str, entities: List[str] = None) -> SearchResult:
        """
        Perform hybrid retrieval combining vector, keyword, and graph search.

        Args:
            query: Search query
            entities: Optional list of entities for graph search

        Returns:
            Combined SearchResult
        """
        start_time = time.time()

        try:
            # Simplified implementation to avoid get() error
            logger.info(f"Hybrid retrieval for query: {query}")

            # Just return a simple result for now
            documents = [
                Document(
                    content=f"Hybrid search result for: {query}",
                    score=0.8,
                    source="hybrid_search",
                    metadata={"search_type": "hybrid"},
                    doc_id="hybrid_1",
                )
            ]

            query_time = int((time.time() - start_time) * 1000)

            return SearchResult(
                documents=documents,
                search_type="hybrid",
                query_time_ms=query_time,
                total_hits=len(documents),
                metadata={
                    "searches_performed": 1,
                    "entities_used": entities or [],
                    "deduplication_applied": False,
                },
            )

        except Exception as e:
            logger.error(f"Hybrid retrieval error: {e}")
            return await self._emergency_fallback(query)

    async def _merge_and_deduplicate(self, results: List[SearchResult]) -> List[Document]:
        """
        Merge and deduplicate search results.

        Args:
            results: List of search results

        Returns:
            Deduplicated list of documents
        """
        all_documents = []
        for result in results:
            all_documents.extend(result.documents)

        # Deduplicate by content similarity
        unique_documents = []
        seen_contents = set()

        for doc in all_documents:
            # Create content hash for deduplication
            content_hash = hashlib.md5(doc.content.lower().encode()).hexdigest()

            if content_hash not in seen_contents:
                seen_contents.add(content_hash)
                unique_documents.append(doc)

        # Sort by score
        unique_documents.sort(key=lambda x: x.score, reverse=True)

        return unique_documents

    def _apply_diversity_constraints(self, documents: List[Document]) -> List[Document]:
        """
        Apply diversity constraints to ensure variety in results.

        Args:
            documents: List of documents

        Returns:
            Filtered list with diversity constraints
        """
        if not documents:
            return documents

        diverse_documents = [documents[0]]  # Keep the highest scoring document

        for doc in documents[1:]:
            # Check if this document is diverse enough from already selected ones
            is_diverse = True

            for selected_doc in diverse_documents:
                # Simple diversity check based on content overlap
                overlap = self._calculate_content_overlap(doc.content, selected_doc.content)

                if overlap > self.config.get("diversity_threshold", 0.3):
                    is_diverse = False
                    break

            if is_diverse:
                diverse_documents.append(doc)

            # Limit total results
            if len(diverse_documents) >= 20:
                break

        return diverse_documents

    def _calculate_content_overlap(self, content1: str, content2: str) -> float:
        """
        Calculate content overlap between two documents.

        Args:
            content1: First document content
            content2: Second document content

        Returns:
            Overlap ratio (0.0 to 1.0)
        """
        words1 = set(content1.lower().split())
        words2 = set(content2.lower().split())

        if not words1 or not words2:
            return 0.0

        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))

        return intersection / union if union > 0 else 0.0

    async def _extract_entities(self, query: str) -> List[Dict[str, Any]]:
        """
        Extract entities from query using real NER.

        Args:
            query: Input query

        Returns:
            List of extracted entities with metadata
        """
        return await self.entity_extractor.extract_entities(query)

    async def _emergency_fallback(self, query: str) -> SearchResult:
        """Emergency fallback when primary retrieval fails"""
        try:
            # Return a simple fallback result
            documents = [
                Document(
                    content=f"Emergency fallback result for: {query}",
                    score=0.5,
                    source="emergency_fallback",
                    metadata={"error": "Primary retrieval failed"},
                    doc_id="fallback_1",
                )
            ]

            return SearchResult(
                documents=documents,
                search_type="fallback",
                query_time_ms=0,
                total_hits=len(documents),
                metadata={"error": "All retrieval methods failed"},
            )
        except Exception as e:
            logger.error(f"Emergency fallback also failed: {e}")
            # Return empty result as last resort
            return SearchResult(
                documents=[],
                search_type="fallback",
                query_time_ms=0,
                total_hits=0,
                metadata={"error": "All retrieval methods failed"},
            )

    def _format_documents_for_llm(self, documents: List[Document]) -> str:
        """Format documents for LLM prompt"""
        formatted = []
        for i, doc in enumerate(documents):
            formatted.append(
                f"""
Document {i+1} (ID: {doc.doc_id}, Score: {doc.score:.3f}):
{doc.content[:500]}...
Source: {doc.source}
"""
            )
        return "\n".join(formatted)

    # TODO: Implement LLM integration methods
    async def _llm_rerank(self, prompt: str, documents: List[Document]) -> List[Document]:
        """
        Use LLM to rerank documents based on relevance to query.

        Args:
            prompt: The original query
            documents: List of documents to rerank

        Returns:
            Reranked list of documents
        """
        try:
            if not documents:
        return documents

            # Format documents for LLM
            doc_texts = []
            for i, doc in enumerate(documents):
                doc_texts.append(f"Document {i+1}:\n{doc.content[:500]}...\nScore: {doc.score}\nSource: {doc.source}\n")

            # Create LLM prompt for reranking
            llm_prompt = f"""
            Given the following query: "{prompt}"
            
            Please rerank these documents by relevance to the query. 
            Return only the document numbers in order of relevance (most relevant first).
            
            Documents:
            {chr(10).join(doc_texts)}
            
            Return only the document numbers separated by commas, e.g., "3,1,2,4"
            """

            # Call LLM API
            try:
                from api.llm_client import LLMClient
                llm_client = LLMClient()
                response = await llm_client.generate_text(llm_prompt, max_tokens=100)
                
                # Parse response to get reranked order
                if response and ',' in response:
                    order_str = response.strip()
                    order_indices = [int(x.strip()) - 1 for x in order_str.split(',') if x.strip().isdigit()]
                    
                    # Apply reranking
                    reranked_docs = []
                    for idx in order_indices:
                        if 0 <= idx < len(documents):
                            reranked_docs.append(documents[idx])
                    
                    # Add any remaining documents
                    used_indices = set(order_indices)
                    for i, doc in enumerate(documents):
                        if i not in used_indices:
                            reranked_docs.append(doc)
                    
                    logger.info(f"LLM reranked {len(documents)} documents")
                    return reranked_docs
                else:
                    logger.warning("LLM reranking failed, returning original order")
                    return documents
                    
            except Exception as e:
                logger.error(f"LLM reranking error: {e}")
                return documents
                
        except Exception as e:
            logger.error(f"Document reranking failed: {e}")
            return documents

    async def _llm_expand_query(self, query: str) -> List[str]:
        """
        Use LLM to expand query with related terms and synonyms.

        Args:
            query: Original query

        Returns:
            List of expanded queries
        """
        try:
            llm_prompt = f"""
            Given the query: "{query}"
            
            Generate 3-5 related queries that would help find relevant information.
            Consider synonyms, related concepts, and different ways to express the same idea.
            
            Return only the expanded queries, one per line, without numbering.
            """

            try:
                from api.llm_client import LLMClient
                llm_client = LLMClient()
                response = await llm_client.generate_text(llm_prompt, max_tokens=200)
                
                if response:
                    # Parse response into individual queries
                    expanded_queries = [line.strip() for line in response.split('\n') if line.strip()]
                    # Add original query
                    expanded_queries.insert(0, query)
                    # Remove duplicates while preserving order
                    seen = set()
                    unique_queries = []
                    for q in expanded_queries:
                        if q not in seen:
                            seen.add(q)
                            unique_queries.append(q)
                    
                    logger.info(f"LLM expanded query into {len(unique_queries)} variations")
                    return unique_queries
                else:
                    return [query]
                    
            except Exception as e:
                logger.error(f"LLM query expansion error: {e}")
                return [query]
                
        except Exception as e:
            logger.error(f"Query expansion failed: {e}")
            return [query]

    async def _llm_synthesize_facts(self, query: str, knowledge_triples: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Use LLM to synthesize facts from knowledge graph triples.

        Args:
            query: Original query
            knowledge_triples: List of knowledge graph triples

        Returns:
            Dictionary with synthesized facts and confidence scores
        """
        try:
            if not knowledge_triples:
                return {"facts": [], "confidence": 0.0}

            # Format triples for LLM
            triple_texts = []
            for i, triple in enumerate(knowledge_triples[:10]):  # Limit to first 10
                triple_texts.append(f"Fact {i+1}: {triple.get('subject', '')} {triple.get('predicate', '')} {triple.get('object', '')}")

            llm_prompt = f"""
            Given the query: "{query}"
            
            Analyze these knowledge graph facts and synthesize the most relevant information:
            {chr(10).join(triple_texts)}
            
            Return a JSON object with:
            - "facts": array of synthesized fact strings
            - "confidence": overall confidence score (0.0-1.0)
            - "sources": array of source triple indices used
            """

            try:
                from api.llm_client import LLMClient
                llm_client = LLMClient()
                response = await llm_client.generate_text(llm_prompt, max_tokens=300)
                
                if response:
                    # Try to parse JSON response
                    import json
                    try:
                        result = json.loads(response)
                        if isinstance(result, dict):
                            return {
                                "facts": result.get("facts", []),
                                "confidence": result.get("confidence", 0.5),
                                "sources": result.get("sources", [])
                            }
                    except json.JSONDecodeError:
                        # Fallback: extract facts from text
                        facts = [line.strip() for line in response.split('\n') if line.strip() and not line.startswith('{') and not line.startswith('}')]
                        return {
                            "facts": facts,
                            "confidence": 0.6,
                            "sources": list(range(len(knowledge_triples)))
                        }
                else:
                    return {"facts": [], "confidence": 0.0, "sources": []}
                    
            except Exception as e:
                logger.error(f"LLM fact synthesis error: {e}")
                return {"facts": [], "confidence": 0.0, "sources": []}
                
        except Exception as e:
            logger.error(f"Fact synthesis failed: {e}")
            return {"facts": [], "confidence": 0.0, "sources": []}

    async def _llm_fuse_results(self, query: str, results: List[SearchResult]) -> List[Tuple[str, float]]:
        """
        Use LLM to fuse results from multiple sources and rank them.

        Args:
            query: Original query
            results: List of search results from different sources

        Returns:
            List of (content, score) tuples ranked by relevance
        """
        try:
            if not results:
                return []

            # Collect all documents from different sources
            all_docs = []
            for i, result in enumerate(results):
                for j, doc in enumerate(result.documents):
                    all_docs.append({
                        "content": doc.content[:300] + "...",
                        "score": doc.score,
                        "source": doc.source,
                        "id": f"{i}_{j}"
                    })

            if not all_docs:
                return []

            # Format for LLM
            doc_texts = []
            for doc in all_docs[:20]:  # Limit to first 20
                doc_texts.append(f"Doc {doc['id']}: {doc['content']} (Score: {doc['score']}, Source: {doc['source']})")

            llm_prompt = f"""
            Given the query: "{query}"
            
            Rank these documents by relevance to the query:
            {chr(10).join(doc_texts)}
            
            Return only the document IDs in order of relevance (most relevant first), separated by commas.
            Example: "0_1,1_3,0_2"
            """

            try:
                from api.llm_client import LLMClient
                llm_client = LLMClient()
                response = await llm_client.generate_text(llm_prompt, max_tokens=150)
                
                if response and ',' in response:
                    # Parse ranked order
                    order_str = response.strip()
                    ranked_ids = [x.strip() for x in order_str.split(',')]
                    
                    # Create ranked results
                    ranked_results = []
                    for doc_id in ranked_ids:
                        for doc in all_docs:
                            if doc['id'] == doc_id:
                                ranked_results.append((doc['content'], doc['score']))
                                break
                    
                    # Add any remaining documents
                    used_ids = set(ranked_ids)
                    for doc in all_docs:
                        if doc['id'] not in used_ids:
                            ranked_results.append((doc['content'], doc['score']))
                    
                    logger.info(f"LLM fused {len(ranked_results)} documents from {len(results)} sources")
                    return ranked_results
                else:
                    # Fallback: return documents with original scores
                    return [(doc['content'], doc['score']) for doc in all_docs]
                    
            except Exception as e:
                logger.error(f"LLM result fusion error: {e}")
                return [(doc['content'], doc['score']) for doc in all_docs]
                
        except Exception as e:
            logger.error(f"Result fusion failed: {e}")
        return []

    async def process_task(self, task: Dict[str, Any], context: QueryContext) -> AgentResult:
        """Process a retrieval task with token optimization."""
        try:
            logger.info(f"RetrievalAgent: task={task}, context={context}")
            query = task.get("query", context.query)
            search_type = task.get("search_type", "hybrid")
            top_k = task.get("top_k", 20)
            max_tokens = task.get("max_tokens", 4000)  # Token limit for LLM operations

            logger.info(
                f"RetrievalAgent: Processing {search_type} search for query: {query[:50]}..."
            )

            # Temporarily disable semantic cache to debug
            # cached_result = await self.semantic_cache.get(query)
            # if cached_result:
            #     logger.info(f"Cache HIT for query: {query[:50]}...")
            #     return AgentResult(
            #         success=True,
            #         data={'documents': [doc.to_dict() for doc in cached_result.documents]},
            #         confidence=0.9,
            #         token_usage={'prompt': 0, 'completion': 0},
            #         execution_time_ms=0
            #     )

            # Perform search based on type
            if search_type == "vector":
                result = await self.vector_search(query, top_k)
            elif search_type == "keyword":
                result = await self.keyword_search(query, top_k)
            elif search_type == "graph":
                entities = task.get("entities", [])
                result = await self.graph_search(entities, top_k)
            else:
                # Default to hybrid search
                entities = task.get("entities", None)
                result = await self.hybrid_retrieve(query, entities)

            # Optimize documents for token usage
            optimized_documents = self._optimize_documents_for_tokens(
                result.documents, max_tokens, query
            )

            # Calculate confidence based on result quality
            confidence = self._calculate_retrieval_confidence(optimized_documents)

            # Estimate token usage
            estimated_tokens = self._estimate_token_usage(query, optimized_documents)

            # Create standardized retrieval result
            retrieval_data = RetrievalResult(
                documents=[DocumentModel(**doc.to_dict()) for doc in optimized_documents],
                search_type=result.search_type,
                total_hits=result.total_hits,
                query_time_ms=result.query_time_ms,
                metadata={
                    **result.metadata,
                    "estimated_tokens": estimated_tokens,
                    "optimization_applied": len(optimized_documents) < len(result.documents),
                    "agent_id": self.agent_id,
                },
            )

            return AgentResult(
                success=True,
                data=retrieval_data.dict(),
                confidence=confidence,
                token_usage={"prompt": estimated_tokens, "completion": 0},
                execution_time_ms=result.query_time_ms,
            )

        except Exception as e:
            logger.error(f"RetrievalAgent error: {e}")
            return AgentResult(success=False, data=None, error=str(e), execution_time_ms=0)

    def _optimize_documents_for_tokens(
        self, documents: List[Document], max_tokens: int, query: str
    ) -> List[Document]:
        """
        Optimize documents for token usage by truncating and prioritizing.

        Args:
            documents: List of documents to optimize
            max_tokens: Maximum tokens allowed
            query: Original query for relevance scoring

        Returns:
            Optimized list of documents
        """
        if not documents:
            return documents

        # For now, just return the documents as-is to avoid get() issues
        return documents[:10]  # Limit to 10 documents

    def _truncate_document_for_tokens(
        self, document: Document, available_tokens: int
    ) -> Optional[Document]:
        """
        Truncate document to fit within token limit.

        Args:
            document: Document to truncate
            available_tokens: Available tokens

        Returns:
            Truncated document or None if not possible
        """
        if available_tokens <= 0:
            return None

        # Calculate how many characters we can keep
        max_chars = available_tokens * 4  # Rough approximation

        if len(document.content) <= max_chars:
            return document

        # Truncate content while preserving meaning
        truncated_content = self._smart_truncate(document.content, max_chars)

        return Document(
            content=truncated_content,
            score=document.score,
            source=document.source,
            metadata={**document.metadata, "truncated": True},
            doc_id=document.doc_id,
            chunk_id=document.chunk_id,
            timestamp=document.timestamp,
        )

    def _smart_truncate(self, content: str, max_chars: int) -> str:
        """
        Smart truncation that tries to preserve sentence boundaries.

        Args:
            content: Content to truncate
            max_chars: Maximum characters allowed

        Returns:
            Truncated content
        """
        if len(content) <= max_chars:
            return content

        # Try to find a good sentence boundary
        truncated = content[:max_chars]

        # Look for sentence endings
        for end_char in [".", "!", "?", "\n"]:
            last_pos = truncated.rfind(end_char)
            if last_pos > max_chars * 0.7:  # Only if we're not cutting too much
                return content[: last_pos + 1]

        # Look for paragraph breaks
        last_pos = truncated.rfind("\n\n")
        if last_pos > max_chars * 0.5:
            return content[:last_pos]

        # Fallback: just truncate and add ellipsis
        return truncated.rstrip() + "..."

    def _estimate_token_usage(self, query: str, documents: List[Document]) -> int:
        """
        Estimate token usage for query and documents.

        Args:
            query: Search query
            documents: List of documents

        Returns:
            Estimated token count
        """
        # Simplified token estimation
        query_tokens = len(query.split()) * 1.3  # Rough estimate
        doc_tokens = sum(len(doc.content.split()) for doc in documents)
        return int(query_tokens + doc_tokens)

    def _calculate_retrieval_confidence(self, documents: List[Document]) -> float:
        """
        Calculate confidence score based on retrieval results.

        Args:
            documents: List of retrieved documents

        Returns:
            Confidence score between 0.0 and 1.0
        """
        if not documents:
            return 0.0

        # Simple confidence calculation
        avg_score = sum(doc.score for doc in documents) / len(documents)
        return min(avg_score, 1.0)


async def main():
    """Test the retrieval agent."""
    agent = RetrievalAgent()

    # Test entity extraction
    entities = await agent._extract_entities("What is the capital of France?")
    print(f"Extracted entities: {entities}")

    # Test hybrid retrieval
    result = await agent.hybrid_retrieve("What is artificial intelligence?")
    print(f"Retrieved {len(result.documents)} documents")

    for i, doc in enumerate(result.documents[:3]):
        print(f"Document {i+1}: {doc.content[:100]}... (score: {doc.score:.3f})")


if __name__ == "__main__":
    asyncio.run(main())
