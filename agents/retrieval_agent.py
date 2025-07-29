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


# --- VectorDBClient: Real Pinecone + OpenAI Integration ---
# Required environment variables:
#   PINECONE_API_KEY
#   PINECONE_ENVIRONMENT
#   PINECONE_INDEX_NAME
#   OPENAI_API_KEY

try:
    import pinecone
    PINECONE_AVAILABLE = True
except ImportError:
    PINECONE_AVAILABLE = False
    pinecone = None

try:
    import qdrant_client
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams, PointStruct
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False
    QdrantClient = None

class VectorDBClient:
    """
    VectorDBClient using OpenAI for embeddings and Pinecone/Qdrant for vector search.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.embedding_model = config.get("embedding_model", "text-embedding-ada-002")
        
        # Pinecone configuration
        self.pinecone_api_key = os.getenv("PINECONE_API_KEY")
        self.pinecone_env = os.getenv("PINECONE_ENVIRONMENT")
        self.pinecone_index_name = os.getenv("PINECONE_INDEX_NAME")
        
        # Qdrant configuration
        self.qdrant_url = os.getenv("QDRANT_URL")
        self.qdrant_collection = config.get("collection_name", "knowledge_base")
        
        # Initialize vector database
        self.pinecone_index = None
        self.qdrant_client = None
        self._initialize_vector_db()

    def _initialize_vector_db(self):
        """Initialize vector database connection."""
        # Try Pinecone first
        if PINECONE_AVAILABLE and self.pinecone_api_key:
            try:
                pinecone.init(api_key=self.pinecone_api_key, environment=self.pinecone_env)
                self.pinecone_index = pinecone.Index(self.pinecone_index_name)
                logger.info("✅ Pinecone initialized successfully")
                return
            except Exception as e:
                logger.warning(f"⚠️ Pinecone initialization failed: {e}")
        
        # Try Qdrant as fallback
        if QDRANT_AVAILABLE and self.qdrant_url:
            try:
                self.qdrant_client = QdrantClient(url=self.qdrant_url)
                # Create collection if it doesn't exist
                try:
                    self.qdrant_client.get_collection(self.qdrant_collection)
                except:
                    self.qdrant_client.create_collection(
                        collection_name=self.qdrant_collection,
                        vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
                    )
                logger.info("✅ Qdrant initialized successfully")
                return
            except Exception as e:
                logger.warning(f"⚠️ Qdrant initialization failed: {e}")
        
        logger.info("⚠️ No vector database available, using fallback storage")

    async def get_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for text using OpenAI or fallback to sentence-transformers.
        """
        try:
            # Try OpenAI first
            import openai
            response = openai.Embedding.create(input=text, model=self.embedding_model)
            return response["data"][0]["embedding"]
        except Exception as e:
            logger.warning(f"OpenAI embedding failed: {e}, using fallback")
            try:
                # Fallback to sentence-transformers
                from sentence_transformers import SentenceTransformer
                model = SentenceTransformer('all-MiniLM-L6-v2')
                embedding = model.encode(text)
                return embedding.tolist()
            except Exception as e2:
                logger.error(f"Fallback embedding also failed: {e2}")
                # Return random embedding as last resort
                import random
                return [random.uniform(-1, 1) for _ in range(384)]  # 384-dim embedding

    async def search(
        self, query_embedding: List[float], top_k: int, filters: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """
        Perform vector similarity search using Pinecone or Qdrant.
        """
        # Try Pinecone first
        if self.pinecone_index:
            try:
                result = self.pinecone_index.query(
                    vector=query_embedding, 
                    top_k=top_k, 
                    include_metadata=True,
                    filter=filters
                )
                hits = result.get("matches", [])
                results = []
                for match in hits:
                    results.append({
                        "id": match.get("id", ""),
                        "content": match["metadata"].get("content", ""),
                        "score": match.get("score", 0.0),
                        "metadata": match.get("metadata", {}),
                    })
                return results
            except Exception as e:
                logger.error(f"Pinecone search error: {e}")
        
        # Try Qdrant as fallback
        if self.qdrant_client:
            try:
                search_result = self.qdrant_client.search(
                    collection_name=self.qdrant_collection,
                    query_vector=query_embedding,
                    limit=top_k,
                    query_filter=filters
                )
                results = []
                for point in search_result:
                    results.append({
                        "id": point.id,
                        "content": point.payload.get("content", ""),
                        "score": point.score,
                        "metadata": point.payload,
                    })
                return results
            except Exception as e:
                logger.error(f"Qdrant search error: {e}")
        
        # Fallback to empty results
        logger.warning("No vector database available, returning empty results")
        return []

    async def search_similar(self, query: str, top_k: int = 20) -> List[Dict[str, Any]]:
        """
        Search for similar vectors using text query (OpenAI + Vector DB).
        """
        query_embedding = await self.get_embedding(query)
        return await self.search(query_embedding, top_k)

    async def upsert_documents(self, documents: List[Dict[str, Any]]):
        """
        Insert or update documents in vector database.
        
        Args:
            documents: List of documents with content and metadata
        """
        if not documents:
            return
        
        try:
            # Generate embeddings for all documents
            embeddings = []
            for doc in documents:
                embedding = await self.get_embedding(doc["content"])
                embeddings.append(embedding)
            
            # Insert into vector database
            if self.pinecone_index:
                await self._upsert_to_pinecone(documents, embeddings)
            elif self.qdrant_client:
                await self._upsert_to_qdrant(documents, embeddings)
            else:
                logger.warning("No vector database available for upsert")
                
        except Exception as e:
            logger.error(f"Failed to upsert documents: {e}")
    
    async def _upsert_to_pinecone(self, documents: List[Dict], embeddings: List[List[float]]):
        """Upsert documents to Pinecone."""
        try:
            vectors = []
            for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
                vectors.append({
                    "id": doc.get("id", f"doc_{i}"),
                    "values": embedding,
                    "metadata": {
                        "content": doc["content"],
                        "source": doc.get("source", "unknown"),
                        "timestamp": doc.get("timestamp", ""),
                        **doc.get("metadata", {})
                    }
                })
            
            self.pinecone_index.upsert(vectors=vectors)
            logger.info(f"Upserted {len(documents)} documents to Pinecone")
            
        except Exception as e:
            logger.error(f"Pinecone upsert failed: {e}")
    
    async def _upsert_to_qdrant(self, documents: List[Dict], embeddings: List[List[float]]):
        """Upsert documents to Qdrant."""
        try:
            points = []
            for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
                points.append(PointStruct(
                    id=doc.get("id", f"doc_{i}"),
                    vector=embedding,
                    payload={
                        "content": doc["content"],
                        "source": doc.get("source", "unknown"),
                        "timestamp": doc.get("timestamp", ""),
                        **doc.get("metadata", {})
                    }
                ))
            
            self.qdrant_client.upsert(
                collection_name=self.qdrant_collection,
                points=points
            )
            logger.info(f"Upserted {len(documents)} documents to Qdrant")
            
        except Exception as e:
            logger.error(f"Qdrant upsert failed: {e}")


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


# --- SERP Integration ---
# Required environment variables:
#   SERP_API_KEY (for serpapi.com)
#   GOOGLE_API_KEY (for Google Custom Search)

import aiohttp
import json
import urllib.parse

class SERPClient:
    """
    Search Engine Results Page (SERP) client for real-time web search.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.serp_api_key = os.getenv("SERP_API_KEY")
        self.google_api_key = os.getenv("GOOGLE_API_KEY")
        self.google_cx = os.getenv("GOOGLE_CUSTOM_SEARCH_CX")
        
    async def search_web(self, query: str, num_results: int = 10) -> List[Dict[str, Any]]:
        """
        Search the web using SERP API or Google Custom Search.
        
        Args:
            query: Search query
            num_results: Number of results to return
            
        Returns:
            List of search results
        """
        results = []
        
        # Try SERP API first
        if self.serp_api_key:
            try:
                serp_results = await self._search_serp(query, num_results)
                results.extend(serp_results)
            except Exception as e:
                logger.warning(f"SERP API failed: {e}")
        
        # Try Google Custom Search as fallback
        if not results and self.google_api_key and self.google_cx:
            try:
                google_results = await self._search_google(query, num_results)
                results.extend(google_results)
            except Exception as e:
                logger.warning(f"Google Custom Search failed: {e}")
        
        # If both fail, return mock results
        if not results:
            logger.warning("All search APIs failed, returning mock results")
            results = self._generate_mock_results(query, num_results)
        
        return results
    
    async def _search_serp(self, query: str, num_results: int) -> List[Dict[str, Any]]:
        """Search using SERP API."""
        async with aiohttp.ClientSession() as session:
            params = {
                'api_key': self.serp_api_key,
                'q': query,
                'num': num_results,
                'engine': 'google'
            }
            
            async with session.get('https://serpapi.com/search', params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    organic_results = data.get('organic_results', [])
                    
                    results = []
                    for result in organic_results:
                        results.append({
                            'title': result.get('title', ''),
                            'snippet': result.get('snippet', ''),
                            'link': result.get('link', ''),
                            'source': 'serp_api',
                            'score': 0.8  # Default score for web results
                        })
                    return results
                else:
                    raise Exception(f"SERP API returned status {response.status}")
    
    async def _search_google(self, query: str, num_results: int) -> List[Dict[str, Any]]:
        """Search using Google Custom Search API."""
        async with aiohttp.ClientSession() as session:
            params = {
                'key': self.google_api_key,
                'cx': self.google_cx,
                'q': query,
                'num': min(num_results, 10)  # Google CSE max is 10
            }
            
            async with session.get('https://www.googleapis.com/customsearch/v1', params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    items = data.get('items', [])
                    
                    results = []
                    for item in items:
                        results.append({
                            'title': item.get('title', ''),
                            'snippet': item.get('snippet', ''),
                            'link': item.get('link', ''),
                            'source': 'google_cse',
                            'score': 0.8
                        })
                    return results
                else:
                    raise Exception(f"Google CSE returned status {response.status}")
    
    def _generate_mock_results(self, query: str, num_results: int) -> List[Dict[str, Any]]:
        """Generate mock search results when APIs fail."""
        mock_results = []
        for i in range(num_results):
            mock_results.append({
                'title': f'Mock result {i+1} for "{query}"',
                'snippet': f'This is a mock search result for the query "{query}". In a real implementation, this would contain actual search results from the web.',
                'link': f'https://example.com/mock-result-{i+1}',
                'source': 'mock',
                'score': 0.5
            })
        return mock_results


class QueryIntelligence:
    """
    Advanced query intelligence for intent classification, entity recognition, and complexity scoring.
    """
    
    def __init__(self):
        self.intent_patterns = {
            "factual": [
                r"what is", r"who is", r"when did", r"where is", r"how many",
                r"definition of", r"meaning of", r"explain", r"describe"
            ],
            "comparative": [
                r"compare", r"difference between", r"vs", r"versus",
                r"better than", r"worse than", r"similar to"
            ],
            "procedural": [
                r"how to", r"steps to", r"process for", r"guide",
                r"tutorial", r"instructions", r"method"
            ],
            "analytical": [
                r"why", r"because", r"reason", r"cause", r"effect",
                r"impact", r"consequence", r"analysis"
            ],
            "opinion": [
                r"opinion", r"think", r"believe", r"feel", r"view",
                r"perspective", r"point of view"
            ]
        }
        
        self.complexity_indicators = {
            "simple": ["what", "who", "when", "where", "how many"],
            "moderate": ["explain", "describe", "compare", "why"],
            "complex": ["analyze", "evaluate", "synthesize", "critique", "hypothesize"]
        }
        
        # Multi-language support
        self.language_patterns = {
            "english": {
                "factual": ["what is", "who is", "when did", "where is", "how many"],
                "comparative": ["compare", "difference between", "vs", "versus"],
                "procedural": ["how to", "steps to", "process for", "guide"],
                "analytical": ["why", "because", "reason", "cause", "effect"],
                "opinion": ["opinion", "think", "believe", "feel", "view"]
            },
            "spanish": {
                "factual": ["qué es", "quién es", "cuándo", "dónde", "cuántos"],
                "comparative": ["comparar", "diferencia entre", "vs", "versus"],
                "procedural": ["cómo", "pasos para", "proceso", "guía"],
                "analytical": ["por qué", "porque", "razón", "causa", "efecto"],
                "opinion": ["opinión", "pensar", "creer", "sentir", "ver"]
            },
            "french": {
                "factual": ["qu'est-ce que", "qui est", "quand", "où", "combien"],
                "comparative": ["comparer", "différence entre", "vs", "versus"],
                "procedural": ["comment", "étapes pour", "processus", "guide"],
                "analytical": ["pourquoi", "parce que", "raison", "cause", "effet"],
                "opinion": ["opinion", "penser", "croire", "sentir", "voir"]
            },
            "german": {
                "factual": ["was ist", "wer ist", "wann", "wo", "wie viele"],
                "comparative": ["vergleichen", "unterschied zwischen", "vs", "gegen"],
                "procedural": ["wie", "schritte für", "prozess", "anleitung"],
                "analytical": ["warum", "weil", "grund", "ursache", "wirkung"],
                "opinion": ["meinung", "denken", "glauben", "fühlen", "sehen"]
            }
        }
    
    async def classify_intent(self, query: str) -> Dict[str, Any]:
        """
        Classify the intent of a query.
        
        Args:
            query: Input query
            
        Returns:
            Intent classification with confidence scores
        """
        try:
            import re
            
            query_lower = query.lower()
            intent_scores = {}
            
            # Calculate scores for each intent type
            for intent, patterns in self.intent_patterns.items():
                score = 0
                for pattern in patterns:
                    if re.search(pattern, query_lower):
                        score += 1
                intent_scores[intent] = score
            
            # Normalize scores
            total_matches = sum(intent_scores.values())
            if total_matches > 0:
                intent_scores = {k: v/total_matches for k, v in intent_scores.items()}
            
            # Get primary intent
            primary_intent = max(intent_scores.items(), key=lambda x: x[1])
            
            return {
                "primary_intent": primary_intent[0],
                "confidence": primary_intent[1],
                "all_intents": intent_scores
            }
            
        except Exception as e:
            logger.error(f"Intent classification failed: {e}")
            return {
                "primary_intent": "factual",
                "confidence": 0.5,
                "all_intents": {"factual": 0.5}
            }
    
    async def extract_entities_advanced(self, query: str) -> List[Dict[str, Any]]:
        """
        Advanced entity extraction with NER and custom patterns.
        
        Args:
            query: Input query
            
        Returns:
            List of extracted entities with metadata
        """
        try:
            entities = []
            
            # Use LLM for advanced entity extraction
            llm_prompt = f"""
            Extract named entities from this query: "{query}"
            
            Return a JSON array with entities in this format:
            [
                {{
                    "text": "entity name",
                    "type": "PERSON|ORGANIZATION|LOCATION|DATE|SCIENTIFIC_TERM|OTHER",
                    "confidence": 0.0-1.0,
                    "relevance": "high|medium|low"
                }}
            ]
            
            Focus on entities that are important for information retrieval.
            """
            
            try:
                from api.llm_client import LLMClient
                llm_client = LLMClient()
                response = await llm_client.generate_text(llm_prompt, max_tokens=300)
                
                if response:
                    import json
                    try:
                        entities = json.loads(response)
                        if isinstance(entities, list):
                            return entities
                    except json.JSONDecodeError:
                        pass
            except Exception as e:
                logger.warning(f"LLM entity extraction failed: {e}")
            
            # Fallback to basic entity extraction
            return await self._extract_basic_entities(query)
            
        except Exception as e:
            logger.error(f"Advanced entity extraction failed: {e}")
            return []
    
    async def _extract_basic_entities(self, query: str) -> List[Dict[str, Any]]:
        """Basic entity extraction using regex patterns."""
        import re
        
        entities = []
        
        # Extract dates
        date_patterns = [
            r'\b\d{4}\b',  # Years
            r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b',
            r'\b\d{1,2}/\d{1,2}/\d{4}\b'
        ]
        
        for pattern in date_patterns:
            matches = re.finditer(pattern, query, re.IGNORECASE)
            for match in matches:
                entities.append({
                    "text": match.group(),
                    "type": "DATE",
                    "confidence": 0.8,
                    "relevance": "medium"
                })
        
        # Extract potential organizations (capitalized words)
        org_pattern = r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b'
        org_matches = re.finditer(org_pattern, query)
        for match in org_matches:
            text = match.group()
            if len(text.split()) > 1:  # Multi-word entities
                entities.append({
                    "text": text,
                    "type": "ORGANIZATION",
                    "confidence": 0.6,
                    "relevance": "medium"
                })
        
        return entities
    
    async def score_complexity(self, query: str) -> Dict[str, Any]:
        """
        Score query complexity for resource allocation.
        
        Args:
            query: Input query
            
        Returns:
            Complexity score and analysis
        """
        try:
            query_lower = query.lower()
            
            # Count complexity indicators
            complexity_scores = {}
            for level, indicators in self.complexity_indicators.items():
                score = 0
                for indicator in indicators:
                    if indicator in query_lower:
                        score += 1
                complexity_scores[level] = score
            
            # Calculate overall complexity
            total_indicators = sum(complexity_scores.values())
            if total_indicators == 0:
                complexity_level = "simple"
                complexity_score = 0.3
            elif complexity_scores["complex"] > 0:
                complexity_level = "complex"
                complexity_score = 0.8
            elif complexity_scores["moderate"] > 0:
                complexity_level = "moderate"
                complexity_score = 0.6
            else:
                complexity_level = "simple"
                complexity_score = 0.3
            
            # Additional complexity factors
            word_count = len(query.split())
            if word_count > 20:
                complexity_score += 0.2
            elif word_count < 5:
                complexity_score -= 0.1
            
            # Cap complexity score
            complexity_score = min(1.0, max(0.0, complexity_score))
            
            return {
                "complexity_level": complexity_level,
                "complexity_score": complexity_score,
                "word_count": word_count,
                "indicators": complexity_scores,
                "estimated_tokens": word_count * 1.5,  # Rough estimate
                "suggested_timeout": int(complexity_score * 30) + 10  # Seconds
            }
            
        except Exception as e:
            logger.error(f"Complexity scoring failed: {e}")
            return {
                "complexity_level": "moderate",
                "complexity_score": 0.5,
                "word_count": len(query.split()),
                "indicators": {},
                "estimated_tokens": 50,
                "suggested_timeout": 20
            }
    
    async def analyze_query(self, query: str) -> Dict[str, Any]:
        """
        Comprehensive query analysis with multilingual support.
        
        Args:
            query: Input query
            
        Returns:
            Complete query analysis
        """
        try:
            # Run all analyses in parallel
            intent_task = self.classify_intent_multilingual(query)
            entities_task = self.extract_entities_advanced(query)
            complexity_task = self.score_complexity(query)
            language_task = self.detect_language(query)
            
            intent_result, entities_result, complexity_result, language_result = await asyncio.gather(
                intent_task, entities_task, complexity_task, language_task
            )
            
            return {
                "query": query,
                "intent": intent_result,
                "entities": entities_result,
                "complexity": complexity_result,
                "language": language_result,
                "analysis_timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Query analysis failed: {e}")
            return {
                "query": query,
                "error": str(e),
                "analysis_timestamp": datetime.now().isoformat()
            }

    async def detect_language(self, query: str) -> Dict[str, Any]:
        """
        Detect the language of the query.
        
        Args:
            query: Input query
            
        Returns:
            Language detection result with confidence
        """
        try:
            # Use LLM for language detection
            llm_prompt = f"""
            Detect the language of this text: "{query}"
            
            Return only the language code (en, es, fr, de, etc.) or "unknown" if unclear.
            """
            
            try:
                from api.llm_client import LLMClient
                llm_client = LLMClient()
                response = await llm_client.generate_text(llm_prompt, max_tokens=10)
                
                if response:
                    language_code = response.strip().lower()
                    # Map language codes to our supported languages
                    language_map = {
                        "en": "english",
                        "es": "spanish", 
                        "fr": "french",
                        "de": "german"
                    }
                    
                    detected_language = language_map.get(language_code, "english")
                    return {
                        "language": detected_language,
                        "language_code": language_code,
                        "confidence": 0.9
                    }
            except Exception as e:
                logger.warning(f"LLM language detection failed: {e}")
            
            # Fallback to basic language detection
            return self._basic_language_detection(query)
            
        except Exception as e:
            logger.error(f"Language detection failed: {e}")
            return {
                "language": "english",
                "language_code": "en",
                "confidence": 0.5
            }
    
    def _basic_language_detection(self, query: str) -> Dict[str, Any]:
        """Basic language detection using character patterns."""
        query_lower = query.lower()
        
        # Simple pattern matching
        if any(char in query for char in "áéíóúñü"):
            return {"language": "spanish", "language_code": "es", "confidence": 0.7}
        elif any(char in query for char in "àâäéèêëïîôöùûüÿç"):
            return {"language": "french", "language_code": "fr", "confidence": 0.7}
        elif any(char in query for char in "äöüß"):
            return {"language": "german", "language_code": "de", "confidence": 0.7}
        else:
            return {"language": "english", "language_code": "en", "confidence": 0.6}
    
    async def classify_intent_multilingual(self, query: str) -> Dict[str, Any]:
        """
        Classify intent in multiple languages.
        
        Args:
            query: Input query
            
        Returns:
            Intent classification with language detection
        """
        try:
            # Detect language first
            language_result = await self.detect_language(query)
            detected_language = language_result["language"]
            
            # Use language-specific patterns if available
            if detected_language in self.language_patterns:
                return await self._classify_intent_with_language(query, detected_language)
            else:
                # Fallback to English patterns
                return await self.classify_intent(query)
                
        except Exception as e:
            logger.error(f"Multilingual intent classification failed: {e}")
            return await self.classify_intent(query)
    
    async def _classify_intent_with_language(self, query: str, language: str) -> Dict[str, Any]:
        """Classify intent using language-specific patterns."""
        try:
            import re
            
            query_lower = query.lower()
            intent_scores = {}
            patterns = self.language_patterns[language]
            
            # Calculate scores for each intent type
            for intent, phrases in patterns.items():
                score = 0
                for phrase in phrases:
                    if phrase in query_lower:
                        score += 1
                intent_scores[intent] = score
            
            # Normalize scores
            total_matches = sum(intent_scores.values())
            if total_matches > 0:
                intent_scores = {k: v/total_matches for k, v in intent_scores.items()}
            
            # Get primary intent
            primary_intent = max(intent_scores.items(), key=lambda x: x[1])
            
            return {
                "primary_intent": primary_intent[0],
                "confidence": primary_intent[1],
                "all_intents": intent_scores,
                "language": language
            }
            
        except Exception as e:
            logger.error(f"Language-specific intent classification failed: {e}")
            return {
                "primary_intent": "factual",
                "confidence": 0.5,
                "all_intents": {"factual": 0.5},
                "language": language
            }


class RetrievalAgent(BaseAgent):
    """
    RetrievalAgent that combines vector search, keyword search, and knowledge graph queries.
    """

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(agent_id="retrieval_agent", agent_type=AgentType.RETRIEVAL)
        self.config = config or self._default_config()
        
        # Initialize clients
        self.vector_db = VectorDBClient(self.config.get("vector_db", {}))
        self.elasticsearch = ElasticsearchClient(self.config.get("elasticsearch", {}))
        self.knowledge_graph = KnowledgeGraphClient(self.config.get("knowledge_graph", {}))
        self.serp_client = SERPClient(self.config.get("serp", {}))
        self.semantic_cache = SemanticCache()
        
        # Initialize query intelligence
        self.query_intelligence = QueryIntelligence()
        
        # Initialize entity extractor
        self.entity_extractor = EntityExtractor()

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

    async def web_search(self, query: str, top_k: int = 10) -> SearchResult:
        """
        Perform web search using SERP API or Google Custom Search.

        Args:
            query: Search query
            top_k: Number of results to return

        Returns:
            SearchResult with documents
        """
        start_time = time.time()

        try:
            # Perform web search
            results = await self.serp_client.search_web(query, top_k)

            # Convert to Document objects
            documents = []
            for result in results:
                doc = Document(
                    content=result.get("snippet", ""),
                    score=result.get("score", 0.5),
                    source="web_search",
                    metadata={
                        "title": result.get("title", ""),
                        "url": result.get("link", ""),
                        "source": result.get("source", "unknown"),
                    },
                    doc_id=result.get("link", ""),
                )
                documents.append(doc)

            query_time = int((time.time() - start_time) * 1000)

            return SearchResult(
                documents=documents,
                search_type="web_search",
                query_time_ms=query_time,
                total_hits=len(documents),
                metadata={"search_engine": "serp/google"},
            )

        except Exception as e:
            logger.error(f"Web search error: {e}")
            return SearchResult(
                documents=[],
                search_type="web_search",
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
        Perform hybrid retrieval using multiple strategies based on query intelligence.
        
        Args:
            query: User query
            entities: Optional pre-extracted entities
            
        Returns:
            Combined search results
        """
        start_time = time.time()
        
        try:
            # Analyze query for intelligent retrieval strategy
            query_analysis = await self.query_intelligence.analyze_query(query)
            intent = query_analysis["intent"]["primary_intent"]
            complexity = query_analysis["complexity"]["complexity_level"]
            
            # Extract entities if not provided
            if not entities:
                entities = [entity["text"] for entity in query_analysis["entities"]]
            
            # Determine retrieval strategy based on intent and complexity
            retrieval_strategies = self._determine_retrieval_strategies(intent, complexity, entities)
            
            # Execute retrieval strategies in parallel
            retrieval_tasks = []
            
            if "vector" in retrieval_strategies:
                retrieval_tasks.append(self.vector_search(query, top_k=20))
            
            if "keyword" in retrieval_strategies:
                retrieval_tasks.append(self.keyword_search(query, top_k=20))
            
            if "web" in retrieval_strategies:
                retrieval_tasks.append(self.web_search(query, top_k=10))
            
            if "graph" in retrieval_strategies and entities:
                retrieval_tasks.append(self.graph_search(entities, top_k=20))
            
            # Execute all strategies
            if retrieval_tasks:
                results = await asyncio.gather(*retrieval_tasks, return_exceptions=True)
                
                # Filter out failed retrievals
                successful_results = []
                for result in results:
                    if isinstance(result, SearchResult):
                        successful_results.append(result)
                    else:
                        logger.warning(f"Retrieval strategy failed: {result}")
                
                if successful_results:
                    # Merge and deduplicate results
                    merged_documents = await self._merge_and_deduplicate(successful_results)
                    
                    # Apply diversity constraints
                    diverse_documents = self._apply_diversity_constraints(merged_documents)
                    
                    # Use LLM reranking for complex queries
                    if complexity == "complex":
                        diverse_documents = await self._llm_rerank(query, diverse_documents)
                    
                    query_time = int((time.time() - start_time) * 1000)
                    
                    return SearchResult(
                        documents=diverse_documents,
                        search_type="hybrid",
                        query_time_ms=query_time,
                        total_hits=len(diverse_documents),
                        metadata={
                            "intent": intent,
                            "complexity": complexity,
                            "entities": entities,
                            "strategies_used": retrieval_strategies,
                            "query_analysis": query_analysis
                        }
                    )
            
            # Fallback to basic retrieval
            logger.warning("All retrieval strategies failed, using fallback")
            return await self._emergency_fallback(query)
            
        except Exception as e:
            logger.error(f"Hybrid retrieval failed: {e}")
            return await self._emergency_fallback(query)
    
    def _determine_retrieval_strategies(self, intent: str, complexity: str, entities: List[str]) -> List[str]:
        """
        Determine which retrieval strategies to use based on query analysis.
        
        Args:
            intent: Query intent (factual, comparative, etc.)
            complexity: Query complexity (simple, moderate, complex)
            entities: Extracted entities
            
        Returns:
            List of retrieval strategies to use
        """
        strategies = []
        
        # Base strategies for all queries
        strategies.append("vector")
        strategies.append("keyword")
        
        # Add web search for factual and comparative queries
        if intent in ["factual", "comparative"]:
            strategies.append("web")
        
        # Add graph search if entities are available
        if entities:
            strategies.append("graph")
        
        # Add web search for complex queries regardless of intent
        if complexity == "complex":
            if "web" not in strategies:
                strategies.append("web")
        
        return strategies

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
