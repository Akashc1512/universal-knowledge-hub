"""
Pinecone vector database client for storing and retrieving embeddings.
Uses Pinecone's new API with integrated embedding models.
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from pinecone import Pinecone, ServerlessSpec
from core.config import config
from core.llm_client import llm_client

logger = logging.getLogger(__name__)


@dataclass
class VectorRecord:
    """Vector record with metadata."""
    id: str
    vector: List[float]
    metadata: Dict[str, Any]
    score: Optional[float] = None


class PineconeClient:
    """
    Pinecone vector database client using new API.
    Handles initialization, upserting, and querying vectors with integrated embedding model.
    """
    
    def __init__(self):
        self.client = None
        self.index = None
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize Pinecone client and index."""
        try:
            if not config.pinecone_api_key:
                logger.warning("Pinecone API key not configured")
                return
            
            # Initialize Pinecone with new API
            self.client = Pinecone(api_key=config.pinecone_api_key)
            
            # Create index if it doesn't exist
            if not self.client.has_index(config.pinecone_index_name):
                logger.info(f"Creating Pinecone index: {config.pinecone_index_name}")
                from pinecone import ServerlessSpec
                
                self.client.create_index(
                    name=config.pinecone_index_name,
                    spec=ServerlessSpec(
                        cloud="aws",
                        region="us-east-1"
                    ),
                    dimension=config.vector_dimension,
                    metric="cosine"
                )
            
            self.index = self.client.Index(config.pinecone_index_name)
            logger.info("✅ Pinecone client initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Pinecone client: {e}")
            self.client = None
            self.index = None
    
    def is_available(self) -> bool:
        """Check if Pinecone client is available."""
        return self.index is not None
    
    async def upsert_vectors(
        self,
        vectors: List[VectorRecord],
        namespace: Optional[str] = None
    ) -> bool:
        """
        Upsert vectors to Pinecone index using integrated embedding model.
        
        Args:
            vectors: List of VectorRecord objects
            namespace: Optional namespace for the vectors
        
        Returns:
            True if successful, False otherwise
        """
        if not self.is_available():
            logger.error("Pinecone client not available")
            return False
        
        try:
            # Convert to Pinecone format with text for integrated embedding
            pinecone_records = []
            for record in vectors:
                pinecone_records.append({
                    "id": record.id,
                    "values": record.vector,
                    "metadata": record.metadata
                })
            
            # Upsert vectors
            await asyncio.to_thread(
                self.index.upsert,
                vectors=pinecone_records,
                namespace=namespace
            )
            
            logger.info(f"✅ Upserted {len(vectors)} vectors to Pinecone")
            return True
            
        except Exception as e:
            logger.error(f"Error upserting vectors to Pinecone: {e}")
            return False
    
    async def query_vectors(
        self,
        query_vector: List[float],
        top_k: int = 10,
        namespace: Optional[str] = None,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[VectorRecord]:
        """
        Query vectors from Pinecone index.
        
        Args:
            query_vector: Query vector
            top_k: Number of results to return
            namespace: Optional namespace to query
            filter: Optional metadata filter
        
        Returns:
            List of VectorRecord objects
        """
        if not self.is_available():
            logger.error("Pinecone client not available")
            return []
        
        try:
            # Query vectors
            response = await asyncio.to_thread(
                self.index.query,
                vector=query_vector,
                top_k=top_k,
                namespace=namespace,
                filter=filter,
                include_metadata=True
            )
            
            # Convert response to VectorRecord objects
            results = []
            for match in response.matches:
                results.append(VectorRecord(
                    id=match.id,
                    vector=match.values,
                    metadata=match.metadata,
                    score=match.score
                ))
            
            logger.info(f"✅ Retrieved {len(results)} vectors from Pinecone")
            return results
            
        except Exception as e:
            logger.error(f"Error querying vectors from Pinecone: {e}")
            return []
    
    async def query_text(
        self,
        query_text: str,
        top_k: int = 10,
        namespace: Optional[str] = None,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[VectorRecord]:
        """
        Query using text (integrated embedding model will handle vectorization).
        
        Args:
            query_text: Query text
            top_k: Number of results to return
            namespace: Optional namespace to query
            filter: Optional metadata filter
        
        Returns:
            List of VectorRecord objects
        """
        if not self.is_available():
            logger.error("Pinecone client not available")
            return []
        
        try:
            # For now, use the regular vector query approach
            # TODO: Implement proper text query when Pinecone API supports it
            logger.warning("Text query not yet implemented, using vector query instead")
            return await self.query_vectors([], top_k, namespace, filter)
            
        except Exception as e:
            logger.error(f"Error querying text from Pinecone: {e}")
            return []
    
    async def delete_vectors(
        self,
        vector_ids: List[str],
        namespace: Optional[str] = None
    ) -> bool:
        """
        Delete vectors from Pinecone index.
        
        Args:
            vector_ids: List of vector IDs to delete
            namespace: Optional namespace
        
        Returns:
            True if successful, False otherwise
        """
        if not self.is_available():
            logger.error("Pinecone client not available")
            return False
        
        try:
            await asyncio.to_thread(
                self.index.delete,
                ids=vector_ids,
                namespace=namespace
            )
            
            logger.info(f"✅ Deleted {len(vector_ids)} vectors from Pinecone")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting vectors from Pinecone: {e}")
            return False
    
    async def get_index_stats(self) -> Dict[str, Any]:
        """
        Get index statistics.
        
        Returns:
            Dictionary with index statistics
        """
        if not self.is_available():
            logger.error("Pinecone client not available")
            return {}
        
        try:
            stats = await asyncio.to_thread(self.index.describe_index_stats)
            return {
                "total_vector_count": stats.total_vector_count,
                "dimension": stats.dimension,
                "index_fullness": stats.index_fullness,
                "namespaces": stats.namespaces
            }
        except Exception as e:
            logger.error(f"Error getting index stats: {e}")
            return {}


class VectorDatabaseManager:
    """
    High-level vector database manager.
    Handles text embedding and vector operations.
    """
    
    def __init__(self):
        self.pinecone_client = PineconeClient()
        self.llm_client = llm_client
    
    async def store_documents(
        self,
        documents: List[Dict[str, Any]],
        namespace: Optional[str] = None
    ) -> bool:
        """
        Store documents as vectors in the database using integrated embedding model.
        
        Args:
            documents: List of documents with 'id', 'content', and optional 'metadata'
            namespace: Optional namespace
        
        Returns:
            True if successful, False otherwise
        """
        if not self.pinecone_client.is_available():
            logger.error("Vector database not available")
            return False
        
        try:
            # Generate embeddings for documents
            vectors = []
            for doc in documents:
                # Generate embedding for document content
                embedding = await self.llm_client.generate_embeddings(doc['content'])
                
                # Create vector record
                metadata = {
                    'content': doc['content'],
                    **doc.get('metadata', {})
                }
                
                # Only add timestamp if it's not None
                if doc.get('timestamp'):
                    metadata['timestamp'] = doc['timestamp']
                
                vector_record = VectorRecord(
                    id=doc['id'],
                    vector=embedding,
                    metadata=metadata
                )
                vectors.append(vector_record)
            
            # Store vectors
            return await self.pinecone_client.upsert_vectors(vectors, namespace)
            
        except Exception as e:
            logger.error(f"Error storing documents: {e}")
            return False
    
    async def search_similar(
        self,
        query: str,
        top_k: int = 10,
        namespace: Optional[str] = None,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar documents using integrated embedding model.
        
        Args:
            query: Search query
            top_k: Number of results to return
            namespace: Optional namespace
            filter: Optional metadata filter
        
        Returns:
            List of similar documents with scores
        """
        if not self.pinecone_client.is_available():
            logger.error("Vector database not available")
            return []
        
        try:
            # Generate embedding for query using LLM client
            query_embedding = await self.llm_client.generate_embeddings(query)
            
            # Search using vector query
            vector_results = await self.pinecone_client.query_vectors(
                query_embedding,
                top_k=top_k,
                namespace=namespace,
                filter=filter
            )
            
            # Convert to document format
            results = []
            for record in vector_results:
                results.append({
                    'id': record.id,
                    'content': record.metadata.get('content', ''),
                    'score': record.score,
                    'metadata': record.metadata
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Error searching similar documents: {e}")
            return []
    
    async def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        return await self.pinecone_client.get_index_stats()


# Global vector database manager instance
vector_db = VectorDatabaseManager() 