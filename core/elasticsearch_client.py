"""
Elasticsearch client for Elastic Cloud integration.
Handles authentication, indexing, and search operations.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional
from elasticsearch import AsyncElasticsearch
from core.config import config

logger = logging.getLogger(__name__)


class ElasticsearchClient:
    """
    Elasticsearch client for Elastic Cloud.
    Handles authentication, indexing, and search operations.
    """
    
    def __init__(self):
        self.client = None
        self.index_name = config.elasticsearch_index
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize Elasticsearch client with proper authentication."""
        try:
            if not config.elasticsearch_url:
                logger.warning("Elasticsearch URL not configured")
                return
            
            # Configure authentication
            auth_config = {}
            if config.elasticsearch_api_key:
                auth_config["api_key"] = config.elasticsearch_api_key
                logger.info("Using API key authentication for Elasticsearch")
            elif config.elasticsearch_username and config.elasticsearch_password:
                auth_config["basic_auth"] = (config.elasticsearch_username, config.elasticsearch_password)
                logger.info("Using basic authentication for Elasticsearch")
            else:
                logger.warning("No Elasticsearch authentication configured")
            
            # Initialize client
            self.client = AsyncElasticsearch(
                [config.elasticsearch_url],
                **auth_config,
                verify_certs=True,
                ssl_show_warn=False
            )
            
            logger.info("✅ Elasticsearch client initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Elasticsearch client: {e}")
            self.client = None
    
    def is_available(self) -> bool:
        """Check if Elasticsearch client is available."""
        return self.client is not None
    
    async def test_connection(self) -> bool:
        """Test connection to Elasticsearch."""
        if not self.is_available():
            logger.error("Elasticsearch client not available")
            return False
        
        try:
            info = await self.client.info()
            logger.info(f"✅ Connected to Elasticsearch: {info['version']['number']}")
            return True
        except Exception as e:
            logger.error(f"Elasticsearch connection test failed: {e}")
            return False
    
    async def create_index(self, index_name: Optional[str] = None) -> bool:
        """
        Create Elasticsearch index with proper mappings.
        
        Args:
            index_name: Optional index name, uses default if not provided
        
        Returns:
            True if successful, False otherwise
        """
        if not self.is_available():
            logger.error("Elasticsearch client not available")
            return False
        
        index_name = index_name or self.index_name
        
        try:
            # Check if index exists
            exists = await self.client.indices.exists(index=index_name)
            if exists:
                logger.info(f"Index {index_name} already exists")
                return True
            
            # Create index with mappings (serverless-compatible)
            mapping = {
                "mappings": {
                    "properties": {
                        "content": {
                            "type": "text",
                            "analyzer": "standard"
                        },
                        "title": {
                            "type": "text",
                            "analyzer": "standard"
                        },
                        "source": {
                            "type": "keyword"
                        },
                        "category": {
                            "type": "keyword"
                        },
                        "timestamp": {
                            "type": "date"
                        },
                        "metadata": {
                            "type": "object",
                            "enabled": True
                        }
                    }
                }
                # Note: No settings for serverless mode
            }
            
            await self.client.indices.create(index=index_name, body=mapping)
            logger.info(f"✅ Created Elasticsearch index: {index_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error creating Elasticsearch index: {e}")
            return False
    
    async def index_document(
        self,
        doc_id: str,
        content: str,
        title: Optional[str] = None,
        source: Optional[str] = None,
        category: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        index_name: Optional[str] = None
    ) -> bool:
        """
        Index a document in Elasticsearch.
        
        Args:
            doc_id: Document ID
            content: Document content
            title: Document title
            source: Document source
            category: Document category
            metadata: Additional metadata
            index_name: Optional index name
        
        Returns:
            True if successful, False otherwise
        """
        if not self.is_available():
            logger.error("Elasticsearch client not available")
            return False
        
        index_name = index_name or self.index_name
        
        try:
            document = {
                "content": content,
                "timestamp": "now"
            }
            
            if title:
                document["title"] = title
            if source:
                document["source"] = source
            if category:
                document["category"] = category
            if metadata:
                document["metadata"] = metadata
            
            await self.client.index(
                index=index_name,
                id=doc_id,
                body=document
            )
            
            logger.info(f"✅ Indexed document {doc_id} in {index_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error indexing document {doc_id}: {e}")
            return False
    
    async def search_documents(
        self,
        query: str,
        top_k: int = 10,
        index_name: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search documents in Elasticsearch.
        
        Args:
            query: Search query
            top_k: Number of results to return
            index_name: Optional index name
            filters: Optional filters
        
        Returns:
            List of search results
        """
        if not self.is_available():
            logger.error("Elasticsearch client not available")
            return []
        
        index_name = index_name or self.index_name
        
        try:
            # Build search query
            search_body = {
                "query": {
                    "bool": {
                        "must": [
                            {
                                "multi_match": {
                                    "query": query,
                                    "fields": ["content^2", "title"],
                                    "type": "best_fields",
                                    "fuzziness": "AUTO"
                                }
                            }
                        ]
                    }
                },
                "size": top_k,
                "_source": ["content", "title", "source", "category", "metadata"]
            }
            
            # Add filters if provided
            if filters:
                search_body["query"]["bool"]["filter"] = []
                for field, value in filters.items():
                    search_body["query"]["bool"]["filter"].append({
                        "term": {field: value}
                    })
            
            response = await self.client.search(
                index=index_name,
                body=search_body
            )
            
            # Process results
            results = []
            for hit in response["hits"]["hits"]:
                results.append({
                    "id": hit["_id"],
                    "score": hit["_score"],
                    "content": hit["_source"].get("content", ""),
                    "title": hit["_source"].get("title", ""),
                    "source": hit["_source"].get("source", ""),
                    "category": hit["_source"].get("category", ""),
                    "metadata": hit["_source"].get("metadata", {})
                })
            
            logger.info(f"✅ Found {len(results)} documents in Elasticsearch")
            return results
            
        except Exception as e:
            logger.error(f"Error searching Elasticsearch: {e}")
            return []
    
    async def get_index_stats(self, index_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get index statistics.
        
        Args:
            index_name: Optional index name
        
        Returns:
            Dictionary with index statistics
        """
        if not self.is_available():
            logger.error("Elasticsearch client not available")
            return {}
        
        index_name = index_name or self.index_name
        
        try:
            # For serverless mode, use count API instead of stats
            count_response = await self.client.count(index=index_name)
            return {
                "total_docs": count_response["count"],
                "index_name": index_name,
                "serverless_mode": True
            }
        except Exception as e:
            logger.error(f"Error getting index stats: {e}")
            return {}


# Global Elasticsearch client instance
elasticsearch_client = ElasticsearchClient() 