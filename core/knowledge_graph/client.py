"""
Neo4j Knowledge Graph Client
Provides high-performance access to the knowledge graph for recommendation engine.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import time
from contextlib import asynccontextmanager

try:
    from neo4j import AsyncGraphDatabase
    from neo4j.exceptions import ServiceUnavailable, AuthError, ClientError
except ImportError:
    AsyncGraphDatabase = None
    ServiceUnavailable = Exception
    AuthError = Exception
    ClientError = Exception

logger = logging.getLogger(__name__)


@dataclass
class GraphQuery:
    """Graph query with parameters and metadata."""
    cypher: str
    parameters: Dict[str, Any]
    timeout: int = 30
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class GraphResult:
    """Result from graph query execution."""
    success: bool
    data: List[Dict[str, Any]]
    execution_time: float
    records_count: int
    error: Optional[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class Neo4jClient:
    """High-performance Neo4j client for knowledge graph operations."""
    
    def __init__(self, uri: str, username: str, password: str, database: str = "neo4j"):
        self.uri = uri
        self.username = username
        self.password = password
        self.database = database
        self.driver = None
        self._connection_pool_size = 50
        self._max_connection_lifetime = 3600  # 1 hour
        self._connection_timeout = 30
        self._query_timeout = 30
        
        # Performance tracking
        self.query_count = 0
        self.total_execution_time = 0.0
        self.error_count = 0
        
    async def connect(self) -> bool:
        """Establish connection to Neo4j database."""
        try:
            if AsyncGraphDatabase is None:
                logger.error("Neo4j driver not available. Install neo4j package.")
                return False
            
            self.driver = AsyncGraphDatabase.driver(
                self.uri,
                auth=(self.username, self.password),
                max_connection_pool_size=self._connection_pool_size,
                max_connection_lifetime=self._max_connection_lifetime,
                connection_timeout=self._connection_timeout
            )
            
            # Test connection
            async with self.driver.session(database=self.database) as session:
                result = await session.run("RETURN 1 as test")
                await result.single()
            
            logger.info(f"✅ Connected to Neo4j database: {self.uri}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to connect to Neo4j: {e}")
            return False
    
    async def disconnect(self):
        """Close connection to Neo4j database."""
        if self.driver:
            await self.driver.close()
            logger.info("🔌 Disconnected from Neo4j database")
    
    @asynccontextmanager
    async def get_session(self):
        """Get Neo4j session with automatic cleanup."""
        if not self.driver:
            raise RuntimeError("Neo4j driver not initialized. Call connect() first.")
        
        session = self.driver.session(database=self.database)
        try:
            yield session
        finally:
            await session.close()
    
    async def execute_query(self, query: GraphQuery) -> GraphResult:
        """Execute a Cypher query with performance tracking."""
        start_time = time.time()
        
        try:
            async with self.get_session() as session:
                result = await session.run(
                    query.cypher,
                    query.parameters,
                    timeout=query.timeout
                )
                
                records = await result.data()
                execution_time = time.time() - start_time
                
                # Update performance metrics
                self.query_count += 1
                self.total_execution_time += execution_time
                
                logger.debug(f"Query executed in {execution_time:.3f}s: {query.cypher[:100]}...")
                
                return GraphResult(
                    success=True,
                    data=records,
                    execution_time=execution_time,
                    records_count=len(records),
                    metadata=query.metadata
                )
                
        except Exception as e:
            execution_time = time.time() - start_time
            self.error_count += 1
            
            logger.error(f"Query failed after {execution_time:.3f}s: {e}")
            
            return GraphResult(
                success=False,
                data=[],
                execution_time=execution_time,
                records_count=0,
                error=str(e),
                metadata=query.metadata
            )
    
    async def create_node(self, node_type: str, properties: Dict[str, Any]) -> GraphResult:
        """Create a node in the knowledge graph."""
        cypher = f"CREATE (n:{node_type} $properties) RETURN n"
        
        query = GraphQuery(
            cypher=cypher,
            parameters={"properties": properties},
            metadata={"operation": "create_node", "node_type": node_type}
        )
        
        return await self.execute_query(query)
    
    async def create_relationship(self, source_id: str, target_id: str, 
                                relationship_type: str, properties: Dict[str, Any] = None) -> GraphResult:
        """Create a relationship between two nodes."""
        if properties is None:
            properties = {}
        
        cypher = """
        MATCH (a), (b)
        WHERE a.id = $source_id AND b.id = $target_id
        CREATE (a)-[r:$relationship_type $properties]->(b)
        RETURN r
        """
        
        query = GraphQuery(
            cypher=cypher,
            parameters={
                "source_id": source_id,
                "target_id": target_id,
                "relationship_type": relationship_type,
                "properties": properties
            },
            metadata={
                "operation": "create_relationship",
                "relationship_type": relationship_type
            }
        )
        
        return await self.execute_query(query)
    
    async def find_user_recommendations(self, user_id: str, limit: int = 10) -> GraphResult:
        """Find document recommendations for a user based on graph patterns."""
        cypher = """
        MATCH (u:User {id: $user_id})-[:VIEWED]->(d1:Document)
        MATCH (d1)-[:SIMILAR_TO]-(d2:Document)
        WHERE NOT EXISTS((u)-[:VIEWED]->(d2))
        WITH d2, count(d1) as similarity_score
        ORDER BY similarity_score DESC
        LIMIT $limit
        RETURN d2.id as document_id, d2.title as title, similarity_score
        """
        
        query = GraphQuery(
            cypher=cypher,
            parameters={"user_id": user_id, "limit": limit},
            metadata={"operation": "user_recommendations", "user_id": user_id}
        )
        
        return await self.execute_query(query)
    
    async def find_semantic_recommendations(self, document_id: str, limit: int = 10) -> GraphResult:
        """Find semantically similar documents."""
        cypher = """
        MATCH (d1:Document {id: $document_id})-[:CONTAINS]->(c1:Concept)
        MATCH (c1)-[:SEMANTICALLY_SIMILAR]-(c2:Concept)
        MATCH (c2)<-[:CONTAINS]-(d2:Document)
        WHERE d1.id <> d2.id
        WITH d2, count(c2) as semantic_score
        ORDER BY semantic_score DESC
        LIMIT $limit
        RETURN d2.id as document_id, d2.title as title, semantic_score
        """
        
        query = GraphQuery(
            cypher=cypher,
            parameters={"document_id": document_id, "limit": limit},
            metadata={"operation": "semantic_recommendations", "document_id": document_id}
        )
        
        return await self.execute_query(query)
    
    async def find_collaborative_recommendations(self, user_id: str, limit: int = 10) -> GraphResult:
        """Find collaborative filtering recommendations."""
        cypher = """
        MATCH (u1:User {id: $user_id})-[:VIEWED]->(d:Document)
        MATCH (u2:User)-[:VIEWED]->(d)
        WHERE u1.id <> u2.id
        MATCH (u2)-[:VIEWED]->(d2:Document)
        WHERE NOT EXISTS((u1)-[:VIEWED]->(d2))
        WITH d2, count(u2) as collaborative_score
        ORDER BY collaborative_score DESC
        LIMIT $limit
        RETURN d2.id as document_id, d2.title as title, collaborative_score
        """
        
        query = GraphQuery(
            cypher=cypher,
            parameters={"user_id": user_id, "limit": limit},
            metadata={"operation": "collaborative_recommendations", "user_id": user_id}
        )
        
        return await self.execute_query(query)
    
    async def get_user_profile(self, user_id: str) -> GraphResult:
        """Get comprehensive user profile with preferences."""
        cypher = """
        MATCH (u:User {id: $user_id})
        OPTIONAL MATCH (u)-[:VIEWED]->(d:Document)
        OPTIONAL MATCH (d)-[:BELONGS_TO]->(t:Topic)
        OPTIONAL MATCH (d)-[:CONTAINS]->(c:Concept)
        RETURN u,
               collect(DISTINCT d) as viewed_documents,
               collect(DISTINCT t) as preferred_topics,
               collect(DISTINCT c) as preferred_concepts
        """
        
        query = GraphQuery(
            cypher=cypher,
            parameters={"user_id": user_id},
            metadata={"operation": "user_profile", "user_id": user_id}
        )
        
        return await self.execute_query(query)
    
    async def get_document_insights(self, document_id: str) -> GraphResult:
        """Get comprehensive document insights and relationships."""
        cypher = """
        MATCH (d:Document {id: $document_id})
        OPTIONAL MATCH (d)-[:BELONGS_TO]->(t:Topic)
        OPTIONAL MATCH (d)-[:CONTAINS]->(c:Concept)
        OPTIONAL MATCH (d)-[:SIMILAR_TO]-(d2:Document)
        OPTIONAL MATCH (u:User)-[:VIEWED]->(d)
        RETURN d,
               collect(DISTINCT t) as topics,
               collect(DISTINCT c) as concepts,
               collect(DISTINCT d2) as similar_documents,
               count(DISTINCT u) as view_count
        """
        
        query = GraphQuery(
            cypher=cypher,
            parameters={"document_id": document_id},
            metadata={"operation": "document_insights", "document_id": document_id}
        )
        
        return await self.execute_query(query)
    
    async def track_user_interaction(self, user_id: str, document_id: str, 
                                   interaction_type: str, metadata: Dict[str, Any] = None) -> GraphResult:
        """Track user interaction with document."""
        if metadata is None:
            metadata = {}
        
        # Add timestamp
        metadata["timestamp"] = datetime.now().isoformat()
        
        cypher = """
        MATCH (u:User {id: $user_id})
        MATCH (d:Document {id: $document_id})
        CREATE (u)-[r:$interaction_type $metadata]->(d)
        RETURN r
        """
        
        query = GraphQuery(
            cypher=cypher,
            parameters={
                "user_id": user_id,
                "document_id": document_id,
                "interaction_type": interaction_type,
                "metadata": metadata
            },
            metadata={
                "operation": "track_interaction",
                "interaction_type": interaction_type
            }
        )
        
        return await self.execute_query(query)
    
    async def get_graph_statistics(self) -> GraphResult:
        """Get comprehensive graph statistics."""
        cypher = """
        MATCH (n)
        RETURN labels(n)[0] as node_type, count(n) as count
        UNION ALL
        MATCH ()-[r]->()
        RETURN type(r) as relationship_type, count(r) as count
        """
        
        query = GraphQuery(
            cypher=cypher,
            parameters={},
            metadata={"operation": "graph_statistics"}
        )
        
        return await self.execute_query(query)
    
    async def health_check(self) -> bool:
        """Perform health check on Neo4j database."""
        try:
            result = await self.execute_query(GraphQuery(
                cypher="RETURN 1 as health",
                parameters={},
                timeout=5
            ))
            return result.success
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for monitoring."""
        avg_execution_time = (self.total_execution_time / self.query_count 
                            if self.query_count > 0 else 0)
        
        return {
            "total_queries": self.query_count,
            "total_execution_time": self.total_execution_time,
            "average_execution_time": avg_execution_time,
            "error_count": self.error_count,
            "error_rate": (self.error_count / self.query_count 
                          if self.query_count > 0 else 0),
            "success_rate": ((self.query_count - self.error_count) / self.query_count 
                           if self.query_count > 0 else 0)
        } 