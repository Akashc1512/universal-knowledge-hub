"""
Knowledge Graph Data Migration System
Handles migration of existing content and user data to Neo4j knowledge graph.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Generator
from datetime import datetime
import json
import time
from dataclasses import dataclass
from pathlib import Path
import hashlib

from core.knowledge_graph.client import Neo4jClient, GraphQuery, GraphResult
from core.knowledge_graph.schema import KnowledgeGraphSchema, NodeType, RelationshipType

logger = logging.getLogger(__name__)


@dataclass
class MigrationConfig:
    """Configuration for data migration."""
    batch_size: int = 1000
    max_workers: int = 4
    retry_attempts: int = 3
    timeout: int = 300  # 5 minutes
    dry_run: bool = False


@dataclass
class MigrationStats:
    """Migration statistics and progress tracking."""
    total_nodes: int = 0
    total_relationships: int = 0
    nodes_created: int = 0
    relationships_created: int = 0
    errors: int = 0
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    
    @property
    def duration(self) -> float:
        """Get migration duration in seconds."""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return 0.0
    
    @property
    def success_rate(self) -> float:
        """Get migration success rate."""
        total = self.total_nodes + self.total_relationships
        if total == 0:
            return 0.0
        successful = self.nodes_created + self.relationships_created
        return (successful / total) * 100


class DataMigrator:
    """Comprehensive data migration system for knowledge graph."""
    
    def __init__(self, graph_client: Neo4jClient, config: MigrationConfig = None):
        self.graph_client = graph_client
        self.config = config or MigrationConfig()
        self.schema = KnowledgeGraphSchema()
        self.stats = MigrationStats()
        
    async def initialize_graph(self) -> bool:
        """Initialize the knowledge graph with schema and constraints."""
        try:
            logger.info("🔧 Initializing knowledge graph schema...")
            
            # Create constraints and indexes
            schema_script = self.schema.get_cypher_schema()
            
            # Split into individual statements
            statements = [stmt.strip() for stmt in schema_script.split('\n') if stmt.strip()]
            
            for statement in statements:
                if statement.startswith('CREATE'):
                    query = GraphQuery(
                        cypher=statement,
                        parameters={},
                        timeout=self.config.timeout,
                        metadata={"operation": "schema_initialization"}
                    )
                    
                    result = await self.graph_client.execute_query(query)
                    if not result.success:
                        logger.warning(f"Schema statement failed: {statement[:100]}...")
            
            logger.info("✅ Knowledge graph schema initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize knowledge graph: {e}")
            return False
    
    async def migrate_users(self, users_data: List[Dict[str, Any]]) -> MigrationStats:
        """Migrate user data to knowledge graph."""
        logger.info(f"👥 Migrating {len(users_data)} users...")
        
        stats = MigrationStats()
        stats.start_time = datetime.now()
        stats.total_nodes = len(users_data)
        
        for i, user_data in enumerate(users_data):
            try:
                # Validate user data against schema
                if not self.schema.validate_node_properties(NodeType.USER, user_data):
                    logger.warning(f"Invalid user data: {user_data.get('id', 'unknown')}")
                    stats.errors += 1
                    continue
                
                # Create user node
                result = await self.graph_client.create_node("User", user_data)
                
                if result.success:
                    stats.nodes_created += 1
                    if (i + 1) % 100 == 0:
                        logger.info(f"✅ Created {i + 1}/{len(users_data)} users")
                else:
                    stats.errors += 1
                    logger.error(f"Failed to create user {user_data.get('id')}: {result.error}")
                
            except Exception as e:
                stats.errors += 1
                logger.error(f"Error migrating user {user_data.get('id', 'unknown')}: {e}")
        
        stats.end_time = datetime.now()
        logger.info(f"✅ User migration complete: {stats.nodes_created} created, {stats.errors} errors")
        return stats
    
    async def migrate_documents(self, documents_data: List[Dict[str, Any]]) -> MigrationStats:
        """Migrate document data to knowledge graph."""
        logger.info(f"📄 Migrating {len(documents_data)} documents...")
        
        stats = MigrationStats()
        stats.start_time = datetime.now()
        stats.total_nodes = len(documents_data)
        
        for i, doc_data in enumerate(documents_data):
            try:
                # Validate document data
                if not self.schema.validate_node_properties(NodeType.DOCUMENT, doc_data):
                    logger.warning(f"Invalid document data: {doc_data.get('id', 'unknown')}")
                    stats.errors += 1
                    continue
                
                # Create document node
                result = await self.graph_client.create_node("Document", doc_data)
                
                if result.success:
                    stats.nodes_created += 1
                    if (i + 1) % 100 == 0:
                        logger.info(f"✅ Created {i + 1}/{len(documents_data)} documents")
                else:
                    stats.errors += 1
                    logger.error(f"Failed to create document {doc_data.get('id')}: {result.error}")
                
            except Exception as e:
                stats.errors += 1
                logger.error(f"Error migrating document {doc_data.get('id', 'unknown')}: {e}")
        
        stats.end_time = datetime.now()
        logger.info(f"✅ Document migration complete: {stats.nodes_created} created, {stats.errors} errors")
        return stats
    
    async def migrate_topics(self, topics_data: List[Dict[str, Any]]) -> MigrationStats:
        """Migrate topic data to knowledge graph."""
        logger.info(f"🏷️ Migrating {len(topics_data)} topics...")
        
        stats = MigrationStats()
        stats.start_time = datetime.now()
        stats.total_nodes = len(topics_data)
        
        for i, topic_data in enumerate(topics_data):
            try:
                # Validate topic data
                if not self.schema.validate_node_properties(NodeType.TOPIC, topic_data):
                    logger.warning(f"Invalid topic data: {topic_data.get('id', 'unknown')}")
                    stats.errors += 1
                    continue
                
                # Create topic node
                result = await self.graph_client.create_node("Topic", topic_data)
                
                if result.success:
                    stats.nodes_created += 1
                    if (i + 1) % 100 == 0:
                        logger.info(f"✅ Created {i + 1}/{len(topics_data)} topics")
                else:
                    stats.errors += 1
                    logger.error(f"Failed to create topic {topic_data.get('id')}: {result.error}")
                
            except Exception as e:
                stats.errors += 1
                logger.error(f"Error migrating topic {topic_data.get('id', 'unknown')}: {e}")
        
        stats.end_time = datetime.now()
        logger.info(f"✅ Topic migration complete: {stats.nodes_created} created, {stats.errors} errors")
        return stats
    
    async def create_relationships(self, relationships_data: List[Dict[str, Any]]) -> MigrationStats:
        """Create relationships between nodes in the knowledge graph."""
        logger.info(f"🔗 Creating {len(relationships_data)} relationships...")
        
        stats = MigrationStats()
        stats.start_time = datetime.now()
        stats.total_relationships = len(relationships_data)
        
        for i, rel_data in enumerate(relationships_data):
            try:
                source_id = rel_data.get('source_id')
                target_id = rel_data.get('target_id')
                relationship_type = rel_data.get('relationship_type')
                properties = rel_data.get('properties', {})
                
                if not all([source_id, target_id, relationship_type]):
                    logger.warning(f"Incomplete relationship data: {rel_data}")
                    stats.errors += 1
                    continue
                
                # Create relationship
                result = await self.graph_client.create_relationship(
                    source_id=source_id,
                    target_id=target_id,
                    relationship_type=relationship_type,
                    properties=properties
                )
                
                if result.success:
                    stats.relationships_created += 1
                    if (i + 1) % 100 == 0:
                        logger.info(f"✅ Created {i + 1}/{len(relationships_data)} relationships")
                else:
                    stats.errors += 1
                    logger.error(f"Failed to create relationship: {result.error}")
                
            except Exception as e:
                stats.errors += 1
                logger.error(f"Error creating relationship: {e}")
        
        stats.end_time = datetime.now()
        logger.info(f"✅ Relationship creation complete: {stats.relationships_created} created, {stats.errors} errors")
        return stats
    
    async def discover_relationships(self) -> MigrationStats:
        """Automatically discover relationships based on content analysis."""
        logger.info("🔍 Discovering relationships automatically...")
        
        stats = MigrationStats()
        stats.start_time = datetime.now()
        
        # Discover document-topic relationships
        topic_discovery_query = GraphQuery(
            cypher="""
            MATCH (d:Document)
            MATCH (t:Topic)
            WHERE d.content CONTAINS t.name OR d.title CONTAINS t.name
            CREATE (d)-[:BELONGS_TO {confidence: 0.8, algorithm: 'keyword_matching'}]->(t)
            RETURN count(*) as relationships_created
            """,
            parameters={},
            metadata={"operation": "topic_discovery"}
        )
        
        result = await self.graph_client.execute_query(topic_discovery_query)
        if result.success and result.data:
            stats.relationships_created += result.data[0].get('relationships_created', 0)
        
        # Discover document similarity relationships
        similarity_discovery_query = GraphQuery(
            cypher="""
            MATCH (d1:Document)
            MATCH (d2:Document)
            WHERE d1.id < d2.id
            AND d1.content_type = d2.content_type
            AND d1.language = d2.language
            WITH d1, d2, 
                 size([word IN split(d1.title, ' ') WHERE word IN split(d2.title, ' ')]) as common_words
            WHERE common_words > 2
            CREATE (d1)-[:SIMILAR_TO {score: common_words/10.0, algorithm: 'title_similarity'}]-(d2)
            RETURN count(*) as relationships_created
            """,
            parameters={},
            metadata={"operation": "similarity_discovery"}
        )
        
        result = await self.graph_client.execute_query(similarity_discovery_query)
        if result.success and result.data:
            stats.relationships_created += result.data[0].get('relationships_created', 0)
        
        stats.end_time = datetime.now()
        logger.info(f"✅ Relationship discovery complete: {stats.relationships_created} relationships discovered")
        return stats
    
    async def validate_migration(self) -> Dict[str, Any]:
        """Validate the migration by checking graph statistics."""
        logger.info("🔍 Validating migration...")
        
        validation_query = GraphQuery(
            cypher="""
            MATCH (n)
            RETURN labels(n)[0] as node_type, count(n) as count
            UNION ALL
            MATCH ()-[r]->()
            RETURN type(r) as relationship_type, count(r) as count
            """,
            parameters={},
            metadata={"operation": "migration_validation"}
        )
        
        result = await self.graph_client.execute_query(validation_query)
        
        if result.success:
            validation_data = {}
            for record in result.data:
                if 'node_type' in record:
                    validation_data[f"nodes_{record['node_type']}"] = record['count']
                elif 'relationship_type' in record:
                    validation_data[f"relationships_{record['relationship_type']}"] = record['count']
            
            logger.info(f"✅ Migration validation complete: {validation_data}")
            return validation_data
        else:
            logger.error(f"❌ Migration validation failed: {result.error}")
            return {}
    
    async def run_full_migration(self, data_source: Dict[str, Any]) -> Dict[str, Any]:
        """Run complete migration process."""
        logger.info("🚀 Starting full knowledge graph migration...")
        
        overall_stats = MigrationStats()
        overall_stats.start_time = datetime.now()
        
        try:
            # Initialize graph schema
            if not await self.initialize_graph():
                raise Exception("Failed to initialize graph schema")
            
            # Migrate users
            if 'users' in data_source:
                user_stats = await self.migrate_users(data_source['users'])
                overall_stats.nodes_created += user_stats.nodes_created
                overall_stats.errors += user_stats.errors
            
            # Migrate documents
            if 'documents' in data_source:
                doc_stats = await self.migrate_documents(data_source['documents'])
                overall_stats.nodes_created += doc_stats.nodes_created
                overall_stats.errors += doc_stats.errors
            
            # Migrate topics
            if 'topics' in data_source:
                topic_stats = await self.migrate_topics(data_source['topics'])
                overall_stats.nodes_created += topic_stats.nodes_created
                overall_stats.errors += topic_stats.errors
            
            # Create explicit relationships
            if 'relationships' in data_source:
                rel_stats = await self.create_relationships(data_source['relationships'])
                overall_stats.relationships_created += rel_stats.relationships_created
                overall_stats.errors += rel_stats.errors
            
            # Discover automatic relationships
            discovery_stats = await self.discover_relationships()
            overall_stats.relationships_created += discovery_stats.relationships_created
            overall_stats.errors += discovery_stats.errors
            
            # Validate migration
            validation_data = await self.validate_migration()
            
            overall_stats.end_time = datetime.now()
            
            migration_report = {
                "success": overall_stats.success_rate > 95.0,
                "duration_seconds": overall_stats.duration,
                "nodes_created": overall_stats.nodes_created,
                "relationships_created": overall_stats.relationships_created,
                "errors": overall_stats.errors,
                "success_rate": overall_stats.success_rate,
                "validation_data": validation_data
            }
            
            logger.info(f"✅ Migration complete: {migration_report}")
            return migration_report
            
        except Exception as e:
            logger.error(f"❌ Migration failed: {e}")
            overall_stats.end_time = datetime.now()
            return {
                "success": False,
                "error": str(e),
                "duration_seconds": overall_stats.duration,
                "nodes_created": overall_stats.nodes_created,
                "relationships_created": overall_stats.relationships_created,
                "errors": overall_stats.errors
            }


class SampleDataGenerator:
    """Generate sample data for testing and development."""
    
    @staticmethod
    def generate_sample_users(count: int = 100) -> List[Dict[str, Any]]:
        """Generate sample user data."""
        users = []
        departments = ["Engineering", "Marketing", "Sales", "HR", "Finance", "Operations"]
        roles = ["Manager", "Senior", "Junior", "Lead", "Director"]
        
        for i in range(count):
            user = {
                "id": f"user_{i:04d}",
                "email": f"user{i:04d}@company.com",
                "name": f"User {i:04d}",
                "department": departments[i % len(departments)],
                "role": roles[i % len(roles)],
                "preferences": {
                    "topics": ["technology", "business", "innovation"],
                    "languages": ["en", "es"],
                    "content_types": ["document", "video", "article"]
                },
                "created_at": datetime.now().isoformat(),
                "last_active": datetime.now().isoformat(),
                "profile_completeness": 0.8
            }
            users.append(user)
        
        return users
    
    @staticmethod
    def generate_sample_documents(count: int = 500) -> List[Dict[str, Any]]:
        """Generate sample document data."""
        documents = []
        content_types = ["pdf", "docx", "txt", "html", "markdown"]
        languages = ["en", "es", "fr", "de"]
        topics = ["technology", "business", "science", "health", "education", "finance"]
        
        for i in range(count):
            doc = {
                "id": f"doc_{i:06d}",
                "title": f"Sample Document {i:06d}",
                "content": f"This is sample content for document {i:06d} with some interesting information.",
                "content_type": content_types[i % len(content_types)],
                "file_size": 1024 * (i % 1000 + 1),
                "language": languages[i % len(languages)],
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat(),
                "author": f"user_{(i % 100):04d}",
                "version": "1.0",
                "access_level": "public",
                "summary": f"Summary for document {i:06d}",
                "keywords": [topics[i % len(topics)], "sample", "document"]
            }
            documents.append(doc)
        
        return documents
    
    @staticmethod
    def generate_sample_topics(count: int = 50) -> List[Dict[str, Any]]:
        """Generate sample topic data."""
        topics = []
        categories = ["technology", "business", "science", "health", "education", "finance", "arts", "sports"]
        
        for i in range(count):
            topic = {
                "id": f"topic_{i:03d}",
                "name": f"Topic {i:03d}",
                "description": f"Description for topic {i:03d}",
                "category": categories[i % len(categories)],
                "importance": (i % 10) / 10.0,
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat()
            }
            topics.append(topic)
        
        return topics
    
    @staticmethod
    def generate_sample_relationships() -> List[Dict[str, Any]]:
        """Generate sample relationship data."""
        relationships = []
        
        # User-Document relationships (VIEWED)
        for i in range(1000):
            rel = {
                "source_id": f"user_{(i % 100):04d}",
                "target_id": f"doc_{(i % 500):06d}",
                "relationship_type": "VIEWED",
                "properties": {
                    "timestamp": datetime.now().isoformat(),
                    "duration": (i % 300) + 30,
                    "session_id": f"session_{i:06d}"
                }
            }
            relationships.append(rel)
        
        # Document-Topic relationships (BELONGS_TO)
        for i in range(500):
            rel = {
                "source_id": f"doc_{i:06d}",
                "target_id": f"topic_{(i % 50):03d}",
                "relationship_type": "BELONGS_TO",
                "properties": {
                    "confidence": 0.8 + (i % 20) / 100.0,
                    "algorithm": "keyword_matching"
                }
            }
            relationships.append(rel)
        
        return relationships 