"""
Knowledge Graph Schema for Universal Knowledge Platform
Defines the graph structure for recommendation engine and relationship discovery.
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class NodeType(Enum):
    """Types of nodes in the knowledge graph."""
    USER = "User"
    DOCUMENT = "Document"
    TOPIC = "Topic"
    TAG = "Tag"
    DEPARTMENT = "Department"
    CONCEPT = "Concept"
    ENTITY = "Entity"
    EVENT = "Event"
    LOCATION = "Location"
    ORGANIZATION = "Organization"


class RelationshipType(Enum):
    """Types of relationships in the knowledge graph."""
    # User relationships
    VIEWED = "VIEWED"
    LIKED = "LIKED"
    SHARED = "SHARED"
    BOOKMARKED = "BOOKMARKED"
    COMMENTED = "COMMENTED"
    
    # Document relationships
    CONTAINS = "CONTAINS"
    REFERENCES = "REFERENCES"
    SIMILAR_TO = "SIMILAR_TO"
    VERSION_OF = "VERSION_OF"
    TRANSLATED_TO = "TRANSLATED_TO"
    
    # Topic relationships
    BELONGS_TO = "BELONGS_TO"
    RELATED_TO = "RELATED_TO"
    PARENT_OF = "PARENT_OF"
    CHILD_OF = "CHILD_OF"
    
    # Entity relationships
    MENTIONED_IN = "MENTIONED_IN"
    WORKS_FOR = "WORKS_FOR"
    LOCATED_IN = "LOCATED_IN"
    PART_OF = "PART_OF"
    
    # Semantic relationships
    SEMANTICALLY_SIMILAR = "SEMANTICALLY_SIMILAR"
    CO_OCCURS_WITH = "CO_OCCURS_WITH"
    IMPLIES = "IMPLIES"
    CONTRADICTS = "CONTRADICTS"


@dataclass
class NodeSchema:
    """Schema definition for a graph node."""
    node_type: NodeType
    properties: Dict[str, Any]
    constraints: List[str]
    indexes: List[str]
    required_properties: List[str]


@dataclass
class RelationshipSchema:
    """Schema definition for a graph relationship."""
    relationship_type: RelationshipType
    source_node: NodeType
    target_node: NodeType
    properties: Dict[str, Any]
    constraints: List[str]


class KnowledgeGraphSchema:
    """Comprehensive knowledge graph schema for recommendation engine."""
    
    def __init__(self):
        self.nodes = self._define_nodes()
        self.relationships = self._define_relationships()
        self.constraints = self._define_constraints()
        self.indexes = self._define_indexes()
    
    def _define_nodes(self) -> Dict[NodeType, NodeSchema]:
        """Define all node schemas."""
        return {
            NodeType.USER: NodeSchema(
                node_type=NodeType.USER,
                properties={
                    "id": "string",
                    "email": "string",
                    "name": "string",
                    "department": "string",
                    "role": "string",
                    "preferences": "map",
                    "created_at": "datetime",
                    "last_active": "datetime",
                    "profile_completeness": "float"
                },
                constraints=[
                    "UNIQUE(user.id)",
                    "UNIQUE(user.email)"
                ],
                indexes=[
                    "user.id",
                    "user.email",
                    "user.department"
                ],
                required_properties=["id", "email", "name"]
            ),
            
            NodeType.DOCUMENT: NodeSchema(
                node_type=NodeType.DOCUMENT,
                properties={
                    "id": "string",
                    "title": "string",
                    "content": "text",
                    "content_type": "string",
                    "file_size": "long",
                    "language": "string",
                    "created_at": "datetime",
                    "updated_at": "datetime",
                    "author": "string",
                    "version": "string",
                    "access_level": "string",
                    "embedding": "vector",
                    "summary": "text",
                    "keywords": "list"
                },
                constraints=[
                    "UNIQUE(document.id)"
                ],
                indexes=[
                    "document.id",
                    "document.title",
                    "document.content_type",
                    "document.language",
                    "document.author"
                ],
                required_properties=["id", "title", "content_type"]
            ),
            
            NodeType.TOPIC: NodeSchema(
                node_type=NodeType.TOPIC,
                properties={
                    "id": "string",
                    "name": "string",
                    "description": "text",
                    "category": "string",
                    "importance": "float",
                    "created_at": "datetime",
                    "updated_at": "datetime",
                    "embedding": "vector"
                },
                constraints=[
                    "UNIQUE(topic.id)",
                    "UNIQUE(topic.name)"
                ],
                indexes=[
                    "topic.id",
                    "topic.name",
                    "topic.category"
                ],
                required_properties=["id", "name"]
            ),
            
            NodeType.TAG: NodeSchema(
                node_type=NodeType.TAG,
                properties={
                    "id": "string",
                    "name": "string",
                    "category": "string",
                    "usage_count": "long",
                    "created_at": "datetime"
                },
                constraints=[
                    "UNIQUE(tag.id)",
                    "UNIQUE(tag.name)"
                ],
                indexes=[
                    "tag.id",
                    "tag.name",
                    "tag.category"
                ],
                required_properties=["id", "name"]
            ),
            
            NodeType.DEPARTMENT: NodeSchema(
                node_type=NodeType.DEPARTMENT,
                properties={
                    "id": "string",
                    "name": "string",
                    "description": "text",
                    "manager": "string",
                    "created_at": "datetime"
                },
                constraints=[
                    "UNIQUE(department.id)",
                    "UNIQUE(department.name)"
                ],
                indexes=[
                    "department.id",
                    "department.name"
                ],
                required_properties=["id", "name"]
            ),
            
            NodeType.CONCEPT: NodeSchema(
                node_type=NodeType.CONCEPT,
                properties={
                    "id": "string",
                    "name": "string",
                    "definition": "text",
                    "category": "string",
                    "complexity": "float",
                    "embedding": "vector",
                    "created_at": "datetime"
                },
                constraints=[
                    "UNIQUE(concept.id)",
                    "UNIQUE(concept.name)"
                ],
                indexes=[
                    "concept.id",
                    "concept.name",
                    "concept.category"
                ],
                required_properties=["id", "name"]
            ),
            
            NodeType.ENTITY: NodeSchema(
                node_type=NodeType.ENTITY,
                properties={
                    "id": "string",
                    "name": "string",
                    "type": "string",
                    "confidence": "float",
                    "source": "string",
                    "created_at": "datetime"
                },
                constraints=[
                    "UNIQUE(entity.id)"
                ],
                indexes=[
                    "entity.id",
                    "entity.name",
                    "entity.type"
                ],
                required_properties=["id", "name", "type"]
            )
        }
    
    def _define_relationships(self) -> List[RelationshipSchema]:
        """Define all relationship schemas."""
        return [
            # User-Document relationships
            RelationshipSchema(
                relationship_type=RelationshipType.VIEWED,
                source_node=NodeType.USER,
                target_node=NodeType.DOCUMENT,
                properties={
                    "timestamp": "datetime",
                    "duration": "long",
                    "session_id": "string"
                },
                constraints=[]
            ),
            
            RelationshipSchema(
                relationship_type=RelationshipType.LIKED,
                source_node=NodeType.USER,
                target_node=NodeType.DOCUMENT,
                properties={
                    "timestamp": "datetime",
                    "rating": "int"
                },
                constraints=[]
            ),
            
            RelationshipSchema(
                relationship_type=RelationshipType.SHARED,
                source_node=NodeType.USER,
                target_node=NodeType.DOCUMENT,
                properties={
                    "timestamp": "datetime",
                    "platform": "string",
                    "audience": "string"
                },
                constraints=[]
            ),
            
            # Document relationships
            RelationshipSchema(
                relationship_type=RelationshipType.CONTAINS,
                source_node=NodeType.DOCUMENT,
                target_node=NodeType.CONCEPT,
                properties={
                    "frequency": "int",
                    "relevance": "float"
                },
                constraints=[]
            ),
            
            RelationshipSchema(
                relationship_type=RelationshipType.REFERENCES,
                source_node=NodeType.DOCUMENT,
                target_node=NodeType.DOCUMENT,
                properties={
                    "reference_type": "string",
                    "context": "text"
                },
                constraints=[]
            ),
            
            RelationshipSchema(
                relationship_type=RelationshipType.SIMILAR_TO,
                source_node=NodeType.DOCUMENT,
                target_node=NodeType.DOCUMENT,
                properties={
                    "similarity_score": "float",
                    "algorithm": "string"
                },
                constraints=[]
            ),
            
            # Topic relationships
            RelationshipSchema(
                relationship_type=RelationshipType.BELONGS_TO,
                source_node=NodeType.DOCUMENT,
                target_node=NodeType.TOPIC,
                properties={
                    "confidence": "float",
                    "algorithm": "string"
                },
                constraints=[]
            ),
            
            RelationshipSchema(
                relationship_type=RelationshipType.RELATED_TO,
                source_node=NodeType.TOPIC,
                target_node=NodeType.TOPIC,
                properties={
                    "strength": "float",
                    "relationship_type": "string"
                },
                constraints=[]
            ),
            
            # Entity relationships
            RelationshipSchema(
                relationship_type=RelationshipType.MENTIONED_IN,
                source_node=NodeType.ENTITY,
                target_node=NodeType.DOCUMENT,
                properties={
                    "frequency": "int",
                    "context": "text",
                    "sentiment": "float"
                },
                constraints=[]
            ),
            
            # Semantic relationships
            RelationshipSchema(
                relationship_type=RelationshipType.SEMANTICALLY_SIMILAR,
                source_node=NodeType.CONCEPT,
                target_node=NodeType.CONCEPT,
                properties={
                    "similarity_score": "float",
                    "algorithm": "string"
                },
                constraints=[]
            )
        ]
    
    def _define_constraints(self) -> List[str]:
        """Define database constraints."""
        return [
            # User constraints
            "CREATE CONSTRAINT user_id_unique IF NOT EXISTS FOR (u:User) REQUIRE u.id IS UNIQUE",
            "CREATE CONSTRAINT user_email_unique IF NOT EXISTS FOR (u:User) REQUIRE u.email IS UNIQUE",
            
            # Document constraints
            "CREATE CONSTRAINT document_id_unique IF NOT EXISTS FOR (d:Document) REQUIRE d.id IS UNIQUE",
            
            # Topic constraints
            "CREATE CONSTRAINT topic_id_unique IF NOT EXISTS FOR (t:Topic) REQUIRE t.id IS UNIQUE",
            "CREATE CONSTRAINT topic_name_unique IF NOT EXISTS FOR (t:Topic) REQUIRE t.name IS UNIQUE",
            
            # Tag constraints
            "CREATE CONSTRAINT tag_id_unique IF NOT EXISTS FOR (t:Tag) REQUIRE t.id IS UNIQUE",
            "CREATE CONSTRAINT tag_name_unique IF NOT EXISTS FOR (t:Tag) REQUIRE t.name IS UNIQUE",
            
            # Department constraints
            "CREATE CONSTRAINT department_id_unique IF NOT EXISTS FOR (d:Department) REQUIRE d.id IS UNIQUE",
            "CREATE CONSTRAINT department_name_unique IF NOT EXISTS FOR (d:Department) REQUIRE d.name IS UNIQUE",
            
            # Concept constraints
            "CREATE CONSTRAINT concept_id_unique IF NOT EXISTS FOR (c:Concept) REQUIRE c.id IS UNIQUE",
            "CREATE CONSTRAINT concept_name_unique IF NOT EXISTS FOR (c:Concept) REQUIRE c.name IS UNIQUE",
            
            # Entity constraints
            "CREATE CONSTRAINT entity_id_unique IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE"
        ]
    
    def _define_indexes(self) -> List[str]:
        """Define database indexes for performance."""
        return [
            # User indexes
            "CREATE INDEX user_email_index IF NOT EXISTS FOR (u:User) ON (u.email)",
            "CREATE INDEX user_department_index IF NOT EXISTS FOR (u:User) ON (u.department)",
            "CREATE INDEX user_last_active_index IF NOT EXISTS FOR (u:User) ON (u.last_active)",
            
            # Document indexes
            "CREATE INDEX document_title_index IF NOT EXISTS FOR (d:Document) ON (d.title)",
            "CREATE INDEX document_content_type_index IF NOT EXISTS FOR (d:Document) ON (d.content_type)",
            "CREATE INDEX document_language_index IF NOT EXISTS FOR (d:Document) ON (d.language)",
            "CREATE INDEX document_author_index IF NOT EXISTS FOR (d:Document) ON (d.author)",
            "CREATE INDEX document_created_at_index IF NOT EXISTS FOR (d:Document) ON (d.created_at)",
            
            # Topic indexes
            "CREATE INDEX topic_category_index IF NOT EXISTS FOR (t:Topic) ON (t.category)",
            "CREATE INDEX topic_importance_index IF NOT EXISTS FOR (t:Topic) ON (t.importance)",
            
            # Tag indexes
            "CREATE INDEX tag_category_index IF NOT EXISTS FOR (t:Tag) ON (t.category)",
            "CREATE INDEX tag_usage_count_index IF NOT EXISTS FOR (t:Tag) ON (t.usage_count)",
            
            # Concept indexes
            "CREATE INDEX concept_category_index IF NOT EXISTS FOR (c:Concept) ON (c.category)",
            "CREATE INDEX concept_complexity_index IF NOT EXISTS FOR (c:Concept) ON (c.complexity)",
            
            # Entity indexes
            "CREATE INDEX entity_type_index IF NOT EXISTS FOR (e:Entity) ON (e.type)",
            "CREATE INDEX entity_confidence_index IF NOT EXISTS FOR (e:Entity) ON (e.confidence)",
            
            # Relationship indexes
            "CREATE INDEX viewed_timestamp_index IF NOT EXISTS FOR ()-[r:VIEWED]-() ON (r.timestamp)",
            "CREATE INDEX liked_timestamp_index IF NOT EXISTS FOR ()-[r:LIKED]-() ON (r.timestamp)",
            "CREATE INDEX contains_frequency_index IF NOT EXISTS FOR ()-[r:CONTAINS]-() ON (r.frequency)",
            "CREATE INDEX similar_score_index IF NOT EXISTS FOR ()-[r:SIMILAR_TO]-() ON (r.similarity_score)"
        ]
    
    def get_cypher_schema(self) -> str:
        """Generate Cypher schema creation script."""
        schema_script = []
        
        # Add constraints
        schema_script.extend(self.constraints)
        
        # Add indexes
        schema_script.extend(self.indexes)
        
        return "\n".join(schema_script)
    
    def validate_node_properties(self, node_type: NodeType, properties: Dict[str, Any]) -> bool:
        """Validate node properties against schema."""
        if node_type not in self.nodes:
            return False
        
        schema = self.nodes[node_type]
        required_props = set(schema.required_properties)
        provided_props = set(properties.keys())
        
        return required_props.issubset(provided_props)
    
    def validate_relationship_properties(self, relationship_type: RelationshipType, properties: Dict[str, Any]) -> bool:
        """Validate relationship properties against schema."""
        for rel_schema in self.relationships:
            if rel_schema.relationship_type == relationship_type:
                # Basic validation - could be enhanced
                return True
        return False 