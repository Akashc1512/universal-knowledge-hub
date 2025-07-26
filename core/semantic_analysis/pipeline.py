"""
Semantic Analysis Pipeline for Universal Knowledge Platform
Implements NLP pipeline for entity extraction, semantic similarity, and relationship discovery.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import time
import hashlib
import json
from datetime import datetime

logger = logging.getLogger(__name__)

try:
    import spacy
    from spacy.tokens import Doc
    from spacy.language import Language
except ImportError:
    spacy = None
    Doc = None
    Language = None

try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
except ImportError:
    SentenceTransformer = None
    np = None

from core.knowledge_graph.client import Neo4jClient, GraphQuery


@dataclass
class Entity:
    """Extracted entity from text."""
    text: str
    label: str
    start: int
    end: int
    confidence: float
    source: str


@dataclass
class SemanticSimilarity:
    """Semantic similarity between two documents."""
    doc1_id: str
    doc2_id: str
    similarity_score: float
    algorithm: str
    metadata: Dict[str, Any]


@dataclass
class ContentAnalysis:
    """Complete content analysis result."""
    document_id: str
    entities: List[Entity]
    topics: List[str]
    keywords: List[str]
    summary: str
    sentiment: float
    complexity: float
    processing_time: float


class SemanticAnalysisPipeline:
    """Comprehensive semantic analysis pipeline."""
    
    def __init__(self, graph_client: Neo4jClient, model_name: str = "en_core_web_sm"):
        self.graph_client = graph_client
        self.model_name = model_name
        self.nlp = None
        self.sentence_transformer = None
        self.initialized = False
        
        # Performance tracking
        self.processed_documents = 0
        self.total_processing_time = 0.0
        self.entity_count = 0
        
    async def initialize(self) -> bool:
        """Initialize the semantic analysis pipeline."""
        try:
            logger.info("🔧 Initializing semantic analysis pipeline...")
            
            # Initialize spaCy
            if spacy is not None:
                try:
                    self.nlp = spacy.load(self.model_name)
                    logger.info(f"✅ Loaded spaCy model: {self.model_name}")
                except OSError:
                    logger.warning(f"spaCy model {self.model_name} not found, downloading...")
                    spacy.cli.download(self.model_name)
                    self.nlp = spacy.load(self.model_name)
            else:
                logger.warning("spaCy not available, entity extraction disabled")
            
            # Initialize sentence transformer
            if SentenceTransformer is not None:
                self.sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')
                logger.info("✅ Loaded sentence transformer model")
            else:
                logger.warning("sentence-transformers not available, semantic similarity disabled")
            
            self.initialized = True
            logger.info("✅ Semantic analysis pipeline initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize semantic analysis pipeline: {e}")
            return False
    
    def extract_entities(self, text: str) -> List[Entity]:
        """Extract named entities from text using spaCy."""
        if not self.nlp:
            return []
        
        try:
            doc = self.nlp(text)
            entities = []
            
            for ent in doc.ents:
                entity = Entity(
                    text=ent.text,
                    label=ent.label_,
                    start=ent.start_char,
                    end=ent.end_char,
                    confidence=0.8,  # spaCy doesn't provide confidence scores
                    source="spacy"
                )
                entities.append(entity)
            
            return entities
            
        except Exception as e:
            logger.error(f"Entity extraction failed: {e}")
            return []
    
    def calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity between two texts."""
        if not self.sentence_transformer:
            return 0.0
        
        try:
            # Generate embeddings
            embeddings = self.sentence_transformer.encode([text1, text2])
            
            # Calculate cosine similarity
            similarity = np.dot(embeddings[0], embeddings[1]) / (
                np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
            )
            
            return float(similarity)
            
        except Exception as e:
            logger.error(f"Semantic similarity calculation failed: {e}")
            return 0.0
    
    def extract_keywords(self, text: str, max_keywords: int = 10) -> List[str]:
        """Extract keywords from text using spaCy."""
        if not self.nlp:
            return []
        
        try:
            doc = self.nlp(text)
            
            # Extract nouns, adjectives, and verbs
            keywords = []
            for token in doc:
                if (token.pos_ in ['NOUN', 'ADJ', 'VERB'] and 
                    not token.is_stop and 
                    len(token.text) > 2):
                    keywords.append(token.text.lower())
            
            # Remove duplicates and limit
            keywords = list(set(keywords))[:max_keywords]
            return keywords
            
        except Exception as e:
            logger.error(f"Keyword extraction failed: {e}")
            return []
    
    def analyze_content(self, document_id: str, text: str) -> ContentAnalysis:
        """Perform comprehensive content analysis."""
        start_time = time.time()
        
        try:
            # Extract entities
            entities = self.extract_entities(text)
            
            # Extract keywords
            keywords = self.extract_keywords(text)
            
            # Generate topics from keywords
            topics = keywords[:5]  # Use top 5 keywords as topics
            
            # Simple summary (first 200 characters)
            summary = text[:200] + "..." if len(text) > 200 else text
            
            # Simple sentiment analysis (placeholder)
            sentiment = 0.5  # Neutral sentiment
            
            # Calculate complexity (average word length)
            words = text.split()
            complexity = sum(len(word) for word in words) / len(words) if words else 0
            
            processing_time = time.time() - start_time
            
            # Update statistics
            self.processed_documents += 1
            self.total_processing_time += processing_time
            self.entity_count += len(entities)
            
            return ContentAnalysis(
                document_id=document_id,
                entities=entities,
                topics=topics,
                keywords=keywords,
                summary=summary,
                sentiment=sentiment,
                complexity=complexity,
                processing_time=processing_time
            )
            
        except Exception as e:
            logger.error(f"Content analysis failed for document {document_id}: {e}")
            return ContentAnalysis(
                document_id=document_id,
                entities=[],
                topics=[],
                keywords=[],
                summary="",
                sentiment=0.0,
                complexity=0.0,
                processing_time=time.time() - start_time
            )
    
    async def process_document(self, document_id: str, text: str) -> bool:
        """Process a single document and update the knowledge graph."""
        try:
            # Analyze content
            analysis = self.analyze_content(document_id, text)
            
            # Update document with analysis results
            update_query = GraphQuery(
                cypher="""
                MATCH (d:Document {id: $document_id})
                SET d.keywords = $keywords,
                    d.topics = $topics,
                    d.summary = $summary,
                    d.sentiment = $sentiment,
                    d.complexity = $complexity,
                    d.analysis_timestamp = $timestamp
                RETURN d
                """,
                parameters={
                    "document_id": document_id,
                    "keywords": analysis.keywords,
                    "topics": analysis.topics,
                    "summary": analysis.summary,
                    "sentiment": analysis.sentiment,
                    "complexity": analysis.complexity,
                    "timestamp": datetime.now().isoformat()
                },
                metadata={"operation": "document_analysis_update"}
            )
            
            result = await self.graph_client.execute_query(update_query)
            if not result.success:
                logger.error(f"Failed to update document {document_id}: {result.error}")
                return False
            
            # Create entity nodes and relationships
            for entity in analysis.entities:
                await self._create_entity_node(entity, document_id)
            
            # Create topic relationships
            for topic in analysis.topics:
                await self._create_topic_relationship(document_id, topic)
            
            logger.debug(f"✅ Processed document {document_id}: {len(analysis.entities)} entities, {len(analysis.topics)} topics")
            return True
            
        except Exception as e:
            logger.error(f"Document processing failed for {document_id}: {e}")
            return False
    
    async def _create_entity_node(self, entity: Entity, document_id: str) -> bool:
        """Create entity node and relationship to document."""
        try:
            # Create entity node
            entity_data = {
                "id": f"entity_{hashlib.md5(entity.text.encode()).hexdigest()[:8]}",
                "name": entity.text,
                "type": entity.label,
                "confidence": entity.confidence,
                "source": entity.source
            }
            
            create_entity_query = GraphQuery(
                cypher="""
                MERGE (e:Entity {id: $entity_id})
                SET e.name = $name, e.type = $type, e.confidence = $confidence, e.source = $source
                RETURN e
                """,
                parameters=entity_data,
                metadata={"operation": "entity_creation"}
            )
            
            result = await self.graph_client.execute_query(create_entity_query)
            if not result.success:
                return False
            
            # Create relationship to document
            rel_query = GraphQuery(
                cypher="""
                MATCH (e:Entity {id: $entity_id})
                MATCH (d:Document {id: $document_id})
                MERGE (e)-[r:MENTIONED_IN {frequency: 1}]->(d)
                ON CREATE SET r.frequency = 1
                ON MATCH SET r.frequency = r.frequency + 1
                RETURN r
                """,
                parameters={
                    "entity_id": entity_data["id"],
                    "document_id": document_id
                },
                metadata={"operation": "entity_document_relationship"}
            )
            
            result = await self.graph_client.execute_query(rel_query)
            return result.success
            
        except Exception as e:
            logger.error(f"Failed to create entity node: {e}")
            return False
    
    async def _create_topic_relationship(self, document_id: str, topic_name: str) -> bool:
        """Create relationship between document and topic."""
        try:
            # Create topic node if it doesn't exist
            topic_data = {
                "id": f"topic_{hashlib.md5(topic_name.encode()).hexdigest()[:8]}",
                "name": topic_name,
                "category": "auto_generated",
                "importance": 0.5
            }
            
            create_topic_query = GraphQuery(
                cypher="""
                MERGE (t:Topic {id: $topic_id})
                SET t.name = $name, t.category = $category, t.importance = $importance
                RETURN t
                """,
                parameters=topic_data,
                metadata={"operation": "topic_creation"}
            )
            
            result = await self.graph_client.execute_query(create_topic_query)
            if not result.success:
                return False
            
            # Create relationship
            rel_query = GraphQuery(
                cypher="""
                MATCH (d:Document {id: $document_id})
                MATCH (t:Topic {id: $topic_id})
                MERGE (d)-[r:BELONGS_TO {confidence: 0.8, algorithm: 'semantic_analysis'}]->(t)
                RETURN r
                """,
                parameters={
                    "document_id": document_id,
                    "topic_id": topic_data["id"]
                },
                metadata={"operation": "document_topic_relationship"}
            )
            
            result = await self.graph_client.execute_query(rel_query)
            return result.success
            
        except Exception as e:
            logger.error(f"Failed to create topic relationship: {e}")
            return False
    
    async def find_semantic_similarities(self, document_id: str, threshold: float = 0.7) -> List[SemanticSimilarity]:
        """Find semantically similar documents."""
        try:
            # Get document content
            doc_query = GraphQuery(
                cypher="""
                MATCH (d:Document {id: $document_id})
                RETURN d.title as title, d.content as content
                """,
                parameters={"document_id": document_id},
                metadata={"operation": "document_retrieval"}
            )
            
            result = await self.graph_client.execute_query(doc_query)
            if not result.success or not result.data:
                return []
            
            doc_content = result.data[0]
            doc_text = f"{doc_content['title']} {doc_content['content']}"
            
            # Get other documents for comparison
            other_docs_query = GraphQuery(
                cypher="""
                MATCH (d:Document)
                WHERE d.id <> $document_id
                RETURN d.id as id, d.title as title, d.content as content
                LIMIT 100
                """,
                parameters={"document_id": document_id},
                metadata={"operation": "similar_documents_retrieval"}
            )
            
            result = await self.graph_client.execute_query(other_docs_query)
            if not result.success:
                return []
            
            similarities = []
            for other_doc in result.data:
                other_text = f"{other_doc['title']} {other_doc['content']}"
                similarity_score = self.calculate_semantic_similarity(doc_text, other_text)
                
                if similarity_score >= threshold:
                    similarity = SemanticSimilarity(
                        doc1_id=document_id,
                        doc2_id=other_doc['id'],
                        similarity_score=similarity_score,
                        algorithm="sentence_transformer",
                        metadata={"threshold": threshold}
                    )
                    similarities.append(similarity)
            
            # Sort by similarity score
            similarities.sort(key=lambda x: x.similarity_score, reverse=True)
            return similarities[:10]  # Return top 10 similar documents
            
        except Exception as e:
            logger.error(f"Semantic similarity search failed: {e}")
            return []
    
    async def create_semantic_relationships(self, document_id: str) -> bool:
        """Create semantic similarity relationships in the knowledge graph."""
        try:
            similarities = await self.find_semantic_similarities(document_id)
            
            for similarity in similarities:
                # Create similarity relationship
                rel_query = GraphQuery(
                    cypher="""
                    MATCH (d1:Document {id: $doc1_id})
                    MATCH (d2:Document {id: $doc2_id})
                    MERGE (d1)-[r:SEMANTICALLY_SIMILAR {score: $score, algorithm: $algorithm}]-(d2)
                    RETURN r
                    """,
                    parameters={
                        "doc1_id": similarity.doc1_id,
                        "doc2_id": similarity.doc2_id,
                        "score": similarity.similarity_score,
                        "algorithm": similarity.algorithm
                    },
                    metadata={"operation": "semantic_relationship_creation"}
                )
                
                result = await self.graph_client.execute_query(rel_query)
                if not result.success:
                    logger.warning(f"Failed to create semantic relationship: {result.error}")
            
            logger.info(f"✅ Created {len(similarities)} semantic relationships for document {document_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create semantic relationships: {e}")
            return False
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for the semantic analysis pipeline."""
        avg_processing_time = (self.total_processing_time / self.processed_documents 
                             if self.processed_documents > 0 else 0)
        
        return {
            "processed_documents": self.processed_documents,
            "total_processing_time": self.total_processing_time,
            "average_processing_time": avg_processing_time,
            "total_entities_extracted": self.entity_count,
            "average_entities_per_document": (self.entity_count / self.processed_documents 
                                            if self.processed_documents > 0 else 0),
            "pipeline_initialized": self.initialized,
            "spacy_available": self.nlp is not None,
            "sentence_transformer_available": self.sentence_transformer is not None
        }


class BatchProcessor:
    """Batch processing for semantic analysis."""
    
    def __init__(self, pipeline: SemanticAnalysisPipeline, batch_size: int = 10):
        self.pipeline = pipeline
        self.batch_size = batch_size
    
    async def process_documents_batch(self, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Process a batch of documents."""
        start_time = time.time()
        processed = 0
        errors = 0
        
        for i in range(0, len(documents), self.batch_size):
            batch = documents[i:i + self.batch_size]
            
            # Process batch concurrently
            tasks = []
            for doc in batch:
                task = self.pipeline.process_document(doc['id'], doc['content'])
                tasks.append(task)
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for result in results:
                if isinstance(result, Exception):
                    errors += 1
                    logger.error(f"Batch processing error: {result}")
                elif result:
                    processed += 1
                else:
                    errors += 1
        
        duration = time.time() - start_time
        
        return {
            "total_documents": len(documents),
            "processed": processed,
            "errors": errors,
            "duration_seconds": duration,
            "documents_per_second": len(documents) / duration if duration > 0 else 0
        } 