"""
Advanced Recommendation Engine for Universal Knowledge Platform
Implements hybrid recommendation system with collaborative and content-based filtering.
"""

import asyncio
import logging
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import time
from collections import defaultdict
import json

from core.knowledge_graph.client import Neo4jClient, GraphQuery

logger = logging.getLogger(__name__)


@dataclass
class Recommendation:
    """Individual recommendation with metadata."""
    document_id: str
    title: str
    score: float
    algorithm: str
    confidence: float
    explanation: str
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class RecommendationResult:
    """Complete recommendation result for a user."""
    user_id: str
    recommendations: List[Recommendation]
    total_count: int
    execution_time: float
    algorithms_used: List[str]
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class CollaborativeFiltering:
    """Collaborative filtering recommendation algorithm."""
    
    def __init__(self, graph_client: Neo4jClient):
        self.graph_client = graph_client
        self.user_item_matrix = None
        self.similarity_matrix = None
        
    async def build_user_item_matrix(self) -> bool:
        """Build user-item interaction matrix from graph data."""
        try:
            # Get all user-document interactions
            query = GraphQuery(
                cypher="""
                MATCH (u:User)-[r:VIEWED]->(d:Document)
                RETURN u.id as user_id, d.id as document_id, count(r) as interaction_count
                """,
                parameters={},
                metadata={"operation": "build_user_item_matrix"}
            )
            
            result = await self.graph_client.execute_query(query)
            if not result.success:
                return False
            
            # Build matrix
            user_ids = set()
            document_ids = set()
            interactions = {}
            
            for record in result.data:
                user_id = record["user_id"]
                document_id = record["document_id"]
                count = record["interaction_count"]
                
                user_ids.add(user_id)
                document_ids.add(document_id)
                interactions[(user_id, document_id)] = count
            
            # Create matrix
            user_list = list(user_ids)
            document_list = list(document_ids)
            
            self.user_item_matrix = np.zeros((len(user_list), len(document_list)))
            
            for (user_id, document_id), count in interactions.items():
                i = user_list.index(user_id)
                j = document_list.index(document_id)
                self.user_item_matrix[i, j] = count
            
            self.user_ids = user_list
            self.document_ids = document_list
            
            logger.info(f"Built user-item matrix: {len(user_list)} users, {len(document_list)} documents")
            return True
            
        except Exception as e:
            logger.error(f"Failed to build user-item matrix: {e}")
            return False
    
    def compute_user_similarity(self, user_id: str) -> List[Tuple[str, float]]:
        """Compute similarity between users using cosine similarity."""
        if self.user_item_matrix is None:
            return []
        
        try:
            user_idx = self.user_ids.index(user_id)
            user_vector = self.user_item_matrix[user_idx]
            
            similarities = []
            for i, other_user_id in enumerate(self.user_ids):
                if i != user_idx:
                    other_vector = self.user_item_matrix[i]
                    similarity = self._cosine_similarity(user_vector, other_vector)
                    if similarity > 0:
                        similarities.append((other_user_id, similarity))
            
            # Sort by similarity
            similarities.sort(key=lambda x: x[1], reverse=True)
            return similarities[:10]  # Top 10 similar users
            
        except ValueError:
            logger.warning(f"User {user_id} not found in matrix")
            return []
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0
        
        return dot_product / (norm1 * norm2)
    
    async def get_recommendations(self, user_id: str, limit: int = 10) -> List[Recommendation]:
        """Get collaborative filtering recommendations."""
        try:
            # Get similar users
            similar_users = self.compute_user_similarity(user_id)
            if not similar_users:
                return []
            
            # Get documents viewed by similar users
            similar_user_ids = [user_id for user_id, _ in similar_users[:5]]
            
            query = GraphQuery(
                cypher="""
                MATCH (u:User)-[:VIEWED]->(d:Document)
                WHERE u.id IN $similar_user_ids
                AND NOT EXISTS((
                    MATCH (target:User {id: $user_id})-[:VIEWED]->(d)
                    RETURN target
                ))
                WITH d, count(u) as collaborative_score
                ORDER BY collaborative_score DESC
                LIMIT $limit
                RETURN d.id as document_id, d.title as title, collaborative_score
                """,
                parameters={
                    "user_id": user_id,
                    "similar_user_ids": similar_user_ids,
                    "limit": limit
                },
                metadata={"operation": "collaborative_recommendations"}
            )
            
            result = await self.graph_client.execute_query(query)
            if not result.success:
                return []
            
            recommendations = []
            for record in result.data:
                score = record["collaborative_score"]
                recommendation = Recommendation(
                    document_id=record["document_id"],
                    title=record["title"],
                    score=float(score),
                    algorithm="collaborative_filtering",
                    confidence=min(score / 5.0, 1.0),  # Normalize confidence
                    explanation=f"Viewed by {score} similar users",
                    metadata={"similar_users_count": score}
                )
                recommendations.append(recommendation)
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Collaborative filtering failed: {e}")
            return []


class ContentBasedFiltering:
    """Content-based filtering recommendation algorithm."""
    
    def __init__(self, graph_client: Neo4jClient):
        self.graph_client = graph_client
    
    async def get_user_preferences(self, user_id: str) -> Dict[str, float]:
        """Extract user preferences from viewed documents."""
        try:
            query = GraphQuery(
                cypher="""
                MATCH (u:User {id: $user_id})-[:VIEWED]->(d:Document)
                MATCH (d)-[:BELONGS_TO]->(t:Topic)
                WITH t, count(d) as topic_count
                RETURN t.name as topic, topic_count
                ORDER BY topic_count DESC
                """,
                parameters={"user_id": user_id},
                metadata={"operation": "user_preferences"}
            )
            
            result = await self.graph_client.execute_query(query)
            if not result.success:
                return {}
            
            preferences = {}
            total_documents = sum(record["topic_count"] for record in result.data)
            
            for record in result.data:
                topic = record["topic"]
                count = record["topic_count"]
                preferences[topic] = count / total_documents if total_documents > 0 else 0
            
            return preferences
            
        except Exception as e:
            logger.error(f"Failed to get user preferences: {e}")
            return {}
    
    async def get_recommendations(self, user_id: str, limit: int = 10) -> List[Recommendation]:
        """Get content-based recommendations based on user preferences."""
        try:
            # Get user preferences
            preferences = await self.get_user_preferences(user_id)
            if not preferences:
                return []
            
            # Get documents in preferred topics
            preferred_topics = list(preferences.keys())[:5]  # Top 5 topics
            
            query = GraphQuery(
                cypher="""
                MATCH (d:Document)-[:BELONGS_TO]->(t:Topic)
                WHERE t.name IN $preferred_topics
                AND NOT EXISTS((
                    MATCH (u:User {id: $user_id})-[:VIEWED]->(d)
                    RETURN u
                ))
                WITH d, t.name as topic, $preferences[t.name] as topic_preference
                ORDER BY topic_preference DESC, d.created_at DESC
                LIMIT $limit
                RETURN d.id as document_id, d.title as title, topic, topic_preference
                """,
                parameters={
                    "user_id": user_id,
                    "preferred_topics": preferred_topics,
                    "preferences": preferences,
                    "limit": limit
                },
                metadata={"operation": "content_based_recommendations"}
            )
            
            result = await self.graph_client.execute_query(query)
            if not result.success:
                return []
            
            recommendations = []
            for record in result.data:
                topic_preference = record["topic_preference"]
                recommendation = Recommendation(
                    document_id=record["document_id"],
                    title=record["title"],
                    score=float(topic_preference),
                    algorithm="content_based_filtering",
                    confidence=topic_preference,
                    explanation=f"Matches your interest in {record['topic']}",
                    metadata={"topic": record["topic"], "topic_preference": topic_preference}
                )
                recommendations.append(recommendation)
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Content-based filtering failed: {e}")
            return []


class SemanticFiltering:
    """Semantic similarity-based recommendation algorithm."""
    
    def __init__(self, graph_client: Neo4jClient):
        self.graph_client = graph_client
    
    async def get_recommendations(self, user_id: str, limit: int = 10) -> List[Recommendation]:
        """Get semantic recommendations based on document similarity."""
        try:
            # Get user's recently viewed documents
            query = GraphQuery(
                cypher="""
                MATCH (u:User {id: $user_id})-[:VIEWED]->(d:Document)
                WITH d ORDER BY d.last_viewed DESC LIMIT 5
                MATCH (d)-[:SIMILAR_TO]-(d2:Document)
                WHERE NOT EXISTS((
                    MATCH (u2:User {id: $user_id})-[:VIEWED]->(d2)
                    RETURN u2
                ))
                WITH d2, count(d) as semantic_score
                ORDER BY semantic_score DESC
                LIMIT $limit
                RETURN d2.id as document_id, d2.title as title, semantic_score
                """,
                parameters={"user_id": user_id, "limit": limit},
                metadata={"operation": "semantic_recommendations"}
            )
            
            result = await self.graph_client.execute_query(query)
            if not result.success:
                return []
            
            recommendations = []
            for record in result.data:
                semantic_score = record["semantic_score"]
                recommendation = Recommendation(
                    document_id=record["document_id"],
                    title=record["title"],
                    score=float(semantic_score),
                    algorithm="semantic_filtering",
                    confidence=min(semantic_score / 5.0, 1.0),
                    explanation=f"Semantically similar to {semantic_score} of your recent documents",
                    metadata={"semantic_score": semantic_score}
                )
                recommendations.append(recommendation)
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Semantic filtering failed: {e}")
            return []


class HybridRecommendationEngine:
    """Hybrid recommendation engine combining multiple algorithms."""
    
    def __init__(self, graph_client: Neo4jClient):
        self.graph_client = graph_client
        self.collaborative_filtering = CollaborativeFiltering(graph_client)
        self.content_based_filtering = ContentBasedFiltering(graph_client)
        self.semantic_filtering = SemanticFiltering(graph_client)
        
        # Algorithm weights for hybrid scoring
        self.algorithm_weights = {
            "collaborative_filtering": 0.4,
            "content_based_filtering": 0.3,
            "semantic_filtering": 0.3
        }
    
    async def get_recommendations(self, user_id: str, limit: int = 10) -> RecommendationResult:
        """Get hybrid recommendations combining multiple algorithms."""
        start_time = time.time()
        
        try:
            # Get recommendations from all algorithms
            collaborative_recs = await self.collaborative_filtering.get_recommendations(user_id, limit)
            content_based_recs = await self.content_based_filtering.get_recommendations(user_id, limit)
            semantic_recs = await self.semantic_filtering.get_recommendations(user_id, limit)
            
            # Combine and score recommendations
            all_recommendations = {}
            
            # Add collaborative filtering recommendations
            for rec in collaborative_recs:
                if rec.document_id not in all_recommendations:
                    all_recommendations[rec.document_id] = {
                        "document_id": rec.document_id,
                        "title": rec.title,
                        "scores": {},
                        "explanations": [],
                        "metadata": {}
                    }
                
                all_recommendations[rec.document_id]["scores"]["collaborative"] = rec.score
                all_recommendations[rec.document_id]["explanations"].append(rec.explanation)
                all_recommendations[rec.document_id]["metadata"].update(rec.metadata)
            
            # Add content-based recommendations
            for rec in content_based_recs:
                if rec.document_id not in all_recommendations:
                    all_recommendations[rec.document_id] = {
                        "document_id": rec.document_id,
                        "title": rec.title,
                        "scores": {},
                        "explanations": [],
                        "metadata": {}
                    }
                
                all_recommendations[rec.document_id]["scores"]["content_based"] = rec.score
                all_recommendations[rec.document_id]["explanations"].append(rec.explanation)
                all_recommendations[rec.document_id]["metadata"].update(rec.metadata)
            
            # Add semantic recommendations
            for rec in semantic_recs:
                if rec.document_id not in all_recommendations:
                    all_recommendations[rec.document_id] = {
                        "document_id": rec.document_id,
                        "title": rec.title,
                        "scores": {},
                        "explanations": [],
                        "metadata": {}
                    }
                
                all_recommendations[rec.document_id]["scores"]["semantic"] = rec.score
                all_recommendations[rec.document_id]["explanations"].append(rec.explanation)
                all_recommendations[rec.document_id]["metadata"].update(rec.metadata)
            
            # Calculate hybrid scores
            final_recommendations = []
            for doc_id, rec_data in all_recommendations.items():
                hybrid_score = 0.0
                total_weight = 0.0
                
                for algorithm, weight in self.algorithm_weights.items():
                    if algorithm in rec_data["scores"]:
                        hybrid_score += rec_data["scores"][algorithm] * weight
                        total_weight += weight
                
                if total_weight > 0:
                    hybrid_score /= total_weight
                    
                    # Create final recommendation
                    final_rec = Recommendation(
                        document_id=rec_data["document_id"],
                        title=rec_data["title"],
                        score=hybrid_score,
                        algorithm="hybrid",
                        confidence=hybrid_score,
                        explanation="; ".join(rec_data["explanations"]),
                        metadata={
                            "individual_scores": rec_data["scores"],
                            "algorithms_used": list(rec_data["scores"].keys()),
                            **rec_data["metadata"]
                        }
                    )
                    final_recommendations.append(final_rec)
            
            # Sort by hybrid score
            final_recommendations.sort(key=lambda x: x.score, reverse=True)
            
            execution_time = time.time() - start_time
            
            return RecommendationResult(
                user_id=user_id,
                recommendations=final_recommendations[:limit],
                total_count=len(final_recommendations),
                execution_time=execution_time,
                algorithms_used=list(self.algorithm_weights.keys()),
                metadata={
                    "collaborative_count": len(collaborative_recs),
                    "content_based_count": len(content_based_recs),
                    "semantic_count": len(semantic_recs)
                }
            )
            
        except Exception as e:
            logger.error(f"Hybrid recommendation failed: {e}")
            execution_time = time.time() - start_time
            
            return RecommendationResult(
                user_id=user_id,
                recommendations=[],
                total_count=0,
                execution_time=execution_time,
                algorithms_used=[],
                metadata={"error": str(e)}
            )
    
    async def update_algorithm_weights(self, user_id: str, feedback: Dict[str, float]):
        """Update algorithm weights based on user feedback."""
        # Simple weight adjustment based on feedback
        # In production, this could use machine learning
        
        total_feedback = sum(feedback.values())
        if total_feedback > 0:
            for algorithm, score in feedback.items():
                if algorithm in self.algorithm_weights:
                    # Adjust weight based on feedback
                    adjustment = (score - 0.5) * 0.1  # Small adjustment
                    self.algorithm_weights[algorithm] = max(0.1, 
                        self.algorithm_weights[algorithm] + adjustment)
            
            # Normalize weights
            total_weight = sum(self.algorithm_weights.values())
            for algorithm in self.algorithm_weights:
                self.algorithm_weights[algorithm] /= total_weight
            
            logger.info(f"Updated algorithm weights for user {user_id}: {self.algorithm_weights}")
    
    def get_algorithm_weights(self) -> Dict[str, float]:
        """Get current algorithm weights."""
        return self.algorithm_weights.copy() 