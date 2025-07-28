"""
Comprehensive tests for the recommendation system.
Tests knowledge graph, recommendation engine, and API functionality.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, List, Any
import time

from core.knowledge_graph.schema import KnowledgeGraphSchema, NodeType, RelationshipType
from core.knowledge_graph.client import Neo4jClient, GraphQuery, GraphResult
from core.recommendation.engine import (
    CollaborativeFiltering,
    ContentBasedFiltering,
    SemanticFiltering,
    HybridRecommendationEngine,
    Recommendation,
    RecommendationResult,
)


class TestKnowledgeGraphSchema:
    """Test knowledge graph schema functionality."""

    def test_schema_initialization(self):
        """Test schema initialization and structure."""
        schema = KnowledgeGraphSchema()

        # Test node definitions
        assert NodeType.USER in schema.nodes
        assert NodeType.DOCUMENT in schema.nodes
        assert NodeType.TOPIC in schema.nodes
        assert NodeType.TAG in schema.nodes

        # Test relationship definitions
        assert len(schema.relationships) > 0
        assert any(r.relationship_type == RelationshipType.VIEWED for r in schema.relationships)
        assert any(r.relationship_type == RelationshipType.LIKED for r in schema.relationships)

        # Test constraints and indexes
        assert len(schema.constraints) > 0
        assert len(schema.indexes) > 0

    def test_node_validation(self):
        """Test node property validation."""
        schema = KnowledgeGraphSchema()

        # Valid user properties
        valid_user_props = {"id": "user123", "email": "user@example.com", "name": "Test User"}
        assert schema.validate_node_properties(NodeType.USER, valid_user_props)

        # Invalid user properties (missing required)
        invalid_user_props = {
            "id": "user123"
            # Missing email and name
        }
        assert not schema.validate_node_properties(NodeType.USER, invalid_user_props)

    def test_cypher_schema_generation(self):
        """Test Cypher schema generation."""
        schema = KnowledgeGraphSchema()
        cypher_schema = schema.get_cypher_schema()

        assert "CREATE CONSTRAINT" in cypher_schema
        assert "CREATE INDEX" in cypher_schema
        assert "user.id" in cypher_schema
        assert "document.id" in cypher_schema


class TestNeo4jClient:
    """Test Neo4j client functionality."""

    @pytest.fixture
    def mock_client(self):
        """Create mock Neo4j client."""
        client = Mock(spec=Neo4jClient)
        client.execute_query = AsyncMock()
        return client

    @pytest.mark.asyncio
    async def test_health_check(self, mock_client):
        """Test health check functionality."""
        # Mock successful health check
        mock_client.execute_query.return_value = GraphResult(
            success=True, data=[{"health": 1}], execution_time=0.1, records_count=1
        )

        # Test health check
        result = await mock_client.health_check()
        assert result is True

    @pytest.mark.asyncio
    async def test_create_node(self, mock_client):
        """Test node creation."""
        # Mock successful node creation
        mock_client.execute_query.return_value = GraphResult(
            success=True,
            data=[{"n": {"id": "doc123", "title": "Test Document"}}],
            execution_time=0.1,
            records_count=1,
        )

        # Test node creation
        result = await mock_client.create_node(
            "Document", {"id": "doc123", "title": "Test Document"}
        )

        assert result.success
        assert len(result.data) == 1
        assert result.data[0]["n"]["id"] == "doc123"

    @pytest.mark.asyncio
    async def test_create_relationship(self, mock_client):
        """Test relationship creation."""
        # Mock successful relationship creation
        mock_client.execute_query.return_value = GraphResult(
            success=True,
            data=[{"r": {"type": "VIEWED", "timestamp": "2024-01-01"}}],
            execution_time=0.1,
            records_count=1,
        )

        # Test relationship creation
        result = await mock_client.create_relationship(
            "user123", "doc123", "VIEWED", {"timestamp": "2024-01-01"}
        )

        assert result.success
        assert len(result.data) == 1


class TestCollaborativeFiltering:
    """Test collaborative filtering algorithm."""

    @pytest.fixture
    def mock_graph_client(self):
        """Create mock graph client."""
        client = Mock(spec=Neo4jClient)
        client.execute_query = AsyncMock()
        return client

    @pytest.mark.asyncio
    async def test_build_user_item_matrix(self, mock_graph_client):
        """Test user-item matrix building."""
        cf = CollaborativeFiltering(mock_graph_client)

        # Mock matrix data
        mock_data = [
            {"user_id": "user1", "document_id": "doc1", "interaction_count": 2},
            {"user_id": "user1", "document_id": "doc2", "interaction_count": 1},
            {"user_id": "user2", "document_id": "doc1", "interaction_count": 1},
        ]

        mock_graph_client.execute_query.return_value = GraphResult(
            success=True, data=mock_data, execution_time=0.1, records_count=len(mock_data)
        )

        # Test matrix building
        success = await cf.build_user_item_matrix()
        assert success
        assert cf.user_item_matrix is not None
        assert len(cf.user_ids) == 2
        assert len(cf.document_ids) == 2

    def test_cosine_similarity(self):
        """Test cosine similarity calculation."""
        cf = CollaborativeFiltering(Mock())

        # Test identical vectors
        vec1 = np.array([1, 2, 3])
        vec2 = np.array([1, 2, 3])
        similarity = cf._cosine_similarity(vec1, vec2)
        assert similarity == 1.0

        # Test orthogonal vectors
        vec1 = np.array([1, 0, 0])
        vec2 = np.array([0, 1, 0])
        similarity = cf._cosine_similarity(vec1, vec2)
        assert similarity == 0.0

    @pytest.mark.asyncio
    async def test_get_recommendations(self, mock_graph_client):
        """Test collaborative filtering recommendations."""
        cf = CollaborativeFiltering(mock_graph_client)

        # Mock recommendation data
        mock_recommendations = [
            {"document_id": "doc1", "title": "Test Doc 1", "collaborative_score": 3},
            {"document_id": "doc2", "title": "Test Doc 2", "collaborative_score": 2},
        ]

        mock_graph_client.execute_query.return_value = GraphResult(
            success=True,
            data=mock_recommendations,
            execution_time=0.1,
            records_count=len(mock_recommendations),
        )

        # Test recommendations
        recommendations = await cf.get_recommendations("user123", limit=5)

        assert len(recommendations) == 2
        assert recommendations[0].document_id == "doc1"
        assert recommendations[0].algorithm == "collaborative_filtering"
        assert recommendations[0].score == 3.0


class TestContentBasedFiltering:
    """Test content-based filtering algorithm."""

    @pytest.fixture
    def mock_graph_client(self):
        """Create mock graph client."""
        client = Mock(spec=Neo4jClient)
        client.execute_query = AsyncMock()
        return client

    @pytest.mark.asyncio
    async def test_get_user_preferences(self, mock_graph_client):
        """Test user preference extraction."""
        cbf = ContentBasedFiltering(mock_graph_client)

        # Mock preference data
        mock_preferences = [
            {"topic": "technology", "topic_count": 5},
            {"topic": "science", "topic_count": 3},
        ]

        mock_graph_client.execute_query.return_value = GraphResult(
            success=True,
            data=mock_preferences,
            execution_time=0.1,
            records_count=len(mock_preferences),
        )

        # Test preference extraction
        preferences = await cbf.get_user_preferences("user123")

        assert "technology" in preferences
        assert "science" in preferences
        assert preferences["technology"] == 5 / 8  # 5/(5+3)
        assert preferences["science"] == 3 / 8

    @pytest.mark.asyncio
    async def test_get_recommendations(self, mock_graph_client):
        """Test content-based recommendations."""
        cbf = ContentBasedFiltering(mock_graph_client)

        # Mock recommendation data
        mock_recommendations = [
            {
                "document_id": "doc1",
                "title": "Tech Doc",
                "topic": "technology",
                "topic_preference": 0.8,
            },
            {
                "document_id": "doc2",
                "title": "Science Doc",
                "topic": "science",
                "topic_preference": 0.6,
            },
        ]

        mock_graph_client.execute_query.return_value = GraphResult(
            success=True,
            data=mock_recommendations,
            execution_time=0.1,
            records_count=len(mock_recommendations),
        )

        # Test recommendations
        recommendations = await cbf.get_recommendations("user123", limit=5)

        assert len(recommendations) == 2
        assert recommendations[0].document_id == "doc1"
        assert recommendations[0].algorithm == "content_based_filtering"
        assert recommendations[0].score == 0.8


class TestHybridRecommendationEngine:
    """Test hybrid recommendation engine."""

    @pytest.fixture
    def mock_graph_client(self):
        """Create mock graph client."""
        client = Mock(spec=Neo4jClient)
        client.execute_query = AsyncMock()
        return client

    @pytest.fixture
    def hybrid_engine(self, mock_graph_client):
        """Create hybrid recommendation engine."""
        return HybridRecommendationEngine(mock_graph_client)

    @pytest.mark.asyncio
    async def test_get_recommendations(self, hybrid_engine, mock_graph_client):
        """Test hybrid recommendations."""
        # Mock collaborative filtering results
        collaborative_recs = [
            Recommendation(
                document_id="doc1",
                title="Collaborative Doc",
                score=0.8,
                algorithm="collaborative_filtering",
                confidence=0.8,
                explanation="Viewed by similar users",
            )
        ]

        # Mock content-based results
        content_recs = [
            Recommendation(
                document_id="doc2",
                title="Content Doc",
                score=0.7,
                algorithm="content_based_filtering",
                confidence=0.7,
                explanation="Matches your interests",
            )
        ]

        # Mock semantic results
        semantic_recs = [
            Recommendation(
                document_id="doc3",
                title="Semantic Doc",
                score=0.6,
                algorithm="semantic_filtering",
                confidence=0.6,
                explanation="Semantically similar",
            )
        ]

        # Patch the individual algorithms
        with (
            patch.object(
                hybrid_engine.collaborative_filtering,
                "get_recommendations",
                return_value=collaborative_recs,
            ),
            patch.object(
                hybrid_engine.content_based_filtering,
                "get_recommendations",
                return_value=content_recs,
            ),
            patch.object(
                hybrid_engine.semantic_filtering, "get_recommendations", return_value=semantic_recs
            ),
        ):

            # Test hybrid recommendations
            result = await hybrid_engine.get_recommendations("user123", limit=10)

            assert result.user_id == "user123"
            assert len(result.recommendations) == 3
            assert result.algorithms_used == [
                "collaborative_filtering",
                "content_based_filtering",
                "semantic_filtering",
            ]
            assert result.execution_time > 0

    def test_algorithm_weights(self, hybrid_engine):
        """Test algorithm weight management."""
        # Test initial weights
        weights = hybrid_engine.get_algorithm_weights()
        assert "collaborative_filtering" in weights
        assert "content_based_filtering" in weights
        assert "semantic_filtering" in weights

        # Test weight update
        feedback = {
            "collaborative_filtering": 0.8,
            "content_based_filtering": 0.6,
            "semantic_filtering": 0.4,
        }

        asyncio.run(hybrid_engine.update_algorithm_weights("user123", feedback))

        # Verify weights were updated
        new_weights = hybrid_engine.get_algorithm_weights()
        assert new_weights != weights


class TestRecommendationAPI:
    """Test recommendation API endpoints."""

    @pytest.fixture
    def mock_recommendation_service(self):
        """Create mock recommendation service."""
        from api.recommendation_service import RecommendationService

        service = RecommendationService()
        return service

    def test_recommendation_request_validation(self):
        """Test recommendation request validation."""
        from api.recommendation_service import RecommendationRequest

        # Valid request
        valid_request = RecommendationRequest(
            user_id="user123", limit=10, algorithm="hybrid", include_explanations=True
        )
        assert valid_request.user_id == "user123"
        assert valid_request.limit == 10

        # Test limit validation
        with pytest.raises(ValueError):
            RecommendationRequest(user_id="user123", limit=100)  # Too high

    def test_recommendation_response_structure(self):
        """Test recommendation response structure."""
        from api.recommendation_service import RecommendationResponse

        response = RecommendationResponse(
            user_id="user123",
            recommendations=[
                {
                    "document_id": "doc1",
                    "title": "Test Doc",
                    "score": 0.8,
                    "algorithm": "hybrid",
                    "confidence": 0.8,
                    "explanation": "Test explanation",
                }
            ],
            total_count=1,
            execution_time=0.1,
            algorithms_used=["collaborative_filtering", "content_based_filtering"],
            metadata={"test": "data"},
        )

        assert response.user_id == "user123"
        assert len(response.recommendations) == 1
        assert response.total_count == 1
        assert response.execution_time == 0.1


@pytest.mark.asyncio
async def test_full_recommendation_pipeline():
    """Test complete recommendation pipeline."""
    # This test would require a real Neo4j instance
    # For now, we'll test the components individually

    # Test schema
    schema = KnowledgeGraphSchema()
    assert schema is not None

    # Test recommendation creation
    rec = Recommendation(
        document_id="doc123",
        title="Test Document",
        score=0.8,
        algorithm="hybrid",
        confidence=0.8,
        explanation="Test recommendation",
    )
    assert rec.document_id == "doc123"
    assert rec.score == 0.8

    # Test result creation
    result = RecommendationResult(
        user_id="user123",
        recommendations=[rec],
        total_count=1,
        execution_time=0.1,
        algorithms_used=["hybrid"],
    )
    assert result.user_id == "user123"
    assert len(result.recommendations) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
