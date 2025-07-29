"""
Standardized Data Models for Multi-Agent Knowledge Platform
This module defines consistent data structures for inter-agent communication.
"""

from pydantic import BaseModel, Field
from typing import Any, Optional, Union
from enum import Enum
from dataclasses import dataclass
from datetime import datetime


class AgentDataModel(BaseModel):
    """Base model for agent data structures with common validation."""

    class Config:
        """Pydantic configuration for data models."""

        extra = "forbid"  # Prevent additional fields
        validate_assignment = True  # Validate on assignment


class DocumentModel(AgentDataModel):
    """Standardized document model for retrieval results."""

    content: str = Field(..., description="Document content")
    score: float = Field(..., ge=0.0, le=1.0, description="Relevance score")
    source: str = Field(..., description="Document source")
    doc_id: str = Field(default="", description="Document ID")
    chunk_id: Optional[str] = Field(default=None, description="Chunk ID")
    timestamp: Optional[str] = Field(default=None, description="Document timestamp")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class VerifiedFactModel(AgentDataModel):
    """Standardized verified fact model for fact-checking results."""

    claim: str = Field(..., description="The verified claim")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    source: str = Field(..., description="Source of verification")
    evidence: list[str] = Field(default_factory=list, description="Supporting evidence")
    contradicting_evidence: list[str] = Field(
        default_factory=list, description="Contradicting evidence"
    )
    verification_method: str = Field(..., description="Method used for verification")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class CitationModel(AgentDataModel):
    """Standardized citation model."""

    id: str = Field(..., description="Citation ID")
    text: str = Field(..., description="Citation text")
    url: Optional[str] = Field(default=None, description="Source URL")
    title: Optional[str] = Field(default=None, description="Source title")
    author: Optional[str] = Field(default=None, description="Source author")
    date: Optional[str] = Field(default=None, description="Source date")
    source: Optional[str] = Field(default=None, description="Source type")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Citation confidence")


class SynthesisResult(AgentDataModel):
    """Standardized synthesis result model."""

    answer: str = Field(..., description="Synthesized answer")
    synthesis_method: str = Field(..., description="Method used for synthesis")
    fact_count: int = Field(..., ge=0, description="Number of facts used")
    processing_time_ms: int = Field(..., ge=0, description="Processing time in milliseconds")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class CitationResult(AgentDataModel):
    """Standardized citation result model."""

    cited_content: str = Field(..., description="Content with inline citations")
    bibliography: list[str] = Field(default_factory=list, description="Full bibliography")
    in_text_citations: list[dict[str, Any]] = Field(
        default_factory=list, description="Inline citations"
    )
    citation_style: str = Field(default="APA", description="Citation style used")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class RetrievalResult(AgentDataModel):
    """Standardized retrieval result model."""

    documents: list[DocumentModel] = Field(default_factory=list, description="Retrieved documents")
    search_type: str = Field(..., description="Type of search performed")
    total_hits: int = Field(..., ge=0, description="Total number of hits")
    query_time_ms: int = Field(..., ge=0, description="Query execution time")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class FactCheckResult(AgentDataModel):
    """Standardized fact-checking result model."""

    verified_facts: list[VerifiedFactModel] = Field(
        default_factory=list, description="Verified facts"
    )
    contested_claims: list[dict[str, Any]] = Field(
        default_factory=list, description="Contested claims"
    )
    verification_method: str = Field(..., description="Verification method used")
    total_claims: int = Field(..., ge=0, description="Total number of claims processed")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class AgentTaskModel(AgentDataModel):
    """Standardized task model for agent communication."""

    query: str = Field(..., description="The query to process")
    task_type: str = Field(..., description="Type of task")
    parameters: dict[str, Any] = Field(default_factory=dict, description="Task parameters")
    context: dict[str, Any] = Field(default_factory=dict, description="Additional context")
    priority: int = Field(default=2, ge=1, le=5, description="Task priority (1-5)")
    timeout_ms: int = Field(default=5000, ge=0, description="Task timeout in milliseconds")


class AgentResponseModel(AgentDataModel):
    """Standardized response model for agent results."""

    success: bool = Field(..., description="Whether the task was successful")
    data: Optional[dict[str, Any]] = Field(default=None, description="Response data")
    confidence: float = Field(default=0.0, ge=0.0, le=1.0, description="Confidence score")
    error: Optional[str] = Field(default=None, description="Error message if failed")
    execution_time_ms: int = Field(default=0, ge=0, description="Execution time in milliseconds")
    token_usage: dict[str, int] = Field(
        default_factory=lambda: {"prompt": 0, "completion": 0}, description="Token usage"
    )
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


# Legacy compatibility models for backward compatibility
class LegacySynthesisResult(AgentDataModel):
    """Legacy synthesis result with both 'answer' and 'response' keys for compatibility."""

    answer: str = Field(..., description="Synthesized answer")
    response: str = Field(..., description="Legacy response field (same as answer)")
    synthesis_method: str = Field(..., description="Method used for synthesis")
    fact_count: int = Field(..., ge=0, description="Number of facts used")
    processing_time_ms: int = Field(..., ge=0, description="Processing time in milliseconds")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class LegacyCitationResult(AgentDataModel):
    """Legacy citation result with both 'bibliography' and 'citations' keys for compatibility."""

    cited_content: str = Field(..., description="Content with inline citations")
    bibliography: list[str] = Field(default_factory=list, description="Full bibliography")
    citations: list[str] = Field(
        default_factory=list, description="Legacy citations field (same as bibliography)"
    )
    citation_style: str = Field(default="APA", description="Citation style used")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


# Utility functions for data conversion
def convert_to_standard_synthesis(data: dict[str, Any]) -> SynthesisResult:
    """Convert legacy synthesis data to standardized format."""
    # Handle both 'answer' and 'response' keys
    answer = data.get("answer") or data.get("response", "")

    return SynthesisResult(
        answer=answer,
        synthesis_method=data.get("synthesis_method", "rule_based"),
        fact_count=data.get("fact_count", 0),
        processing_time_ms=data.get("processing_time_ms", 0),
        metadata=data.get("metadata", {}),
    )


def convert_to_standard_citation(data: dict[str, Any]) -> CitationResult:
    """Convert legacy citation data to standardized format."""
    # Handle both 'bibliography' and 'citations' keys
    bibliography = data.get("bibliography", [])
    if not bibliography:
        bibliography = data.get("citations", [])

    return CitationResult(
        cited_content=data.get("cited_content", ""),
        bibliography=bibliography,
        in_text_citations=data.get("in_text_citations", []),
        citation_style=data.get("citation_style", "APA"),
        metadata=data.get("metadata", {}),
    )


def convert_to_standard_retrieval(data: dict[str, Any]) -> RetrievalResult:
    """Convert legacy retrieval data to standardized format."""
    documents = []
    for doc_data in data.get("documents", []):
        if isinstance(doc_data, dict):
            documents.append(DocumentModel(**doc_data))
        else:
            # Handle case where documents might be Document objects
            documents.append(
                DocumentModel(
                    content=getattr(doc_data, "content", ""),
                    score=getattr(doc_data, "score", 0.0),
                    source=getattr(doc_data, "source", ""),
                    doc_id=getattr(doc_data, "doc_id", ""),
                    chunk_id=getattr(doc_data, "chunk_id", None),
                    timestamp=getattr(doc_data, "timestamp", None),
                    metadata=getattr(doc_data, "metadata", {}),
                )
            )

    return RetrievalResult(
        documents=documents,
        search_type=data.get("search_type", "hybrid"),
        total_hits=data.get("total_hits", 0),
        query_time_ms=data.get("query_time_ms", 0),
        metadata=data.get("metadata", {}),
    )


def convert_to_standard_factcheck(data: dict[str, Any]) -> FactCheckResult:
    """Convert legacy fact-checking data to standardized format."""
    verified_facts = []
    for fact_data in data.get("verified_facts", []):
        if isinstance(fact_data, dict):
            verified_facts.append(VerifiedFactModel(**fact_data))
        else:
            # Handle case where facts might be objects
            verified_facts.append(
                VerifiedFactModel(
                    claim=getattr(fact_data, "claim", ""),
                    confidence=getattr(fact_data, "confidence", 0.0),
                    source=getattr(fact_data, "source", ""),
                    evidence=getattr(fact_data, "evidence", []),
                    contradicting_evidence=getattr(fact_data, "contradicting_evidence", []),
                    verification_method=getattr(fact_data, "verification_method", ""),
                    metadata=getattr(fact_data, "metadata", {}),
                )
            )

    return FactCheckResult(
        verified_facts=verified_facts,
        contested_claims=data.get("contested_claims", []),
        verification_method=data.get("verification_method", ""),
        total_claims=data.get("total_claims", len(verified_facts)),
        metadata=data.get("metadata", {}),
    )


# Data validation functions
def validate_agent_data(data: dict[str, Any], expected_type: str) -> bool:
    """Validate agent data against expected type."""
    try:
        if expected_type == "synthesis":
            convert_to_standard_synthesis(data)
        elif expected_type == "citation":
            convert_to_standard_citation(data)
        elif expected_type == "retrieval":
            convert_to_standard_retrieval(data)
        elif expected_type == "factcheck":
            convert_to_standard_factcheck(data)
        else:
            return False
        return True
    except Exception:
        return False


def get_standardized_keys() -> dict[str, list[str]]:
    """Get standardized keys for each agent type."""
    return {
        "synthesis": ["answer", "synthesis_method", "fact_count", "processing_time_ms"],
        "citation": ["cited_content", "bibliography", "in_text_citations", "citation_style"],
        "retrieval": ["documents", "search_type", "total_hits", "query_time_ms"],
        "factcheck": ["verified_facts", "contested_claims", "verification_method", "total_claims"],
    }
