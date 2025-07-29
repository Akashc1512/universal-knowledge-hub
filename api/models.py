from typing import Any, Optional
from pydantic import BaseModel, Field
from datetime import datetime


class QueryRequest(BaseModel):
    """Request model for knowledge queries."""

    query: str = Field(
        ..., min_length=1, max_length=10000, description="The user's question or query"
    )
    max_tokens: Optional[int] = Field(
        1000, ge=100, le=4000, description="Maximum tokens for response"
    )
    confidence_threshold: Optional[float] = Field(
        0.8, ge=0.0, le=1.0, description="Minimum confidence score"
    )
    user_context: Optional[dict[str, Any]] = Field(
        default_factory=dict, description="Additional user context"
    )


class QueryResponse(BaseModel):
    """Response model for knowledge queries."""

    answer: str = Field(..., description="The AI-generated answer")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    citations: list[dict[str, Any]] = Field(default_factory=list, description="Source citations")
    query_id: str = Field(..., description="Unique query identifier")
    processing_time: Optional[float] = Field(None, description="Processing time in seconds")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class FeedbackRequest(BaseModel):
    """Request model for user feedback."""
    
    query_id: str = Field(..., description="Query ID to provide feedback for")
    feedback_type: str = Field(..., description="Type of feedback (helpful/not-helpful)")
    details: Optional[str] = Field(None, description="Additional feedback details")


class FeedbackResponse(BaseModel):
    """Response model for feedback submission."""
    
    success: bool = Field(..., description="Whether feedback was recorded successfully")
    message: str = Field(..., description="Response message")
    feedback_id: Optional[str] = Field(None, description="Unique feedback identifier")


class QueryUpdateRequest(BaseModel):
    """Request model for updating a query."""
    
    query: Optional[str] = Field(None, min_length=1, max_length=10000, description="Updated query text")
    max_tokens: Optional[int] = Field(None, ge=100, le=4000, description="Updated max tokens")
    confidence_threshold: Optional[float] = Field(None, ge=0.0, le=1.0, description="Updated confidence threshold")
    user_context: Optional[dict[str, Any]] = Field(None, description="Updated user context")
    reprocess: Optional[bool] = Field(False, description="Whether to reprocess the query")


class QueryListResponse(BaseModel):
    """Response model for listing queries."""
    
    queries: list[dict[str, Any]] = Field(..., description="List of query records")
    total: int = Field(..., description="Total number of queries")
    page: int = Field(..., description="Current page number")
    page_size: int = Field(..., description="Number of queries per page")
    has_next: bool = Field(..., description="Whether there are more pages")


class QueryDetailResponse(BaseModel):
    """Response model for detailed query information."""
    
    query_id: str = Field(..., description="Unique query identifier")
    query: str = Field(..., description="Original query text")
    answer: str = Field(..., description="Generated answer")
    confidence: float = Field(..., description="Confidence score")
    citations: list[dict[str, Any]] = Field(..., description="Source citations")
    metadata: dict[str, Any] = Field(..., description="Query metadata")
    created_at: datetime = Field(..., description="Query creation timestamp")
    updated_at: Optional[datetime] = Field(None, description="Last update timestamp")
    processing_time: Optional[float] = Field(None, description="Processing time in seconds")
    user_id: str = Field(..., description="User who created the query")
    status: str = Field(..., description="Query status (completed, processing, failed)")


class QueryStatusResponse(BaseModel):
    """Response model for query status."""
    
    query_id: str = Field(..., description="Query identifier")
    status: str = Field(..., description="Current status")
    message: Optional[str] = Field(None, description="Status message")
    progress: Optional[float] = Field(None, ge=0.0, le=1.0, description="Processing progress")
    estimated_completion: Optional[datetime] = Field(None, description="Estimated completion time")


class HealthResponse(BaseModel):
    """Health check response model."""

    status: str = Field(..., description="Service status (healthy/unhealthy)")
    version: str = Field(..., description="API version")
    timestamp: float = Field(..., description="Current timestamp")
    uptime: float = Field(..., description="Service uptime in seconds")
    components: dict[str, str] = Field(default_factory=dict, description="Component health status")
    error: Optional[str] = Field(None, description="Error message if unhealthy")


class Citation(BaseModel):
    """Citation model for source references."""

    id: str = Field(..., description="Citation identifier")
    title: Optional[str] = Field(None, description="Source title")
    url: Optional[str] = Field(None, description="Source URL")
    author: Optional[str] = Field(None, description="Author name")
    date: Optional[str] = Field(None, description="Publication date")
    source: Optional[str] = Field(None, description="Source type")
    confidence: float = Field(1.0, ge=0.0, le=1.0, description="Citation confidence")


class AgentStatus(BaseModel):
    """Agent status model."""

    name: str = Field(..., description="Agent name")
    status: str = Field(..., description="Agent status")
    last_heartbeat: Optional[datetime] = Field(None, description="Last heartbeat")
    error_count: int = Field(0, description="Error count")
    success_rate: float = Field(1.0, ge=0.0, le=1.0, description="Success rate")


class MetricsResponse(BaseModel):
    """Metrics response model."""

    request_counter: int = Field(..., description="Total request count")
    error_counter: int = Field(..., description="Total error count")
    cache_hits: int = Field(..., description="Cache hit count")
    cache_misses: int = Field(..., description="Cache miss count")
    average_response_time: float = Field(..., description="Average response time")
    active_requests: int = Field(..., description="Currently active requests")
    version: str = Field(..., description="Application version")
    uptime: float = Field(..., description="Service uptime")


class AnalyticsResponse(BaseModel):
    """Analytics response model."""

    total_queries: int = Field(..., description="Total queries processed")
    successful_queries: int = Field(..., description="Successful queries")
    failed_queries: int = Field(..., description="Failed queries")
    average_confidence: float = Field(..., description="Average confidence score")
    top_queries: list[dict[str, Any]] = Field(
        default_factory=list, description="Most popular queries"
    )
    user_activity: dict[str, int] = Field(
        default_factory=dict, description="User activity by user ID"
    )
    time_period: dict[str, Any] = Field(
        default_factory=dict, description="Time period for analytics"
    )


class ErrorResponse(BaseModel):
    """Error response model."""

    error: str = Field(..., description="Error message")
    request_id: str = Field(..., description="Request ID for tracing")
    timestamp: float = Field(..., description="Error timestamp")
    details: Optional[dict[str, Any]] = Field(None, description="Error details")


class CacheStats(BaseModel):
    """Cache statistics model."""

    size: int = Field(..., description="Current cache size")
    max_size: int = Field(..., description="Maximum cache size")
    hit_rate: float = Field(..., description="Cache hit rate")
    evictions: int = Field(..., description="Number of evictions")
    oldest_entry: Optional[float] = Field(None, description="Oldest entry timestamp")
    newest_entry: Optional[float] = Field(None, description="Newest entry timestamp")


class SecurityInfo(BaseModel):
    """Security information model."""

    threats_blocked: int = Field(..., description="Number of threats blocked")
    rate_limit_violations: int = Field(..., description="Rate limit violations")
    suspicious_ips: list[str] = Field(default_factory=list, description="Suspicious IP addresses")
    last_security_scan: Optional[datetime] = Field(None, description="Last security scan timestamp")
    security_score: float = Field(..., ge=0.0, le=100.0, description="Security score")


class SystemInfo(BaseModel):
    """System information model."""

    version: str = Field(..., description="Application version")
    uptime: float = Field(..., description="Service uptime")
    memory_usage: float = Field(..., description="Memory usage percentage")
    cpu_usage: float = Field(..., description="CPU usage percentage")
    active_connections: int = Field(..., description="Active connections")
    total_requests: int = Field(..., description="Total requests processed")
    error_rate: float = Field(..., description="Error rate percentage")
