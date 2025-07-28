"""
API v1 Endpoints for Universal Knowledge Platform
Contains all v1 API endpoints with backward compatibility.
"""

from fastapi import APIRouter, Depends, HTTPException, Request
from typing import Dict, Any
import logging

from api.models import (
    QueryRequest,
    QueryResponse,
    FeedbackRequest,
    FeedbackResponse,
)
from api.validators import QueryRequestValidator, FeedbackRequestValidator
from api.auth import get_current_user
from api.versioning import version_deprecated

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/query", response_model=QueryResponse)
async def process_query_v1(
    request: QueryRequestValidator,
    http_request: Request,
    current_user=Depends(get_current_user)
):
    """
    Process a knowledge query (v1 API).
    
    This is the original query endpoint with basic functionality.
    """
    # Import main query handler
    from api.main import process_query as main_process_query
    
    # Call the main handler
    return await main_process_query(request, http_request, current_user)


@router.post("/feedback", response_model=FeedbackResponse)
async def submit_feedback_v1(
    request: FeedbackRequestValidator,
    current_user=Depends(get_current_user)
):
    """Submit feedback for a query (v1 API)."""
    # Import main feedback handler
    from api.main import submit_feedback as main_submit_feedback
    
    # Call the main handler
    return await main_submit_feedback(request, current_user)


@router.get("/search")
@version_deprecated(sunset_date="2025-12-31")
async def search_v1(
    q: str,
    limit: int = 10,
    current_user=Depends(get_current_user)
):
    """
    Simple search endpoint (v1 API).
    
    DEPRECATED: Use /query endpoint instead.
    """
    logger.warning(f"Deprecated search endpoint called by user {current_user.user_id}")
    
    # Convert to query format
    query_request = QueryRequestValidator(
        query=q,
        max_tokens=1000,
        confidence_threshold=0.7
    )
    
    # Use the query endpoint
    from api.main import orchestrator
    
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Service temporarily unavailable")
    
    result = await orchestrator.process_query({
        "query": query_request.query,
        "max_tokens": query_request.max_tokens,
        "confidence_threshold": query_request.confidence_threshold,
        "user_id": current_user.user_id
    })
    
    # Format for v1 search response
    return {
        "results": result.get("documents", [])[:limit],
        "query": q,
        "total": len(result.get("documents", []))
    }


@router.get("/status")
async def get_status_v1():
    """Get API status (v1 API)."""
    return {
        "status": "operational",
        "version": "1.0.0",
        "api_version": "v1"
    } 