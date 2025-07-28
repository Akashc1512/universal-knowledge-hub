"""
API v2 Endpoints for Universal Knowledge Platform
Enhanced API with additional features and improved structure.
"""

from fastapi import APIRouter, Depends, HTTPException, Request, BackgroundTasks
from fastapi.responses import StreamingResponse
from typing import Dict, Any, List, Optional
import logging
import asyncio
import json

from api.models import (
    QueryRequest,
    QueryResponse,
    FeedbackRequest,
    FeedbackResponse,
)
from api.validators import QueryRequestValidator, FeedbackRequestValidator
from api.auth import get_current_user
from api.versioning import get_feature_flag, VersionedResponse

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/query", response_model=QueryResponse)
async def process_query_v2(
    request: QueryRequestValidator,
    http_request: Request,
    background_tasks: BackgroundTasks,
    stream: bool = False,
    current_user=Depends(get_current_user)
):
    """
    Process a knowledge query (v2 API).
    
    Enhanced features:
    - Streaming responses
    - Background processing
    - Multi-language support
    - Advanced analytics
    """
    # Check feature flags
    if stream and not get_feature_flag("v2", "streaming_responses"):
        raise HTTPException(
            status_code=400,
            detail="Streaming responses not available in this version"
        )
    
    if request.language != "en" and not get_feature_flag("v2", "multi_language"):
        raise HTTPException(
            status_code=400,
            detail="Multi-language support not available in this version"
        )
    
    # Import main query handler
    from api.main import orchestrator
    
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Service temporarily unavailable")
    
    # Stream response if requested
    if stream:
        return StreamingResponse(
            stream_query_response(request, current_user),
            media_type="application/x-ndjson"
        )
    
    # Process normally
    result = await orchestrator.process_query({
        "query": request.query,
        "max_tokens": request.max_tokens,
        "confidence_threshold": request.confidence_threshold,
        "language": request.language,
        "user_id": current_user.user_id,
        "metadata": request.metadata
    })
    
    # Add background analytics task
    background_tasks.add_task(
        track_advanced_analytics,
        query=request.query,
        result=result,
        user_id=current_user.user_id
    )
    
    # Format response for v2
    return VersionedResponse.format_response(result, "v2")


@router.post("/batch/query")
async def batch_query_v2(
    requests: List[QueryRequestValidator],
    http_request: Request,
    current_user=Depends(get_current_user)
):
    """
    Process multiple queries in batch (v2 API).
    
    Enhanced feature for processing multiple queries efficiently.
    """
    if not get_feature_flag("v2", "batch_processing"):
        raise HTTPException(
            status_code=400,
            detail="Batch processing not available in this version"
        )
    
    if len(requests) > 10:
        raise HTTPException(
            status_code=400,
            detail="Maximum 10 queries allowed in a batch"
        )
    
    from api.main import orchestrator
    
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Service temporarily unavailable")
    
    # Process queries concurrently
    tasks = []
    for req in requests:
        task = orchestrator.process_query({
            "query": req.query,
            "max_tokens": req.max_tokens,
            "confidence_threshold": req.confidence_threshold,
            "language": req.language,
            "user_id": current_user.user_id,
            "metadata": req.metadata
        })
        tasks.append(task)
    
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Format batch response
    batch_results = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            batch_results.append({
                "index": i,
                "success": False,
                "error": str(result)
            })
        else:
            batch_results.append({
                "index": i,
                "success": True,
                "result": result
            })
    
    return VersionedResponse.format_response({
        "batch_id": http_request.state.request_id,
        "total": len(requests),
        "successful": sum(1 for r in batch_results if r["success"]),
        "failed": sum(1 for r in batch_results if not r["success"]),
        "results": batch_results
    }, "v2")


@router.get("/analytics/advanced")
async def get_advanced_analytics_v2(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    metrics: Optional[List[str]] = None,
    current_user=Depends(get_current_user)
):
    """
    Get advanced analytics (v2 API).
    
    Enhanced analytics with more detailed insights.
    """
    if not get_feature_flag("v2", "advanced_analytics"):
        raise HTTPException(
            status_code=400,
            detail="Advanced analytics not available in this version"
        )
    
    # Import analytics handler
    from api.analytics import get_advanced_analytics
    
    analytics_data = await get_advanced_analytics(
        start_date=start_date,
        end_date=end_date,
        metrics=metrics or ["all"],
        user_id=current_user.user_id
    )
    
    return VersionedResponse.format_response(analytics_data, "v2")


@router.post("/feedback", response_model=FeedbackResponse)
async def submit_feedback_v2(
    request: FeedbackRequestValidator,
    background_tasks: BackgroundTasks,
    current_user=Depends(get_current_user)
):
    """
    Submit feedback for a query (v2 API).
    
    Enhanced with background processing and analytics.
    """
    # Import main feedback handler
    from api.main import submit_feedback as main_submit_feedback
    
    result = await main_submit_feedback(request, current_user)
    
    # Add background task for feedback analytics
    background_tasks.add_task(
        process_feedback_analytics,
        feedback=request,
        user_id=current_user.user_id
    )
    
    return VersionedResponse.format_response(result, "v2")


@router.websocket("/query/stream")
async def websocket_query_v2(websocket, current_user=Depends(get_current_user)):
    """
    WebSocket endpoint for real-time query streaming (v2 API).
    
    Allows bidirectional communication for interactive queries.
    """
    if not get_feature_flag("v2", "streaming_responses"):
        await websocket.close(code=1008, reason="Feature not available")
        return
    
    await websocket.accept()
    
    try:
        while True:
            # Receive query
            data = await websocket.receive_json()
            
            # Validate query
            try:
                query_request = QueryRequestValidator(**data)
            except Exception as e:
                await websocket.send_json({
                    "error": "Invalid query format",
                    "details": str(e)
                })
                continue
            
            # Process query with streaming
            from api.main import orchestrator
            
            # Send acknowledgment
            await websocket.send_json({
                "type": "acknowledgment",
                "query_id": data.get("query_id", "unknown")
            })
            
            # Stream results
            async for chunk in orchestrator.stream_query(query_request.dict()):
                await websocket.send_json({
                    "type": "result_chunk",
                    "chunk": chunk
                })
            
            # Send completion
            await websocket.send_json({
                "type": "complete",
                "query_id": data.get("query_id", "unknown")
            })
            
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await websocket.close(code=1011, reason="Internal error")


# Helper functions

async def stream_query_response(request: QueryRequestValidator, user):
    """Generate streaming response for a query."""
    from api.main import orchestrator
    
    # Send initial acknowledgment
    yield json.dumps({
        "type": "start",
        "query": request.query[:100] + "..." if len(request.query) > 100 else request.query
    }) + "\n"
    
    # Stream results
    async for chunk in orchestrator.stream_query({
        "query": request.query,
        "max_tokens": request.max_tokens,
        "confidence_threshold": request.confidence_threshold,
        "language": request.language,
        "user_id": user.user_id
    }):
        yield json.dumps({
            "type": "chunk",
            "data": chunk
        }) + "\n"
    
    # Send completion
    yield json.dumps({
        "type": "complete",
        "timestamp": asyncio.get_event_loop().time()
    }) + "\n"


async def track_advanced_analytics(query: str, result: Dict[str, Any], user_id: str):
    """Track advanced analytics for v2 queries."""
    logger.info(f"Tracking advanced analytics for user {user_id}")
    # Implementation would go here
    pass


async def process_feedback_analytics(feedback: FeedbackRequestValidator, user_id: str):
    """Process feedback analytics in background."""
    logger.info(f"Processing feedback analytics for user {user_id}")
    # Implementation would go here
    pass 