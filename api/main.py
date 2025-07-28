import asyncio
import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import Dict, Any, Optional
from datetime import datetime, timedelta

import uvicorn
from fastapi import FastAPI, HTTPException, Request, Response, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format='{"timestamp": "%(asctime)s", "level": "%(levelname)s", "logger": "%(name)s", "message": "%(message)s", "service": "sarvanom-api", "version": "1.0.0"}',
    handlers=[logging.StreamHandler()],
)


# Create custom formatter that handles missing fields
class SafeJSONFormatter(logging.Formatter):
    def format(self, record):
        # Add default values for common fields
        record.request_id = getattr(record, "request_id", "unknown")
        record.user_id = getattr(record, "user_id", "unknown")
        record.service = "sarvanom-api"
        record.version = "1.0.0"
        
        # Use JSON format with all fields
        log_obj = {
            "timestamp": self.formatTime(record),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "request_id": record.request_id,
            "user_id": record.user_id,
            "service": record.service,
            "version": record.version
        }
        
        import json
        return json.dumps(log_obj)

# Apply custom formatter to all handlers
root_logger = logging.getLogger()
formatter = SafeJSONFormatter()
for handler in root_logger.handlers:
    handler.setFormatter(formatter)

logger = logging.getLogger(__name__)

# Import after logging setup
from agents.lead_orchestrator import LeadOrchestrator, AgentType
from api.models import (
    QueryRequest,
    QueryResponse,
    QueryUpdateRequest,
    QueryListResponse,
    QueryDetailResponse,
    QueryStatusResponse,
    HealthResponse,
    MetricsResponse,
    AnalyticsResponse,
    FeedbackRequest,
    FeedbackResponse,
)
from api.validators import (
    QueryRequestValidator,
    FeedbackRequestValidator,
    SearchRequestValidator,
    AnalyticsRequestValidator,
    ConfigUpdateValidator,
)
from api.auth import get_current_user
from api.analytics import track_query
from api.cache import _query_cache
from agents.base_agent import QueryContext
from api.metrics import record_request_metrics, record_error_metrics, record_business_metrics

# Global variables
orchestrator = None
startup_time = None
app_version = "1.0.0"

# In-memory query storage (for demonstration - replace with database in production)
query_storage = {}
query_index = 0


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager with proper startup and shutdown."""
    global orchestrator, startup_time

    # Startup
    startup_time = time.time()
    logger.info("🚀 Starting SarvanOM - Your Own Knowledge Hub Powered by AI")

    try:
        # Setup graceful shutdown handler
        from api.shutdown_handler import get_shutdown_handler
        
        shutdown_handler = get_shutdown_handler()
        shutdown_handler.setup_signal_handlers()
        logger.info("✅ Shutdown handlers configured")
        
        # Initialize connection pools
        from api.connection_pool import get_pool_manager
        
        pool_manager = await get_pool_manager()
        logger.info("✅ Connection pools initialized")
        
        # Initialize rate limiter
        from api.rate_limiter import get_rate_limiter
        
        rate_limiter = await get_rate_limiter()
        logger.info("✅ Rate limiter initialized")

        # Initialize caches
        from api.cache import initialize_caches

        await initialize_caches()

        # Start integration monitoring
        from api.integration_monitor import start_integration_monitoring

        await start_integration_monitoring()

        # Initialize orchestrator
        orchestrator = LeadOrchestrator()
        logger.info("✅ SarvanOM initialized successfully")
    except Exception as e:
        logger.error(f"❌ Failed to initialize SarvanOM: {e}")
        raise

    yield

    # Shutdown
    logger.info("🛑 Shutting down SarvanOM - Starting graceful shutdown")
    
    # Create shutdown tasks list
    shutdown_tasks = []
    shutdown_errors = []
    
    # 1. Stop accepting new requests (handled by FastAPI)
    
    # 2. Wait for ongoing requests to complete (with timeout)
    logger.info("⏳ Waiting for ongoing requests to complete...")
    await asyncio.sleep(2)  # Give requests 2 seconds to complete
    
    # 3. Shutdown orchestrator
    if orchestrator:
        try:
            logger.info("🔄 Shutting down orchestrator...")
            await orchestrator.shutdown()
            logger.info("✅ Orchestrator shut down")
        except Exception as e:
            error_msg = f"Error during orchestrator shutdown: {e}"
            logger.error(error_msg)
            shutdown_errors.append(error_msg)

    # 4. Shutdown rate limiter
    try:
        from api.rate_limiter import get_rate_limiter
        
        logger.info("🔄 Shutting down rate limiter...")
        rate_limiter = await get_rate_limiter()
        await rate_limiter.shutdown()
        logger.info("✅ Rate limiter shut down")
    except Exception as e:
        error_msg = f"Error during rate limiter shutdown: {e}"
        logger.error(error_msg)
        shutdown_errors.append(error_msg)

    # 5. Shutdown connection pools
    try:
        from api.connection_pool import shutdown_pools
        
        logger.info("🔄 Shutting down connection pools...")
        await shutdown_pools()
        logger.info("✅ Connection pools shut down")
    except Exception as e:
        error_msg = f"Error during connection pool shutdown: {e}"
        logger.error(error_msg)
        shutdown_errors.append(error_msg)

    # 6. Flush and shutdown caches
    try:
        from api.cache import shutdown_caches

        logger.info("🔄 Flushing and shutting down caches...")
        await shutdown_caches()
        logger.info("✅ Caches flushed and shut down")
    except Exception as e:
        error_msg = f"Error during cache shutdown: {e}"
        logger.error(error_msg)
        shutdown_errors.append(error_msg)
    
    # 7. Stop monitoring tasks
    try:
        from api.integration_monitor import stop_integration_monitoring
        
        logger.info("🔄 Stopping integration monitoring...")
        await stop_integration_monitoring()
        logger.info("✅ Integration monitoring stopped")
    except Exception as e:
        error_msg = f"Error stopping integration monitoring: {e}"
        logger.error(error_msg)
        shutdown_errors.append(error_msg)
    
    # 8. Final cleanup
    if shutdown_errors:
        logger.warning(f"⚠️ Shutdown completed with {len(shutdown_errors)} errors:")
        for error in shutdown_errors:
            logger.warning(f"  - {error}")
    else:
        logger.info("✅ Graceful shutdown completed successfully")
    
    # Log final shutdown time
    shutdown_duration = time.time() - startup_time if startup_time else 0
    logger.info(f"📊 Total runtime: {shutdown_duration:.2f} seconds")


# Create FastAPI app with lifespan
app = FastAPI(
    title="SarvanOM - Your Own Knowledge Hub",
    description="AI-powered knowledge platform with multi-agent architecture",
    version="1.0.0",
    lifespan=lifespan,
)

# Add CORS origins from environment
cors_origins = os.getenv("CORS_ORIGINS", "http://localhost:3000,http://localhost:3001").split(",")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "X-API-Key", "Authorization"],
)

# Add rate limiting middleware
from api.rate_limiter import rate_limit_middleware
app.middleware("http")(rate_limit_middleware)

# Add authentication endpoints
from api.auth_endpoints import router as auth_router
app.include_router(auth_router)


# Request ID middleware
@app.middleware("http")
async def add_request_id(request: Request, call_next):
    """Add request ID to all requests for tracing."""
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id

    # Add request ID to response headers
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id

    return response


# Logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all requests with timing and request ID."""
    start_time = time.time()
    request_id = getattr(request.state, "request_id", "unknown")

    # Extract user info
    user_id = "anonymous"
    try:
        current_user = await get_current_user(request)
        user_id = current_user.user_id
    except:
        pass

    # Log request
    logger.info(
        f"📥 {request.method} {request.url.path} from {request.client.host}",
        extra={
            "request_id": request_id,
            "user_id": user_id,
            "method": request.method,
            "path": request.url.path,
            "client_ip": request.client.host,
        },
    )

    try:
        response = await call_next(request)
        process_time = time.time() - start_time

        # Log response
        logger.info(
            f"📤 {request.method} {request.url.path} -> {response.status_code} ({process_time:.3f}s)",
            extra={
                "request_id": request_id,
                "user_id": user_id,
                "status_code": response.status_code,
                "process_time": process_time,
            },
        )

        return response
    except Exception as e:
        process_time = time.time() - start_time
        logger.error(
            f"❌ {request.method} {request.url.path} -> ERROR ({process_time:.3f}s): {str(e)}",
            extra={
                "request_id": request_id,
                "user_id": user_id,
                "error": str(e),
                "process_time": process_time,
            },
            exc_info=True,
        )
        raise


# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler with structured logging."""
    request_id = getattr(request.state, "request_id", "unknown")

    logger.error(
        f"Unhandled exception: {str(exc)}",
        extra={
            "request_id": request_id,
            "exception_type": type(exc).__name__,
            "exception_message": str(exc),
        },
        exc_info=True,
    )

    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "request_id": request_id,
            "timestamp": time.time(),
        },
    )


@app.exception_handler(ValidationError)
async def validation_exception_handler(request: Request, exc: ValidationError):
    """Handle Pydantic validation errors."""
    request_id = getattr(request.state, "request_id", "unknown")

    logger.warning(
        f"Validation error: {exc.errors()}",
        extra={"request_id": request_id, "validation_errors": exc.errors()},
    )

    return JSONResponse(
        status_code=422,
        content={"error": "Validation error", "details": exc.errors(), "request_id": request_id},
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions."""
    request_id = getattr(request.state, "request_id", "unknown")

    logger.warning(
        f"HTTP {exc.status_code}: {exc.detail}",
        extra={"request_id": request_id, "status_code": exc.status_code, "detail": exc.detail},
    )

    return JSONResponse(
        status_code=exc.status_code, content={"error": exc.detail, "request_id": request_id}
    )


@app.get("/", response_model=Dict[str, Any])
async def root():
    """Root endpoint with application info."""
    return {
        "name": "SarvanOM - Your Own Knowledge Hub",
        "version": "1.0.0",
        "status": "running",
        "uptime": time.time() - startup_time if startup_time else 0,
        "endpoints": {
            "health": "/health",
            "docs": "/docs",
            "query": "/query",
            "metrics": "/metrics",
            "analytics": "/analytics",
        },
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint with detailed status."""
    try:
        # Import health checks module
        from api.health_checks import check_all_services
        
        # Check orchestrator health
        orchestrator_healthy = orchestrator is not None

        # Check cache health
        cache_healthy = _query_cache is not None

        # Check external integrations with actual health checks
        external_health = await check_all_services()
        
        # Extract individual service statuses
        integration_status = {}
        for service_name, service_health in external_health["services"].items():
            integration_status[service_name] = "healthy" if service_health.get("healthy", False) else "unhealthy"

        # Overall health - consider core components critical, external services as warnings
        healthy = orchestrator_healthy and cache_healthy

        return HealthResponse(
            status="healthy" if healthy else "unhealthy",
            version=app_version,
            timestamp=time.time(),
            uptime=time.time() - startup_time if startup_time else 0,
            components={
                "orchestrator": "healthy" if orchestrator_healthy else "unhealthy",
                "cache": "healthy" if cache_healthy else "unhealthy",
                **integration_status,
            },
            metadata={
                "external_services_detail": external_health["services"],
                "total_check_latency_ms": external_health.get("total_latency_ms", 0)
            }
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthResponse(
            status="unhealthy",
            version=app_version,
            timestamp=time.time(),
            uptime=time.time() - startup_time if startup_time else 0,
            error=str(e),
        )


@app.post("/query", response_model=QueryResponse)
async def process_query(
    request: QueryRequestValidator, http_request: Request, current_user=Depends(get_current_user)
):
    """Process a knowledge query through the multi-agent system."""
    request_id = getattr(http_request.state, "request_id", "unknown")

    # Add request ID to logging context
    logger.info(
        f"Processing query: {request.query[:50]}...",
        extra={
            "request_id": request_id,
            "user_id": current_user.user_id,
            "query_length": len(request.query),
            "max_tokens": request.max_tokens,
            "confidence_threshold": request.confidence_threshold,
        },
    )

    start_time = time.time()

    try:
        # Check cache first
        cache_key = f"{current_user.user_id}:{request.query}"
        cached_result = await _query_cache.get(cache_key, user_context=request.user_context)

        if cached_result:
            logger.info(
                f"Cache HIT for query: {request.query[:50]}...",
                extra={
                    "request_id": request_id,
                    "user_id": current_user.user_id,
                    "cache_hit": True,
                },
            )

            # Track analytics for cache hit
            await track_query(
                query=request.query,
                execution_time=time.time() - start_time,
                confidence=cached_result.get("confidence", 0.0),
                client_ip=http_request.client.host,
                user_id=current_user.user_id,
                cache_hit=True,
            )

            # Record metrics
            record_request_metrics("POST", "/query", 200, time.time() - start_time)
            record_business_metrics(
                "cache_hit",
                cached_result.get("confidence", 0.0),
                len(cached_result.get("answer", "")),
                "/query",
            )

            return QueryResponse(
                answer=cached_result.get("answer", ""),
                confidence=cached_result.get("confidence", 0.0),
                citations=cached_result.get("citations", []),
                query_id=cached_result.get("query_id", str(uuid.uuid4())),
                metadata={
                    "cache_hit": True,
                    "request_id": request_id,
                    "execution_time_ms": int((time.time() - start_time) * 1000),
                },
            )

        # Create query context
        query_context = QueryContext(
            query=request.query,
            user_id=current_user.user_id,
            user_context={
                **(request.user_context or {}),
                "max_tokens": request.max_tokens,
                "confidence_threshold": request.confidence_threshold,
            },
            token_budget=request.max_tokens or 4000,
        )

        # Process query through orchestrator
        result = await orchestrator.process_query(query_context)
        process_time = time.time() - start_time

        # Check for partial failures
        agent_results = result.get("metadata", {}).get("agent_results", {})
        failed_agents = []
        successful_agents = []

        for agent_type, agent_result in agent_results.items():
            if agent_result.get("status") == "failed":
                failed_agents.append(agent_type)
            elif agent_result.get("status") == "success":
                successful_agents.append(agent_type)

        # Determine if this is a partial failure
        is_partial_failure = len(failed_agents) > 0 and len(successful_agents) > 0
        is_complete_failure = len(successful_agents) == 0

        # Adjust success flag based on failure analysis
        if is_complete_failure:
            result["success"] = False
        elif is_partial_failure:
            result["success"] = True  # Still successful but with partial results
            result["metadata"]["partial_failure"] = True
            result["metadata"]["failed_agents"] = failed_agents
            result["metadata"]["successful_agents"] = successful_agents

        # Cache the result
        await _query_cache.put(cache_key, result, user_context=request.user_context)

        # Log success with detailed information
        logger.info(
            f"Query processed successfully in {process_time:.3f}s",
            extra={
                "request_id": request_id,
                "user_id": current_user.user_id,
                "execution_time": process_time,
                "confidence": result.get("confidence", 0.0),
                "cache_hit": False,
                "partial_failure": is_partial_failure,
                "failed_agents": failed_agents,
                "successful_agents": successful_agents,
            },
        )

        # Track analytics
        await track_query(
            query=request.query,
            execution_time=process_time,
            confidence=result.get("confidence", 0.0),
            client_ip=http_request.client.host,
            user_id=current_user.user_id,
            cache_hit=False,
            agent_results=agent_results,
        )

        # Record metrics
        record_request_metrics("POST", "/query", 200, process_time)
        record_business_metrics(
            "query_processed",
            result.get("confidence", 0.0),
            len(result.get("answer", "")),
            "/query",
        )

        # Generate query ID and store query
        query_id = str(uuid.uuid4())
        
        # Store query in storage
        global query_index
        query_index += 1
        
        query_record = {
            "query_id": query_id,
            "query": request.query,
            "answer": result.get("answer", ""),
            "confidence": result.get("confidence", 0.0),
            "citations": result.get("citations", []),
            "metadata": {
                "request_id": request_id,
                "execution_time_ms": int(process_time * 1000),
                "cache_hit": False,
                "agent_results": agent_results,
                "token_usage": result.get("metadata", {}).get("token_usage", {}),
                "partial_failure": is_partial_failure,
                "failed_agents": failed_agents,
                "successful_agents": successful_agents,
            },
            "created_at": datetime.now(),
            "updated_at": None,
            "processing_time": process_time,
            "user_id": current_user.user_id,
            "status": "completed",
            "max_tokens": request.max_tokens,
            "confidence_threshold": request.confidence_threshold,
            "user_context": request.user_context
        }
        
        query_storage[query_id] = query_record

        return QueryResponse(
            answer=result.get("answer", ""),
            confidence=result.get("confidence", 0.0),
            citations=result.get("citations", []),
            query_id=query_id,
            processing_time=process_time,
            metadata={
                "request_id": request_id,
                "execution_time_ms": int(process_time * 1000),
                "cache_hit": False,
                "agent_results": agent_results,
                "token_usage": result.get("metadata", {}).get("token_usage", {}),
                "partial_failure": is_partial_failure,
                "failed_agents": failed_agents,
                "successful_agents": successful_agents,
            },
        )

    except Exception as e:
        process_time = time.time() - start_time

        logger.error(
            f"❌ Query processing failed: {str(e)}",
            extra={
                "request_id": request_id,
                "user_id": current_user.user_id,
                "execution_time": process_time,
                "error": str(e),
            },
            exc_info=True,
        )

        # Track failed query
        await track_query(
            query=request.query,
            execution_time=process_time,
            confidence=0.0,
            client_ip=http_request.client.host,
            user_id=current_user.user_id,
            error=str(e),
        )

        # Record error metrics
        record_error_metrics("query_processing_error", "/query")
        record_request_metrics("POST", "/query", 500, process_time)

        raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")


@app.post("/feedback", response_model=FeedbackResponse)
async def submit_feedback(
    feedback: FeedbackRequest, 
    http_request: Request, 
    current_user=Depends(get_current_user)
):
    """Submit user feedback for query results."""
    request_id = getattr(http_request.state, "request_id", "unknown")
    
    logger.info(
        f"Feedback received for query {feedback.query_id}: {feedback.feedback_type}",
        extra={
            "request_id": request_id,
            "user_id": current_user.user_id,
            "query_id": feedback.query_id,
            "feedback_type": feedback.feedback_type,
        },
    )
    
    try:
        # Generate feedback ID
        feedback_id = f"feedback_{feedback.query_id}_{int(time.time())}"
        
        # Store feedback (placeholder - implement actual storage)
        feedback_data = {
            "feedback_id": feedback_id,
            "query_id": feedback.query_id,
            "user_id": current_user.user_id,
            "feedback_type": feedback.feedback_type,
            "details": feedback.details,
            "timestamp": datetime.now().isoformat(),
            "request_id": request_id,
        }
        
        # Store in database/cache for analytics
        try:
            from api.analytics import store_feedback
            await store_feedback(feedback_data)
            logger.info(f"Feedback stored in analytics: {feedback_id}")
        except Exception as e:
            logger.warning(f"Failed to store feedback in analytics: {e}")
        
        logger.info(f"Feedback stored: {feedback_data}")
        
        return FeedbackResponse(
            success=True,
            message="Feedback recorded successfully",
            feedback_id=feedback_id
        )
        
    except Exception as e:
        logger.error(f"Failed to process feedback: {e}", extra={"request_id": request_id})
        raise HTTPException(
            status_code=500,
            detail=f"Failed to record feedback: {str(e)}"
        )


@app.get("/metrics")
async def get_metrics():
    """Get application metrics in Prometheus format."""
    try:
        from api.metrics import get_metrics_collector

        collector = get_metrics_collector()
        metrics_dict = collector.get_metrics_dict()

        # Add application-specific metrics
        metrics_dict.update(
            {
                "sarvanom_version": app_version,
                "sarvanom_uptime_seconds": time.time() - startup_time if startup_time else 0,
                "sarvanom_requests_total": metrics_dict.get("request_counter", 0),
                "sarvanom_errors_total": metrics_dict.get("error_counter", 0),
                "sarvanom_cache_hits_total": metrics_dict.get("cache_hits", 0),
                "sarvanom_cache_misses_total": metrics_dict.get("cache_misses", 0),
                "sarvanom_average_response_time_seconds": metrics_dict.get(
                    "average_response_time", 0.0
                ),
                "sarvanom_active_users": len(metrics_dict.get("user_activity", {})),
                "sarvanom_partial_failures_total": metrics_dict.get("partial_failures", 0),
                "sarvanom_complete_failures_total": metrics_dict.get("complete_failures", 0),
            }
        )

        # Add integration status metrics
        integration_status = {
            "sarvanom_integration_vector_db_status": 1,  # 1 = healthy, 0 = unhealthy
            "sarvanom_integration_elasticsearch_status": 1,
            "sarvanom_integration_knowledge_graph_status": 1,
            "sarvanom_integration_llm_api_status": 1,
        }
        metrics_dict.update(integration_status)

        return metrics_dict
    except Exception as e:
        logger.error(f"Failed to get metrics: {e}")
        return {"error": "Failed to get metrics"}


@app.get("/analytics")
async def get_analytics():
    """Get detailed analytics data."""
    try:
        from api.analytics import get_analytics

        analytics = await get_analytics()
        return analytics
    except Exception as e:
        logger.error(f"Failed to get analytics: {e}")
        return {"error": "Failed to get analytics"}


@app.get("/integrations")
async def get_integration_status():
    """Get status of all external integrations."""
    try:
        from api.integration_monitor import get_integration_monitor

        monitor = await get_integration_monitor()
        status = await monitor.get_integration_status()

        return {
            "timestamp": time.time(),
            "integrations": status,
            "summary": {
                "total": len(status),
                "healthy": len([s for s in status.values() if s["status"] == "healthy"]),
                "unhealthy": len([s for s in status.values() if s["status"] == "unhealthy"]),
                "not_configured": len(
                    [s for s in status.values() if s["status"] == "not_configured"]
                ),
            },
        }
    except Exception as e:
        logger.error(f"Failed to get integration status: {e}")
        return {"error": "Failed to get integration status"}


# ============================================================================
# COMPREHENSIVE QUERY MANAGEMENT ENDPOINTS
# ============================================================================

@app.get("/queries", response_model=QueryListResponse)
async def list_queries(
    page: int = 1,
    page_size: int = 20,
    user_filter: Optional[str] = None,
    status_filter: Optional[str] = None,
    http_request: Request = None,
    current_user=Depends(get_current_user)
):
    """List all queries with pagination and filtering."""
    request_id = getattr(http_request.state, "request_id", "unknown")
    
    try:
        # Filter queries for current user (unless admin)
        user_queries = []
        for query_record in query_storage.values():
            # Basic user filter - only show user's own queries unless admin
            if query_record["user_id"] == current_user.user_id or current_user.permissions.get("admin", False):
                # Apply additional filters
                if user_filter and user_filter not in query_record["user_id"]:
                    continue
                if status_filter and query_record["status"] != status_filter:
                    continue
                user_queries.append(query_record)
        
        # Sort by creation date (newest first)
        user_queries.sort(key=lambda x: x["created_at"], reverse=True)
        
        # Implement pagination
        total = len(user_queries)
        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size
        paginated_queries = user_queries[start_idx:end_idx]
        
        # Convert to simplified format for listing
        query_list = []
        for query in paginated_queries:
            query_list.append({
                "query_id": query["query_id"],
                "query": query["query"][:100] + "..." if len(query["query"]) > 100 else query["query"],
                "status": query["status"],
                "confidence": query["confidence"],
                "created_at": query["created_at"].isoformat(),
                "processing_time": query["processing_time"]
            })
        
        has_next = end_idx < total
        
        logger.info(
            f"Listed {len(query_list)} queries for user {current_user.user_id}",
            extra={"request_id": request_id, "user_id": current_user.user_id, "total": total}
        )
        
        return QueryListResponse(
            queries=query_list,
            total=total,
            page=page,
            page_size=page_size,
            has_next=has_next
        )
        
    except Exception as e:
        logger.error(f"Failed to list queries: {e}", extra={"request_id": request_id})
        raise HTTPException(status_code=500, detail=f"Failed to list queries: {str(e)}")


@app.get("/queries/{query_id}", response_model=QueryDetailResponse)
async def get_query(
    query_id: str,
    http_request: Request,
    current_user=Depends(get_current_user)
):
    """Get detailed information about a specific query."""
    request_id = getattr(http_request.state, "request_id", "unknown")
    
    try:
        if query_id not in query_storage:
            raise HTTPException(status_code=404, detail="Query not found")
        
        query_record = query_storage[query_id]
        
        # Check authorization - users can only see their own queries unless admin
        if query_record["user_id"] != current_user.user_id and not current_user.permissions.get("admin", False):
            raise HTTPException(status_code=403, detail="Access denied")
        
        logger.info(
            f"Retrieved query {query_id}",
            extra={"request_id": request_id, "user_id": current_user.user_id, "query_id": query_id}
        )
        
        return QueryDetailResponse(
            query_id=query_record["query_id"],
            query=query_record["query"],
            answer=query_record["answer"],
            confidence=query_record["confidence"],
            citations=query_record["citations"],
            metadata=query_record["metadata"],
            created_at=query_record["created_at"],
            updated_at=query_record["updated_at"],
            processing_time=query_record["processing_time"],
            user_id=query_record["user_id"],
            status=query_record["status"]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get query {query_id}: {e}", extra={"request_id": request_id})
        raise HTTPException(status_code=500, detail=f"Failed to get query: {str(e)}")


@app.put("/queries/{query_id}", response_model=QueryDetailResponse)
async def update_query(
    query_id: str,
    update_request: QueryUpdateRequest,
    http_request: Request,
    current_user=Depends(get_current_user)
):
    """Update an existing query and optionally reprocess it."""
    request_id = getattr(http_request.state, "request_id", "unknown")
    
    try:
        if query_id not in query_storage:
            raise HTTPException(status_code=404, detail="Query not found")
        
        query_record = query_storage[query_id]
        
        # Check authorization
        if query_record["user_id"] != current_user.user_id and not current_user.permissions.get("admin", False):
            raise HTTPException(status_code=403, detail="Access denied")
        
        # Update fields
        if update_request.query is not None:
            query_record["query"] = update_request.query
        if update_request.max_tokens is not None:
            query_record["max_tokens"] = update_request.max_tokens
        if update_request.confidence_threshold is not None:
            query_record["confidence_threshold"] = update_request.confidence_threshold
        if update_request.user_context is not None:
            query_record["user_context"] = update_request.user_context
        
        query_record["updated_at"] = datetime.now()
        
        # Reprocess if requested
        if update_request.reprocess and update_request.query:
            query_record["status"] = "processing"
            query_storage[query_id] = query_record
            
            # Create new query request
            new_request = QueryRequest(
                query=query_record["query"],
                max_tokens=query_record["max_tokens"],
                confidence_threshold=query_record["confidence_threshold"],
                user_context=query_record["user_context"]
            )
            
            # Process through orchestrator
            start_time = time.time()
            query_context = QueryContext(
                query=new_request.query,
                user_id=current_user.user_id,
                user_context={
                    **(new_request.user_context or {}),
                    "max_tokens": new_request.max_tokens,
                    "confidence_threshold": new_request.confidence_threshold,
                },
                token_budget=new_request.max_tokens or 4000,
            )
            
            result = await orchestrator.process_query(query_context)
            process_time = time.time() - start_time
            
            # Update with new results
            query_record.update({
                "answer": result.get("answer", ""),
                "confidence": result.get("confidence", 0.0),
                "citations": result.get("citations", []),
                "processing_time": process_time,
                "status": "completed",
                "metadata": {
                    **query_record["metadata"],
                    "reprocessed": True,
                    "reprocess_time": process_time,
                    "execution_time_ms": int(process_time * 1000)
                }
            })
        
        query_storage[query_id] = query_record
        
        logger.info(
            f"Updated query {query_id}",
            extra={
                "request_id": request_id, 
                "user_id": current_user.user_id, 
                "query_id": query_id,
                "reprocessed": update_request.reprocess
            }
        )
        
        return QueryDetailResponse(
            query_id=query_record["query_id"],
            query=query_record["query"],
            answer=query_record["answer"],
            confidence=query_record["confidence"],
            citations=query_record["citations"],
            metadata=query_record["metadata"],
            created_at=query_record["created_at"],
            updated_at=query_record["updated_at"],
            processing_time=query_record["processing_time"],
            user_id=query_record["user_id"],
            status=query_record["status"]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update query {query_id}: {e}", extra={"request_id": request_id})
        raise HTTPException(status_code=500, detail=f"Failed to update query: {str(e)}")


@app.delete("/queries/{query_id}")
async def delete_query(
    query_id: str,
    http_request: Request,
    current_user=Depends(get_current_user)
):
    """Delete a specific query."""
    request_id = getattr(http_request.state, "request_id", "unknown")
    
    try:
        if query_id not in query_storage:
            raise HTTPException(status_code=404, detail="Query not found")
        
        query_record = query_storage[query_id]
        
        # Check authorization
        if query_record["user_id"] != current_user.user_id and not current_user.permissions.get("admin", False):
            raise HTTPException(status_code=403, detail="Access denied")
        
        # Delete the query
        del query_storage[query_id]
        
        logger.info(
            f"Deleted query {query_id}",
            extra={"request_id": request_id, "user_id": current_user.user_id, "query_id": query_id}
        )
        
        return {"message": "Query deleted successfully", "query_id": query_id}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete query {query_id}: {e}", extra={"request_id": request_id})
        raise HTTPException(status_code=500, detail=f"Failed to delete query: {str(e)}")


@app.get("/queries/{query_id}/status", response_model=QueryStatusResponse)
async def get_query_status(
    query_id: str,
    http_request: Request,
    current_user=Depends(get_current_user)
):
    """Get the processing status of a specific query."""
    request_id = getattr(http_request.state, "request_id", "unknown")
    
    try:
        if query_id not in query_storage:
            raise HTTPException(status_code=404, detail="Query not found")
        
        query_record = query_storage[query_id]
        
        # Check authorization
        if query_record["user_id"] != current_user.user_id and not current_user.permissions.get("admin", False):
            raise HTTPException(status_code=403, detail="Access denied")
        
        # Calculate progress for processing queries
        progress = None
        estimated_completion = None
        message = None
        
        if query_record["status"] == "processing":
            # Simulate progress calculation (in real implementation, this would be tracked)
            progress = 0.5  # 50% complete
            estimated_completion = datetime.now() + timedelta(seconds=30)
            message = "Query is being processed by the multi-agent system"
        elif query_record["status"] == "completed":
            progress = 1.0
            message = "Query processing completed successfully"
        elif query_record["status"] == "failed":
            progress = 0.0
            message = "Query processing failed"
        
        return QueryStatusResponse(
            query_id=query_id,
            status=query_record["status"],
            message=message,
            progress=progress,
            estimated_completion=estimated_completion
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get query status {query_id}: {e}", extra={"request_id": request_id})
        raise HTTPException(status_code=500, detail=f"Failed to get query status: {str(e)}")


@app.patch("/queries/{query_id}/reprocess")
async def reprocess_query(
    query_id: str,
    http_request: Request,
    current_user=Depends(get_current_user)
):
    """Reprocess an existing query with the same parameters."""
    request_id = getattr(http_request.state, "request_id", "unknown")
    
    try:
        if query_id not in query_storage:
            raise HTTPException(status_code=404, detail="Query not found")
        
        query_record = query_storage[query_id]
        
        # Check authorization
        if query_record["user_id"] != current_user.user_id and not current_user.permissions.get("admin", False):
            raise HTTPException(status_code=403, detail="Access denied")
        
        # Mark as processing
        query_record["status"] = "processing"
        query_record["updated_at"] = datetime.now()
        query_storage[query_id] = query_record
        
        # Create query request from stored data
        reprocess_request = QueryRequest(
            query=query_record["query"],
            max_tokens=query_record["max_tokens"],
            confidence_threshold=query_record["confidence_threshold"],
            user_context=query_record["user_context"]
        )
        
        # Process through orchestrator
        start_time = time.time()
        query_context = QueryContext(
            query=reprocess_request.query,
            user_id=current_user.user_id,
            user_context={
                **(reprocess_request.user_context or {}),
                "max_tokens": reprocess_request.max_tokens,
                "confidence_threshold": reprocess_request.confidence_threshold,
            },
            token_budget=reprocess_request.max_tokens or 4000,
        )
        
        result = await orchestrator.process_query(query_context)
        process_time = time.time() - start_time
        
        # Update with new results
        query_record.update({
            "answer": result.get("answer", ""),
            "confidence": result.get("confidence", 0.0),
            "citations": result.get("citations", []),
            "processing_time": process_time,
            "status": "completed",
            "updated_at": datetime.now(),
            "metadata": {
                **query_record["metadata"],
                "reprocessed": True,
                "reprocess_count": query_record["metadata"].get("reprocess_count", 0) + 1,
                "last_reprocess_time": process_time
            }
        })
        
        query_storage[query_id] = query_record
        
        logger.info(
            f"Reprocessed query {query_id}",
            extra={
                "request_id": request_id, 
                "user_id": current_user.user_id, 
                "query_id": query_id,
                "processing_time": process_time
            }
        )
        
        return {
            "message": "Query reprocessed successfully", 
            "query_id": query_id,
            "processing_time": process_time,
            "new_confidence": result.get("confidence", 0.0)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to reprocess query {query_id}: {e}", extra={"request_id": request_id})
        raise HTTPException(status_code=500, detail=f"Failed to reprocess query: {str(e)}")


if __name__ == "__main__":
    uvicorn.run(
        "api.main:app",
        host=os.getenv("UKP_HOST", "0.0.0.0"),
        port=int(os.getenv("UKP_PORT", "8002")),
        reload=os.getenv("UKP_RELOAD", "false").lower() == "true",
        log_level=os.getenv("UKP_LOG_LEVEL", "info"),
    )
