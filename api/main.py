from fastapi import FastAPI, HTTPException, Request, Response, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ValidationError, Field, validator
from typing import Dict, List, Optional, Any
import asyncio
import logging
import time
from datetime import datetime
from contextlib import asynccontextmanager
from collections import defaultdict
import asyncio
from datetime import datetime, timedelta
from dotenv import load_dotenv
import os
import uuid
import json

# Load environment variables from .env file
load_dotenv()

# Import advanced features
from api.cache import get_cached_result, cache_result, get_cache_stats
from api.analytics import track_query, get_analytics_summary
from api.security import check_security, get_security_summary
from api.auth import get_current_user, require_read, require_write, require_admin, log_request_safely, User
from api.metrics import (
    record_request_metrics, record_agent_metrics, record_cache_metrics,
    record_security_metrics, record_token_metrics, record_error_metrics,
    record_business_metrics, get_metrics_collector
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global request tracking
request_counter = 0
error_counter = 0

# Global orchestrator instance
orchestrator = None

# Rate limiting storage
rate_limit_store = defaultdict(list)

# Concurrency control
MAX_CONCURRENT_REQUESTS = int(os.getenv('MAX_CONCURRENT_REQUESTS', '10'))
request_semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

# Request tracking for concurrency monitoring
active_requests = set()
request_start_times = {}

def check_rate_limit(client_ip: str, limit: int = 60, window: int = 60) -> bool:
    """Check if client has exceeded rate limit."""
    now = time.time()
    
    # Clean old entries
    rate_limit_store[client_ip] = [
        timestamp for timestamp in rate_limit_store[client_ip]
        if now - timestamp < window
    ]
    
    # Check limit
    if len(rate_limit_store[client_ip]) >= limit:
        return False
    
    # Add current request
    rate_limit_store[client_ip].append(now)
    return True

def get_client_ip(request: Request) -> str:
    """Extract client IP from request."""
    forwarded_for = request.headers.get("X-Forwarded-For")
    if forwarded_for:
        return forwarded_for.split(",")[0].strip()
    return request.client.host if request.client else "unknown"

async def track_request(request_id: str):
    """Track active request for concurrency monitoring."""
    active_requests.add(request_id)
    request_start_times[request_id] = time.time()

async def untrack_request(request_id: str):
    """Remove request from tracking."""
    active_requests.discard(request_id)
    request_start_times.pop(request_id, None)

def get_concurrency_stats() -> Dict[str, Any]:
    """Get concurrency statistics."""
    now = time.time()
    active_count = len(active_requests)
    avg_request_time = 0
    
    if active_count > 0:
        request_times = [now - start_time for start_time in request_start_times.values()]
        avg_request_time = sum(request_times) / len(request_times)
    
    return {
        'active_requests': active_count,
        'max_concurrent': MAX_CONCURRENT_REQUESTS,
        'available_slots': MAX_CONCURRENT_REQUESTS - active_count,
        'avg_request_time': avg_request_time,
        'semaphore_value': request_semaphore._value
    }

class ErrorResponse(BaseModel):
    """Standard error response format."""
    error: str
    detail: Optional[str] = None
    timestamp: str  # Changed from datetime to str for JSON serialization
    request_id: str

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for FastAPI app."""
    # Startup
    logger.info("🚀 Starting Universal Knowledge Platform")
    
    # Initialize core components
    try:
        # Initialize orchestrator
        import sys
        import os
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        
        from agents.lead_orchestrator import LeadOrchestrator
        global orchestrator # Declare global to allow modification
        orchestrator = LeadOrchestrator()
        
        # Initialize other components as needed
        logger.info("✅ Universal Knowledge Platform initialized successfully")
    except Exception as e:
        logger.error(f"❌ Failed to initialize platform: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down Universal Knowledge Platform")

app = FastAPI(
    title="Universal Knowledge Platform API",
    description="Multi-agent knowledge retrieval and synthesis platform",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add trusted host middleware
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*"]  # Configure appropriately for production
)

@app.exception_handler(ValidationError)
async def validation_exception_handler(request: Request, exc: ValidationError):
    """Handle validation errors."""
    return JSONResponse(
        status_code=422,
        content=ErrorResponse(
            error="Validation Error",
            detail=str(exc),
            timestamp=datetime.now().isoformat(),
            request_id=str(uuid.uuid4())
        ).dict()
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal Server Error",
            detail="An unexpected error occurred",
            timestamp=datetime.now().isoformat(),
            request_id=str(uuid.uuid4())
        ).dict()
    )

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """Add process time header to response and record metrics."""
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    
    # Record metrics
    record_request_metrics(
        method=request.method,
        endpoint=request.url.path,
        status_code=response.status_code,
        duration=process_time
    )
    
    response.headers["X-Process-Time"] = str(process_time)
    return response

@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log requests safely and record business metrics."""
    # Get user if authenticated
    user = None
    try:
        user = await get_current_user(request, None)
    except:
        pass  # User not authenticated, which is fine for some endpoints
    
    # Log request safely
    log_request_safely(request, user)
    
    response = await call_next(request)
    
    # Record business metrics for query endpoints
    if request.url.path == "/query" and response.status_code == 200:
        try:
            # Extract response length and confidence from response body
            response_body = b""
            async for chunk in response.body_iterator:
                response_body += chunk
            
            # Reconstruct response
            response = Response(
                content=response_body,
                status_code=response.status_code,
                headers=dict(response.headers),
                media_type=response.media_type
            )
            
            # Record business metrics
            record_business_metrics(
                category="query",
                confidence=0.8,  # Default confidence
                response_length=len(response_body),
                endpoint="/query"
            )
        except Exception as e:
            logger.warning(f"Failed to record business metrics: {e}")
    
    return response

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=10000, description="The query to process")
    user_context: Optional[Dict[str, Any]] = None
    max_tokens: Optional[int] = Field(default=1000, ge=1, le=10000, description="Maximum tokens for response")
    confidence_threshold: Optional[float] = Field(default=0.7, ge=0.0, le=1.0, description="Confidence threshold")
    
    @validator('query')
    def validate_query(cls, v):
        if not v or not v.strip():
            raise ValueError('Query cannot be empty')
        return v.strip()
    
    @validator('max_tokens')
    def validate_max_tokens(cls, v):
        if v is not None and (v < 1 or v > 10000):
            raise ValueError('max_tokens must be between 1 and 10,000')
        return v
    
    @validator('confidence_threshold')
    def validate_confidence_threshold(cls, v):
        if v is not None and (v < 0 or v > 1):
            raise ValueError('confidence_threshold must be between 0 and 1')
        return v

class QueryResponse(BaseModel):
    query: str
    answer: str
    confidence: float
    citations: List[Dict[str, Any]]
    execution_time: float
    timestamp: str  # Changed from datetime to str for JSON serialization
    metadata: Dict[str, Any]

class HealthResponse(BaseModel):
    status: str
    version: str
    timestamp: str  # Changed from datetime to str for JSON serialization
    agents_status: Dict[str, str]

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with basic info."""
    return {
        "message": "Universal Knowledge Platform API",
        "version": "1.0.0",
        "status": "running"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        timestamp=datetime.now().isoformat(),
        agents_status={
            "retrieval": "ready" if orchestrator else "not_initialized",
            "factcheck": "ready" if orchestrator else "not_initialized", 
            "synthesis": "ready" if orchestrator else "not_initialized",
            "citation": "ready" if orchestrator else "not_initialized",
            "orchestrator": "ready" if orchestrator else "not_initialized"
        }
    )

@app.post("/query", response_model=QueryResponse)
async def process_query(
    request: QueryRequest, 
    http_request: Request,
    current_user: User = Depends(require_read())  # Require read permission
):
    """Process a knowledge query through the multi-agent pipeline with authentication."""
    request_id = str(uuid.uuid4())
    
    # Concurrency control
    async with request_semaphore:
        await track_request(request_id)
        
        try:
            # Rate limiting (now per-user based on authentication)
            client_ip = get_client_ip(http_request)
            user_rate_limit = getattr(current_user, 'rate_limit', 60)
            
            if not check_rate_limit(client_ip, user_rate_limit):
                raise HTTPException(
                    status_code=429,
                    detail=f"Rate limit exceeded. Limit: {user_rate_limit} requests/minute"
                )
            
            # Security check (non-blocking)
            security_result = await check_security(
                request.query, 
                client_ip, 
                current_user.user_id,
                0.0  # Will be updated after processing
            )
            
            if security_result.get('blocked'):
                raise HTTPException(
                    status_code=403,
                    detail="Request blocked due to security concerns."
                )
            
            # Input validation
            if not request.query or not request.query.strip():
                raise HTTPException(
                    status_code=400,
                    detail="Query cannot be empty"
                )
            
            if len(request.query) > 10000:  # 10KB limit
                raise HTTPException(
                    status_code=400,
                    detail="Query too long. Maximum 10,000 characters allowed."
                )
            
            if request.max_tokens and (request.max_tokens < 1 or request.max_tokens > 10000):
                raise HTTPException(
                    status_code=400,
                    detail="max_tokens must be between 1 and 10,000"
                )
            
            if request.confidence_threshold and (request.confidence_threshold < 0 or request.confidence_threshold > 1):
                raise HTTPException(
                    status_code=400,
                    detail="confidence_threshold must be between 0 and 1"
                )
            
            if not orchestrator:
                raise HTTPException(
                    status_code=503, 
                    detail="Orchestrator not initialized. Service temporarily unavailable."
                )
            
            # Check cache first (non-blocking)
            cached_result = await get_cached_result(request.query, request.user_context)
            if cached_result:
                return QueryResponse(
                    query=request.query,
                    answer=cached_result.get('answer', 'Cached response'),
                    confidence=cached_result.get('confidence', 0.8),
                    citations=cached_result.get('citations', []),
                    execution_time=0.01,  # Cache hit is very fast
                    timestamp=datetime.now().isoformat(),
                    metadata={
                        'cached': True,
                        'request_id': request_id,
                        'user_id': current_user.user_id,
                        'concurrency_stats': get_concurrency_stats()
                    }
                )
            
            # Process query through orchestrator
            start_time = time.time()
            
            # Add token limits to the request
            user_context = request.user_context or {}
            user_context['max_tokens'] = request.max_tokens
            user_context['confidence_threshold'] = request.confidence_threshold
            user_context['user_id'] = current_user.user_id
            user_context['user_role'] = current_user.role
            
            result = await orchestrator.process_query(request.query, user_context)
            
            execution_time = time.time() - start_time
            
            # Update security check with actual response time
            await check_security(
                request.query, 
                client_ip, 
                current_user.user_id,
                execution_time
            )
            
            # Cache successful results
            if result.get('success', False):
                await cache_result(request.query, result, request.user_context)
            
            # Track analytics
            await track_query(
                request.query, 
                execution_time, 
                result.get('confidence', 0.0),
                client_ip,
                current_user.user_id
            )
            
            return QueryResponse(
                query=request.query,
                answer=result.get('answer', ''),
                confidence=result.get('confidence', 0.0),
                citations=result.get('citations', []),
                execution_time=execution_time,
                timestamp=datetime.now().isoformat(),
                metadata={
                    'success': result.get('success', False),
                    'error': result.get('error'),
                    'agents_used': result.get('metadata', {}).get('agents_used', []),
                    'request_id': request_id,
                    'user_id': current_user.user_id,
                    'user_role': current_user.role,
                    'concurrency_stats': get_concurrency_stats(),
                    'estimated_tokens': result.get('metadata', {}).get('estimated_tokens', 0),
                    'optimization_applied': result.get('metadata', {}).get('optimization_applied', False)
                }
            )
            
        except HTTPException:
            # Re-raise HTTP exceptions as-is
            raise
        except Exception as e:
            # Log error and return error response
            logger.error(f"Query processing error: {str(e)}")
            global error_counter
            error_counter += 1
            
            return QueryResponse(
                query=request.query,
                answer="",
                confidence=0.0,
                citations=[],
                execution_time=time.time() - start_time if 'start_time' in locals() else 0,
                timestamp=datetime.now().isoformat(),
                metadata={
                    'success': False,
                    'error': str(e),
                    'request_id': request_id,
                    'user_id': current_user.user_id if 'current_user' in locals() else None,
                    'concurrency_stats': get_concurrency_stats()
                }
            )
        finally:
            await untrack_request(request_id)

@app.get("/agents")
async def list_agents(current_user: User = Depends(require_read())):
    """List available agents and their status."""
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Orchestrator not initialized")
    
    agents_info = {}
    for agent_type, agent in orchestrator.agents.items():
        agents_info[agent_type.value] = {
            "status": "ready" if agent else "not_initialized",
            "type": agent_type.value,
            "description": f"{agent_type.value.title()} agent"
        }
    
    return {
        "agents": agents_info,
        "total_agents": len(agents_info),
        "user_id": current_user.user_id,
        "user_role": current_user.role
    }

@app.get("/metrics")
async def get_prometheus_metrics():
    """Get Prometheus metrics."""
    from api.metrics import get_metrics_collector
    from fastapi.responses import Response
    
    metrics = get_metrics_collector().get_metrics()
    return Response(content=metrics, media_type="text/plain")

@app.get("/metrics/json")
async def get_metrics_json(current_user: User = Depends(require_admin())):
    """Get metrics as JSON (admin only)."""
    metrics_data = get_metrics_collector().get_metrics_dict()
    metrics_data.update({
        "user_id": current_user.user_id,
        "user_role": current_user.role
    })
    return metrics_data

@app.get("/analytics")
async def get_analytics(current_user: User = Depends(require_admin())):
    """Get analytics data (admin only)."""
    analytics_data = await get_analytics_summary()
    analytics_data.update({
        "user_id": current_user.user_id,
        "user_role": current_user.role
    })
    return analytics_data

@app.get("/security")
async def get_security_info(current_user: User = Depends(require_admin())):
    """Get security information (admin only)."""
    security_data = get_security_summary()
    security_data.update({
        "user_id": current_user.user_id,
        "user_role": current_user.role
    })
    return security_data

@app.get("/cache/stats")
async def get_cache_statistics(current_user: User = Depends(require_admin())):
    """Get cache statistics (admin only)."""
    cache_stats = await get_cache_stats()
    cache_stats.update({
        "user_id": current_user.user_id,
        "user_role": current_user.role
    })
    return cache_stats

@app.get("/concurrency")
async def get_concurrency_info(current_user: User = Depends(require_admin())):
    """Get concurrency and performance information (admin only)."""
    return {
        "concurrency_stats": get_concurrency_stats(),
        "rate_limits": {
            "active_ips": len(rate_limit_store),
            "total_requests": sum(len(timestamps) for timestamps in rate_limit_store.values())
        },
        "performance": {
            "total_requests": request_counter,
            "total_errors": error_counter,
            "error_rate": error_counter / max(request_counter, 1)
        },
        "cache_stats": await get_cache_stats(),
        "security_stats": get_security_summary(),
        "user_id": current_user.user_id,
        "user_role": current_user.role
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 