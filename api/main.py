from fastapi import FastAPI, HTTPException, Request, Response
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

# Import advanced features
from api.cache import get_cached_result, cache_result, get_cache_stats
from api.analytics import track_query, get_analytics_summary
from api.security import check_security, get_security_summary

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global request tracking
request_counter = 0
error_counter = 0

# Rate limiting storage
rate_limit_store = defaultdict(list)

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

class ErrorResponse(BaseModel):
    """Standard error response format."""
    error: str
    detail: Optional[str] = None
    timestamp: datetime
    request_id: str

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for FastAPI app."""
    # Startup
    logger.info("🚀 Starting Universal Knowledge Platform")
    
    # Initialize core components
    try:
        # Initialize orchestrator
        global orchestrator
        import sys
        import os
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        
        from agents.lead_orchestrator import LeadOrchestrator
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

@app.exception_handler(ValidationError)
async def validation_exception_handler(request: Request, exc: ValidationError):
    """Handle Pydantic validation errors."""
    global error_counter
    error_counter += 1
    
    return JSONResponse(
        status_code=422,
        content=ErrorResponse(
            error="Validation Error",
            detail=str(exc),
            timestamp=datetime.now(),
            request_id=str(request_counter)
        ).dict()
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle all unhandled exceptions."""
    global error_counter
    error_counter += 1
    
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal Server Error",
            detail="An unexpected error occurred",
            timestamp=datetime.now().isoformat(),
            request_id=str(request_counter)
        ).dict()
    )

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """Add processing time header to all responses."""
    global request_counter
    request_counter += 1
    
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    
    response.headers["X-Process-Time"] = str(process_time)
    response.headers["X-Request-ID"] = str(request_counter)
    
    return response

@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all requests for monitoring."""
    start_time = time.time()
    
    # Log request
    logger.info(f"Request: {request.method} {request.url.path} from {request.client.host}")
    
    response = await call_next(request)
    
    # Log response
    process_time = time.time() - start_time
    logger.info(f"Response: {response.status_code} took {process_time:.3f}s")
    
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
    timestamp: datetime
    metadata: Dict[str, Any]

class HealthResponse(BaseModel):
    status: str
    version: str
    timestamp: datetime
    agents_status: Dict[str, str]

# Global orchestrator instance
orchestrator = None

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
        timestamp=datetime.now(),
        agents_status={
            "retrieval": "ready" if orchestrator else "not_initialized",
            "factcheck": "ready" if orchestrator else "not_initialized", 
            "synthesis": "ready" if orchestrator else "not_initialized",
            "citation": "ready" if orchestrator else "not_initialized",
            "orchestrator": "ready" if orchestrator else "not_initialized"
        }
    )

@app.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest, http_request: Request):
    """Process a knowledge query through the multi-agent pipeline."""
    # Rate limiting
    client_ip = get_client_ip(http_request)
    if not check_rate_limit(client_ip):
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded. Please try again later."
        )
    
    # Security check
    security_result = await check_security(
        request.query, 
        client_ip, 
        request.user_context.get("user_id") if request.user_context else None,
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
    
    try:
        # Check cache first
        cached_result = await get_cached_result(request.query, request.user_context)
        if cached_result:
            return QueryResponse(
                query=request.query,
                answer=cached_result.get('response', 'Cached response'),
                confidence=cached_result.get('confidence', 0.0),
                citations=cached_result.get('citations', []),
                execution_time=0.1,  # Fast cache response
                timestamp=datetime.now(),
                metadata={
                    "cache_hit": True,
                    "agents_used": cached_result.get('agent_results', {}).keys(),
                    "tokens_used": cached_result.get('token_usage', {}).get('total', 0),
                    "client_ip": client_ip
                }
            )
        
        # Create query context with correct parameters
        from agents.base_agent import QueryContext
        context = QueryContext(
            query=request.query.strip(),
            user_id=request.user_context.get("user_id") if request.user_context else None,
            token_budget=request.max_tokens or 1000,
            metadata={
                "confidence_threshold": request.confidence_threshold or 0.7,
                "max_tokens": request.max_tokens or 1000,
                "user_context": request.user_context or {}
            }
        )
        
        # Process query through orchestrator with timeout
        start_time = datetime.now()
        try:
            result = await asyncio.wait_for(
                orchestrator.process_query(request.query, request.user_context),
                timeout=30.0  # 30 second timeout
            )
        except asyncio.TimeoutError:
            raise HTTPException(
                status_code=408,
                detail="Request timeout. Please try a simpler query."
            )
        
        execution_time = (datetime.now() - start_time).total_seconds()
        
        # Validate result
        if not result:
            raise HTTPException(
                status_code=500,
                detail="No response generated from orchestrator"
            )
        
        # Cache the result
        await cache_result(request.query, result, request.user_context)
        
        # Track analytics
        await track_query(
            query=request.query,
            user_id=request.user_context.get("user_id") if request.user_context else None,
            execution_time=execution_time,
            response_size=len(str(result)),
            confidence=result.get('confidence', 0.0),
            cache_hit=False,
            error_occurred=False,
            error_type=None,
            agent_usage=result.get('agent_results', {}),
            token_usage=result.get('token_usage', {}),
            user_agent=http_request.headers.get("User-Agent"),
            ip_address=client_ip
        )
        
        # Handle result format (result is a dict, not an object)
        return QueryResponse(
            query=request.query,
            answer=result.get('response', 'No response generated'),
            confidence=result.get('confidence', 0.0),
            citations=result.get('citations', []),
            execution_time=execution_time,
            timestamp=datetime.now(),
            metadata={
                "agents_used": list(result.get('agent_results', {}).keys()),
                "tokens_used": result.get('token_usage', {}).get('total', 0),
                "cache_hit": False,
                "execution_pattern": result.get('execution_pattern', 'unknown'),
                "client_ip": client_ip,
                "security_monitored": security_result.get('monitored', False)
            }
        )
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Error processing query: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Internal server error. Please try again later."
        )

@app.get("/agents")
async def list_agents():
    """List available agents and their capabilities."""
    return {
        "agents": {
            "retrieval": {
                "name": "RetrievalAgent",
                "description": "Hybrid search combining semantic and keyword matching",
                "capabilities": ["semantic_search", "keyword_search", "hybrid_ranking"]
            },
            "factcheck": {
                "name": "FactCheckAgent", 
                "description": "Claim verification and fact-checking",
                "capabilities": ["claim_verification", "source_validation", "confidence_scoring"]
            },
            "synthesis": {
                "name": "SynthesisAgent",
                "description": "Answer generation and content synthesis",
                "capabilities": ["answer_generation", "content_synthesis", "confidence_assessment"]
            },
            "citation": {
                "name": "CitationAgent",
                "description": "Multi-format citation generation",
                "capabilities": ["citation_generation", "format_conversion", "source_tracking"]
            },
            "orchestrator": {
                "name": "LeadOrchestrator",
                "description": "Multi-agent coordination and workflow management",
                "capabilities": ["workflow_orchestration", "agent_coordination", "result_aggregation"]
            }
        },
        "total_agents": 5,
        "status": "operational" if orchestrator else "initializing"
    }

@app.get("/metrics")
async def get_metrics():
    """Get comprehensive system metrics."""
    return {
        "requests_processed": request_counter,
        "errors_encountered": error_counter,
        "average_response_time": 0.0,  # Will be calculated from analytics
        "cache_hit_rate": 0.0,  # Will be calculated from cache stats
        "active_agents": 5 if orchestrator else 0,
        "system_health": "healthy" if orchestrator else "degraded"
    }

@app.get("/analytics")
async def get_analytics():
    """Get comprehensive analytics data."""
    return get_analytics_summary()

@app.get("/security")
async def get_security_info():
    """Get security information and statistics."""
    return get_security_summary()

@app.get("/cache/stats")
async def get_cache_statistics():
    """Get cache statistics."""
    return get_cache_stats()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 