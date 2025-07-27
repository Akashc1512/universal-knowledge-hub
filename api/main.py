import asyncio
import logging
import time
from contextlib import asynccontextmanager
from typing import Dict, Any, Optional
from datetime import datetime

from fastapi import FastAPI, HTTPException, Depends, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from dotenv import load_dotenv
import os

# Import our modules
from api.auth import require_read, require_admin, User
from api.security import check_security, get_security_summary
from api.cache import QueryCache, SemanticCache
from api.analytics import track_query, get_analytics_summary
from api.metrics import (
    request_counter, request_duration_seconds, error_counter,
    active_connections_gauge, system_memory_bytes_gauge, system_cpu_percent_gauge
)
from agents.lead_orchestrator import LeadOrchestrator
from agents.base_agent import QueryContext

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables
orchestrator: Optional[LeadOrchestrator] = None
request_semaphore = asyncio.Semaphore(100)  # Limit concurrent requests

# Pydantic models
class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=10000, description="The user's question")
    max_tokens: Optional[int] = Field(1000, ge=100, le=4000, description="Maximum tokens for response")
    confidence_threshold: Optional[float] = Field(0.8, ge=0.0, le=1.0, description="Minimum confidence score")
    user_context: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional user context")

class QueryResponse(BaseModel):
    answer: str = Field(..., description="The AI-generated answer")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    citations: list = Field(default_factory=list, description="Source citations")
    query_id: str = Field(..., description="Unique query identifier")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

class HealthResponse(BaseModel):
    status: str = Field(..., description="Service status")
    timestamp: str = Field(..., description="Current timestamp")
    version: str = Field(..., description="API version")
    agents: Dict[str, bool] = Field(..., description="Agent status")

class ConcurrencyInfo(BaseModel):
    active_requests: int = Field(..., description="Number of active requests")
    max_concurrent: int = Field(..., description="Maximum concurrent requests")
    semaphore_value: int = Field(..., description="Current semaphore value")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    global orchestrator
    
    # Startup
    logger.info("🚀 Starting SarvanOM - Your Own Knowledge Hub Powered by AI")
    
    try:
        orchestrator = LeadOrchestrator()
        logger.info("✅ SarvanOM initialized successfully")
    except Exception as e:
        logger.error(f"❌ Failed to initialize SarvanOM: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down SarvanOM")

# Create FastAPI app
app = FastAPI(
    title="SarvanOM API",
    description="Your Own Knowledge Hub Powered by AI - Advanced AI-powered knowledge platform with multi-agent orchestration",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*"]  # Configure appropriately for production
)

# Request tracking
active_requests = 0

async def track_request():
    """Track active requests"""
    global active_requests
    active_requests += 1
    active_connections_gauge.set(active_requests)

async def untrack_request():
    """Untrack active requests"""
    global active_requests
    active_requests -= 1
    active_connections_gauge.set(active_requests)

# Middleware for request timing
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response

# Middleware for request logging
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    
    # Log request details
    logger.info(f"📥 {request.method} {request.url.path} from {request.client.host}")
    
    response = await call_next(request)
    
    # Log response details
    process_time = time.time() - start_time
    logger.info(f"📤 {request.method} {request.url.path} -> {response.status_code} ({process_time:.3f}s)")
    
    return response

# Error handlers
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    global error_counter
    error_counter += 1
    
    logger.error(f"❌ Unhandled exception: {exc}")
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": "An unexpected error occurred",
            "timestamp": datetime.utcnow().isoformat()
        }
    )

# Root endpoint
@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with service information"""
    return {
        "name": "SarvanOM",
        "tagline": "Your Own Knowledge Hub Powered by AI",
        "version": "1.0.0",
        "status": "operational",
        "website": "https://sarvanom.ai",
        "documentation": "https://docs.sarvanom.ai",
        "api_docs": "/docs"
    }

# Health check endpoint
@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint"""
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Orchestrator not initialized")
    
    # Check agent status
    agents_status = {
        "retrieval_agent": orchestrator.retrieval_agent is not None,
        "factcheck_agent": orchestrator.factcheck_agent is not None,
        "synthesis_agent": orchestrator.synthesis_agent is not None,
        "citation_agent": orchestrator.citation_agent is not None
    }
    
    return HealthResponse(
        status="healthy" if all(agents_status.values()) else "degraded",
        timestamp=datetime.utcnow().isoformat(),
        version="1.0.0",
        agents=agents_status
    )

# Main query endpoint
@app.post("/query", response_model=QueryResponse, tags=["Query"])
async def process_query(
    request: QueryRequest,
    current_user: User = Depends(require_read())
):
    """Process a knowledge query using the multi-agent system"""
    global orchestrator, error_counter
    
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Orchestrator not initialized")
    
    # Track request
    await track_request()
    
    try:
        # Security check
        security_result = await check_security(request.query, request.client.host)
        if security_result.threat_detected:
            raise HTTPException(
                status_code=403,
                detail=f"Security threat detected: {security_result.threat_type}"
            )
        
        # Check cache first
        cache_key = f"{request.query}:{request.max_tokens}:{request.confidence_threshold}"
        cached_result = QueryCache.get(cache_key)
        
        if cached_result:
            logger.info(f"🎯 Cache hit for query: {request.query[:50]}...")
            return QueryResponse(**cached_result)
        
        # Process with orchestrator
        start_time = time.time()
        
        # Create query context
        query_context = QueryContext(
            query=request.query,
            max_tokens=request.max_tokens,
            confidence_threshold=request.confidence_threshold,
            user_id=current_user.user_id,
            user_role=current_user.role
        )
        
        # Process query
        result = await orchestrator.process_query(query_context)
        
        process_time = time.time() - start_time
        request_duration_seconds.observe(process_time)
        
        # Track analytics
        await track_query(
            query=request.query,
            user_id=current_user.user_id,
            response_time=process_time,
            success=result.get('success', True)
        )
        
        # Increment request counter
        request_counter.inc()
        
        # Cache successful results
        if result.get('success', True):
            cache_data = {
                'answer': result['answer'],
                'confidence': result['confidence'],
                'citations': result['citations'],
                'metadata': result.get('metadata', {})
            }
            QueryCache.put(cache_key, cache_data)
        
        return QueryResponse(
            answer=result['answer'],
            confidence=result['confidence'],
            citations=result['citations'],
            query_id=result.get('query_id', f"query_{int(time.time())}"),
            metadata=result.get('metadata', {})
        )
        
    except HTTPException:
        raise
    except Exception as e:
        global error_counter
        error_counter += 1
        
        logger.error(f"❌ Query processing failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Query processing failed: {str(e)}"
        )
    finally:
        await untrack_request()

# Agents endpoint
@app.get("/agents", tags=["Agents"])
async def list_agents(current_user: User = Depends(require_read())):
    """List available AI agents and their status"""
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Orchestrator not initialized")
    
    return {
        "agents": {
            "retrieval_agent": {
                "name": "Retrieval Agent",
                "description": "Intelligent document search and retrieval",
                "status": "active" if orchestrator.retrieval_agent else "inactive"
            },
            "factcheck_agent": {
                "name": "Fact-Check Agent", 
                "description": "Verification of claims and information accuracy",
                "status": "active" if orchestrator.factcheck_agent else "inactive"
            },
            "synthesis_agent": {
                "name": "Synthesis Agent",
                "description": "AI-powered answer generation and synthesis", 
                "status": "active" if orchestrator.synthesis_agent else "inactive"
            },
            "citation_agent": {
                "name": "Citation Agent",
                "description": "Automatic source citation and attribution",
                "status": "active" if orchestrator.citation_agent else "inactive"
            }
        },
        "user_id": current_user.user_id,
        "timestamp": datetime.utcnow().isoformat()
    }

# Metrics endpoint
@app.get("/metrics", tags=["Monitoring"])
async def get_metrics():
    """Get Prometheus metrics"""
    from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
    
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )

# Analytics endpoint (admin only)
@app.get("/analytics", tags=["Analytics"])
async def get_analytics(current_user: User = Depends(require_admin())):
    """Get analytics summary (admin only)"""
    return await get_analytics_summary()

# Security endpoint (admin only)
@app.get("/security", tags=["Security"])
async def get_security_info(current_user: User = Depends(require_admin())):
    """Get security summary (admin only)"""
    return await get_security_summary()

# Cache stats endpoint (admin only)
@app.get("/cache/stats", tags=["Cache"])
async def get_cache_stats(current_user: User = Depends(require_admin())):
    """Get cache statistics (admin only)"""
    return {
        "query_cache": {
            "size": len(QueryCache.cache),
            "hits": QueryCache.hits,
            "misses": QueryCache.misses
        },
        "semantic_cache": {
            "size": len(SemanticCache.cache),
            "hits": SemanticCache.hits,
            "misses": SemanticCache.misses
        }
    }

# Concurrency info endpoint
@app.get("/concurrency", response_model=ConcurrencyInfo, tags=["Monitoring"])
async def get_concurrency_info():
    """Get concurrency information"""
    return ConcurrencyInfo(
        active_requests=active_requests,
        max_concurrent=100,
        semaphore_value=request_semaphore._value
    )

# Import Response for metrics
from fastapi.responses import Response 