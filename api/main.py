import asyncio
import logging
import time
import uuid
import psutil
from contextlib import asynccontextmanager
from typing import Any, Optional
from datetime import datetime, timedelta

import uvicorn
from fastapi import FastAPI, HTTPException, Request, Response, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import ValidationError, Field, BaseModel

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
from api.auth import get_current_user, login_user, register_user, generate_api_key, revoke_api_key, get_user_api_keys, require_read
from api.analytics import track_query
from api.cache import _query_cache
from agents.base_agent import QueryContext
from api.metrics import record_request_metrics, record_error_metrics, record_business_metrics, record_agent_metrics, record_cache_metrics, record_security_metrics, record_token_metrics
from api.rate_limiter import RateLimiter, RateLimitConfig, rate_limit_middleware, rate_limit
from api.exceptions import UKPHTTPException, AuthenticationError, AuthorizationError, RateLimitExceededError

# Add expert review models
class ExpertReviewRequest(BaseModel):
    review_id: str
    expert_id: str
    verdict: str  # "supported", "contradicted", "unclear"
    notes: str
    confidence: float = Field(ge=0.0, le=1.0)

class ExpertReviewResponse(BaseModel):
    review_id: str
    status: str
    expert_id: str
    verdict: str
    notes: str
    confidence: float
    completed_at: str

# Rate limiting configurations
QUERY_RATE_LIMIT = RateLimitConfig(
    requests_per_minute=60,
    requests_per_hour=1000,
    burst_size=10
)

AUTH_RATE_LIMIT = RateLimitConfig(
    requests_per_minute=10,
    requests_per_hour=100,
    burst_size=5
)

# Add authentication models
from pydantic import BaseModel
from typing import Optional

class LoginRequest(BaseModel):
    username: str
    password: str

class RegisterRequest(BaseModel):
    username: str
    password: str
    role: str = "user"

class AuthResponse(BaseModel):
    access_token: str
    token_type: str
    api_key: str
    user_id: str
    role: str
    permissions: list[str]

class APIKeyResponse(BaseModel):
    api_key: str
    user_id: str
    role: str
    permissions: list[str]
    description: str
    created_at: str

# Global variables
orchestrator = None
startup_time = None
app_version = "1.0.0"

# Request concurrency control
request_semaphore = asyncio.Semaphore(100)  # Limit to 100 concurrent requests

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
        
        rate_limiter = get_rate_limiter()
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
        rate_limiter = get_rate_limiter()
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

# Add rate limiting middleware (temporarily disabled due to Redis issues)
# app.middleware("http")(rate_limit_middleware)

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
        # Only try to get user for endpoints that require authentication
        if request.url.path not in ["/", "/health", "/metrics", "/analytics", "/integrations", "/query"]:
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

        # Record metrics
        try:
            from api.metrics import record_request_metrics
            record_request_metrics(
                method=request.method,
                endpoint=request.url.path,
                status_code=response.status_code,
                duration=process_time
            )
        except Exception as e:
            logger.warning(f"Failed to record metrics: {e}")

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


# Security middleware
@app.middleware("http")
async def security_check(request: Request, call_next):
    """Security middleware that checks requests for threats."""
    request_id = getattr(request.state, "request_id", "unknown")
    client_ip = request.client.host
    user_id = "anonymous"
    
    try:
        # Extract user ID if available
        if request.url.path not in ["/", "/health", "/metrics", "/analytics", "/integrations"]:
            current_user = await get_current_user(request)
            user_id = current_user.user_id
    except:
        pass

    # Extract query for security check (only for query endpoint)
    query = ""
    if request.url.path == "/query" and request.method == "POST":
        try:
            body = await request.body()
            if body:
                import json
                data = json.loads(body)
                query = data.get("query", "")
        except:
            pass

    # Perform security check
    try:
        from api.security import check_security
        security_result = await check_security(
            query=query,
            client_ip=client_ip,
            user_id=user_id,
            initial_confidence=0.0
        )
        
        # If blocked, return error response
        if security_result.get("blocked", False):
            logger.warning(
                f"Request blocked by security check",
                extra={
                    "request_id": request_id,
                    "user_id": user_id,
                    "client_ip": client_ip,
                    "threats": security_result.get("threats", [])
                }
            )
            
            # Record security metrics
            threats = security_result.get("threats", [])
            for threat in threats:
                record_security_metrics(
                    threat_type=threat.get("type", "unknown"),
                    severity=threat.get("severity", "medium"),
                    blocked=True
                )
            
            return JSONResponse(
                status_code=403,
                content={
                    "error": "Request blocked by security check",
                    "request_id": request_id,
                    "threats": security_result.get("threats", [])
                }
            )
        
        # Record security metrics for monitored threats
        threats = security_result.get("threats", [])
        if threats:
            for threat in threats:
                record_security_metrics(
                    threat_type=threat.get("type", "unknown"),
                    severity=threat.get("severity", "medium"),
                    blocked=False
                )
            
    except Exception as e:
        logger.error(
            f"Security check failed: {str(e)}",
            extra={
                "request_id": request_id,
                "user_id": user_id,
                "client_ip": client_ip,
                "error": str(e)
            }
        )
        # Continue processing on security check failure (fail open)
    
    # Continue with request processing
    response = await call_next(request)
    return response


# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler with structured logging and secure error messages."""
    request_id = getattr(request.state, "request_id", "unknown")

    # Handle custom UKP exceptions
    if isinstance(exc, UKPHTTPException):
        logger.error(
            f"UKP HTTP Exception: {exc.internal_message}",
            extra={
                "request_id": request_id,
                "status_code": exc.status_code,
                "exception_type": type(exc).__name__,
            }
        )
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": exc.detail,
                "request_id": request_id,
                "timestamp": time.time(),
            },
        )

    # Handle validation errors
    if isinstance(exc, ValidationError):
        logger.warning(
            f"Validation error: {str(exc)}",
            extra={
                "request_id": request_id,
                "exception_type": type(exc).__name__,
            }
        )
        return JSONResponse(
            status_code=422,
            content={
                "error": "Validation failed",
                "request_id": request_id,
                "timestamp": time.time(),
            },
        )

    # Handle authentication/authorization errors
    if isinstance(exc, (AuthenticationError, AuthorizationError)):
        logger.warning(
            f"Auth error: {str(exc)}",
            extra={
                "request_id": request_id,
                "exception_type": type(exc).__name__,
            }
        )
        return JSONResponse(
            status_code=exc.status_code if hasattr(exc, 'status_code') else 401,
            content={
                "error": "Authentication or authorization failed",
                "request_id": request_id,
                "timestamp": time.time(),
            },
        )

    # Handle rate limiting
    if isinstance(exc, RateLimitExceededError):
        logger.info(
            f"Rate limit exceeded",
            extra={
                "request_id": request_id,
                "exception_type": type(exc).__name__,
            }
        )
        return JSONResponse(
            status_code=429,
            content={
                "error": exc.detail,
                "request_id": request_id,
                "timestamp": time.time(),
            },
            headers=exc.headers or {}
        )

    # Handle unexpected exceptions securely
    logger.error(
        f"Unhandled exception: {str(exc)}",
        extra={
            "request_id": request_id,
            "exception_type": type(exc).__name__,
            "exception_message": str(exc),
        },
        exc_info=True,
    )

    # Return generic error message to avoid information leakage
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
    """Health check endpoint."""
    try:
        # Check basic system health
        uptime = time.time() - startup_time if startup_time else 0
        memory_usage = psutil.virtual_memory().percent
        cpu_usage = psutil.cpu_percent(interval=1)
        
        # Check cache health
        cache_status = "healthy"
        try:
            await _query_cache.get("health_check")
        except Exception as e:
            cache_status = f"unhealthy: {str(e)}"
        
        # Check orchestrator health
        orchestrator_status = "healthy"
        if orchestrator is None:
            orchestrator_status = "unhealthy: not initialized"
        
        return HealthResponse(
            status="healthy",
            version=app_version,
            timestamp=time.time(),
            uptime=float(uptime),
            components={
                "cache": cache_status,
                "orchestrator": orchestrator_status,
                "memory_usage_percent": str(memory_usage),
                "cpu_usage_percent": str(cpu_usage),
            },
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthResponse(
            status="unhealthy",
            version=app_version,
            timestamp=time.time(),
            uptime=0.0,
            components={"error": str(e)},
        )

@app.post("/auth/login", response_model=AuthResponse)
# @rate_limit(AUTH_RATE_LIMIT)  # Temporarily disabled
async def login(request: LoginRequest):
    """Login endpoint for user authentication."""
    try:
        result = await login_user(request.username, request.password)
        if not result:
            raise HTTPException(status_code=401, detail="Invalid credentials")
        
        logger.info(f"User logged in: {request.username}")
        return AuthResponse(**result)
    except Exception as e:
        logger.error(f"Login failed: {e}")
        raise HTTPException(status_code=500, detail="Login failed")

@app.post("/auth/register", response_model=AuthResponse)
# @rate_limit(AUTH_RATE_LIMIT)  # Temporarily disabled
async def register(request: RegisterRequest):
    """Register a new user."""
    try:
        api_key = await register_user(request.username, request.password, request.role)
        if not api_key:
            raise HTTPException(status_code=400, detail="User already exists")
        
        # Login the user after registration
        result = await login_user(request.username, request.password)
        
        logger.info(f"User registered: {request.username}")
        return AuthResponse(**result)
    except Exception as e:
        logger.error(f"Registration failed: {e}")
        raise HTTPException(status_code=500, detail="Registration failed")

@app.post("/auth/api-key", response_model=APIKeyResponse)
async def create_api_key(current_user=Depends(get_current_user)):
    """Create a new API key for the current user."""
    try:
        api_key = generate_api_key(
            current_user.user_id, 
            current_user.role, 
            current_user.permissions
        )
        
        key_info = API_KEY_REGISTRY[api_key]
        
        logger.info(f"API key created for user: {current_user.user_id}")
        return APIKeyResponse(
            api_key=api_key,
            user_id=key_info["user_id"],
            role=key_info["role"],
            permissions=key_info["permissions"],
            description=key_info["description"],
            created_at=key_info["created_at"].isoformat(),
        )
    except Exception as e:
        logger.error(f"API key creation failed: {e}")
        raise HTTPException(status_code=500, detail="API key creation failed")

@app.delete("/auth/api-key/{api_key}")
async def revoke_api_key_endpoint(api_key: str, current_user=Depends(get_current_user)):
    """Revoke an API key."""
    try:
        # Check if user owns this API key
        user_keys = get_user_api_keys(current_user.user_id)
        if api_key not in user_keys:
            raise HTTPException(status_code=403, detail="Not authorized to revoke this API key")
        
        success = revoke_api_key(api_key)
        if not success:
            raise HTTPException(status_code=404, detail="API key not found")
        
        logger.info(f"API key revoked for user: {current_user.user_id}")
        return {"message": "API key revoked successfully"}
    except Exception as e:
        logger.error(f"API key revocation failed: {e}")
        raise HTTPException(status_code=500, detail="API key revocation failed")

@app.get("/auth/api-keys", response_model=list[APIKeyResponse])
async def list_api_keys(current_user=Depends(get_current_user)):
    """List all API keys for the current user."""
    try:
        user_keys = get_user_api_keys(current_user.user_id)
        key_responses = []
        
        for key in user_keys:
            if key in API_KEY_REGISTRY:
                key_info = API_KEY_REGISTRY[key]
                key_responses.append(APIKeyResponse(
                    api_key=key,
                    user_id=key_info["user_id"],
                    role=key_info["role"],
                    permissions=key_info["permissions"],
                    description=key_info["description"],
                    created_at=key_info["created_at"].isoformat(),
                ))
        
        return key_responses
    except Exception as e:
        logger.error(f"API key listing failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to list API keys")

# Expert Review Endpoints
@app.get("/expert-reviews/pending", response_model=list[dict[str, Any]])
async def get_pending_reviews(current_user=Depends(get_current_user)):
    """Get list of pending expert reviews."""
    try:
        # Check if user has expert permissions
        if not current_user.has_permission("expert_review"):
            raise HTTPException(status_code=403, detail="Insufficient permissions for expert review")
        
        # Get fact check agent to access pending reviews
        from agents.factcheck_agent import FactCheckAgent
        factcheck_agent = FactCheckAgent()
        pending_reviews = await factcheck_agent.get_pending_reviews()
        
        return pending_reviews
    except Exception as e:
        logger.error(f"Failed to get pending reviews: {e}")
        raise HTTPException(status_code=500, detail="Failed to get pending reviews")

@app.post("/expert-reviews/{review_id}", response_model=ExpertReviewResponse)
async def submit_expert_review(
    review_id: str,
    review: ExpertReviewRequest,
    current_user=Depends(get_current_user)
):
    """Submit expert review decision."""
    try:
        # Check if user has expert permissions
        if not current_user.has_permission("expert_review"):
            raise HTTPException(status_code=403, detail="Insufficient permissions for expert review")
        
        # Validate verdict
        valid_verdicts = ["supported", "contradicted", "unclear"]
        if review.verdict not in valid_verdicts:
            raise HTTPException(status_code=400, detail=f"Invalid verdict. Must be one of: {valid_verdicts}")
        
        # Get fact check agent to update review
        from agents.factcheck_agent import FactCheckAgent
        factcheck_agent = FactCheckAgent()
        
        decision = {
            "expert_id": review.expert_id,
            "notes": review.notes,
            "verdict": review.verdict,
            "confidence": review.confidence
        }
        
        await factcheck_agent.update_review_decision(review_id, decision)
        
        logger.info(f"Expert review submitted for {review_id} by {current_user.user_id}")
        
        return ExpertReviewResponse(
            review_id=review_id,
            status="completed",
            expert_id=review.expert_id,
            verdict=review.verdict,
            notes=review.notes,
            confidence=review.confidence,
            completed_at=datetime.now().isoformat()
        )
    except Exception as e:
        logger.error(f"Failed to submit expert review: {e}")
        raise HTTPException(status_code=500, detail="Failed to submit expert review")

@app.get("/expert-reviews/{review_id}", response_model=Dict[str, Any])
async def get_review_details(
    review_id: str,
    current_user=Depends(get_current_user)
):
    """Get details of a specific expert review."""
    try:
        # Check if user has expert permissions
        if not current_user.has_permission("expert_review"):
            raise HTTPException(status_code=403, detail="Insufficient permissions for expert review")
        
        # Load review from file system
        import json
        import os
        
        review_dir = "data/manual_reviews"
        filepath = os.path.join(review_dir, f"{review_id}.json")
        
        if not os.path.exists(filepath):
            raise HTTPException(status_code=404, detail="Review not found")
        
        with open(filepath, 'r') as f:
            review = json.load(f)
        
        return review
    except Exception as e:
        logger.error(f"Failed to get review details: {e}")
        raise HTTPException(status_code=500, detail="Failed to get review details")


@app.post("/query", response_model=QueryResponse)
# @rate_limit(QUERY_RATE_LIMIT)  # Temporarily disabled
async def process_query(
    request: QueryRequestValidator, http_request: Request, current_user=None
):
    """Process a knowledge query through the multi-agent system."""
    request_id = getattr(http_request.state, "request_id", "unknown")
    
    # Create default user if none provided
    if current_user is None:
        from api.auth import User
        current_user = User(
            user_id="default_user",
            role="user",
            permissions=["read", "write"]
        )

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

    # SECURITY CHECK: Add security validation before processing
    try:
        from api.security import check_security
        security_result = await check_security(
            query=request.query,
            client_ip=http_request.client.host,
            user_id=current_user.user_id,
            initial_confidence=0.0
        )
        
        # Check if request should be blocked
        if security_result.get('status') == 'threat_detected' or security_result.get('status') == 'blocked':
            logger.warning(
                f"Query blocked by security check",
                extra={
                    "request_id": request_id,
                    "user_id": current_user.user_id,
                    "client_ip": http_request.client.host,
                    "threats": security_result.get("threats", [])
                }
            )
            
            # Record security metrics for blocked request
            threats = security_result.get("threats", [])
            for threat in threats:
                record_security_metrics(
                    threat_type=threat.get("type", "unknown"),
                    severity=threat.get("severity", "medium"),
                    blocked=True
                )
                
            # Also record for any block_reason
            if security_result.get("block_reason"):
                record_security_metrics(
                    threat_type="ip_blocked",
                    severity="high",
                    blocked=True
                )
            
            raise HTTPException(
                status_code=403,
                detail=f"Request blocked by security check: {security_result.get('block_reason', 'Security violation detected')}"
            )
        
        # Record security metrics for monitored (but not blocked) threats
        threats = security_result.get("threats", [])
        for threat in threats:
            record_security_metrics(
                threat_type=threat.get("type", "unknown"),
                severity=threat.get("severity", "low"),
                blocked=False
            )
            
    except HTTPException:
        # Re-raise HTTPException (security blocks)
        raise
    except Exception as e:
        logger.error(
            f"Security check failed: {e}",
            extra={
                "request_id": request_id,
                "user_id": current_user.user_id,
                "error": str(e)
            },
            exc_info=True,
        )
        # Continue processing if security check fails (fail-open for availability)

    # Acquire semaphore to limit concurrent requests
    async with request_semaphore:
        logger.info(
            f"Acquired semaphore for query processing",
            extra={
                "request_id": request_id,
                "user_id": current_user.user_id,
                "semaphore_available": request_semaphore._value,
            },
        )

        try:
            # Check cache first
            cache_key = f"{current_user.user_id}:{request.query}"
            cached_result = await _query_cache.get(cache_key)

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

                # Record comprehensive metrics
                execution_time = time.time() - start_time
                record_request_metrics("POST", "/query", 200, execution_time)
                record_cache_metrics("query", True, None)
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
                        "execution_time_ms": int(execution_time * 1000),
                    },
                )

            # FIXED: Create user_context properly and call orchestrator with string query
            user_context = request.user_context or {}
            if request.max_tokens:
                user_context["max_tokens"] = request.max_tokens
            if request.confidence_threshold:
                user_context["confidence_threshold"] = request.confidence_threshold
            user_context["trace_id"] = request_id
            user_context["user_id"] = current_user.user_id

            # Process query through orchestrator - FIXED: pass string and dict instead of QueryContext
            result = await orchestrator.process_query(request.query, user_context)
            process_time = time.time() - start_time

            # Check if processing was successful
            if not result.get('success', True):
                error_msg = result.get('error', 'Processing failed')
                logger.error(
                    f"Query processing failed: {error_msg}",
                    extra={
                        "request_id": request_id,
                        "user_id": current_user.user_id,
                        "error": error_msg
                    },
                    exc_info=True,
                )
                raise HTTPException(status_code=500, detail=error_msg)

            # Record comprehensive metrics for orchestrator processing
            record_request_metrics("POST", "/query", 200, process_time)
            record_cache_metrics("query", False, None)
            
            # Record agent metrics if available
            agent_results = result.get("metadata", {}).get("agent_results", {})
            if agent_results:
                for agent_type, agent_result in agent_results.items():
                    if isinstance(agent_result, dict):
                        record_agent_metrics(
                            agent_type=str(agent_type),
                            status="success" if agent_result.get("success", False) else "error",
                            duration=agent_result.get("execution_time_ms", 0) / 1000.0
                        )
            
            # Record token usage if available
            token_usage = result.get("metadata", {}).get("token_usage", {})
            if token_usage:
                for agent_type, tokens in token_usage.items():
                    if isinstance(tokens, dict):
                        record_token_metrics(
                            agent_type=str(agent_type),
                            prompt_tokens=tokens.get("prompt_tokens", 0),
                            completion_tokens=tokens.get("completion_tokens", 0)
                        )
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
            await _query_cache.set(cache_key, result)

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
        
        # Store feedback using the new feedback storage system
        try:
            from api.feedback_storage import get_feedback_storage, FeedbackRequest, FeedbackType, FeedbackPriority
            
            # Create feedback request
            feedback_request = FeedbackRequest(
                query_id=feedback.query_id,
                user_id=current_user.user_id,
                feedback_type=FeedbackType(feedback.feedback_type),
                details=feedback.details,
                priority=FeedbackPriority.MEDIUM,
                metadata={
                    "request_id": request_id,
                    "timestamp": datetime.now().isoformat(),
                    "source": "api"
                }
            )
            
            # Store feedback
            feedback_storage = get_feedback_storage()
            stored_feedback = await feedback_storage.store_feedback(feedback_request)
            
            logger.info(f"Feedback stored successfully: {stored_feedback.id}")
            
        except Exception as e:
            logger.error(f"Failed to store feedback: {e}")
            # Don't fail the request if feedback storage fails
        
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
async def get_metrics(current_user=Depends(get_current_user)):
    """Get Prometheus metrics (admin only)."""
    # Check admin permissions
    if not current_user.has_permission("admin"):
        raise AuthorizationError("Admin permission required for metrics endpoint")
    
    # Additional security check for production
    if os.getenv("ENVIRONMENT") == "production" and not os.getenv("ENABLE_METRICS_ENDPOINT", "").lower() == "true":
        raise AuthorizationError("Metrics endpoint disabled in production")
    
    try:
        from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
        return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
    except Exception as e:
        logger.error(f"Failed to generate metrics: {e}")
        raise HTTPException(status_code=500, detail="Metrics generation failed")

@app.get("/analytics", response_model=Dict[str, Any])
async def get_analytics(current_user=Depends(require_read())):
    """Get analytics data (any authenticated user)."""
    # TODO: Restrict to admin in future when proper user accounts exist
    try:
        analytics_data = await _analytics_collector.get_summary()
        # Remove sensitive data
        safe_analytics = {
            "total_requests": analytics_data.get("total_requests", 0),
            "total_errors": analytics_data.get("total_errors", 0),
            "average_response_time": analytics_data.get("average_response_time", 0),
            "cache_hit_rate": analytics_data.get("cache_hit_rate", 0),
            "popular_queries": analytics_data.get("popular_query_categories", {}),
            "timestamp": datetime.now().isoformat()
        }
        return safe_analytics
    except Exception as e:
        logger.error(f"Failed to get analytics: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Analytics retrieval failed")

@app.get("/security", response_model=Dict[str, Any])
async def get_security_status(current_user=Depends(get_current_user)):
    """Get security status (admin only)."""
    # Check admin permissions
    if not current_user.has_permission("admin"):
        raise AuthorizationError("Admin permission required for security endpoint")
    
    # Additional security check for production
    if os.getenv("ENVIRONMENT") == "production" and not os.getenv("ENABLE_SECURITY_ENDPOINT", "").lower() == "true":
        raise AuthorizationError("Security endpoint disabled in production")
    
    try:
        from api.security import get_security_summary
        security_summary = get_security_summary()
        # Remove sensitive security details
        safe_summary = {
            "status": security_summary.get("status", "unknown"),
            "threats_detected_today": security_summary.get("threat_stats", {}).get("daily_count", 0),
            "requests_blocked_today": security_summary.get("threat_stats", {}).get("blocked_today", 0),
            "security_level": security_summary.get("security_level", "normal"),
            "timestamp": datetime.now().isoformat()
        }
        return safe_summary
    except Exception as e:
        logger.error(f"Failed to get security status: {e}")
        raise HTTPException(status_code=500, detail="Security status retrieval failed")


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
