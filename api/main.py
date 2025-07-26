from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import asyncio
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Universal Knowledge Platform API",
    description="AI-driven knowledge hub with multi-agent orchestration",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for API
class QueryRequest(BaseModel):
    query: str
    user_context: Optional[Dict[str, Any]] = None
    max_tokens: Optional[int] = 1000
    confidence_threshold: Optional[float] = 0.7

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

@app.on_event("startup")
async def startup_event():
    """Initialize the orchestrator on startup."""
    global orchestrator
    try:
        # Import here to avoid circular imports
        import sys
        import os
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        
        from agents.lead_orchestrator import LeadOrchestrator
        orchestrator = LeadOrchestrator()
        logger.info("✅ Universal Knowledge Platform initialized successfully")
    except Exception as e:
        logger.error(f"❌ Failed to initialize orchestrator: {e}")
        # Don't raise - allow API to start without orchestrator for testing
        logger.info("⚠️  Starting API without orchestrator for testing")

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
async def process_query(request: QueryRequest):
    """Process a knowledge query through the multi-agent pipeline."""
    if not orchestrator:
        raise HTTPException(
            status_code=503, 
            detail="Orchestrator not initialized. Service temporarily unavailable."
        )
    
    try:
        # Create query context
        from core.types import QueryContext
        context = QueryContext(
            query=request.query,
            user_context=request.user_context or {},
            max_tokens=request.max_tokens,
            confidence_threshold=request.confidence_threshold
        )
        
        # Process query through orchestrator
        start_time = datetime.now()
        result = await orchestrator.process_query(context)
        execution_time = (datetime.now() - start_time).total_seconds()
        
        return QueryResponse(
            query=request.query,
            answer=result.answer,
            confidence=result.confidence,
            citations=result.citations,
            execution_time=execution_time,
            timestamp=datetime.now(),
            metadata={
                "agents_used": result.metadata.get("agents_used", []),
                "tokens_used": result.metadata.get("tokens_used", 0),
                "cache_hit": result.metadata.get("cache_hit", False)
            }
        )
        
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
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
    """Get system metrics and performance data."""
    return {
        "uptime": "running",
        "requests_processed": 0,  # TODO: Implement metrics collection
        "average_response_time": 0.0,
        "cache_hit_rate": 0.0,
        "active_agents": 5 if orchestrator else 0
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 