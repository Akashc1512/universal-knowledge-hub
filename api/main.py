from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import asyncio
import logging
from datetime import datetime

# Import your existing components
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.lead_orchestrator import LeadOrchestrator
from core.types import QueryContext

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
        orchestrator = LeadOrchestrator()
        logger.info("✅ Universal Knowledge Platform initialized successfully")
    except Exception as e:
        logger.error(f"❌ Failed to initialize orchestrator: {e}")
        raise

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
            "retrieval": "ready",
            "factcheck": "ready", 
            "synthesis": "ready",
            "citation": "ready",
            "orchestrator": "ready"
        }
    )

@app.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    """Process a knowledge query through the multi-agent pipeline."""
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    start_time = datetime.now()
    
    try:
        # Create query context
        context = QueryContext(
            query=request.query,
            user_context=request.user_context or {},
            max_tokens=request.max_tokens,
            confidence_threshold=request.confidence_threshold
        )
        
        # Process query through orchestrator
        result = await orchestrator.process_query(request.query, context)
        
        execution_time = (datetime.now() - start_time).total_seconds()
        
        return QueryResponse(
            query=request.query,
            answer=result.get('response', 'No answer generated'),
            confidence=result.get('confidence', 0.0),
            citations=result.get('citations', []),
            execution_time=execution_time,
            timestamp=datetime.now(),
            metadata={
                'verified_claims': result.get('verified_claims', 0),
                'sources_consulted': result.get('sources_consulted', 0),
                'agent_pipeline': result.get('agent_pipeline', [])
            }
        )
        
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")

@app.get("/agents")
async def list_agents():
    """List available agents and their capabilities."""
    return {
        "agents": {
            "retrieval": {
                "description": "Hybrid search combining vector, keyword, and graph queries",
                "capabilities": ["semantic_search", "keyword_search", "knowledge_graph"]
            },
            "factcheck": {
                "description": "Claim verification and cross-source validation",
                "capabilities": ["claim_decomposition", "source_verification", "confidence_scoring"]
            },
            "synthesis": {
                "description": "Answer construction from verified facts",
                "capabilities": ["answer_generation", "confidence_assessment", "context_integration"]
            },
            "citation": {
                "description": "Multi-format citation generation",
                "capabilities": ["apa_citations", "mla_citations", "source_tracking"]
            },
            "orchestrator": {
                "description": "Multi-agent coordination and pipeline management",
                "capabilities": ["agent_orchestration", "execution_patterns", "result_aggregation"]
            }
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 