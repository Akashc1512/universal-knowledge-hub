"""
Fixes for LeadOrchestrator Pipeline Issues

This module contains patches and improvements to fix:
1. Missing error handling for agent failures
2. Proper None/empty result handling
3. Cascading failure prevention
4. Improved data flow between agents
"""

import logging
from typing import Dict, List, Any, Optional
from agents.base_agent import AgentResult, AgentType, QueryContext

logger = logging.getLogger(__name__)


class OrchestrationError(Exception):
    """Custom exception for orchestration failures."""
    pass


def create_safe_agent_result(
    success: bool = False,
    data: Optional[Dict[str, Any]] = None,
    error: Optional[str] = None,
    confidence: float = 0.0
) -> AgentResult:
    """
    Create a safe AgentResult with proper defaults.
    
    Args:
        success: Whether the operation succeeded
        data: Result data (will be made safe)
        error: Error message if failed
        confidence: Confidence score
        
    Returns:
        Safe AgentResult that won't cause downstream issues
    """
    safe_data = data or {}
    
    # Ensure required fields exist with safe defaults
    if "documents" not in safe_data and not success:
        safe_data["documents"] = []
    if "verified_facts" not in safe_data and not success:
        safe_data["verified_facts"] = []
    if "answer" not in safe_data and not success:
        safe_data["answer"] = ""
        
    return AgentResult(
        success=success,
        data=safe_data,
        error=error,
        confidence=confidence
    )


def safe_prepare_synthesis_input(
    results: Dict[AgentType, AgentResult], 
    context: QueryContext
) -> Dict[str, Any]:
    """
    Safely prepare input for synthesis agent with proper error handling.
    
    Args:
        results: Results from previous agents
        context: Query context
        
    Returns:
        Safe synthesis input that won't cause failures
    """
    # Extract verified facts safely
    verified_facts = []
    
    # Try to get facts from fact checker
    if AgentType.FACT_CHECK in results:
        fact_result = results[AgentType.FACT_CHECK]
        if fact_result and fact_result.success:
            verified_facts = fact_result.data.get("verified_facts", [])
    
    # If no facts from fact checker, try to get documents from retrieval
    if not verified_facts and AgentType.RETRIEVAL in results:
        retrieval_result = results[AgentType.RETRIEVAL]
        if retrieval_result and retrieval_result.success:
            documents = retrieval_result.data.get("documents", [])
            # Convert documents to facts format
            verified_facts = [
                {
                    "content": doc.get("content", ""),
                    "source": doc.get("source", "Unknown"),
                    "confidence": doc.get("score", 0.5),
                    "verified": False  # Not fact-checked
                }
                for doc in documents[:10]  # Limit to top 10
            ]
    
    # Prepare synthesis parameters
    synthesis_params = {
        "style": "concise",
        "max_length": 1000,
        "fallback_mode": not verified_facts  # Flag for synthesis to use fallback
    }
    
    # Add any prepared synthesis data
    for key, result in results.items():
        if key == "synthesis_prep" and result and result.success:
            synthesis_params["prepared_data"] = result.data
            break
    
    return {
        "verified_facts": verified_facts,
        "query": context.query,
        "synthesis_params": synthesis_params,
        "has_facts": bool(verified_facts),
        "fallback_response": f"I couldn't find specific information about '{context.query}'. Please try rephrasing your question or providing more context."
    }


def handle_agent_failure(
    agent_type: AgentType,
    error: Exception,
    context: QueryContext,
    previous_results: Dict[AgentType, AgentResult]
) -> AgentResult:
    """
    Handle agent failure gracefully with appropriate fallback.
    
    Args:
        agent_type: Type of agent that failed
        error: The exception that occurred
        context: Query context
        previous_results: Results from previous agents
        
    Returns:
        Safe fallback result
    """
    logger.error(f"{agent_type.value} agent failed: {str(error)}")
    
    # Create agent-specific fallback responses
    if agent_type == AgentType.RETRIEVAL:
        return create_safe_agent_result(
            success=False,
            data={
                "documents": [],
                "search_performed": False,
                "fallback": True
            },
            error=f"Retrieval failed: {str(error)}",
            confidence=0.0
        )
    
    elif agent_type == AgentType.FACT_CHECK:
        # If fact check fails, pass through retrieval results
        retrieval_data = previous_results.get(AgentType.RETRIEVAL, {})
        documents = retrieval_data.data.get("documents", []) if retrieval_data else []
        
        return create_safe_agent_result(
            success=False,
            data={
                "verified_facts": [
                    {
                        "content": doc.get("content", ""),
                        "source": doc.get("source", "Unknown"),
                        "confidence": 0.5,  # Lower confidence since not verified
                        "verified": False
                    }
                    for doc in documents[:5]
                ],
                "skipped_verification": True
            },
            error=f"Fact checking skipped: {str(error)}",
            confidence=0.3
        )
    
    elif agent_type == AgentType.SYNTHESIS:
        # Create a basic response from available data
        facts = []
        if AgentType.FACT_CHECK in previous_results:
            facts = previous_results[AgentType.FACT_CHECK].data.get("verified_facts", [])
        elif AgentType.RETRIEVAL in previous_results:
            docs = previous_results[AgentType.RETRIEVAL].data.get("documents", [])
            facts = [{"content": d.get("content", "")} for d in docs[:3]]
        
        if facts:
            # Simple concatenation of facts
            answer = "Based on available information:\n\n"
            for i, fact in enumerate(facts[:3], 1):
                answer += f"{i}. {fact.get('content', '')[:200]}...\n"
        else:
            answer = f"I couldn't process your query about '{context.query}'. Please try again."
        
        return create_safe_agent_result(
            success=False,
            data={
                "answer": answer,
                "response": answer,
                "fallback": True
            },
            error=f"Synthesis failed: {str(error)}",
            confidence=0.2
        )
    
    elif agent_type == AgentType.CITATION:
        # Pass through synthesis result without citations
        synthesis_data = previous_results.get(AgentType.SYNTHESIS, {})
        answer = ""
        if synthesis_data:
            answer = synthesis_data.data.get("answer", synthesis_data.data.get("response", ""))
        
        return create_safe_agent_result(
            success=False,
            data={
                "cited_content": answer,
                "bibliography": [],
                "citations_skipped": True
            },
            error=f"Citation failed: {str(error)}",
            confidence=0.5
        )
    
    # Default fallback
    return create_safe_agent_result(
        success=False,
        data={},
        error=f"{agent_type.value} failed: {str(error)}",
        confidence=0.0
    )


def validate_agent_result(result: Any, agent_type: AgentType) -> AgentResult:
    """
    Validate and sanitize agent result to ensure it's safe for downstream use.
    
    Args:
        result: Raw result from agent
        agent_type: Type of agent that produced the result
        
    Returns:
        Validated and safe AgentResult
    """
    # Handle None result
    if result is None:
        return create_safe_agent_result(
            success=False,
            error=f"{agent_type.value} returned None",
            confidence=0.0
        )
    
    # Handle exceptions
    if isinstance(result, Exception):
        return create_safe_agent_result(
            success=False,
            error=str(result),
            confidence=0.0
        )
    
    # Ensure it's an AgentResult
    if not isinstance(result, AgentResult):
        logger.warning(f"{agent_type.value} returned non-AgentResult: {type(result)}")
        return create_safe_agent_result(
            success=False,
            error=f"Invalid result type: {type(result)}",
            confidence=0.0
        )
    
    # Validate data field
    if result.data is None:
        result.data = {}
    
    # Ensure required fields based on agent type
    if agent_type == AgentType.RETRIEVAL and "documents" not in result.data:
        result.data["documents"] = []
    elif agent_type == AgentType.FACT_CHECK and "verified_facts" not in result.data:
        result.data["verified_facts"] = []
    elif agent_type == AgentType.SYNTHESIS:
        if "answer" not in result.data and "response" not in result.data:
            result.data["answer"] = ""
    elif agent_type == AgentType.CITATION and "cited_content" not in result.data:
        result.data["cited_content"] = ""
        result.data["bibliography"] = []
    
    return result


def create_pipeline_error_response(
    context: QueryContext,
    error: str,
    partial_results: Dict[AgentType, AgentResult]
) -> Dict[str, Any]:
    """
    Create a user-friendly error response when pipeline fails.
    
    Args:
        context: Query context
        error: Error description
        partial_results: Any partial results available
        
    Returns:
        Error response with any salvageable information
    """
    # Try to salvage any useful information
    salvaged_info = []
    
    if AgentType.RETRIEVAL in partial_results:
        docs = partial_results[AgentType.RETRIEVAL].data.get("documents", [])
        if docs:
            salvaged_info.append(f"Found {len(docs)} potentially relevant documents")
    
    if AgentType.FACT_CHECK in partial_results:
        facts = partial_results[AgentType.FACT_CHECK].data.get("verified_facts", [])
        if facts:
            salvaged_info.append(f"Verified {len(facts)} facts")
    
    response = {
        "success": False,
        "error": error,
        "query": context.query,
        "partial_results_available": bool(salvaged_info),
        "salvaged_information": salvaged_info,
        "suggestion": "Please try rephrasing your query or breaking it into smaller questions.",
        "timestamp": context.timestamp
    }
    
    # Add any partial answer if available
    if AgentType.SYNTHESIS in partial_results:
        partial_answer = partial_results[AgentType.SYNTHESIS].data.get("answer", "")
        if partial_answer:
            response["partial_answer"] = partial_answer
    
    return response


# Export functions for use in orchestrator
__all__ = [
    'OrchestrationError',
    'create_safe_agent_result',
    'safe_prepare_synthesis_input',
    'handle_agent_failure',
    'validate_agent_result',
    'create_pipeline_error_response'
] 