"""
Orchestrator Workflow Fixes

This module contains improvements to fix:
1. Proper retrieval result merging using RetrievalAgent's merge functionality
2. Better graceful error handling in pipeline execution
3. Improved data flow to prevent None/stale data issues
"""

import logging
import hashlib
from typing import Dict, List, Any, Optional
from agents.base_agent import AgentResult, AgentType, QueryContext
from agents.retrieval_agent import Document, SearchResult

logger = logging.getLogger(__name__)


def merge_retrieval_results_improved(
    results: List[AgentResult],
    retrieval_agent=None
) -> AgentResult:
    """
    Improved merge of multiple retrieval results using RetrievalAgent's merge functionality.
    
    Args:
        results: List of retrieval AgentResults
        retrieval_agent: Optional RetrievalAgent instance for advanced merging
        
    Returns:
        Merged AgentResult with deduplicated documents
    """
    # Extract all SearchResult objects from AgentResults
    search_results = []
    all_documents = []
    
    for result in results:
        if result and result.success and result.data:
            documents = result.data.get("documents", [])
            
            # Convert to Document objects if they're dicts
            doc_objects = []
            for doc in documents:
                if isinstance(doc, dict):
                    doc_objects.append(Document(
                        content=doc.get("content", ""),
                        score=doc.get("score", 0.0),
                        source=doc.get("source", "unknown"),
                        metadata=doc.get("metadata", {}),
                        doc_id=doc.get("doc_id", doc.get("id", ""))
                    ))
                elif hasattr(doc, 'content'):  # Already a Document object
                    doc_objects.append(doc)
            
            all_documents.extend(doc_objects)
            
            # Create SearchResult for merging
            search_type = result.data.get("search_type", "unknown")
            search_results.append(SearchResult(
                documents=doc_objects,
                search_type=search_type,
                query_time_ms=result.execution_time_ms or 0,
                total_hits=len(doc_objects),
                metadata=result.data.get("metadata", {})
            ))
    
    # If we have a retrieval agent, use its advanced merge
    if retrieval_agent and hasattr(retrieval_agent, '_merge_and_deduplicate'):
        try:
            # Use the agent's merge functionality
            merged_documents = retrieval_agent._merge_and_deduplicate(search_results)
            
            # Apply diversity constraints if available
            if hasattr(retrieval_agent, '_apply_diversity_constraints'):
                merged_documents = retrieval_agent._apply_diversity_constraints(merged_documents)
        except Exception as e:
            logger.warning(f"Failed to use retrieval agent merge: {e}, falling back to simple merge")
            merged_documents = simple_merge_documents(all_documents)
    else:
        # Fallback to simple merge
        merged_documents = simple_merge_documents(all_documents)
    
    # Convert back to dict format for AgentResult
    document_dicts = []
    for doc in merged_documents[:20]:  # Limit to top 20
        doc_dict = {
            "content": doc.content,
            "score": doc.score,
            "source": doc.source,
            "metadata": doc.metadata,
            "doc_id": doc.doc_id
        }
        document_dicts.append(doc_dict)
    
    # Calculate confidence based on document count and scores
    avg_score = sum(d["score"] for d in document_dicts) / len(document_dicts) if document_dicts else 0
    confidence = min(0.9, (len(document_dicts) / 20.0) * 0.5 + avg_score * 0.5)
    
    return AgentResult(
        success=True,
        data={
            "documents": document_dicts,
            "total_retrieved": len(all_documents),
            "total_merged": len(document_dicts),
            "search_types": list(set(r.search_type for r in search_results))
        },
        confidence=confidence,
        error=None
    )


def simple_merge_documents(documents: List[Document]) -> List[Document]:
    """
    Simple document merging with content-based deduplication.
    
    Args:
        documents: List of Document objects
        
    Returns:
        Deduplicated and sorted list of documents
    """
    # Deduplicate by content hash
    seen_hashes = set()
    unique_documents = []
    
    for doc in documents:
        # Create content hash
        content_hash = hashlib.md5(doc.content.lower().encode()).hexdigest()
        
        if content_hash not in seen_hashes:
            seen_hashes.add(content_hash)
            unique_documents.append(doc)
    
    # Sort by score descending
    unique_documents.sort(key=lambda x: x.score, reverse=True)
    
    return unique_documents


def execute_pipeline_improved(
    orchestrator,
    context: QueryContext,
    plan: Dict[str, Any],
    query_budget: int
) -> Dict[AgentType, AgentResult]:
    """
    Improved pipeline execution with better error handling and data flow.
    
    This function should be used as a replacement or wrapper for the
    orchestrator's execute_pipeline method.
    
    Args:
        orchestrator: The LeadOrchestrator instance
        context: Query context
        plan: Execution plan
        query_budget: Token budget
        
    Returns:
        Results from each agent with proper error handling
    """
    results = {}
    last_successful_data = None
    
    # Define pipeline stages with dependencies
    pipeline_stages = [
        {
            "agent": AgentType.RETRIEVAL,
            "required": True,
            "fallback": {"documents": []},
            "prepare_input": lambda: {
                "query": context.query,
                "search_type": plan.get("search_type", "hybrid"),
                "top_k": plan.get("top_k", 20)
            }
        },
        {
            "agent": AgentType.FACT_CHECK,
            "required": False,
            "depends_on": AgentType.RETRIEVAL,
            "skip_on_failure": True,
            "prepare_input": lambda: {
                "documents": last_successful_data.get("documents", []) if last_successful_data else [],
                "query": context.query
            }
        },
        {
            "agent": AgentType.SYNTHESIS,
            "required": True,
            "depends_on": AgentType.FACT_CHECK,
            "use_fallback_on_dependency_failure": True,
            "prepare_input": lambda: prepare_synthesis_input_improved(
                results, context, last_successful_data
            )
        },
        {
            "agent": AgentType.CITATION,
            "required": False,
            "depends_on": AgentType.SYNTHESIS,
            "prepare_input": lambda: {
                "content": (
                    last_successful_data.get("answer", "") or 
                    last_successful_data.get("response", "")
                ) if last_successful_data else "",
                "sources": results.get(AgentType.RETRIEVAL, AgentResult(
                    success=False, data={"documents": []}
                )).data.get("documents", [])
            }
        }
    ]
    
    # Execute pipeline stages
    for stage in pipeline_stages:
        agent_type = stage["agent"]
        
        # Check dependencies
        if "depends_on" in stage:
            dependency = stage["depends_on"]
            if dependency in results and not results[dependency].success:
                if stage.get("skip_on_failure"):
                    logger.info(f"Skipping {agent_type.value} due to {dependency.value} failure")
                    continue
                elif not stage.get("use_fallback_on_dependency_failure"):
                    logger.warning(f"{agent_type.value} dependency {dependency.value} failed")
        
        try:
            # Prepare input
            agent_input = stage["prepare_input"]()
            
            # Execute agent
            result = orchestrator.agents[agent_type].process_task(agent_input, context)
            
            # Validate result
            if result is None:
                raise ValueError(f"{agent_type.value} returned None")
            
            if not hasattr(result, 'success'):
                raise ValueError(f"{agent_type.value} returned invalid result type")
            
            # Store result
            results[agent_type] = result
            
            # Update last successful data if this stage succeeded
            if result.success and result.data:
                last_successful_data = result.data
                
        except Exception as e:
            logger.error(f"{agent_type.value} failed: {str(e)}")
            
            # Create fallback result
            fallback_data = stage.get("fallback", {})
            results[agent_type] = AgentResult(
                success=False,
                data=fallback_data,
                error=f"{agent_type.value} failed: {str(e)}",
                confidence=0.0
            )
            
            # If this is a required stage, we might want to stop
            if stage.get("required") and not stage.get("use_fallback_on_dependency_failure"):
                logger.error(f"Required agent {agent_type.value} failed, stopping pipeline")
                break
    
    return results


def prepare_synthesis_input_improved(
    results: Dict[AgentType, AgentResult],
    context: QueryContext,
    last_successful_data: Optional[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Improved synthesis input preparation with better fallbacks.
    
    Args:
        results: Current pipeline results
        context: Query context
        last_successful_data: Last successful agent's data
        
    Returns:
        Synthesis input that won't cause failures
    """
    # Try to get verified facts from fact checker
    verified_facts = []
    
    if AgentType.FACT_CHECK in results and results[AgentType.FACT_CHECK].success:
        verified_facts = results[AgentType.FACT_CHECK].data.get("verified_facts", [])
    
    # If no verified facts, try to use retrieval documents
    if not verified_facts and AgentType.RETRIEVAL in results:
        retrieval_result = results[AgentType.RETRIEVAL]
        if retrieval_result.success:
            documents = retrieval_result.data.get("documents", [])
            # Convert to fact format
            verified_facts = [
                {
                    "content": doc.get("content", ""),
                    "source": doc.get("source", "Unknown"),
                    "confidence": doc.get("score", 0.5),
                    "verified": False
                }
                for doc in documents[:10]
            ]
    
    # If still no facts, use last successful data
    if not verified_facts and last_successful_data:
        if "documents" in last_successful_data:
            documents = last_successful_data["documents"]
            verified_facts = [
                {
                    "content": doc.get("content", ""),
                    "source": doc.get("source", "Unknown"),
                    "confidence": 0.3,
                    "verified": False
                }
                for doc in documents[:5]
            ]
    
    return {
        "verified_facts": verified_facts,
        "query": context.query,
        "synthesis_params": {
            "style": "concise",
            "max_length": 1000,
            "has_verified_facts": bool(verified_facts),
            "fallback_mode": not verified_facts
        }
    }


# Export functions
__all__ = [
    'merge_retrieval_results_improved',
    'simple_merge_documents',
    'execute_pipeline_improved',
    'prepare_synthesis_input_improved'
] 