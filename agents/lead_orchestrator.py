
"""
Multi-Agent Knowledge Platform - Refactored Implementation
This module provides the core architecture for a multi-agent system designed
for intelligent knowledge retrieval, verification, synthesis, and citation.
"""

import asyncio
import uuid
import time
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import logging
from collections import defaultdict
from functools import lru_cache
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import proper agent implementations
from agents.base_agent import QueryContext, AgentResult, AgentMessage, MessageType, TaskPriority, AgentType, BaseAgent
from agents.retrieval_agent import RetrievalAgent
from agents.factcheck_agent import FactCheckAgent
from agents.synthesis_agent import SynthesisAgent
from agents.citation_agent import CitationAgent

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LeadOrchestrator:
    """
    Refactored LeadOrchestrator that uses proper agent implementations
    and provides clean coordination patterns.
    """
    
    def __init__(self):
        """Initialize orchestrator with proper agent instances."""
        logger.info("🚀 Initializing LeadOrchestrator with proper agent implementations")
        
        # Initialize agents using proper implementations
        self.agents = {
            AgentType.RETRIEVAL: RetrievalAgent(),
            AgentType.FACT_CHECK: FactCheckAgent(),
            AgentType.SYNTHESIS: SynthesisAgent(),
            AgentType.CITATION: CitationAgent()
        }
        
        # Initialize supporting components
        self.token_budget = TokenBudgetController()
        self.semantic_cache = SemanticCacheManager()
        self.response_aggregator = ResponseAggregator()
        
        logger.info("✅ LeadOrchestrator initialized successfully")

    async def process_query(self, query: str, user_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Main entry point for processing queries through the multi-agent pipeline.
        
        Args:
            query: The user's question
            user_context: Optional user context and preferences
            
        Returns:
            Dict containing the answer, confidence, citations, and metadata
        """
        start_time = time.time()
        
        try:
            # Create query context
            context = QueryContext(
                query=query,
                user_context=user_context or {},
                trace_id=str(uuid.uuid4())
            )
            
            # Check cache first
            cached_result = await self.semantic_cache.get_cached_response(query)
            if cached_result:
                logger.info(f"Cache HIT for query: {query[:50]}...")
                return cached_result
            
            # Analyze and plan execution
            plan = await self.analyze_and_plan(context)
            
            # Execute based on plan
            if plan['execution_pattern'] == 'pipeline':
                result = await self.execute_pipeline(context, plan)
            elif plan['execution_pattern'] == 'fork_join':
                result = await self.execute_fork_join(context, plan)
            elif plan['execution_pattern'] == 'scatter_gather':
                result = await self.execute_scatter_gather(context, plan)
            else:
                result = await self.execute_pipeline(context, plan)  # Default to pipeline
            
            # Aggregate results
            final_response = self.response_aggregator.aggregate_pipeline_results(result, context)
            
            # Cache successful response
            if final_response.get('success', False):
                await self.semantic_cache.cache_response(query, final_response)
            
            return final_response
            
        except Exception as e:
            logger.error(f"Query processing failed: {str(e)}")
            return {
                'success': False,
                'error': f'Query processing failed: {str(e)}',
                'answer': '',
                'confidence': 0.0,
                'citations': [],
                'metadata': {
                    'trace_id': context.trace_id if 'context' in locals() else str(uuid.uuid4()),
                    'execution_time_ms': int((time.time() - start_time) * 1000)
                }
            }

    async def analyze_and_plan(self, context: QueryContext) -> Dict[str, Any]:
        """
        Analyze query and create execution plan.
        
        Args:
            context: Query context
            
        Returns:
            Execution plan
        """
        # Simple planning based on query characteristics
        query_lower = context.query.lower()
        
        # Determine execution pattern
        if any(word in query_lower for word in ['compare', 'versus', 'vs', 'difference']):
            execution_pattern = 'fork_join'
        elif any(word in query_lower for word in ['research', 'study', 'analysis']):
            execution_pattern = 'scatter_gather'
        else:
            execution_pattern = 'pipeline'
        
        # Define agent sequence
        agents_sequence = [
            AgentType.RETRIEVAL,
            AgentType.FACT_CHECK,
            AgentType.SYNTHESIS,
            AgentType.CITATION
        ]
        
        return {
            'execution_pattern': execution_pattern,
            'agents_sequence': agents_sequence,
            'estimated_tokens': len(context.query.split()) * 10
        }

    async def execute_pipeline(self, context: QueryContext, plan: Dict[str, Any]) -> Dict[AgentType, AgentResult]:
        """
        Execute agents in an optimized pipeline with parallel execution where possible.
        
        Args:
            context: Query context
            plan: Execution plan
            
        Returns:
            Results from each agent
        """
        results = {}
        
        # Phase 1: Parallel retrieval and initial analysis
        logger.info("Phase 1: Parallel retrieval and analysis")
        phase1_tasks = []
        
        # Start retrieval immediately
        retrieval_task = self.agents[AgentType.RETRIEVAL].process_task({
            'query': context.query,
            'search_type': 'hybrid',
            'top_k': 20
        }, context)
        phase1_tasks.append((AgentType.RETRIEVAL, retrieval_task))
        
        # Start entity extraction in parallel (if not already done by retrieval)
        entity_task = self._extract_entities_parallel(context.query)
        phase1_tasks.append(('entities', entity_task))
        
        # Execute phase 1 tasks in parallel
        phase1_results = await asyncio.gather(*[task for _, task in phase1_tasks], return_exceptions=True)
        
        # Process phase 1 results
        for i, (agent_type, _) in enumerate(phase1_tasks):
            if isinstance(phase1_results[i], Exception):
                logger.error(f"Phase 1 task failed: {phase1_results[i]}")
                if agent_type == AgentType.RETRIEVAL:
                    # Critical failure
                    return {agent_type: AgentResult(
                        success=False,
                        error=f"Retrieval failed: {phase1_results[i]}",
                        data={'documents': []}
                    )}
            else:
                results[agent_type] = phase1_results[i]
        
        # Phase 2: Fact checking and synthesis preparation
        logger.info("Phase 2: Fact checking and synthesis preparation")
        phase2_tasks = []
        
        # Start fact checking with retrieved documents
        if AgentType.RETRIEVAL in results and results[AgentType.RETRIEVAL].success:
            fact_check_task = self.agents[AgentType.FACT_CHECK].process_task({
                'documents': results[AgentType.RETRIEVAL].data.get('documents', []),
                'query': context.query
            }, context)
            phase2_tasks.append((AgentType.FACT_CHECK, fact_check_task))
        
        # Start synthesis preparation (document summarization) in parallel
        if AgentType.RETRIEVAL in results and results[AgentType.RETRIEVAL].success:
            synthesis_prep_task = self._prepare_synthesis_data(
                results[AgentType.RETRIEVAL].data.get('documents', []),
                context.query
            )
            phase2_tasks.append(('synthesis_prep', synthesis_prep_task))
        
        # Execute phase 2 tasks in parallel
        if phase2_tasks:
            phase2_results = await asyncio.gather(*[task for _, task in phase2_tasks], return_exceptions=True)
            
            for i, (agent_type, _) in enumerate(phase2_tasks):
                if isinstance(phase2_results[i], Exception):
                    logger.error(f"Phase 2 task failed: {phase2_results[i]}")
                    if agent_type == AgentType.FACT_CHECK:
                        # Continue without fact checking
                        results[agent_type] = AgentResult(
                            success=False,
                            error=f"Fact checking failed: {phase2_results[i]}",
                            data={'verified_facts': []}
                        )
                else:
                    results[agent_type] = phase2_results[i]
        
        # Phase 3: Synthesis and citation (sequential due to dependencies)
        logger.info("Phase 3: Synthesis and citation")
        
        # Synthesis
        synthesis_input = self._prepare_synthesis_input(results, context)
        synthesis_result = await self.agents[AgentType.SYNTHESIS].process_task(synthesis_input, context)
        results[AgentType.SYNTHESIS] = synthesis_result
        
        if not synthesis_result.success:
            logger.error(f"Synthesis failed: {synthesis_result.error}")
            return results
        
        # Citation (depends on synthesis)
        citation_result = await self.agents[AgentType.CITATION].process_task({
            'content': synthesis_result.data.get('answer', ''),
            'sources': results.get(AgentType.RETRIEVAL, AgentResult(success=False, data={'documents': []})).data.get('documents', [])
        }, context)
        results[AgentType.CITATION] = citation_result
        
        return results
    
    async def _extract_entities_parallel(self, query: str) -> List[Dict[str, Any]]:
        """Extract entities in parallel (if not already done by retrieval agent)."""
        # This is a lightweight operation that can run in parallel
        await asyncio.sleep(0.01)  # Simulate processing
        return [
            {'text': 'parallel_entity', 'type': 'PROPER_NOUN', 'confidence': 0.8}
        ]
    
    async def _prepare_synthesis_data(self, documents: List[Dict[str, Any]], query: str) -> Dict[str, Any]:
        """Prepare synthesis data in parallel."""
        # This could include document summarization, relevance scoring, etc.
        await asyncio.sleep(0.05)  # Simulate processing
        
        return {
            'prepared_documents': documents[:10],  # Top 10 most relevant
            'summary': f"Prepared {len(documents)} documents for synthesis",
            'relevance_scores': [doc.get('score', 0.5) for doc in documents[:10]]
        }
    
    def _prepare_synthesis_input(self, results: Dict[AgentType, AgentResult], context: QueryContext) -> Dict[str, Any]:
        """Prepare input for synthesis agent."""
        verified_facts = []
        if AgentType.FACT_CHECK in results and results[AgentType.FACT_CHECK].success:
            verified_facts = results[AgentType.FACT_CHECK].data.get('verified_facts', [])
        
        # Use prepared synthesis data if available
        synthesis_prep = None
        for key, result in results.items():
            if key == 'synthesis_prep' and result.success:
                synthesis_prep = result.data
                break
        
        return {
            'verified_facts': verified_facts,
            'query': context.query,
            'synthesis_params': {
                'style': 'concise',
                'max_length': 1000,
                'prepared_data': synthesis_prep
            }
        }

    async def execute_fork_join(self, context: QueryContext, plan: Dict[str, Any]) -> Dict[AgentType, AgentResult]:
        """
        Execute retrieval agents in parallel and merge results.
        
        Args:
            context: Query context
            plan: Execution plan
            
        Returns:
            Merged results from parallel execution
        """
        # Create parallel retrieval tasks
        retrieval_tasks = [
            self.agents[AgentType.RETRIEVAL].process_task({
                'query': context.query,
                'search_type': 'vector',
                'top_k': 10
            }, context),
            self.agents[AgentType.RETRIEVAL].process_task({
                'query': context.query,
                'search_type': 'keyword',
                'top_k': 10
            }, context),
            self.agents[AgentType.RETRIEVAL].process_task({
                'query': context.query,
                'search_type': 'graph',
                'top_k': 10
            }, context)
        ]
        
        # Execute in parallel
        logger.info("Executing parallel retrieval tasks")
        retrieval_results = await asyncio.gather(*retrieval_tasks, return_exceptions=True)
        
        # Handle exceptions in retrieval results
        valid_results = []
        for result in retrieval_results:
            if isinstance(result, Exception):
                logger.error(f"Retrieval task failed: {result}")
            elif result.success:
                valid_results.append(result)
        
        if not valid_results:
            logger.error("All retrieval tasks failed")
            return {
                AgentType.RETRIEVAL: AgentResult(
                    success=False,
                    error="All retrieval tasks failed",
                    data={'documents': []}
                )
            }
        
        # Merge retrieval results
        merged_retrieval = self._merge_retrieval_results(valid_results)
        
        # Continue with synthesis and citation
        synthesis_result = await self.agents[AgentType.SYNTHESIS].process_task({
            'verified_facts': merged_retrieval.data.get('documents', []),
            'query': context.query,
            'synthesis_params': {'style': 'concise'}
        }, context)
        
        if not synthesis_result.success:
            logger.error(f"Synthesis failed: {synthesis_result.error}")
            return {
                AgentType.RETRIEVAL: merged_retrieval,
                AgentType.SYNTHESIS: synthesis_result
            }
        
        citation_result = await self.agents[AgentType.CITATION].process_task({
            'content': synthesis_result.data.get('answer', ''),
            'sources': merged_retrieval.data.get('documents', [])
        }, context)
        
        return {
            AgentType.RETRIEVAL: merged_retrieval,
            AgentType.SYNTHESIS: synthesis_result,
            AgentType.CITATION: citation_result
        }

    async def execute_scatter_gather(self, context: QueryContext, plan: Dict[str, Any]) -> Dict[AgentType, AgentResult]:
        """
        Execute domain-specific searches and combine results.
        
        Args:
            context: Query context
            plan: Execution plan
            
        Returns:
            Combined results from domain-specific searches
        """
        # Detect domains
        domains = await self._detect_query_domains(context.query)
        
        if not domains:
            # Fallback to standard pipeline
            return await self.execute_pipeline(context, plan)
        
        # Generate domain-specific queries
        domain_queries = await self._generate_domain_queries(context.query, domains)
        
        # Execute parallel domain searches
        domain_tasks = []
        for domain, domain_query in domain_queries.items():
            task = self.agents[AgentType.RETRIEVAL].process_task({
                'query': domain_query,
                'search_type': 'hybrid',
                'top_k': 15
            }, context)
            domain_tasks.append(task)
        
        # Execute domain searches
        domain_results = await asyncio.gather(*domain_tasks, return_exceptions=True)
        
        # Combine domain results
        all_documents = []
        for result in domain_results:
            if isinstance(result, Exception):
                logger.error(f"Domain search failed: {result}")
            elif result.success:
                all_documents.extend(result.data.get('documents', []))
        
        # Create merged retrieval result
        merged_retrieval = self._merge_retrieval_results([
            AgentResult(success=True, data={'documents': all_documents})
        ])
        
        # Continue with synthesis and citation
        synthesis_result = await self.agents[AgentType.SYNTHESIS].process_task({
            'verified_facts': merged_retrieval.data.get('documents', []),
            'query': context.query,
            'synthesis_params': {'style': 'comprehensive'}
        }, context)
        
        citation_result = await self.agents[AgentType.CITATION].process_task({
            'content': synthesis_result.data.get('answer', ''),
            'sources': merged_retrieval.data.get('documents', [])
        }, context)
        
        return {
            AgentType.RETRIEVAL: merged_retrieval,
            AgentType.SYNTHESIS: synthesis_result,
            AgentType.CITATION: citation_result
        }

    def _prepare_task_for_agent(self, agent_type: AgentType, previous_result: Optional[AgentResult], context: QueryContext) -> Dict[str, Any]:
        """
        Prepare task data for a specific agent.
        
        Args:
            agent_type: Type of agent to prepare task for
            previous_result: Result from previous agent (if any)
            context: Query context
            
        Returns:
            Task data for the agent
        """
        if agent_type == AgentType.RETRIEVAL:
            return {
                'query': context.query,
                'search_type': 'hybrid',
                'top_k': 20
            }
        elif agent_type == AgentType.FACT_CHECK:
            documents = previous_result.data.get('documents', []) if previous_result else []
            return {
                'documents': documents,
                'query': context.query
            }
        elif agent_type == AgentType.SYNTHESIS:
            verified_facts = previous_result.data.get('verified_facts', []) if previous_result else []
            return {
                'verified_facts': verified_facts,
                'query': context.query,
                'synthesis_params': {'style': 'concise', 'max_length': 1000}
            }
        elif agent_type == AgentType.CITATION:
            answer = previous_result.data.get('answer', '') if previous_result else ''
            sources = previous_result.data.get('documents', []) if previous_result else []
            return {
                'content': answer,
                'sources': sources,
                'style': 'APA'
            }
        else:
            return {'query': context.query}

    def _merge_retrieval_results(self, results: List[AgentResult]) -> AgentResult:
        """
        Merge multiple retrieval results with deduplication and ranking.
        
        Args:
            results: List of retrieval results
            
        Returns:
            Merged result with deduplicated documents
        """
        all_documents = []
        
        for result in results:
            if result.success:
                all_documents.extend(result.data.get('documents', []))
        
        # Deduplicate by content
        seen_contents = set()
        unique_documents = []
        
        for doc in all_documents:
            content_hash = hash(doc.get('content', ''))
            if content_hash not in seen_contents:
                seen_contents.add(content_hash)
                unique_documents.append(doc)
        
        # Sort by relevance score
        unique_documents.sort(key=lambda x: x.get('score', 0), reverse=True)
        
        # Take top documents
        final_documents = unique_documents[:20]
        
        return AgentResult(
            success=True,
            data={'documents': final_documents},
            confidence=min(0.9, len(final_documents) / 20.0)
        )

    async def _detect_query_domains(self, query: str) -> List[str]:
        """
        Detect relevant domains for a query.
        
        Args:
            query: User query
            
        Returns:
            List of relevant domains
        """
        # Simple domain detection based on keywords
        query_lower = query.lower()
        domains = []
        
        if any(word in query_lower for word in ['science', 'research', 'study', 'experiment']):
            domains.append('scientific')
        
        if any(word in query_lower for word in ['news', 'current', 'recent', 'latest']):
            domains.append('news')
        
        if any(word in query_lower for word in ['business', 'market', 'company', 'industry']):
            domains.append('business')
        
        if any(word in query_lower for word in ['academic', 'scholarly', 'paper', 'journal']):
            domains.append('academic')
        
        if any(word in query_lower for word in ['technology', 'tech', 'software', 'digital']):
            domains.append('technology')
        
        return domains

    async def _generate_domain_queries(self, query: str, domains: List[str]) -> Dict[str, str]:
        """
        Generate domain-specific queries.
        
        Args:
            query: Original query
            domains: Detected domains
            
        Returns:
            Domain-specific queries
        """
        domain_queries = {}
        
        for domain in domains:
            if domain == 'scientific':
                domain_queries[domain] = f"scientific research {query}"
            elif domain == 'news':
                domain_queries[domain] = f"recent news {query}"
            elif domain == 'technology':
                domain_queries[domain] = f"technology {query}"
            elif domain == 'academic':
                domain_queries[domain] = f"academic sources {query}"
            elif domain == 'business':
                domain_queries[domain] = f"business information {query}"
            else:
                domain_queries[domain] = query
        
        return domain_queries


# ============================================================================
# Supporting Classes (Simplified)
# ============================================================================

class TokenBudgetController:
    """Simplified token budget controller."""
    
    def __init__(self, daily_budget: int = None):
        self.daily_budget = daily_budget or 1000000
        self.used_tokens = defaultdict(int)
    
    def allocate_budget_for_query(self, query: str) -> int:
        """Allocate token budget for a query."""
        return min(10000, len(query.split()) * 10)
    
    def track_usage(self, agent_type: AgentType, tokens_used: int):
        """Track token usage for an agent."""
        self.used_tokens[agent_type] += tokens_used


class SemanticCacheManager:
    """Simplified semantic cache manager with thread safety."""
    
    def __init__(self, similarity_threshold: float = None):
        self.similarity_threshold = similarity_threshold or 0.92
        self.cache = {}
        self._lock = asyncio.Lock()  # Add lock for thread safety
    
    async def get_cached_response(self, query: str) -> Optional[Dict[str, Any]]:
        """Get cached response for a query."""
        async with self._lock:
            # Simple exact match for now
            return self.cache.get(query)
    
    async def cache_response(self, query: str, response: Dict[str, Any]):
        """Cache a response for a query."""
        async with self._lock:
            self.cache[query] = response


class ResponseAggregator:
    """Simplified response aggregator with fixed data format handling."""
    
    def __init__(self):
        pass
    
    def aggregate_pipeline_results(self, results: Dict[AgentType, AgentResult], context: QueryContext) -> Dict[str, Any]:
        """
        Aggregate results from pipeline execution.
        
        Args:
            results: Results from each agent
            context: Query context
            
        Returns:
            Final response with answer, confidence, and citations
        """
        # Check for critical failures
        critical_failures = []
        for agent_type, result in results.items():
            if not result.success and agent_type in [AgentType.RETRIEVAL, AgentType.SYNTHESIS]:
                critical_failures.append(f"{agent_type.value}: {result.error}")
        
        if critical_failures:
            return {
                'success': False,
                'error': f"Critical agent failures: {'; '.join(critical_failures)}",
                'answer': '',
                'confidence': 0.0,
                'citations': [],
                'metadata': {
                    'agents_used': [agent_type.value for agent_type in results.keys()],
                    'trace_id': context.trace_id,
                    'failed_agents': critical_failures
                }
            }
        
        # Extract synthesis result
        synthesis_result = results.get(AgentType.SYNTHESIS)
        citation_result = results.get(AgentType.CITATION)
        
        if not synthesis_result or not synthesis_result.success:
            return {
                'success': False,
                'error': 'Synthesis failed',
                'answer': '',
                'confidence': 0.0,
                'citations': [],
                'metadata': {
                    'agents_used': [agent_type.value for agent_type in results.keys()],
                    'trace_id': context.trace_id
                }
            }
        
        # Extract answer and confidence
        answer = synthesis_result.data.get('answer', '')
        confidence = synthesis_result.confidence
        
        # Extract citations - handle both old and new formats
        citations = []
        if citation_result and citation_result.success:
            # Handle the correct data format from CitationAgent
            citation_data = citation_result.data
            if isinstance(citation_data, dict):
                # New format: citations are in the 'citations' key
                citations = citation_data.get('citations', [])
            else:
                # Fallback for old format
                citations = citation_result.data or []
        
        return {
            'success': True,
            'answer': answer,
            'confidence': confidence,
            'citations': citations,
            'metadata': {
                'agents_used': [agent_type.value for agent_type in results.keys()],
                'trace_id': context.trace_id,
                'synthesis_method': synthesis_result.data.get('synthesis_method', 'unknown'),
                'fact_count': synthesis_result.data.get('fact_count', 0)
            }
        }



