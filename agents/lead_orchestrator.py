"""
LeadOrchestrator Implementation for Multi-Agent Knowledge Platform
This module implements the central orchestrator that coordinates all agents.
"""

import asyncio
import logging
import time
import os
import uuid
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum
from datetime import datetime, timedelta
from collections import defaultdict, deque

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from agents.base_agent import AgentResult, QueryContext, AgentType
from agents.retrieval_agent import RetrievalAgent
from agents.factcheck_agent import FactCheckAgent
from agents.synthesis_agent import SynthesisAgent
from agents.citation_agent import CitationAgent
# Data models imported as needed

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
            AgentType.CITATION: CitationAgent(),
        }

        # Initialize supporting components
        self.token_budget = TokenBudgetController()
        self.semantic_cache = SemanticCacheManager()
        self.response_aggregator = ResponseAggregator()

        logger.info("✅ LeadOrchestrator initialized successfully")

    async def process_query(
        self, query: str, user_context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
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
                query=query, user_context=user_context or {}, trace_id=str(uuid.uuid4())
            )

            # Check cache first
            cached_result = await self.semantic_cache.get_cached_response(query)
            if cached_result:
                logger.info(f"Cache HIT for query: {query[:50]}...")
                return cached_result

            # Allocate token budget
            query_budget = await self.token_budget.allocate_budget_for_query(query)
            logger.info(f"Allocated {query_budget} tokens for query")

            # Analyze and plan execution
            plan = await self.analyze_and_plan(context)

            # Execute based on plan
            if plan["execution_pattern"] == "pipeline":
                result = await self.execute_pipeline(context, plan, query_budget)
            elif plan["execution_pattern"] == "fork_join":
                result = await self.execute_fork_join(context, plan, query_budget)
            elif plan["execution_pattern"] == "scatter_gather":
                result = await self.execute_scatter_gather(context, plan, query_budget)
            else:
                result = await self.execute_pipeline(
                    context, plan, query_budget
                )  # Default to pipeline

            # Aggregate results with enhanced aggregator
            final_response = self.response_aggregator.aggregate_pipeline_results(result, context)

            # Track token usage
            total_tokens = final_response.get("metadata", {}).get("token_usage", {}).get("total", 0)
            await self.token_budget.track_usage(AgentType.ORCHESTRATOR, total_tokens)

            # Cache successful response
            if final_response.get("success", False):
                await self.semantic_cache.cache_response(query, final_response)

            return final_response

        except Exception as e:
            logger.error(f"Query processing failed: {str(e)}")
            return {
                "success": False,
                "error": f"Query processing failed: {str(e)}",
                "answer": "",
                "confidence": 0.0,
                "citations": [],
                "metadata": {
                    "trace_id": context.trace_id if "context" in locals() else str(uuid.uuid4()),
                    "execution_time_ms": int((time.time() - start_time) * 1000),
                },
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
        if any(word in query_lower for word in ["compare", "versus", "vs", "difference"]):
            execution_pattern = "fork_join"
        elif any(word in query_lower for word in ["research", "study", "analysis"]):
            execution_pattern = "scatter_gather"
        else:
            execution_pattern = "pipeline"

        # Define agent sequence
        agents_sequence = [
            AgentType.RETRIEVAL,
            AgentType.FACT_CHECK,
            AgentType.SYNTHESIS,
            AgentType.CITATION,
        ]

        return {
            "execution_pattern": execution_pattern,
            "agents_sequence": agents_sequence,
            "estimated_tokens": len(context.query.split()) * 10,
        }

    async def execute_pipeline(
        self, context: QueryContext, plan: Dict[str, Any], query_budget: int
    ) -> Dict[AgentType, AgentResult]:
        """
        Execute agents in an optimized pipeline with parallel execution where possible.

        Args:
            context: Query context
            plan: Execution plan

        Returns:
            Results from each agent
        """
        results = {}

        # Phase 1: Retrieval
        logger.info("Phase 1: Hybrid retrieval and analysis")
        try:
            retrieval_task = {
                "query": context.query,
                "max_tokens": context.user_context.get("max_tokens", 4000),
                "entities": await self._extract_entities_parallel(context.query),
            }
            retrieval_result = await self.agents[AgentType.RETRIEVAL].process_task(
                retrieval_task, context
            )
            results[AgentType.RETRIEVAL] = retrieval_result
        except Exception as e:
            logger.error(f"Retrieval failed: {e}")
            results[AgentType.RETRIEVAL] = AgentResult(
                success=False, error=f"Retrieval failed: {e}", data={"documents": []}
            )

        # Phase 2: Fact checking and synthesis preparation
        logger.info("Phase 2: Fact checking and synthesis preparation")
        phase2_tasks = []

        # Start fact checking with retrieved documents
        if AgentType.RETRIEVAL in results and results[AgentType.RETRIEVAL].success:
            fact_check_task = self.agents[AgentType.FACT_CHECK].process_task(
                {
                    "documents": results[AgentType.RETRIEVAL].data.get("documents", []),
                    "query": context.query,
                },
                context,
            )
            phase2_tasks.append((AgentType.FACT_CHECK, fact_check_task))

        # Execute phase 2 tasks in parallel
        if phase2_tasks:
            phase2_results = await asyncio.gather(
                *[task for _, task in phase2_tasks], return_exceptions=True
            )

            for i, (agent_type, _) in enumerate(phase2_tasks):
                if isinstance(phase2_results[i], Exception):
                    logger.error(f"Phase 2 task failed: {phase2_results[i]}")
                    if agent_type == AgentType.FACT_CHECK:
                        # Continue without fact checking
                        results[agent_type] = AgentResult(
                            success=False,
                            error=f"Fact checking failed: {phase2_results[i]}",
                            data={"verified_facts": []},
                        )
                else:
                    results[agent_type] = phase2_results[i]

        # Phase 3: Synthesis and citation (sequential due to dependencies)
        logger.info("Phase 3: Synthesis and citation")

        # Synthesis
        synthesis_input = self._prepare_synthesis_input(results, context)
        synthesis_result = await self.agents[AgentType.SYNTHESIS].process_task(
            synthesis_input, context
        )
        results[AgentType.SYNTHESIS] = synthesis_result

        if not synthesis_result.success:
            logger.error(f"Synthesis failed: {synthesis_result.error}")
            return results

        # Citation (depends on synthesis)
        citation_result = await self.agents[AgentType.CITATION].process_task(
            {
                "content": synthesis_result.data.get(
                    "response", synthesis_result.data.get("answer", "")
                ),
                "sources": results.get(
                    AgentType.RETRIEVAL, AgentResult(success=False, data={"documents": []})
                ).data.get("documents", []),
            },
            context,
        )
        results[AgentType.CITATION] = citation_result

        return results

    async def _extract_entities_parallel(self, query: str) -> List[Dict[str, Any]]:
        """Extract entities in parallel (if not already done by retrieval agent)."""
        # This is a lightweight operation that can run in parallel
        await asyncio.sleep(0.01)  # Simulate processing
        return [{"text": "parallel_entity", "type": "PROPER_NOUN", "confidence": 0.8}]

    async def _prepare_synthesis_data(
        self, documents: List[Dict[str, Any]], query: str
    ) -> Dict[str, Any]:
        """Prepare synthesis data in parallel."""
        # This could include document summarization, relevance scoring, etc.
        await asyncio.sleep(0.05)  # Simulate processing

        return {
            "prepared_documents": documents[:10],  # Top 10 most relevant
            "summary": f"Prepared {len(documents)} documents for synthesis",
            "relevance_scores": [doc.get("score", 0.5) for doc in documents[:10]],
        }

    def _prepare_synthesis_input(
        self, results: Dict[AgentType, AgentResult], context: QueryContext
    ) -> Dict[str, Any]:
        """Prepare input for synthesis agent."""
        verified_facts = []
        if AgentType.FACT_CHECK in results and results[AgentType.FACT_CHECK].success:
            verified_facts = results[AgentType.FACT_CHECK].data.get("verified_facts", [])

        # Use prepared synthesis data if available
        synthesis_prep = None
        for key, result in results.items():
            if key == "synthesis_prep" and result.success:
                synthesis_prep = result.data
                break

        return {
            "verified_facts": verified_facts,
            "query": context.query,
            "synthesis_params": {
                "style": "concise",
                "max_length": 1000,
                "prepared_data": synthesis_prep,
            },
        }

    async def execute_fork_join(
        self, context: QueryContext, plan: Dict[str, Any], query_budget: int
    ) -> Dict[AgentType, AgentResult]:
        """
        Execute retrieval agents in parallel and merge results.

        Args:
            context: Query context
            plan: Execution plan

        Returns:
            Merged results from parallel execution
        """
        # Create parallel retrieval tasks using process_task
        retrieval_tasks = [
            self.agents[AgentType.RETRIEVAL].process_task(
                {
                    "query": context.query,
                    "search_type": "vector",
                    "max_tokens": context.user_context.get("max_tokens", 4000),
                },
                context,
            ),
            self.agents[AgentType.RETRIEVAL].process_task(
                {
                    "query": context.query,
                    "search_type": "keyword",
                    "max_tokens": context.user_context.get("max_tokens", 4000),
                },
                context,
            ),
            self.agents[AgentType.RETRIEVAL].process_task(
                {
                    "query": context.query,
                    "search_type": "graph",
                    "max_tokens": context.user_context.get("max_tokens", 4000),
                },
                context,
            ),
        ]

        # Execute in parallel
        logger.info("Executing parallel retrieval tasks (vector, keyword, graph)")
        retrieval_results = await asyncio.gather(*retrieval_tasks, return_exceptions=True)

        # Handle exceptions in retrieval results
        valid_results = []
        for result in retrieval_results:
            if isinstance(result, Exception):
                logger.error(f"Retrieval task failed: {result}")
            elif hasattr(result, "data") and result.data.get("documents"):
                valid_results.extend(result.data["documents"])

        if not valid_results:
            logger.error("All retrieval tasks failed")
            return {
                AgentType.RETRIEVAL: AgentResult(
                    success=False, error="All retrieval tasks failed", data={"documents": []}
                )
            }

        # Merge retrieval results (deduplicate by content)
        seen_contents = set()
        unique_documents = []
        for doc in valid_results:
            content_hash = hash(doc.get("content", ""))
            if content_hash not in seen_contents:
                seen_contents.add(content_hash)
                unique_documents.append(doc)
        final_documents = unique_documents[:20]
        merged_retrieval = AgentResult(
            success=True,
            data={"documents": final_documents},
            confidence=min(0.9, len(final_documents) / 20.0),
        )

        # Continue with synthesis and citation
        synthesis_result = await self.agents[AgentType.SYNTHESIS].process_task(
            {
                "verified_facts": merged_retrieval.data.get("documents", []),
                "query": context.query,
                "synthesis_params": {"style": "concise"},
            },
            context,
        )

        if not synthesis_result.success:
            logger.error(f"Synthesis failed: {synthesis_result.error}")
            return {AgentType.RETRIEVAL: merged_retrieval, AgentType.SYNTHESIS: synthesis_result}

        citation_result = await self.agents[AgentType.CITATION].process_task(
            {
                "content": synthesis_result.data.get(
                    "response", synthesis_result.data.get("answer", "")
                ),
                "sources": merged_retrieval.data.get("documents", []),
            },
            context,
        )

        return {
            AgentType.RETRIEVAL: merged_retrieval,
            AgentType.SYNTHESIS: synthesis_result,
            AgentType.CITATION: citation_result,
        }

    async def execute_scatter_gather(
        self, context: QueryContext, plan: Dict[str, Any], query_budget: int
    ) -> Dict[AgentType, AgentResult]:
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
            return await self.execute_pipeline(context, plan, query_budget)

        # Generate domain-specific queries
        domain_queries = await self._generate_domain_queries(context.query, domains)

        # Execute parallel domain searches
        domain_tasks = []
        for domain, domain_query in domain_queries.items():
            task = self.agents[AgentType.RETRIEVAL].process_task(
                {"query": domain_query, "search_type": "hybrid", "top_k": 15}, context
            )
            domain_tasks.append(task)

        # Execute domain searches
        domain_results = await asyncio.gather(*domain_tasks, return_exceptions=True)

        # Combine domain results
        all_documents = []
        for result in domain_results:
            if isinstance(result, Exception):
                logger.error(f"Domain search failed: {result}")
            elif result.success:
                all_documents.extend(result.data.get("documents", []))

        # Create merged retrieval result
        merged_retrieval = self._merge_retrieval_results(
            [AgentResult(success=True, data={"documents": all_documents})]
        )

        # Continue with synthesis and citation
        synthesis_result = await self.agents[AgentType.SYNTHESIS].process_task(
            {
                "verified_facts": merged_retrieval.data.get("documents", []),
                "query": context.query,
                "synthesis_params": {"style": "comprehensive"},
            },
            context,
        )

        citation_result = await self.agents[AgentType.CITATION].process_task(
            {
                "content": synthesis_result.data.get(
                    "response", synthesis_result.data.get("answer", "")
                ),
                "sources": merged_retrieval.data.get("documents", []),
            },
            context,
        )

        return {
            AgentType.RETRIEVAL: merged_retrieval,
            AgentType.SYNTHESIS: synthesis_result,
            AgentType.CITATION: citation_result,
        }

    def _prepare_task_for_agent(
        self, agent_type: AgentType, previous_result: Optional[AgentResult], context: QueryContext
    ) -> Dict[str, Any]:
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
            return {"query": context.query, "search_type": "hybrid", "top_k": 20}
        elif agent_type == AgentType.FACT_CHECK:
            documents = previous_result.data.get("documents", []) if previous_result else []
            return {"documents": documents, "query": context.query}
        elif agent_type == AgentType.SYNTHESIS:
            verified_facts = (
                previous_result.data.get("verified_facts", []) if previous_result else []
            )
            return {
                "verified_facts": verified_facts,
                "query": context.query,
                "synthesis_params": {"style": "concise", "max_length": 1000},
            }
        elif agent_type == AgentType.CITATION:
            answer = (
                previous_result.data.get("response", previous_result.data.get("answer", ""))
                if previous_result
                else ""
            )
            sources = previous_result.data.get("documents", []) if previous_result else []
            return {"content": answer, "sources": sources, "style": "APA"}
        else:
            return {"query": context.query}

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
                all_documents.extend(result.data.get("documents", []))

        # Deduplicate by content
        seen_contents = set()
        unique_documents = []

        for doc in all_documents:
            content_hash = hash(doc.get("content", ""))
            if content_hash not in seen_contents:
                seen_contents.add(content_hash)
                unique_documents.append(doc)

        # Sort by relevance score
        unique_documents.sort(key=lambda x: x.get("score", 0), reverse=True)

        # Take top documents
        final_documents = unique_documents[:20]

        return AgentResult(
            success=True,
            data={"documents": final_documents},
            confidence=min(0.9, len(final_documents) / 20.0),
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

        if any(word in query_lower for word in ["science", "research", "study", "experiment"]):
            domains.append("scientific")

        if any(word in query_lower for word in ["news", "current", "recent", "latest"]):
            domains.append("news")

        if any(word in query_lower for word in ["business", "market", "company", "industry"]):
            domains.append("business")

        if any(word in query_lower for word in ["academic", "scholarly", "paper", "journal"]):
            domains.append("academic")

        if any(word in query_lower for word in ["technology", "tech", "software", "digital"]):
            domains.append("technology")

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
            if domain == "scientific":
                domain_queries[domain] = f"scientific research {query}"
            elif domain == "news":
                domain_queries[domain] = f"recent news {query}"
            elif domain == "technology":
                domain_queries[domain] = f"technology {query}"
            elif domain == "academic":
                domain_queries[domain] = f"academic sources {query}"
            elif domain == "business":
                domain_queries[domain] = f"business information {query}"
            else:
                domain_queries[domain] = query

        return domain_queries


# ============================================================================
# Supporting Classes (Simplified)
# ============================================================================


class TokenBudgetController:
    """Enhanced token budget controller with real usage tracking."""

    def __init__(self, daily_budget: int = None):
        self.daily_budget = daily_budget or int(os.getenv("DAILY_TOKEN_BUDGET", "100000"))
        self.used_today = 0
        self.last_reset = datetime.now().date()
        self.agent_allocations = {
            AgentType.RETRIEVAL: 0.15,  # 15% for retrieval
            AgentType.FACT_CHECK: 0.20,  # 20% for fact-checking
            AgentType.SYNTHESIS: 0.50,  # 50% for synthesis
            AgentType.CITATION: 0.15,  # 15% for citation
        }
        self._lock = asyncio.Lock()

    async def allocate_budget_for_query(self, query: str) -> int:
        """
        Allocate token budget for a query based on length and remaining daily budget.

        Args:
            query: Query string

        Returns:
            Allocated token budget
        """
        async with self._lock:
            # Reset daily usage if it's a new day
            today = datetime.now().date()
            if today > self.last_reset:
                self.used_today = 0
                self.last_reset = today

            # Calculate remaining budget
            remaining_budget = self.daily_budget - self.used_today

            # Allocate based on query length (max 10% of remaining daily budget)
            query_length = len(query.split())
            base_allocation = min(1000, query_length * 10)  # Base allocation
            max_allocation = remaining_budget * 0.1  # Max 10% of remaining

            allocated = min(base_allocation, max_allocation)

            return int(allocated)

    async def track_usage(self, agent_type: AgentType, tokens_used: int):
        """
        Track token usage for an agent.

        Args:
            agent_type: Type of agent that used tokens
            tokens_used: Number of tokens used
        """
        async with self._lock:
            self.used_today += tokens_used

            # Log if usage is high
            if tokens_used > 1000:
                logger.warning(f"High token usage by {agent_type.value}: {tokens_used} tokens")

    async def get_budget_status(self) -> Dict[str, Any]:
        """Get current budget status."""
        async with self._lock:
            return {
                "daily_budget": self.daily_budget,
                "used_today": self.used_today,
                "remaining_today": self.daily_budget - self.used_today,
                "usage_percentage": (self.used_today / self.daily_budget) * 100,
                "last_reset": self.last_reset.isoformat(),
            }

    async def allocate_for_agent(self, agent_type: AgentType, query_budget: int) -> int:
        """
        Allocate portion of query budget for specific agent.

        Args:
            agent_type: Type of agent
            query_budget: Total budget for the query

        Returns:
            Budget allocated for this agent
        """
        allocation_ratio = self.agent_allocations.get(agent_type, 0.1)
        return int(query_budget * allocation_ratio)


class SemanticCacheManager:
    """Enhanced semantic cache manager with embedding-based similarity."""

    def __init__(self, similarity_threshold: float = 0.85, max_cache_size: int = 10000):
        self.cache = {}
        self.similarity_threshold = similarity_threshold
        self.max_cache_size = max_cache_size
        self._lock = asyncio.Lock()
        self.access_times = {}  # Track access times for LRU eviction
        self.hit_counts = {} # Track hit counts for hit rate calculation

    async def get_cached_response(self, query: str) -> Optional[Dict[str, Any]]:
        """
        Get cached response for query using semantic similarity.

        Args:
            query: Query to search for

        Returns:
            Cached response if similar query found, None otherwise
        """
        async with self._lock:
            # Check exact match first
            if query in self.cache:
                self.access_times[query] = time.time()
                self.hit_counts[query] = self.hit_counts.get(query, 0) + 1
                return self.cache[query]

            # Check for similar queries using embedding-based similarity
            try:
                from api.llm_client import LLMClient
                llm_client = LLMClient()
                query_embedding = await llm_client.get_embedding(query)
                
                best_match = None
                best_similarity = 0.0
                
                for cached_query in self.cache.keys():
                    cached_embedding = await llm_client.get_embedding(cached_query)
                    similarity = self._calculate_cosine_similarity(query_embedding, cached_embedding)
                    
                    if similarity > self.similarity_threshold and similarity > best_similarity:
                        best_similarity = similarity
                        best_match = cached_query
                
                if best_match:
                    self.access_times[best_match] = time.time()
                    self.hit_counts[best_match] = self.hit_counts.get(best_match, 0) + 1
                    logger.info(f"Semantic cache hit: '{query}' matched '{best_match}' (similarity: {best_similarity:.3f})")
                    return self.cache[best_match]
                    
            except Exception as e:
                logger.warning(f"Embedding-based similarity failed, falling back to word overlap: {e}")
                # Fallback to word overlap method
                query_words = set(query.lower().split())
                for cached_query in self.cache.keys():
                    cached_words = set(cached_query.lower().split())
                    overlap = len(query_words & cached_words) / len(query_words | cached_words)
                    if overlap > 0.7:  # 70% word overlap threshold
                        self.access_times[cached_query] = time.time()
                        self.hit_counts[cached_query] = self.hit_counts.get(cached_query, 0) + 1
                        return self.cache[cached_query]

            return None

    def _calculate_cosine_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """
        Calculate cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Cosine similarity score (0.0 to 1.0)
        """
        try:
            import numpy as np
            
            # Convert to numpy arrays
            vec1 = np.array(embedding1)
            vec2 = np.array(embedding2)
            
            # Calculate cosine similarity
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
                
            similarity = dot_product / (norm1 * norm2)
            return float(similarity)
            
        except Exception as e:
            logger.error(f"Cosine similarity calculation failed: {e}")
            return 0.0

    async def cache_response(self, query: str, response: Dict[str, Any]):
        """
        Cache response with LRU eviction.

        Args:
            query: Query string
            response: Response to cache
        """
        async with self._lock:
            # Evict least recently used if cache is full
            if len(self.cache) >= self.max_cache_size:
                oldest_query = min(self.access_times.keys(), key=lambda k: self.access_times[k])
                del self.cache[oldest_query]
                del self.access_times[oldest_query]
                if oldest_query in self.hit_counts:
                    del self.hit_counts[oldest_query]

            self.cache[query] = response
            self.access_times[query] = time.time()
            self.hit_counts[query] = 0  # Initialize hit count

    async def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics with hit rate tracking."""
        async with self._lock:
            total_hits = sum(self.hit_counts.values())
            total_requests = len(self.access_times) + total_hits  # Rough estimate
            hit_rate = total_hits / max(total_requests, 1)
            
            return {
                "size": len(self.cache),
                "max_size": self.max_cache_size,
                "hit_rate": hit_rate,
                "total_hits": total_hits,
                "total_requests": total_requests,
                "oldest_entry": min(self.access_times.values()) if self.access_times else None,
                "newest_entry": max(self.access_times.values()) if self.access_times else None,
                "most_hit_query": max(self.hit_counts.items(), key=lambda x: x[1]) if self.hit_counts else None,
            }


class ResponseAggregator:
    """Enhanced response aggregator with proper citation integration."""

    def __init__(self):
        # Weighted confidence calculation based on agent importance
        self.agent_weights = {
            AgentType.RETRIEVAL: 0.25,  # Retrieval quality affects overall confidence
            AgentType.FACT_CHECK: 0.30,  # Fact-checking is critical for accuracy
            AgentType.SYNTHESIS: 0.35,  # Synthesis quality is most important
            AgentType.CITATION: 0.10,  # Citations add credibility but less weight
        }

    def aggregate_pipeline_results(
        self, results: Dict[AgentType, AgentResult], context: QueryContext
    ) -> Dict[str, Any]:
        """
        Aggregate results from all agents with enhanced citation handling.

        Args:
            results: Results from each agent
            context: Query context

        Returns:
            Aggregated response with proper citations
        """
        # Check if synthesis succeeded
        synthesis_result = results.get(AgentType.SYNTHESIS)
        if not synthesis_result or not synthesis_result.success:
            return self._create_partial_response(results, context)

        # Convert to standardized data models
        try:
            synthesis_data = synthesis_result.data
            answer = synthesis_data.answer
        except Exception as e:
            logger.warning(f"Failed to convert synthesis data: {e}")
            answer = synthesis_result.data.get("answer", synthesis_result.data.get("response", ""))

        # Integrate citations if available
        citation_result = results.get(AgentType.CITATION)
        bibliography = []
        cited_content = None
        if citation_result and citation_result.success:
            try:
                citation_data = citation_result.data
                cited_content = citation_data.cited_content
                bibliography = citation_data.bibliography
                # Use cited content if available, otherwise use original answer
                if cited_content:
                    answer = cited_content
            except Exception as e:
                logger.warning(f"Failed to convert citation data: {e}")
                cited_content = citation_result.data.get("cited_content", "")
                bibliography = citation_result.data.get("bibliography", [])
                if cited_content:
                    answer = cited_content

        # Calculate weighted confidence
        confidence = self._calculate_weighted_confidence(results)

        # Compile token usage
        total_tokens = self._compile_token_usage(results)

        return {
            "success": True,
            "answer": answer,
            "cited_content": cited_content if cited_content else answer,
            "confidence": confidence,
            "bibliography": bibliography,
            "metadata": {
                "agent_results": {agent.value: result.success for agent, result in results.items()},
                "token_usage": total_tokens,
                "processing_time_ms": sum(result.execution_time_ms for result in results.values()),
                "trace_id": context.trace_id,
            },
        }

    def _create_partial_response(
        self, results: Dict[AgentType, AgentResult], context: QueryContext
    ) -> Dict[str, Any]:
        """Create response when synthesis fails."""
        return {
            "success": False,
            "answer": "Unable to generate a complete answer due to processing errors.",
            "confidence": 0.0,
            "citations": [],
            "bibliography": [],
            "metadata": {
                "partial_results": True,
                "agent_results": {agent.value: result.success for agent, result in results.items()},
                "errors": [result.error for result in results.values() if result.error],
                "trace_id": context.trace_id,
            },
        }

    def _calculate_weighted_confidence(self, results: Dict[AgentType, AgentResult]) -> float:
        """
        Calculate weighted confidence based on agent importance.

        Args:
            results: Results from each agent

        Returns:
            Weighted confidence score
        """
        total_weight = 0
        weighted_sum = 0

        for agent_type, result in results.items():
            if result.success:
                weight = self.agent_weights.get(agent_type, 0.1)
                weighted_sum += result.confidence * weight
                total_weight += weight

        if total_weight == 0:
            return 0.0

        return weighted_sum / total_weight

    def _compile_token_usage(self, results: Dict[AgentType, AgentResult]) -> Dict[str, int]:
        """
        Compile total token usage across all agents.

        Args:
            results: Results from each agent

        Returns:
            Total token usage by type
        """
        total_prompt = 0
        total_completion = 0

        for result in results.values():
            if result.token_usage:
                total_prompt += result.token_usage.get("prompt", 0)
                total_completion += result.token_usage.get("completion", 0)

        return {
            "prompt": total_prompt,
            "completion": total_completion,
            "total": total_prompt + total_completion,
        }
