"""
Response Aggregator - Merges and formats results from multiple agents.
"""

import logging
from typing import Dict, Any, List, Optional
from agents.base_agent import AgentResult, AgentType, QueryContext

logger = logging.getLogger(__name__)


class ResponseAggregator:
    """
    Merges and formats results from multiple agents into a coherent response.
    """
    
    def __init__(self):
        self.aggregation_strategies = {
            'pipeline': self._aggregate_pipeline_results,
            'fork_join': self._aggregate_fork_join_results,
            'scatter_gather': self._aggregate_scatter_gather_results,
            'feedback': self._aggregate_feedback_pipeline_results
        }
    
    def aggregate_results(self, results: Dict[AgentType, AgentResult], 
                         context: QueryContext, strategy: str = 'pipeline') -> Dict[str, Any]:
        """
        Aggregate results from multiple agents using the specified strategy.
        
        Args:
            results: Dictionary of agent results
            context: Query context
            strategy: Aggregation strategy to use
            
        Returns:
            Aggregated response
        """
        if strategy not in self.aggregation_strategies:
            logger.warning(f"Unknown aggregation strategy: {strategy}, using pipeline")
            strategy = 'pipeline'
        
        aggregator = self.aggregation_strategies[strategy]
        return aggregator(results, context)
    
    def _aggregate_pipeline_results(self, results: Dict[AgentType, AgentResult], 
                                   context: QueryContext) -> Dict[str, Any]:
        """
        Aggregate results from a sequential pipeline execution.
        
        Args:
            results: Results from pipeline agents
            context: Query context
            
        Returns:
            Aggregated response
        """
        # Pipeline results should flow from one agent to the next
        # The final result comes from the last agent in the pipeline
        synthesis_result = results.get(AgentType.SYNTHESIS)
        citation_result = results.get(AgentType.CITATION)
        
        if not synthesis_result or not synthesis_result.success:
            return self._create_error_response("Synthesis failed", context)
        
        # Extract the synthesized answer
        synthesis_data = synthesis_result.data
        answer = synthesis_data.get('answer', 'No answer generated')
        confidence = synthesis_result.confidence
        
        # Add citations if available
        citations = []
        if citation_result and citation_result.success:
            citations = citation_result.data.get('citations', [])
        
        # Add metadata from other agents
        metadata = {
            'retrieval_confidence': results.get(AgentType.RETRIEVAL, AgentResult(False, None)).confidence,
            'fact_check_confidence': results.get(AgentType.FACT_CHECK, AgentResult(False, None)).confidence,
            'synthesis_confidence': synthesis_result.confidence,
            'citation_confidence': citation_result.confidence if citation_result else 0.0,
            'total_execution_time': sum(r.execution_time_ms for r in results.values()),
            'total_tokens_used': self._sum_token_usage(results)
        }
        
        return {
            'query': context.query,
            'answer': answer,
            'confidence': confidence,
            'citations': citations,
            'metadata': metadata,
            'success': True
        }
    
    def _aggregate_fork_join_results(self, results: Dict[AgentType, AgentResult], 
                                    context: QueryContext) -> Dict[str, Any]:
        """
        Aggregate results from a fork-join execution pattern.
        
        Args:
            results: Results from parallel agents
            context: Query context
            
        Returns:
            Aggregated response
        """
        # Fork-join typically has multiple retrieval strategies run in parallel
        # followed by synthesis and citation
        retrieval_results = []
        for agent_type in [AgentType.RETRIEVAL]:
            if agent_type in results and results[agent_type].success:
                retrieval_results.append(results[agent_type])
        
        if not retrieval_results:
            return self._create_error_response("No retrieval results", context)
        
        # Merge retrieval results
        merged_docs = []
        for result in retrieval_results:
            docs = result.data.get('documents', [])
            merged_docs.extend(docs)
        
        # Deduplicate and rank documents
        unique_docs = self._deduplicate_documents(merged_docs)
        
        # Get synthesis result
        synthesis_result = results.get(AgentType.SYNTHESIS)
        if not synthesis_result or not synthesis_result.success:
            return self._create_error_response("Synthesis failed", context)
        
        answer = synthesis_result.data.get('answer', 'No answer generated')
        confidence = synthesis_result.confidence
        
        # Add citations
        citations = []
        citation_result = results.get(AgentType.CITATION)
        if citation_result and citation_result.success:
            citations = citation_result.data.get('citations', [])
        
        metadata = {
            'retrieval_strategies': len(retrieval_results),
            'documents_retrieved': len(merged_docs),
            'unique_documents': len(unique_docs),
            'synthesis_confidence': confidence,
            'total_execution_time': sum(r.execution_time_ms for r in results.values()),
            'total_tokens_used': self._sum_token_usage(results)
        }
        
        return {
            'query': context.query,
            'answer': answer,
            'confidence': confidence,
            'citations': citations,
            'metadata': metadata,
            'success': True
        }
    
    def _aggregate_scatter_gather_results(self, domain_results: List[Dict[str, Any]], 
                                        context: QueryContext) -> Dict[str, Any]:
        """
        Aggregate results from a scatter-gather execution pattern.
        
        Args:
            domain_results: Results from different domains
            context: Query context
            
        Returns:
            Aggregated response
        """
        if not domain_results:
            return self._create_error_response("No domain results", context)
        
        # Combine answers from different domains
        all_answers = []
        all_citations = []
        total_confidence = 0.0
        total_execution_time = 0
        
        for domain_result in domain_results:
            if domain_result.get('success'):
                all_answers.append(domain_result.get('answer', ''))
                all_citations.extend(domain_result.get('citations', []))
                total_confidence += domain_result.get('confidence', 0.0)
                total_execution_time += domain_result.get('metadata', {}).get('total_execution_time', 0)
        
        if not all_answers:
            return self._create_error_response("No successful domain results", context)
        
        # Combine answers (simple concatenation for now)
        combined_answer = " ".join(all_answers)
        avg_confidence = total_confidence / len(domain_results) if domain_results else 0.0
        
        # Deduplicate citations
        unique_citations = self._deduplicate_citations(all_citations)
        
        metadata = {
            'domains_processed': len(domain_results),
            'successful_domains': len([r for r in domain_results if r.get('success')]),
            'average_confidence': avg_confidence,
            'total_execution_time': total_execution_time,
            'total_citations': len(unique_citations)
        }
        
        return {
            'query': context.query,
            'answer': combined_answer,
            'confidence': avg_confidence,
            'citations': unique_citations,
            'metadata': metadata,
            'success': True
        }
    
    def _aggregate_feedback_pipeline_results(self, results: Dict[str, Any], 
                                           context: QueryContext) -> Dict[str, Any]:
        """
        Aggregate results from a feedback pipeline execution.
        
        Args:
            results: Results from feedback pipeline
            context: Query context
            
        Returns:
            Aggregated response
        """
        # Feedback pipeline includes multiple iterations
        iterations = results.get('iterations', [])
        final_result = results.get('final_result')
        
        if not final_result:
            return self._create_error_response("No final result from feedback pipeline", context)
        
        metadata = {
            'iterations': len(iterations),
            'improvement_rounds': len([i for i in iterations if i.get('improved')]),
            'final_confidence': final_result.get('confidence', 0.0),
            'total_execution_time': sum(i.get('execution_time', 0) for i in iterations)
        }
        
        return {
            'query': context.query,
            'answer': final_result.get('answer', 'No answer generated'),
            'confidence': final_result.get('confidence', 0.0),
            'citations': final_result.get('citations', []),
            'metadata': metadata,
            'success': True
        }
    
    def _create_error_response(self, error_message: str, context: QueryContext) -> Dict[str, Any]:
        """Create an error response."""
        return {
            'query': context.query,
            'answer': f"Error: {error_message}",
            'confidence': 0.0,
            'citations': [],
            'metadata': {'error': error_message},
            'success': False
        }
    
    def _deduplicate_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate documents based on content."""
        seen_contents = set()
        unique_docs = []
        
        for doc in documents:
            content_hash = hash(doc.get('content', ''))
            if content_hash not in seen_contents:
                seen_contents.add(content_hash)
                unique_docs.append(doc)
        
        return unique_docs
    
    def _deduplicate_citations(self, citations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate citations based on URL or title."""
        seen_urls = set()
        unique_citations = []
        
        for citation in citations:
            url = citation.get('url', '')
            if url and url not in seen_urls:
                seen_urls.add(url)
                unique_citations.append(citation)
        
        return unique_citations
    
    def _sum_token_usage(self, results: Dict[AgentType, AgentResult]) -> Dict[str, int]:
        """Sum token usage across all agents."""
        total_prompt = 0
        total_completion = 0
        
        for result in results.values():
            usage = result.token_usage
            total_prompt += usage.get('prompt', 0)
            total_completion += usage.get('completion', 0)
        
        return {
            'prompt': total_prompt,
            'completion': total_completion,
            'total': total_prompt + total_completion
        } 