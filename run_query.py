#!/usr/bin/env python3
"""
Universal Knowledge Platform - CLI Entry Point

This script demonstrates the complete system by running a query through all agents.
"""

import asyncio
import sys
import logging
from typing import Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Add the project root to Python path
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agents.lead_orchestrator import LeadOrchestrator
from agents.retrieval_agent import RetrievalAgent
from agents.factcheck_agent import FactCheckAgent
from agents.synthesis_agent import SynthesisAgent
from agents.citation_agent import CitationAgent

logger = logging.getLogger(__name__)


async def run_query(query: str, user_context: Optional[dict] = None) -> dict:
    """
    Run a query through the complete Universal Knowledge Platform.
    
    Args:
        query: The query to process
        user_context: Optional user context
        
    Returns:
        Complete response with answer, confidence, and citations
    """
    logger.info(f"Processing query: {query}")
    
    # Initialize the orchestrator
    orchestrator = LeadOrchestrator()
    
    try:
        # Process the query through all agents
        response = await orchestrator.process_query(query, user_context)
        
        logger.info(f"Query processed successfully")
        logger.info(f"Confidence: {response.get('confidence', 0.0):.2f}")
        logger.info(f"Execution time: {response.get('metadata', {}).get('total_execution_time', 0)}ms")
        
        return response
        
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        return {
            'query': query,
            'answer': f"Error: {str(e)}",
            'confidence': 0.0,
            'citations': [],
            'success': False,
            'error': str(e)
        }


async def test_individual_agents():
    """Test individual agents to ensure they work correctly."""
    logger.info("Testing individual agents...")
    
    # Test RetrievalAgent
    retrieval_agent = RetrievalAgent()
    retrieval_result = await retrieval_agent.hybrid_retrieve("quantum computing", entities=["quantum"])
    logger.info(f"RetrievalAgent: Retrieved {len(retrieval_result.documents)} documents")
    
    # Test FactCheckAgent
    factcheck_agent = FactCheckAgent()
    factcheck_result = await factcheck_agent.verify_claims(
        ["Quantum computing uses quantum bits"], 
        [{"title": "Quantum Computing Basics", "content": "Quantum computing uses quantum bits"}]
    )
    logger.info(f"FactCheckAgent: Verified {len(factcheck_result)} claims")
    
    # Test SynthesisAgent
    from agents.base_agent import QueryContext
    synthesis_agent = SynthesisAgent()
    test_context = QueryContext(query="What is quantum computing?")
    synthesis_result = await synthesis_agent.synthesize(
        factcheck_result, 
        test_context, 
        {"style": "concise"}
    )
    logger.info(f"SynthesisAgent: Generated answer with {synthesis_result.get('verified_claims', 0)} verified claims")
    
    # Test CitationAgent
    citation_agent = CitationAgent()
    citation_result = await citation_agent.generate_citations(
        synthesis_result.get('answer', ''),
        [{"title": "Source", "author": "Author", "url": "http://example.com", "date": "2024"}],
        "APA"
    )
    logger.info(f"CitationAgent: Generated {len(citation_result.get('citations', []))} citations")


def print_response(response: dict):
    """Print a formatted response."""
    print("\n" + "="*80)
    print("UNIVERSAL KNOWLEDGE PLATFORM - QUERY RESPONSE")
    print("="*80)
    
    print(f"\nQuery: {response.get('query', 'Unknown')}")
    print(f"Success: {response.get('success', False)}")
    
    if response.get('success'):
        print(f"\nAnswer:")
        print(f"{response.get('answer', 'No answer generated')}")
        
        print(f"\nConfidence: {response.get('confidence', 0.0):.2f}")
        
        citations = response.get('citations', [])
        if citations:
            print(f"\nCitations ({len(citations)}):")
            for i, citation in enumerate(citations, 1):
                if isinstance(citation, dict):
                    print(f"  {i}. {citation.get('text', 'No citation text')}")
                else:
                    print(f"  {i}. {citation}")
        
        metadata = response.get('metadata', {})
        if metadata:
            print(f"\nMetadata:")
            for key, value in metadata.items():
                print(f"  {key}: {value}")
    else:
        print(f"\nError: {response.get('error', 'Unknown error')}")
    
    print("\n" + "="*80)


async def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        # Default query if none provided
        query = "What are the latest advances in quantum computing?"
        print(f"No query provided, using default: {query}")
    else:
        query = " ".join(sys.argv[1:])
    
    print("Universal Knowledge Platform - MVP Demo")
    print("="*50)
    
    # Test individual agents first
    await test_individual_agents()
    
    # Run the complete query
    response = await run_query(query)
    
    # Print the response
    print_response(response)


if __name__ == "__main__":
    asyncio.run(main())
