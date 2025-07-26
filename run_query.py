import asyncio
from agents.lead_orchestrator import LeadOrchestrator

async def main():
    orchestrator = LeadOrchestrator()
    
    # Process a query using the correct method
    result = await orchestrator.process_query("Explain RAG systems")
    
    print("Query result:")
    print(f"Success: {result.get('success', False)}")
    print(f"Response: {result.get('response', 'No response generated')}")
    print(f"Confidence: {result.get('confidence', 0):.2f}")
    print(f"Execution time: {result.get('execution_time_ms', 0)}ms")
    
    if result.get('error'):
        print(f"Error: {result['error']}")

if __name__ == "__main__":
    asyncio.run(main())
