import asyncio
from agents.lead_orchestrator import LeadOrchestrator

async def main():
    orchestrator = LeadOrchestrator(config={})
    plan = orchestrator.analyze_query("Explain RAG systems")
    print(plan)

if __name__ == "__main__":
    asyncio.run(main())
