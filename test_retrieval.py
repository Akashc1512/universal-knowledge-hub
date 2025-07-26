import asyncio
from agents.retrieval_agent import RetrievalAgent

async def main():
    agent = RetrievalAgent()
    result = await agent.hybrid_retrieve("Climate change impact on agriculture")
    print("Top documents:")
    for doc in result.documents[:3]:
        print(f"- [{doc.source}] {doc.content[:100]}... (score: {doc.score:.2f})")

if __name__ == "__main__":
    asyncio.run(main())
