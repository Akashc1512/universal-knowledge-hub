# Universal Knowledge Platform (UKP)

A next-generation AI-driven knowledge hub that unifies search engine retrieval, Wikipedia/knowledge graph data, and LLM reasoning into a single system to deliver **expert-validated, contextually-aware answers** with high accuracy and transparency.

## Problem Definition & Solution Overview

The **Universal Knowledge Platform (UKP)** addresses the challenge of providing accurate, verifiable answers by combining multiple information sources and AI agents. Key features include:

* **Retrieval-Augmented Generation (RAG)**: Hybrid search combining vector similarity, keyword BM25, and knowledge graph queries
* **Multi-Agent Orchestration**: Specialized AI agents (Retrieval, Fact-Check, Synthesis, Citation) collaborate in a pipeline
* **Knowledge Graph Integration**: Uses Wikidata/DBpedia via SPARQL to fetch structured facts
* **Expert-in-the-Loop Validation**: Supports human expert review workflows (planned for future iterations)
* **Transparency & Trust**: Every answer includes source citations, confidence scores, and provenance tracking

## Architecture

The MVP architecture is modular with distinct layers:

* **Query Intelligence Layer**: Interprets user queries (intent classification, entity recognition, complexity scoring)
* **Retrieval Module (RAG + Graph)**: Hybrid retrieval pipeline combining semantic vector search, keyword search, and knowledge graph queries
* **LLM Orchestration Engine**: LeadOrchestrator agent coordinates multiple specialized sub-agents
* **Agent Pool**: Four main agents handle stages of answering (Retrieval, Fact-Check, Synthesis, Citation)
* **Memory & Caching**: Short-term caches with semantic similarity for query reuse
* **API & Interface Layer**: FastAPI-based web service (planned for future iterations)

## Project Structure

```
universal-knowledge-hub/
├── README.md                         # Project overview and usage instructions
├── requirements.txt                  # Python dependencies
├── run_query.py                     # CLI entry point for testing
├── agents/                          # Agent implementations
│   ├── __init__.py
│   ├── retrieval_agent.py           # Hybrid search implementation
│   ├── factcheck_agent.py           # Claim verification
│   ├── synthesis_agent.py           # Answer construction
│   ├── citation_agent.py            # Citation generation
│   └── lead_orchestrator.py         # Multi-agent coordination
├── architecture/                     # System architecture components
│   ├── multi_agent_patterns.py      # Orchestration patterns
│   ├── orchestration.py             # Core orchestration logic
│   ├── query_intelligence.py        # Query analysis and routing
│   └── rag_graph_rag.py            # RAG + Knowledge Graph integration
├── prompts/                         # Prompt templates
│   ├── query_processing.txt         # Query analysis prompts
│   ├── retrieval.txt                # Search and retrieval prompts
│   ├── synthesis.txt                # Answer synthesis prompts
│   ├── fact_check.txt               # Fact verification prompts
│   ├── citation.txt                 # Citation generation prompts
│   └── factcheck_prompts.yaml       # Structured fact-checking prompts
├── tests/                           # Unit and integration tests
│   ├── __init__.py
│   ├── test_retrieval.py            # Retrieval agent tests
│   ├── test_factcheck.py            # Fact-check agent tests
│   └── test_lead_orchestrator.py    # Orchestrator tests
└── .gitignore                       # Git ignore patterns
```

## Setup Instructions

### Prerequisites

* Python 3.9+
* Git

### Local Development Setup

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd universal-knowledge-hub
   ```

2. **Create and activate virtual environment:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Run a test query:**
   ```bash
   python run_query.py "What are the latest advances in quantum computing?"
   ```

### Docker Setup

1. **Build the Docker image:**
   ```bash
   docker build -t universal-knowledge-platform:latest .
   ```

2. **Run the container:**
   ```bash
   docker run --rm -it universal-knowledge-platform:latest
   ```

### API Service Setup

1. **Start the FastAPI server:**
   ```bash
   python start_api.py
   ```
   
   Or use the management script:
   ```bash
   python manage_api.py start
   ```

2. **Test the API endpoints:**
   ```bash
   python test_api_simple.py
   ```

3. **Check server status:**
   ```bash
   python manage_api.py status
   ```

4. **Access the API documentation:**
   - Swagger UI: http://localhost:8002/docs
   - ReDoc: http://localhost:8002/redoc
   - Health Check: http://localhost:8002/health

## Usage Examples

### Basic Query Processing

```python
from agents.lead_orchestrator import LeadOrchestrator
import asyncio

async def test_query():
    orchestrator = LeadOrchestrator()
    response = await orchestrator.process_query("What is the capital of France?")
    print(f"Answer: {response['response']}")
    print(f"Confidence: {response['confidence']}")
    print(f"Citations: {response['citations']}")

asyncio.run(test_query())
```

### Individual Agent Testing

```python
from agents.retrieval_agent import RetrievalAgent
import asyncio

async def test_retrieval():
    agent = RetrievalAgent()
    results = await agent.hybrid_retrieve("quantum computing", entities=["quantum"])
    for doc in results:
        print(f"Document: {doc.content}")

asyncio.run(test_retrieval())
```

## Development Status

### ✅ Completed (Days 0-8)
- [x] Project structure and modular architecture
- [x] RetrievalAgent with hybrid search (vector + keyword + graph)
- [x] FactCheckAgent with claim decomposition and cross-source verification
- [x] LeadOrchestrator with multi-agent coordination patterns
- [x] Basic caching and token budget management
- [x] Prompt templates for all agent types
- [x] Unit tests for core components

### 🔄 In Progress
- [ ] Integration with real vector databases (Qdrant, Pinecone)
- [ ] Elasticsearch integration for keyword search
- [ ] SPARQL endpoint integration for knowledge graphs
- [ ] LLM API integration (OpenAI, Anthropic)

### 📋 Planned (Next Iterations)
- [ ] Expert review workflow and UI
- [ ] FastAPI web service
- [x] ~~React frontend~~ (Removed - backend-only project)
- [ ] Advanced query intelligence (NLU pipeline)
- [ ] Performance monitoring and logging
- [ ] Cloud deployment (AWS, Azure)

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Testing

Run the test suite:

```bash
python -m pytest tests/
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Built with modern Python async/await patterns
- Inspired by multi-agent systems and RAG architectures
- Designed for enterprise-grade scalability and reliability
