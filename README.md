# Multi-Agent Knowledge Platform

A sophisticated multi-agent system for intelligent knowledge retrieval, verification, synthesis, and citation.

## Features

- **Multi-Agent Architecture**: Orchestrated system with specialized agents:
  - **Retrieval Agent**: Information gathering from multiple sources
  - **Fact-Check Agent**: Verification of claims and information
  - **Synthesis Agent**: Coherent response generation
  - **Citation Agent**: Proper source attribution and citation formatting

- **Multiple Execution Patterns**:
  - Simple pipeline execution
  - Fork-join for parallel processing
  - Scatter-gather for multi-domain queries
  - Feedback loops for quality improvement

- **FastAPI Integration**: RESTful API for query processing
- **Token Budget Management**: Efficient resource allocation
- **Semantic Caching**: Performance optimization
- **Comprehensive Testing**: Unit tests for core components

## Setup

1. **Create virtual environment**:
```bash
python3 -m venv venv
source venv/bin/activate
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

## Usage

### Command Line Interface

Run a single query:
```bash
python run_query.py
```

### FastAPI Server

Start the web server:
```bash
python -m uvicorn agents.lead_orchestrator:app --host 0.0.0.0 --port 8000
```

### API Endpoints

- **POST /query**: Process a query
  ```bash
  curl -X POST "http://localhost:8000/query" \
    -H "Content-Type: application/json" \
    -d '{"query": "What is machine learning?", "user_id": "test_user"}'
  ```

- **GET /health**: Health check
  ```bash
  curl http://localhost:8000/health
  ```

- **GET /metrics**: System metrics
  ```bash
  curl http://localhost:8000/metrics
  ```

## Testing

Run all tests:
```bash
python -m pytest tests/ -v
```

## Project Structure

```
.
├── agents/
│   ├── lead_orchestrator.py    # Main orchestrator and agent implementations
│   ├── synthesis_agent.py      # Standalone synthesis agent
│   ├── citation_agent.py       # Standalone citation agent
│   └── factcheck_agent.py      # Standalone fact-check agent
├── tests/
│   ├── test_lead_orchestrator.py
│   └── test_retrieval.py
├── requirements.txt
├── run_query.py                # CLI interface
└── README.md
```

## Architecture

The system follows a modular, agent-based architecture where:

1. **LeadOrchestrator** coordinates all agents and manages workflow
2. **Specialized Agents** handle specific tasks (retrieval, fact-checking, synthesis, citation)
3. **Message Broker** manages inter-agent communication
4. **Token Controller** manages resource allocation
5. **Cache Manager** optimizes performance through semantic caching
6. **Response Aggregator** combines results from multiple agents

## Configuration

The system is designed to be extensible with:
- Configurable agent parameters
- Multiple execution patterns
- Pluggable external services (vector DBs, LLMs, search engines)
- Comprehensive monitoring and metrics

## Status

✅ **Fixed Issues**:
- Dependencies installed correctly
- Constructor parameter issues resolved
- Import errors fixed
- FastAPI server working
- Tests passing

The application is now fully functional and ready for use!