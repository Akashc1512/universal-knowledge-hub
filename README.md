# 🧠 SarvanOM - Your Own Knowledge Hub Powered by AI

[![SarvanOM](https://img.shields.io/badge/SarvanOM-AI%20Knowledge%20Hub-blue?style=for-the-badge&logo=ai)](https://sarvanom.ai)
[![Python](https://img.shields.io/badge/Python-3.9+-blue?style=for-the-badge&logo=python)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green?style=for-the-badge&logo=fastapi)](https://fastapi.tiangolo.com)
[![Next.js](https://img.shields.io/badge/Next.js-15.4+-black?style=for-the-badge&logo=next.js)](https://nextjs.org)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

> **Your Own Knowledge Hub Powered by AI** - A comprehensive AI-powered knowledge platform that provides accurate, verifiable answers with high transparency and source citations.

## 🌟 Features

### 🤖 **Multi-Agent AI System**
- **Retrieval Agent**: Intelligent document search and retrieval
- **Fact-Check Agent**: Verification of claims and information accuracy
- **Synthesis Agent**: AI-powered answer generation and synthesis
- **Citation Agent**: Automatic source citation and attribution

### 🔍 **Advanced Search & Retrieval**
- Vector database integration (Pinecone, Qdrant)
- Elasticsearch for full-text search
- Knowledge graph integration (SPARQL)
- Semantic caching for improved performance

### 🛡️ **Security & Compliance**
- API key authentication
- Rate limiting and threat detection
- GDPR-compliant data handling
- Comprehensive audit logging

### 📊 **Monitoring & Analytics**
- Real-time Prometheus metrics
- Health monitoring and alerting
- Performance analytics
- Usage tracking and insights

### 🎨 **Modern Frontend**
- Next.js 15 with TypeScript
- Tailwind CSS for responsive design
- Heroicons for beautiful UI components
- Accessibility compliant (WCAG 2.1 AA)

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- Node.js 18+
- Docker (optional)

### Backend Setup
```bash
# Clone the repository
git clone https://github.com/Akashc1512/sarvanom.git
cd sarvanom

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp env.template .env
# Edit .env with your API keys and configuration

# Start the backend
python -m uvicorn api.main:app --host 0.0.0.0 --port 8002 --reload
```

### Frontend Setup
```bash
# Navigate to frontend directory
cd frontend

# Install dependencies
npm install

# Set up environment variables
cp env.example .env.local
# Edit .env.local with your backend API URL

# Start the frontend
npm run dev
```

### Docker Setup
```bash
# Using Docker Compose
docker-compose up -d

# Or build and run manually
docker build -t sarvanom .
docker run -p 8002:8002 sarvanom
```

## 📖 API Documentation

Once the backend is running, visit:
- **API Docs**: http://localhost:8002/docs
- **Health Check**: http://localhost:8002/health
- **Metrics**: http://localhost:8002/metrics

### Example API Usage
```bash
# Query the knowledge hub
curl -X POST "http://localhost:8002/query" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-api-key" \
  -d '{
    "query": "What is artificial intelligence?",
    "max_tokens": 1000,
    "confidence_threshold": 0.8
  }'
```

## 🏗️ Architecture

### Backend Components
- **FastAPI Application**: RESTful API with authentication
- **Multi-Agent Orchestrator**: Coordinates AI agents
- **Security Layer**: Threat detection and rate limiting
- **Caching System**: Multi-level semantic caching
- **Monitoring**: Prometheus metrics and health checks

### Frontend Components
- **Next.js 15**: React framework with App Router
- **TypeScript**: Full type safety
- **Tailwind CSS**: Utility-first styling
- **Component Library**: Modular, reusable UI components

### Infrastructure
- **Docker**: Multi-stage builds for production
- **Kubernetes**: Complete orchestration manifests
- **CI/CD**: GitHub Actions with automated testing
- **Monitoring**: Prometheus metrics and Grafana dashboards

## 🔧 Configuration

### Environment Variables
```bash
# Backend Configuration
UKP_HOST=0.0.0.0
UKP_PORT=8002
OPENAI_API_KEY=your-openai-key
ANTHROPIC_API_KEY=your-anthropic-key

# Database Configuration
VECTOR_DB_URL=your-vector-db-url
ELASTICSEARCH_URL=your-elasticsearch-url
REDIS_URL=your-redis-url

# Security Configuration
API_KEY_SECRET=your-api-key-secret
RATE_LIMIT_PER_MINUTE=60
THREAT_DETECTION_ENABLED=true

# Frontend Configuration
NEXT_PUBLIC_API_BASE_URL=http://localhost:8002
```

## 📊 Performance & Scalability

- **Concurrent Requests**: Up to 100 simultaneous queries
- **Response Time**: < 5 seconds for complex queries
- **Caching**: Multi-level semantic caching
- **Load Balancing**: Kubernetes-ready deployment
- **Monitoring**: Real-time performance metrics

## 🛡️ Security Features

- **Authentication**: API key-based authentication
- **Rate Limiting**: Configurable request limits
- **Threat Detection**: SQL injection, XSS, path traversal protection
- **Data Privacy**: GDPR-compliant data handling
- **Audit Logging**: Comprehensive security event logging

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Setup
```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests/

# Run linting
flake8 api/
black api/
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🌐 Links

- **Website**: [https://sarvanom.ai](https://sarvanom.ai)
- **Documentation**: [https://docs.sarvanom.ai](https://docs.sarvanom.ai)
- **API Reference**: [https://api.sarvanom.ai](https://api.sarvanom.ai)
- **GitHub**: [https://github.com/Akashc1512/sarvanom](https://github.com/Akashc1512/sarvanom)

## 🙏 Acknowledgments

- Built with FastAPI and Next.js
- Powered by OpenAI and Anthropic AI
- Vector search by Pinecone and Qdrant
- Monitoring with Prometheus and Grafana

---

**SarvanOM** - Your Own Knowledge Hub Powered by AI 🧠✨
