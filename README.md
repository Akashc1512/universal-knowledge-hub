# Universal Knowledge Platform

A production-ready AI platform that uses a multi-agent pipeline to provide comprehensive, well-cited answers to complex questions.

**🌐 Domain**: [sarvanom.com](https://sarvanom.com)

## 🚀 Current Status

**Phase**: Working Prototype → Production-Ready AI Platform

The Universal Knowledge Platform is currently a **functional prototype** with a solid foundation ready for AI integration. The system includes:

### ✅ What's Working Now

- **Multi-Agent Architecture**: Functional pipeline with Retrieval, Fact-Check, Synthesis, and Citation agents
- **Modern Web Interface**: React/Next.js frontend with real-time feedback and analytics
- **RESTful API**: FastAPI backend with comprehensive error handling and validation
- **Python 3.13.5**: Latest Python version with enhanced performance and features
- **CI/CD Pipeline**: Automated testing, linting, security scanning, and quality gates
- **Security Foundation**: Input validation, rate limiting, security scanning, and health checks
- **Monitoring**: Health checks, logging, and basic analytics dashboard

### 🔄 Next Steps: AI Integration

The platform is designed to integrate with real AI services. See [ROADMAP.md](./ROADMAP.md) for detailed development phases:

1. **Phase 1**: OpenAI API integration for embeddings and answer generation
2. **Phase 2**: Vector database (Pinecone) for document storage and retrieval
3. **Phase 3**: Expert validation system and advanced UI features
4. **Phase 4**: Production scaling and enterprise features

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Frontend      │    │   Backend       │    │   AI Services   │
│   (Next.js)     │◄──►│   (FastAPI)     │◄──►│   (OpenAI)      │
│                 │    │                 │    │                 │
│ • Query Form    │    │ • Multi-Agent   │    │ • Embeddings    │
│ • Answer Display│    │   Pipeline      │    │ • Text Gen      │
│ • Analytics     │    │ • Rate Limiting │    │ • Web Search    │
│ • Expert UI     │    │ • Security      │    │ • Fact Check    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Vector DB     │    │   SQLite/Redis  │    │   Knowledge     │
│   (Pinecone)    │    │   (Analytics)   │    │   Sources       │
│                 │    │                 │    │                 │
│ • Document      │    │ • User Data     │    │ • Wikipedia     │
│   Storage       │    │ • Query History │    │ • Academic DBs  │
│ • Semantic      │    │ • Analytics     │    │ • News APIs     │
│   Search        │    │ • Feedback      │    │ • Expert Data   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 🚀 Quick Start

### Prerequisites

- **Python 3.13.5** (required)
- **Node.js 18+** (for frontend development)
- **Git** (for version control)

### Development Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-org/universal-knowledge-hub.git
   cd universal-knowledge-hub
   ```

2. **Set up Python environment**:
   ```bash
   # Windows (PowerShell)
   .\scripts\setup_python313.ps1
   
   # macOS/Linux
   chmod +x scripts/setup_python313.sh
   ./scripts/setup_python313.sh
   ```

3. **Configure environment variables**:
   ```bash
   # Copy environment template
   cp env.template .env
   
   # Edit .env with your API keys
   # See GET_API_KEYS_GUIDE.md for required keys
   ```

4. **Start development servers**:
   ```bash
   # Activate virtual environment
   .venv\Scripts\activate  # Windows
   source .venv/bin/activate  # macOS/Linux
   
   # Start backend
   uvicorn api.main:app --reload --port 8000
   
   # Start frontend (in new terminal)
   cd frontend
   npm run dev
   ```

5. **Access the application**:
   - Frontend: http://localhost:3000
   - Backend API: http://localhost:8000
   - API Documentation: http://localhost:8000/docs

### Local Development

1. **Backend Development**:
   ```bash
   # Activate virtual environment
   .venv\Scripts\activate  # Windows
   source .venv/bin/activate  # macOS/Linux
   
   # Run tests
   pytest
   
   # Run with coverage
   pytest --cov=agents --cov=api --cov-report=html
   
   # Start development server
   uvicorn api.main:app --reload --port 8000
   ```

2. **Frontend Development**:
   ```bash
   cd frontend
   
   # Install dependencies
   npm install
   
   # Start development server
   npm run dev
   
   # Run tests
   npm test
   
   # Build for production
   npm run build
   ```

3. **Code Quality**:
   ```bash
   # Format code
   black .
   
   # Lint code
   flake8 .
   
   # Type checking
   mypy .
   
   # Security audit
   bandit -r .
   ```

## 📚 Documentation

- **[LOCAL_DEV_SETUP.md](./LOCAL_DEV_SETUP.md)**: Detailed local development setup
- **[MAANG_CODING_STANDARDS.md](./MAANG_CODING_STANDARDS.md)**: Coding standards and best practices
- **[GET_API_KEYS_GUIDE.md](./GET_API_KEYS_GUIDE.md)**: How to get required API keys
- **[AUTHENTICATION_GUIDE.md](./AUTHENTICATION_GUIDE.md)**: Authentication and security setup
- **[DEPLOYMENT_GUIDE.md](./DEPLOYMENT_GUIDE.md)**: Production deployment guide
- **[ROADMAP.md](./ROADMAP.md)**: Development roadmap and milestones

## 🧪 Testing

### Run All Tests
```bash
# Unit tests
pytest

# Integration tests
pytest -m integration

# Performance tests
pytest tests/performance/

# Bulletproof test suite
python tests/run_bulletproof_tests.py
```

### Test Coverage
```bash
# Generate coverage report
pytest --cov=agents --cov=api --cov-report=html

# View coverage report
# Open htmlcov/index.html in browser
```

## 🔧 Configuration

### Environment Variables
Create a `.env` file with the following variables:

```env
# API Keys (see GET_API_KEYS_GUIDE.md)
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key
PINECONE_API_KEY=your_pinecone_key
ELASTICSEARCH_URL=your_elasticsearch_url

# Database Configuration
DATABASE_URL=sqlite:///./data/app.db

# Security
SECRET_KEY=your_secret_key
JWT_SECRET_KEY=your_jwt_secret

# Redis Configuration (optional)
REDIS_URL=redis://localhost:6379

# Logging
LOG_LEVEL=INFO
```

## 🚀 Deployment

### Production Deployment
See [DEPLOYMENT_GUIDE.md](./DEPLOYMENT_GUIDE.md) for detailed production deployment instructions.

### Docker Deployment (Optional)
For containerized deployment, see the infrastructure directory for Kubernetes and Terraform configurations.

## 🤝 Contributing

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/amazing-feature`
3. **Follow coding standards**: See [MAANG_CODING_STANDARDS.md](./MAANG_CODING_STANDARDS.md)
4. **Write tests**: Ensure all new features have comprehensive tests
5. **Commit changes**: `git commit -m 'Add amazing feature'`
6. **Push to branch**: `git push origin feature/amazing-feature`
7. **Open a Pull Request**

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

- **Documentation**: Check the `documentation/` directory
- **Issues**: Create an issue on GitHub
- **Discussions**: Use GitHub Discussions for questions
- **Setup Help**: Run `python scripts/verify_setup.py` for diagnostics

## 🏆 Acknowledgments

- **FastAPI** for the excellent web framework
- **Next.js** for the React framework
- **OpenAI** for AI capabilities
- **Pinecone** for vector database
- **MAANG-level standards** for code quality
