# Local Development Setup Guide

## Overview
This guide provides instructions for setting up the Universal Knowledge Hub for local development using Python 3.13.5 without Docker containers.

## Prerequisites

### System Requirements
- **Python**: 3.13.5 or higher
- **Node.js**: 18.x or higher (for frontend development)
- **Git**: Latest version
- **Operating System**: Windows 10/11, macOS, or Linux

### Required Software
1. **Python 3.13.5**: Download from [python.org](https://www.python.org/downloads/)
2. **Node.js**: Download from [nodejs.org](https://nodejs.org/)
3. **Git**: Download from [git-scm.com](https://git-scm.com/)

## Installation Steps

### 1. Clone the Repository
```bash
git clone <repository-url>
cd universal-knowledge-hub
```

### 2. Set Up Python Environment

#### Create Virtual Environment
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS/Linux
python3 -m venv .venv
source .venv/bin/activate
```

#### Install Dependencies
```bash
# Upgrade pip
python -m pip install --upgrade pip

# Install core dependencies
pip install -r requirements.txt

# Install development dependencies
pip install -r requirements-dev.txt

# Install test dependencies
pip install -r requirements-test.txt
```

### 3. Set Up Frontend Environment

#### Install Node.js Dependencies
```bash
cd frontend
npm install
```

#### Set Up Environment Variables
```bash
# Copy environment template
cp env.example .env.local

# Edit .env.local with your configuration
# See API_CONFIGURATION.md for details
```

### 4. Configure Environment Variables

#### Backend Configuration
Create a `.env` file in the root directory:
```bash
# Copy template
cp env.template .env

# Edit .env with your settings
```

#### Required Environment Variables
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

# Redis Configuration
REDIS_URL=redis://localhost:6379

# Logging
LOG_LEVEL=INFO
```

### 5. Initialize Database and Services

#### Set Up SQLite Database
```bash
# Create data directory
mkdir -p data

# Initialize database (if using SQLAlchemy)
python -c "from api.database.models import Base, engine; Base.metadata.create_all(bind=engine)"
```

#### Install Redis (Optional for Caching)
```bash
# Windows (using WSL or Docker)
# Install Redis server

# macOS
brew install redis
brew services start redis

# Ubuntu/Debian
sudo apt-get install redis-server
sudo systemctl start redis-server
```

## Running the Application

### 1. Start Backend Server
```bash
# Activate virtual environment
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # macOS/Linux

# Start development server
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

### 2. Start Frontend Development Server
```bash
cd frontend
npm run dev
```

### 3. Access the Application
- **Backend API**: http://localhost:8000
- **Frontend**: http://localhost:3000
- **API Documentation**: http://localhost:8000/docs

## Development Workflow

### Code Quality Tools
```bash
# Format code
black .

# Lint code
flake8 .

# Type checking
mypy .

# Security audit
bandit -r .

# Run tests
pytest

# Run tests with coverage
pytest --cov=agents --cov=api --cov-report=html
```

### Pre-commit Hooks
```bash
# Install pre-commit hooks
pre-commit install

# Run all hooks
pre-commit run --all-files
```

## Testing

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

## Debugging

### Backend Debugging
```bash
# Run with debug logging
LOG_LEVEL=DEBUG uvicorn api.main:app --reload

# Run specific test with debug
pytest tests/test_specific.py -v -s
```

### Frontend Debugging
```bash
# Start with debug mode
cd frontend
npm run dev -- --debug
```

## Performance Monitoring

### Backend Monitoring
```bash
# Run performance tests
python tests/performance/locustfile.py

# Monitor with built-in tools
python -m api.monitoring.performance_monitor
```

### Frontend Monitoring
```bash
# Build for production analysis
cd frontend
npm run build
npm run analyze
```

## Troubleshooting

### Common Issues

#### Python Version Issues
```bash
# Verify Python version
python --version  # Should show 3.13.5

# If using pyenv
pyenv install 3.13.5
pyenv local 3.13.5
```

#### Dependency Issues
```bash
# Clear and reinstall dependencies
pip uninstall -r requirements.txt -y
pip install -r requirements.txt
```

#### Virtual Environment Issues
```bash
# Recreate virtual environment
rm -rf .venv
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

#### Port Conflicts
```bash
# Check what's using port 8000
lsof -i :8000  # macOS/Linux
netstat -ano | findstr :8000  # Windows

# Use different port
uvicorn api.main:app --reload --port 8001
```

### Getting Help
- Check the logs in the `logs/` directory
- Review the documentation in `documentation/`
- Run the diagnostic script: `python scripts/diagnose_setup.py`

## Next Steps
1. Review the [MAANG_CODING_STANDARDS.md](MAANG_CODING_STANDARDS.md)
2. Set up your IDE with the recommended extensions
3. Configure your API keys (see [GET_API_KEYS_GUIDE.md](GET_API_KEYS_GUIDE.md))
4. Run the full test suite to ensure everything is working
5. Start developing!

## Support
For additional help:
- Check the [documentation/](documentation/) directory
- Review [HUMAN_SETUP_GUIDE.md](HUMAN_SETUP_GUIDE.md)
- Run the setup verification script: `python scripts/verify_setup.py`
