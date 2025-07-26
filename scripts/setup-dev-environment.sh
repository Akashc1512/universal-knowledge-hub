#!/bin/bash

# Universal Knowledge Platform - Development Environment Setup
# This script sets up the complete development environment for team members

set -e

echo "🚀 Setting up Universal Knowledge Platform Development Environment"
echo "================================================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if running on Windows
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "cygwin" ]]; then
    print_warning "Running on Windows. Some features may need manual setup."
fi

# Check prerequisites
print_status "Checking prerequisites..."

# Check Python version
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
    print_success "Python $PYTHON_VERSION found"
else
    print_error "Python 3.9+ is required but not installed"
    exit 1
fi

# Check Git
if command -v git &> /dev/null; then
    print_success "Git found"
else
    print_error "Git is required but not installed"
    exit 1
fi

# Check Docker
if command -v docker &> /dev/null; then
    print_success "Docker found"
else
    print_warning "Docker not found. Container features will be limited."
fi

# Check Node.js (for frontend development)
if command -v node &> /dev/null; then
    NODE_VERSION=$(node --version)
    print_success "Node.js $NODE_VERSION found"
else
    print_warning "Node.js not found. Frontend development will be limited."
fi

# Create virtual environment
print_status "Setting up Python virtual environment..."

if [ ! -d ".venv" ]; then
    python3 -m venv .venv
    print_success "Virtual environment created"
else
    print_status "Virtual environment already exists"
fi

# Activate virtual environment
source .venv/bin/activate

# Upgrade pip
print_status "Upgrading pip..."
pip install --upgrade pip

# Install Python dependencies
print_status "Installing Python dependencies..."
pip install -r requirements.txt

# Install development dependencies
print_status "Installing development dependencies..."
pip install pytest pytest-cov pytest-asyncio black flake8 mypy

# Install pre-commit hooks
print_status "Setting up pre-commit hooks..."
pip install pre-commit
pre-commit install

# Create local configuration
print_status "Creating local configuration..."

# Create .env file if it doesn't exist
if [ ! -f ".env" ]; then
    cat > .env << EOF
# Universal Knowledge Platform - Local Development Configuration

# Database Configuration
DATABASE_URL=postgresql://ukp_user:password@localhost:5432/ukp_dev
REDIS_URL=redis://localhost:6379/0

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
LOG_LEVEL=DEBUG
ENVIRONMENT=development

# AI Tool Configuration
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# Search Configuration
ELASTICSEARCH_URL=http://localhost:9200
VECTOR_DIMENSION=768

# Monitoring
ENABLE_METRICS=true
METRICS_PORT=8001
EOF
    print_success "Created .env file"
else
    print_status ".env file already exists"
fi

# Create local directories
print_status "Creating local directories..."
mkdir -p logs
mkdir -p data
mkdir -p temp
mkdir -p docs/generated

# Set up Git hooks
print_status "Setting up Git hooks..."
if [ ! -f ".git/hooks/pre-commit" ]; then
    cat > .git/hooks/pre-commit << 'EOF'
#!/bin/bash
# Pre-commit hook for Universal Knowledge Platform

echo "Running pre-commit checks..."

# Run linting
echo "Running black..."
black --check --diff .

echo "Running flake8..."
flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics

echo "Running mypy..."
mypy . --ignore-missing-imports

# Run tests
echo "Running tests..."
pytest tests/ -v --tb=short

echo "Pre-commit checks completed successfully!"
EOF
    chmod +x .git/hooks/pre-commit
    print_success "Git pre-commit hook installed"
fi

# Create development database setup script
print_status "Creating database setup script..."
cat > scripts/setup-database.sh << 'EOF'
#!/bin/bash

# Database setup script for Universal Knowledge Platform

echo "Setting up development database..."

# Check if PostgreSQL is running
if ! pg_isready -h localhost -p 5432 > /dev/null 2>&1; then
    echo "PostgreSQL is not running. Please start PostgreSQL first."
    exit 1
fi

# Create database and user
psql -h localhost -U postgres << EOF
CREATE DATABASE ukp_dev;
CREATE USER ukp_user WITH PASSWORD 'password';
GRANT ALL PRIVILEGES ON DATABASE ukp_dev TO ukp_user;
\c ukp_dev
GRANT ALL ON SCHEMA public TO ukp_user;
EOF

echo "Database setup completed!"
EOF
chmod +x scripts/setup-database.sh

# Create Docker Compose for local development
print_status "Creating Docker Compose configuration..."
cat > docker-compose.dev.yml << 'EOF'
version: '3.8'

services:
  # PostgreSQL Database
  postgres:
    image: postgres:15
    container_name: ukp-postgres-dev
    environment:
      POSTGRES_DB: ukp_dev
      POSTGRES_USER: ukp_user
      POSTGRES_PASSWORD: password
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U ukp_user -d ukp_dev"]
      interval: 10s
      timeout: 5s
      retries: 5

  # Redis Cache
  redis:
    image: redis:7-alpine
    container_name: ukp-redis-dev
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 5s
      retries: 5

  # Elasticsearch for Search
  elasticsearch:
    image: docker.elastic.co/elasticsearch/elasticsearch:8.8.0
    container_name: ukp-elasticsearch-dev
    environment:
      - discovery.type=single-node
      - xpack.security.enabled=false
      - "ES_JAVA_OPTS=-Xms512m -Xmx512m"
    ports:
      - "9200:9200"
    volumes:
      - elasticsearch_data:/usr/share/elasticsearch/data
    healthcheck:
      test: ["CMD-SHELL", "curl -f http://localhost:9200/_cluster/health || exit 1"]
      interval: 30s
      timeout: 10s
      retries: 5

  # Prometheus for Monitoring
  prometheus:
    image: prom/prometheus:latest
    container_name: ukp-prometheus-dev
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus-config.yaml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/etc/prometheus/console_libraries'
      - '--web.console.templates=/etc/prometheus/consoles'
      - '--storage.tsdb.retention.time=200h'
      - '--web.enable-lifecycle'

  # Grafana for Visualization
  grafana:
    image: grafana/grafana:latest
    container_name: ukp-grafana-dev
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana_data:/var/lib/grafana

volumes:
  postgres_data:
  redis_data:
  elasticsearch_data:
  prometheus_data:
  grafana_data:
EOF

# Create VS Code settings
print_status "Setting up VS Code configuration..."
mkdir -p .vscode
cat > .vscode/settings.json << 'EOF'
{
    "python.defaultInterpreterPath": "./.venv/bin/python",
    "python.linting.enabled": true,
    "python.linting.pylintEnabled": false,
    "python.linting.flake8Enabled": true,
    "python.formatting.provider": "black",
    "python.testing.pytestEnabled": true,
    "python.testing.unittestEnabled": false,
    "python.testing.pytestArgs": [
        "tests"
    ],
    "editor.formatOnSave": true,
    "editor.codeActionsOnSave": {
        "source.organizeImports": true
    },
    "files.exclude": {
        "**/__pycache__": true,
        "**/*.pyc": true,
        ".venv": true,
        "logs": true,
        "temp": true
    },
    "python.analysis.extraPaths": [
        "."
    ]
}
EOF

# Create launch configuration for debugging
cat > .vscode/launch.json << 'EOF'
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Python: Current File",
            "type": "python",
            "request": "launch",
            "program": "${file}",
            "console": "integratedTerminal",
            "cwd": "${workspaceFolder}",
            "env": {
                "PYTHONPATH": "${workspaceFolder}"
            }
        },
        {
            "name": "Python: API Server",
            "type": "python",
            "request": "launch",
            "program": "start_api.py",
            "console": "integratedTerminal",
            "cwd": "${workspaceFolder}",
            "env": {
                "PYTHONPATH": "${workspaceFolder}"
            }
        },
        {
            "name": "Python: Test Current File",
            "type": "python",
            "request": "launch",
            "module": "pytest",
            "args": [
                "${file}",
                "-v"
            ],
            "console": "integratedTerminal",
            "cwd": "${workspaceFolder}"
        }
    ]
}
EOF

# Create tasks for VS Code
cat > .vscode/tasks.json << 'EOF'
{
    "version": "2.0.0",
    "tasks": [
        {
            "label": "Run Tests",
            "type": "shell",
            "command": "pytest",
            "args": [
                "tests/",
                "-v"
            ],
            "group": "test",
            "presentation": {
                "echo": true,
                "reveal": "always",
                "focus": false,
                "panel": "shared"
            }
        },
        {
            "label": "Run Linting",
            "type": "shell",
            "command": "flake8",
            "args": [
                ".",
                "--count",
                "--select=E9,F63,F7,F82",
                "--show-source",
                "--statistics"
            ],
            "group": "build",
            "presentation": {
                "echo": true,
                "reveal": "always",
                "focus": false,
                "panel": "shared"
            }
        },
        {
            "label": "Format Code",
            "type": "shell",
            "command": "black",
            "args": [
                "."
            ],
            "group": "build",
            "presentation": {
                "echo": true,
                "reveal": "always",
                "focus": false,
                "panel": "shared"
            }
        },
        {
            "label": "Start Development Services",
            "type": "shell",
            "command": "docker-compose",
            "args": [
                "-f",
                "docker-compose.dev.yml",
                "up",
                "-d"
            ],
            "group": "build",
            "presentation": {
                "echo": true,
                "reveal": "always",
                "focus": false,
                "panel": "shared"
            }
        }
    ]
}
EOF

# Create development documentation
print_status "Creating development documentation..."
cat > docs/DEVELOPMENT.md << 'EOF'
# Development Guide

## Quick Start

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Akashc1512/universal-knowledge-hub.git
   cd universal-knowledge-hub
   ```

2. **Run the setup script:**
   ```bash
   chmod +x scripts/setup-dev-environment.sh
   ./scripts/setup-dev-environment.sh
   ```

3. **Start development services:**
   ```bash
   docker-compose -f docker-compose.dev.yml up -d
   ```

4. **Set up the database:**
   ```bash
   ./scripts/setup-database.sh
   ```

5. **Start the API server:**
   ```bash
   python start_api.py
   ```

## Development Workflow

### Running Tests
```bash
# Run all tests
pytest

# Run tests with coverage
pytest --cov=agents --cov=core --cov=architecture

# Run specific test file
pytest tests/test_agents.py
```

### Code Quality
```bash
# Format code
black .

# Lint code
flake8 .

# Type checking
mypy .
```

### API Development
```bash
# Start API server
python start_api.py

# Test API endpoints
python test_api.py
```

## Environment Variables

Copy `.env.example` to `.env` and configure:
- Database connection
- API keys for AI services
- Search configuration
- Monitoring settings

## Docker Development

Start all development services:
```bash
docker-compose -f docker-compose.dev.yml up -d
```

Services available:
- PostgreSQL: localhost:5432
- Redis: localhost:6379
- Elasticsearch: localhost:9200
- Prometheus: localhost:9090
- Grafana: localhost:3000

## VS Code Integration

The project includes VS Code configuration for:
- Python interpreter setup
- Debugging configurations
- Task automation
- Code formatting and linting

## AI Tool Integration

This project is optimized for AI-assisted development:
- Cursor IDE Pro for code completion
- GitHub Copilot for pair programming
- Claude Pro for code review
- ChatGPT Plus for documentation

## Monitoring and Observability

- Prometheus metrics available at `/metrics`
- Grafana dashboards for visualization
- Health checks at `/health`
- Performance monitoring enabled

## Troubleshooting

### Common Issues

1. **Database connection errors:**
   - Ensure PostgreSQL is running
   - Check connection string in `.env`

2. **Import errors:**
   - Activate virtual environment
   - Check PYTHONPATH

3. **Docker issues:**
   - Ensure Docker is running
   - Check port conflicts

### Getting Help

- Check the logs in `logs/` directory
- Review the API documentation at `http://localhost:8000/docs`
- Consult the test suite for usage examples
EOF

# Final setup steps
print_status "Finalizing setup..."

# Make scripts executable
chmod +x scripts/*.sh

# Create a development status check
cat > scripts/check-dev-status.sh << 'EOF'
#!/bin/bash

echo "🔍 Checking development environment status..."

# Check Python environment
if [ -d ".venv" ]; then
    echo "✅ Virtual environment exists"
else
    echo "❌ Virtual environment missing"
fi

# Check dependencies
if python -c "import fastapi, uvicorn, requests" 2>/dev/null; then
    echo "✅ Core dependencies installed"
else
    echo "❌ Core dependencies missing"
fi

# Check Docker services
if docker ps | grep -q "ukp-"; then
    echo "✅ Docker services running"
else
    echo "⚠️  Docker services not running (run: docker-compose -f docker-compose.dev.yml up -d)"
fi

# Check database
if pg_isready -h localhost -p 5432 > /dev/null 2>&1; then
    echo "✅ PostgreSQL running"
else
    echo "❌ PostgreSQL not running"
fi

# Check API server
if curl -s http://localhost:8000/health > /dev/null; then
    echo "✅ API server running"
else
    echo "❌ API server not running"
fi

echo "🎉 Development environment check completed!"
EOF
chmod +x scripts/check-dev-status.sh

print_success "Development environment setup completed!"
echo ""
echo "🎉 Universal Knowledge Platform Development Environment is ready!"
echo ""
echo "Next steps:"
echo "1. Configure your .env file with API keys"
echo "2. Start development services: docker-compose -f docker-compose.dev.yml up -d"
echo "3. Set up the database: ./scripts/setup-database.sh"
echo "4. Start the API server: python start_api.py"
echo "5. Check status: ./scripts/check-dev-status.sh"
echo ""
echo "Happy coding! 🚀" 