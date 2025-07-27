# Universal Knowledge Platform - Team Environment Setup (Windows)
# This script sets up the development environment for team members on Windows

param(
    [string]$TeamMember = "developer",
    [string]$Environment = "dev"
)

Write-Host "🚀 Setting up Universal Knowledge Platform Team Environment" -ForegroundColor Green
Write-Host "================================================================" -ForegroundColor Green

# Function to print colored output
function Write-Status {
    param([string]$Message)
    Write-Host "[INFO] $Message" -ForegroundColor Blue
}

function Write-Success {
    param([string]$Message)
    Write-Host "[SUCCESS] $Message" -ForegroundColor Green
}

function Write-Warning {
    param([string]$Message)
    Write-Host "[WARNING] $Message" -ForegroundColor Yellow
}

function Write-Error {
    param([string]$Message)
    Write-Host "[ERROR] $Message" -ForegroundColor Red
}

# Check prerequisites
Write-Status "Checking prerequisites..."

# Check Python
try {
    $pythonVersion = python --version 2>&1
    Write-Success "Python found: $pythonVersion"
} catch {
    Write-Error "Python is required but not installed"
    exit 1
}

# Check Git
try {
    $gitVersion = git --version 2>&1
    Write-Success "Git found: $gitVersion"
} catch {
    Write-Error "Git is required but not installed"
    exit 1
}

# Check Docker
try {
    $dockerVersion = docker --version 2>&1
    Write-Success "Docker found: $dockerVersion"
} catch {
    Write-Warning "Docker not found. Container features will be limited."
}

# Check Node.js
try {
    $nodeVersion = node --version 2>&1
    Write-Success "Node.js found: $nodeVersion"
} catch {
    Write-Warning "Node.js not found. Frontend development will be limited."
}

# Create virtual environment
Write-Status "Setting up Python virtual environment..."

if (!(Test-Path ".venv")) {
    python -m venv .venv
    Write-Success "Virtual environment created"
} else {
    Write-Status "Virtual environment already exists"
}

# Activate virtual environment
Write-Status "Activating virtual environment..."
& .\.venv\Scripts\Activate.ps1

# Upgrade pip
Write-Status "Upgrading pip..."
python -m pip install --upgrade pip

# Install Python dependencies
Write-Status "Installing Python dependencies..."
pip install -r requirements.txt

# Install development dependencies
Write-Status "Installing development dependencies..."
pip install pytest pytest-cov pytest-asyncio black flake8 mypy pre-commit

# Install pre-commit hooks
Write-Status "Setting up pre-commit hooks..."
pre-commit install

# Create local configuration
Write-Status "Creating local configuration..."

# Create .env file if it doesn't exist
if (!(Test-Path ".env")) {
    @"
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

# Team Configuration
TEAM_MEMBER=$TeamMember
ENVIRONMENT=$Environment
"@ | Out-File -FilePath ".env" -Encoding UTF8
    Write-Success "Created .env file"
} else {
    Write-Status ".env file already exists"
}

# Create local directories
Write-Status "Creating local directories..."
New-Item -ItemType Directory -Force -Path "logs" | Out-Null
New-Item -ItemType Directory -Force -Path "data" | Out-Null
New-Item -ItemType Directory -Force -Path "temp" | Out-Null
New-Item -ItemType Directory -Force -Path "docs\generated" | Out-Null

# Set up Git hooks
Write-Status "Setting up Git hooks..."
if (!(Test-Path ".git\hooks\pre-commit")) {
    @"
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
"@ | Out-File -FilePath ".git\hooks\pre-commit" -Encoding UTF8
    Write-Success "Git pre-commit hook installed"
}

# Create VS Code settings
Write-Status "Setting up VS Code configuration..."
New-Item -ItemType Directory -Force -Path ".vscode" | Out-Null

@"
{
    "python.defaultInterpreterPath": "./.venv/Scripts/python.exe",
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
"@ | Out-File -FilePath ".vscode\settings.json" -Encoding UTF8

# Create launch configuration for debugging
@"
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Python: Current File",
            "type": "python",
            "request": "launch",
            "program": "`${file}",
            "console": "integratedTerminal",
            "cwd": "`${workspaceFolder}",
            "env": {
                "PYTHONPATH": "`${workspaceFolder}"
            }
        },
        {
            "name": "Python: API Server",
            "type": "python",
            "request": "launch",
            "program": "start_api.py",
            "console": "integratedTerminal",
            "cwd": "`${workspaceFolder}",
            "env": {
                "PYTHONPATH": "`${workspaceFolder}"
            }
        },
        {
            "name": "Python: Test Current File",
            "type": "python",
            "request": "launch",
            "module": "pytest",
            "args": [
                "`${file}",
                "-v"
            ],
            "console": "integratedTerminal",
            "cwd": "`${workspaceFolder}"
        }
    ]
}
"@ | Out-File -FilePath ".vscode\launch.json" -Encoding UTF8

# Create tasks for VS Code
@"
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
"@ | Out-File -FilePath ".vscode\tasks.json" -Encoding UTF8

# Create team-specific configuration
Write-Status "Creating team-specific configuration..."

@"
# Team Configuration for $TeamMember
# Environment: $Environment

# AI Tool Configuration
CURSOR_IDE_PRO_ENABLED=true
GITHUB_COPILOT_ENABLED=true
CLAUDE_PRO_ENABLED=true
CHATGPT_PLUS_ENABLED=true

# Development Preferences
PREFERRED_EDITOR=vscode
AUTO_FORMAT_ON_SAVE=true
RUN_TESTS_ON_SAVE=false

# Team Communication
SLACK_CHANNEL=#development
JIRA_PROJECT=UKP
DAILY_STANDUP_TIME=09:00

# Performance Settings
PYTHON_OPTIMIZE=1
ENABLE_PROFILING=false
LOG_LEVEL=INFO

# Security Settings
ENABLE_SECURITY_SCANNING=true
AUTO_UPDATE_DEPENDENCIES=true
"@ | Out-File -FilePath "team-config.ini" -Encoding UTF8

# Create development database setup script
Write-Status "Creating database setup script..."
@"
# Database setup script for Universal Knowledge Platform

Write-Host "Setting up development database..."

# Check if PostgreSQL is running
try {
    $pgTest = pg_isready -h localhost -p 5432 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "PostgreSQL is running"
    } else {
        Write-Host "PostgreSQL is not running. Please start PostgreSQL first."
        exit 1
    }
} catch {
    Write-Host "PostgreSQL is not installed or not in PATH"
    exit 1
}

# Create database and user
$sqlCommands = @"
CREATE DATABASE ukp_dev;
CREATE USER ukp_user WITH PASSWORD 'password';
GRANT ALL PRIVILEGES ON DATABASE ukp_dev TO ukp_user;
\c ukp_dev
GRANT ALL ON SCHEMA public TO ukp_user;
"@

# Execute SQL commands
$sqlCommands | psql -h localhost -U postgres

Write-Host "Database setup completed!"
"@ | Out-File -FilePath "scripts\setup-database.ps1" -Encoding UTF8

# Create Docker Compose for local development
Write-Status "Creating Docker Compose configuration..."
@"
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
"@ | Out-File -FilePath "docker-compose.dev.yml" -Encoding UTF8

# Create development status check
Write-Status "Creating development status check..."
@"
Write-Host "🔍 Checking development environment status..." -ForegroundColor Blue

# Check Python environment
if (Test-Path ".venv") {
    Write-Host "✅ Virtual environment exists" -ForegroundColor Green
} else {
    Write-Host "❌ Virtual environment missing" -ForegroundColor Red
}

# Check dependencies
try {
    python -c "import fastapi, uvicorn, requests" 2>$null
    Write-Host "✅ Core dependencies installed" -ForegroundColor Green
} catch {
    Write-Host "❌ Core dependencies missing" -ForegroundColor Red
}

# Check Docker services
try {
    $dockerServices = docker ps --format "table {{.Names}}" | Select-String "ukp-"
    if ($dockerServices) {
        Write-Host "✅ Docker services running" -ForegroundColor Green
    } else {
        Write-Host "⚠️  Docker services not running (run: docker-compose -f docker-compose.dev.yml up -d)" -ForegroundColor Yellow
    }
} catch {
    Write-Host "❌ Docker not available" -ForegroundColor Red
}

# Check database
try {
    $dbTest = pg_isready -h localhost -p 5432 2>$null
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ PostgreSQL running" -ForegroundColor Green
    } else {
        Write-Host "❌ PostgreSQL not running" -ForegroundColor Red
    }
} catch {
    Write-Host "❌ PostgreSQL not available" -ForegroundColor Red
}

# Check API server
try {
    $apiResponse = Invoke-WebRequest -Uri "http://localhost:8000/health" -TimeoutSec 5
    if ($apiResponse.StatusCode -eq 200) {
        Write-Host "✅ API server running" -ForegroundColor Green
    } else {
        Write-Host "❌ API server not responding" -ForegroundColor Red
    }
} catch {
    Write-Host "❌ API server not running" -ForegroundColor Red
}

Write-Host "🎉 Development environment check completed!" -ForegroundColor Green
"@ | Out-File -FilePath "scripts\check-dev-status.ps1" -Encoding UTF8

Write-Success "Development environment setup completed!"
Write-Host ""
Write-Host "🎉 Universal Knowledge Platform Team Environment is ready!" -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "1. Configure your .env file with API keys" -ForegroundColor White
Write-Host "2. Start development services: docker-compose -f docker-compose.dev.yml up -d" -ForegroundColor White
Write-Host "3. Set up the database: .\scripts\setup-database.ps1" -ForegroundColor White
Write-Host "4. Start the API server: python start_api.py" -ForegroundColor White
Write-Host "5. Check status: .\scripts\check-dev-status.ps1" -ForegroundColor White
Write-Host ""
Write-Host "Team Configuration:" -ForegroundColor Yellow
Write-Host "- Team Member: $TeamMember" -ForegroundColor White
Write-Host "- Environment: $Environment" -ForegroundColor White
Write-Host "- AI Tools: Cursor IDE Pro, GitHub Copilot, Claude Pro" -ForegroundColor White
Write-Host ""
Write-Host "Happy coding! 🚀" -ForegroundColor Green 