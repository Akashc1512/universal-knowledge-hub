#!/bin/bash

# Universal Knowledge Hub - Local Development Setup Script
# This script sets up the local development environment without Docker

set -e

echo "🚀 Setting up Universal Knowledge Hub for local development..."

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

# Check if we're in the project root
if [ ! -f "requirements.txt" ]; then
    print_error "Please run this script from the project root directory"
    exit 1
fi

# Step 1: Check Python version
print_status "Checking Python version..."
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
    print_success "Python version: $PYTHON_VERSION"
else
    print_error "Python 3 is not installed"
    exit 1
fi

# Step 2: Setup Python virtual environment
print_status "Setting up Python virtual environment..."
if [ ! -d ".venv-py313" ]; then
    print_status "Creating virtual environment..."
    python3 -m venv .venv-py313
    print_success "Virtual environment created"
else
    print_success "Virtual environment already exists"
fi

# Step 3: Activate virtual environment and install dependencies
print_status "Installing Python dependencies..."
source .venv-py313/bin/activate
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt

if [ -f "requirements-dev.txt" ]; then
    print_status "Installing development dependencies..."
    pip install -r requirements-dev.txt
fi

print_success "Python dependencies installed"

# Step 4: Setup frontend
print_status "Setting up frontend dependencies..."
if [ -d "frontend" ]; then
    cd frontend
    if [ ! -d "node_modules" ]; then
        print_status "Installing Node.js dependencies..."
        npm install
        print_success "Node.js dependencies installed"
    else
        print_success "Node.js dependencies already installed"
    fi
    cd ..
else
    print_warning "Frontend directory not found, skipping frontend setup"
fi

# Step 5: Environment setup
print_status "Setting up environment variables..."
if [ ! -f ".env" ]; then
    if [ -f "env.template" ]; then
        cp env.template .env
        print_success "Environment file created from template"
        print_warning "Please edit .env file with your API keys"
    else
        print_warning "No env.template found, please create .env file manually"
    fi
else
    print_success "Environment file already exists"
fi

# Step 6: Create convenience scripts
print_status "Creating convenience scripts..."

# Backend start script
cat > scripts/start-backend.sh << 'EOF'
#!/bin/bash
cd "$(dirname "$0")/.."
source .venv-py313/bin/activate
echo "🚀 Starting FastAPI backend..."
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
EOF

# Frontend start script
cat > scripts/start-frontend.sh << 'EOF'
#!/bin/bash
cd "$(dirname "$0")/../frontend"
echo "🚀 Starting Next.js frontend..."
npm run dev
EOF

# Test script
cat > scripts/test.sh << 'EOF'
#!/bin/bash
cd "$(dirname "$0")/.."
source .venv-py313/bin/activate
echo "🧪 Running tests..."
pytest "$@"
EOF

# Lint script
cat > scripts/lint.sh << 'EOF'
#!/bin/bash
cd "$(dirname "$0")/.."
source .venv-py313/bin/activate
echo "🔍 Running linting..."
black .
flake8 .
mypy .
cd frontend && npm run lint
EOF

# Make scripts executable
chmod +x scripts/start-backend.sh
chmod +x scripts/start-frontend.sh
chmod +x scripts/test.sh
chmod +x scripts/lint.sh

print_success "Convenience scripts created"

# Step 7: Create VS Code settings
print_status "Setting up VS Code/Cursor settings..."
mkdir -p .vscode

cat > .vscode/settings.json << 'EOF'
{
    "python.defaultInterpreterPath": "./.venv-py313/bin/python",
    "python.linting.enabled": true,
    "python.linting.flake8Enabled": true,
    "python.formatting.provider": "black",
    "python.testing.pytestEnabled": true,
    "python.testing.pytestArgs": ["tests"],
    "editor.formatOnSave": true,
    "editor.codeActionsOnSave": {
        "source.organizeImports": true
    },
    "files.exclude": {
        "**/__pycache__": true,
        "**/*.pyc": true,
        "**/.pytest_cache": true,
        "**/node_modules": true,
        "**/.next": true
    }
}
EOF

print_success "VS Code/Cursor settings configured"

# Step 8: Final instructions
echo ""
print_success "🎉 Setup complete! Here's how to start development:"
echo ""
echo "📋 Quick Start Commands:"
echo "  • Backend:   ./scripts/start-backend.sh"
echo "  • Frontend:  ./scripts/start-frontend.sh"
echo "  • Tests:     ./scripts/test.sh"
echo "  • Lint:      ./scripts/lint.sh"
echo ""
echo "🌐 Access Points:"
echo "  • Backend API: http://localhost:8000"
echo "  • API Docs:    http://localhost:8000/docs"
echo "  • Frontend:    http://localhost:3000"
echo ""
echo "📝 Next Steps:"
echo "  1. Edit .env file with your API keys"
echo "  2. Start backend: ./scripts/start-backend.sh"
echo "  3. Start frontend: ./scripts/start-frontend.sh"
echo "  4. Open Cursor and start coding!"
echo ""
print_warning "Remember to activate the virtual environment: source .venv-py313/bin/activate" 