#!/bin/bash

# Universal Knowledge Hub - Python 3.13.5 Setup Script
# This script sets up the development environment for Python 3.13.5

set -e

echo "🚀 Setting up Universal Knowledge Hub for Python 3.13.5..."

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

# Check if Python 3.13.5 is available
check_python_version() {
    print_status "Checking Python version..."
    
    if command -v python3.13 &> /dev/null; then
        PYTHON_CMD="python3.13"
    elif command -v python &> /dev/null; then
        PYTHON_VERSION=$(python --version 2>&1 | cut -d' ' -f2)
        if [[ "$PYTHON_VERSION" == "3.13"* ]]; then
            PYTHON_CMD="python"
        else
            print_error "Python 3.13.5 is required but not found. Current version: $PYTHON_VERSION"
            print_status "Please install Python 3.13.5 from https://www.python.org/downloads/"
            exit 1
        fi
    else
        print_error "Python is not installed or not in PATH"
        exit 1
    fi
    
    print_success "Found Python: $($PYTHON_CMD --version)"
}

# Create virtual environment
create_venv() {
    print_status "Creating virtual environment..."
    
    if [ -d ".venv" ]; then
        print_warning "Virtual environment already exists. Removing old one..."
        rm -rf .venv
    fi
    
    $PYTHON_CMD -m venv .venv
    print_success "Virtual environment created successfully"
}

# Activate virtual environment
activate_venv() {
    print_status "Activating virtual environment..."
    
    if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
        source .venv/Scripts/activate
    else
        source .venv/bin/activate
    fi
    
    print_success "Virtual environment activated"
}

# Upgrade pip
upgrade_pip() {
    print_status "Upgrading pip..."
    python -m pip install --upgrade pip
    print_success "Pip upgraded successfully"
}

# Install dependencies
install_dependencies() {
    print_status "Installing core dependencies..."
    pip install -r requirements.txt
    
    print_status "Installing development dependencies..."
    pip install -r requirements-dev.txt
    
    print_status "Installing test dependencies..."
    pip install -r requirements-test.txt
    
    print_success "All dependencies installed successfully"
}

# Install pre-commit hooks
install_pre_commit() {
    print_status "Installing pre-commit hooks..."
    pre-commit install
    print_success "Pre-commit hooks installed"
}

# Verify installation
verify_installation() {
    print_status "Verifying installation..."
    
    # Check Python version in venv
    VENV_PYTHON_VERSION=$(python --version 2>&1)
    print_success "Virtual environment Python: $VENV_PYTHON_VERSION"
    
    # Check key packages
    print_status "Checking key packages..."
    python -c "import fastapi; print(f'FastAPI: {fastapi.__version__}')"
    python -c "import uvicorn; print(f'Uvicorn: {uvicorn.__version__}')"
    python -c "import pydantic; print(f'Pydantic: {pydantic.__version__}')"
    python -c "import pytest; print(f'Pytest: {pytest.__version__}')"
    
    print_success "Installation verification completed"
}

# Create necessary directories
create_directories() {
    print_status "Creating necessary directories..."
    
    mkdir -p data
    mkdir -p logs
    mkdir -p frontend/.next
    
    print_success "Directories created"
}

# Setup frontend (if Node.js is available)
setup_frontend() {
    if command -v npm &> /dev/null; then
        print_status "Setting up frontend dependencies..."
        cd frontend
        npm install
        cd ..
        print_success "Frontend dependencies installed"
    else
        print_warning "Node.js/npm not found. Frontend setup skipped."
        print_status "Install Node.js from https://nodejs.org/ to set up frontend"
    fi
}

# Main execution
main() {
    echo "=========================================="
    echo "Universal Knowledge Hub Setup"
    echo "Python 3.13.5 Development Environment"
    echo "=========================================="
    
    check_python_version
    create_venv
    activate_venv
    upgrade_pip
    install_dependencies
    install_pre_commit
    create_directories
    setup_frontend
    verify_installation
    
    echo ""
    echo "=========================================="
    print_success "Setup completed successfully!"
    echo "=========================================="
    echo ""
    echo "Next steps:"
    echo "1. Activate virtual environment:"
    echo "   - Windows: .venv\\Scripts\\activate"
    echo "   - macOS/Linux: source .venv/bin/activate"
    echo ""
    echo "2. Configure environment variables:"
    echo "   cp env.template .env"
    echo "   # Edit .env with your API keys"
    echo ""
    echo "3. Start development servers:"
    echo "   # Backend: uvicorn api.main:app --reload"
    echo "   # Frontend: cd frontend && npm run dev"
    echo ""
    echo "4. Run tests:"
    echo "   pytest"
    echo ""
    echo "For more information, see LOCAL_DEV_SETUP.md"
    echo "=========================================="
}

# Run main function
main "$@" 