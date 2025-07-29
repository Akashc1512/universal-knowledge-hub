#!/bin/bash

# Universal Knowledge Hub - Setup and Deployment Script
# This script sets up the development environment and deploys the application

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_section() {
    echo -e "${BLUE}==========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}==========================================${NC}"
}

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

# Check prerequisites
check_prerequisites() {
    print_section "Checking Prerequisites"
    
    # Check Python
    if command -v python3 &> /dev/null; then
        PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
        print_status "Python 3 found: $PYTHON_VERSION"
    else
        print_error "Python 3 not found. Please install Python 3.13.5+"
        exit 1
    fi
    
    # Check Node.js
    if command -v node &> /dev/null; then
        NODE_VERSION=$(node --version)
        print_status "Node.js found: $NODE_VERSION"
    else
        print_warning "Node.js not found. Frontend features will be limited"
    fi
    
    # Check Docker
    if command -v docker &> /dev/null; then
        DOCKER_VERSION=$(docker --version | cut -d' ' -f3 | cut -d',' -f1)
        print_status "Docker found: $DOCKER_VERSION"
    else
        print_warning "Docker not found. Some services may not be available"
    fi
    
    # Check Redis
    if command -v redis-cli &> /dev/null; then
        print_status "Redis CLI found"
    else
        print_warning "Redis CLI not found. Will use Docker for Redis"
    fi
}

# Setup development environment
setup_dev_environment() {
    print_section "Setting up Development Environment"
    
    # Create virtual environment
    if [ ! -d ".venv" ]; then
        print_status "Creating Python virtual environment..."
        python3 -m venv .venv
        print_success "Virtual environment created"
    else
        print_success "Virtual environment already exists"
    fi
    
    # Activate virtual environment
    print_status "Activating virtual environment..."
    source .venv/bin/activate
    
    # Upgrade pip
    print_status "Upgrading pip..."
    pip install --upgrade pip setuptools wheel
    
    # Install Python dependencies
    print_status "Installing Python dependencies..."
    pip install -r requirements.txt
    
    if [ -f "requirements-dev.txt" ]; then
        print_status "Installing development dependencies..."
        pip install -r requirements-dev.txt
    fi
    
    print_success "Python dependencies installed"
    
    # Setup frontend
    if [ -d "frontend" ]; then
        print_status "Setting up frontend dependencies..."
        cd frontend
        npm install
        cd ..
        print_success "Frontend dependencies installed"
    else
        print_warning "Frontend directory not found"
    fi
}

# Run tests
run_tests() {
    print_section "Running Tests"
    
    # Activate virtual environment
    source .venv/bin/activate
    
    print_status "Running Python tests..."
    python -m pytest tests/ -v --cov=api --cov=agents --cov-report=html
    
    print_status "Running security scan..."
    bandit -r api/ agents/ -f json -o bandit-report.json || true
    
    print_success "Tests completed"
}

# Start development servers
start_dev_servers() {
    print_section "Starting Development Servers"
    
    # Start backend
    print_status "Starting backend server..."
    source .venv/bin/activate
    uvicorn api.main:app --reload --host 0.0.0.0 --port 8000 &
    BACKEND_PID=$!
    print_success "Backend started on http://localhost:8000"
    
    # Start frontend
    if [ -d "frontend" ]; then
        print_status "Starting frontend server..."
        cd frontend
        npm run dev &
        FRONTEND_PID=$!
        cd ..
        print_success "Frontend started on http://localhost:3000"
    fi
    
    print_success "Development servers started"
    print_status "Press Ctrl+C to stop all servers"
    
    # Wait for user to stop
    wait
}

# Deploy to production
deploy_production() {
    print_section "Deploying to Production"
    
    # Check if we're in a production environment
    if [ "$ENVIRONMENT" != "production" ]; then
        print_error "Production deployment requires ENVIRONMENT=production"
        exit 1
    fi
    
    print_status "Building production image..."
    docker build -t universal-knowledge-hub:latest .
    
    print_status "Deploying to Kubernetes..."
    kubectl apply -f k8s/
    
    print_status "Waiting for deployment to be ready..."
    kubectl rollout status deployment/universal-knowledge-hub
    
    print_success "Production deployment completed"
}

# Main function
main() {
    print_section "Universal Knowledge Hub - Setup and Deployment"
    
    case "${1:-setup}" in
        "setup")
            check_prerequisites
            setup_dev_environment
            print_success "Development environment setup completed"
            ;;
        "test")
            run_tests
            ;;
        "dev")
            start_dev_servers
            ;;
        "deploy")
            deploy_production
            ;;
        "all")
            check_prerequisites
            setup_dev_environment
            run_tests
            start_dev_servers
            ;;
        *)
            echo "Usage: $0 {setup|test|dev|deploy|all}"
            echo "  setup   - Setup development environment"
            echo "  test    - Run tests"
            echo "  dev     - Start development servers"
            echo "  deploy  - Deploy to production"
            echo "  all     - Run all steps"
            exit 1
            ;;
    esac
}

# Run main function
main "$@" 