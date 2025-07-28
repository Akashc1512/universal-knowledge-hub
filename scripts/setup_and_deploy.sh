#!/bin/bash
# Universal Knowledge Platform - Setup and Deployment Script
# This script handles environment setup, dependency installation, and deployment

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

print_section() {
    echo -e "\n${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${YELLOW}$1${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
}

# Parse command line arguments
ACTION=${1:-help}
ENVIRONMENT=${2:-development}

# Show usage
show_usage() {
    echo "Universal Knowledge Platform - Setup and Deployment"
    echo ""
    echo "Usage: $0 [action] [environment]"
    echo ""
    echo "Actions:"
    echo "  setup       - Setup development environment"
    echo "  deploy      - Deploy the application"
    echo "  test        - Run tests"
    echo "  check       - Check system health"
    echo "  clean       - Clean up resources"
    echo "  help        - Show this help message"
    echo ""
    echo "Environments:"
    echo "  development - Local development (default)"
    echo "  production  - Production deployment"
    echo "  docker      - Docker deployment"
    echo ""
    echo "Examples:"
    echo "  $0 setup development"
    echo "  $0 deploy production"
    echo "  $0 test"
}

# Check prerequisites
check_prerequisites() {
    print_section "Checking Prerequisites"
    
    # Check Python
    if command -v python3 &> /dev/null; then
        PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
        print_status "Python 3 found: $PYTHON_VERSION"
    else
        print_error "Python 3 not found. Please install Python 3.9+"
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
setup_development() {
    print_section "Setting Up Development Environment"
    
    # Create virtual environment
    if [ ! -d ".venv" ]; then
        print_info "Creating Python virtual environment..."
        python3 -m venv .venv
        print_status "Virtual environment created"
    else
        print_status "Virtual environment already exists"
    fi
    
    # Activate virtual environment
    source .venv/bin/activate
    
    # Upgrade pip
    print_info "Upgrading pip..."
    pip install --upgrade pip --quiet
    
    # Install Python dependencies
    print_info "Installing Python dependencies..."
    pip install -r requirements.txt --quiet
    print_status "Python dependencies installed"
    
    # Setup environment file
    if [ ! -f ".env" ]; then
        print_info "Creating .env file from template..."
        cp env.template .env
        print_status ".env file created"
        print_warning "Please update .env with your configuration"
    else
        print_status ".env file already exists"
    fi
    
    # Create required directories
    print_info "Creating required directories..."
    mkdir -p logs data uploads temp cache
    print_status "Directories created"
    
    # Install frontend dependencies (if Node.js is available)
    if command -v npm &> /dev/null && [ -d "frontend" ]; then
        print_info "Installing frontend dependencies..."
        cd frontend
        npm install --quiet
        cd ..
        print_status "Frontend dependencies installed"
    fi
    
    # Start required services with Docker
    if command -v docker-compose &> /dev/null; then
        print_info "Starting required services (Redis, Elasticsearch)..."
        docker-compose up -d redis elasticsearch
        print_status "Services started"
    else
        print_warning "Docker Compose not found. Please start Redis and Elasticsearch manually"
    fi
    
    print_status "Development environment setup complete!"
}

# Deploy application
deploy_application() {
    print_section "Deploying Application ($ENVIRONMENT)"
    
    case $ENVIRONMENT in
        development)
            print_info "Starting development server..."
            source .venv/bin/activate
            python start_api.py
            ;;
            
        production)
            print_info "Building for production..."
            
            # Build Docker image
            docker build -t sarvanom:latest .
            print_status "Docker image built"
            
            # Run with Docker Compose
            docker-compose -f docker-compose.yml up -d
            print_status "Application deployed"
            
            # Show status
            docker-compose ps
            ;;
            
        docker)
            print_info "Deploying with Docker..."
            docker-compose up -d
            print_status "Docker deployment complete"
            ;;
            
        *)
            print_error "Unknown environment: $ENVIRONMENT"
            exit 1
            ;;
    esac
}

# Run tests
run_tests() {
    print_section "Running Tests"
    
    source .venv/bin/activate
    
    # Run linting
    print_info "Running code linting..."
    flake8 api/ agents/ --count --select=E9,F63,F7,F82 --quiet || true
    
    # Run unit tests
    print_info "Running unit tests..."
    pytest tests/ -v --tb=short || true
    
    # Check imports
    print_info "Checking imports..."
    python -c "
import api.main
import api.health_checks
import api.connection_pool
import api.validators
import api.retry_logic
import api.versioning
import api.rate_limiter
import api.shutdown_handler
print('All imports successful')
    "
    
    print_status "Tests completed"
}

# Check system health
check_health() {
    print_section "Checking System Health"
    
    # Check API health
    if curl -s http://localhost:8002/health > /dev/null 2>&1; then
        print_status "API is running"
        curl -s http://localhost:8002/health | python -m json.tool
    else
        print_error "API is not responding"
    fi
    
    # Check Redis
    if redis-cli ping > /dev/null 2>&1; then
        print_status "Redis is running"
    else
        print_warning "Redis is not responding"
    fi
    
    # Check Elasticsearch
    if curl -s http://localhost:9200/_health > /dev/null 2>&1; then
        print_status "Elasticsearch is running"
    else
        print_warning "Elasticsearch is not responding"
    fi
}

# Clean up resources
cleanup_resources() {
    print_section "Cleaning Up Resources"
    
    # Stop services
    if command -v docker-compose &> /dev/null; then
        print_info "Stopping Docker services..."
        docker-compose down
        print_status "Services stopped"
    fi
    
    # Clean Python cache
    print_info "Cleaning Python cache..."
    find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
    find . -type f -name "*.pyc" -delete 2>/dev/null || true
    print_status "Python cache cleaned"
    
    # Clean logs (optional)
    read -p "Clean log files? (y/N) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -f logs/*.log
        print_status "Log files cleaned"
    fi
}

# Main execution
main() {
    echo "🚀 Universal Knowledge Platform - Setup and Deployment"
    echo "======================================================"
    
    case $ACTION in
        setup)
            check_prerequisites
            setup_development
            ;;
            
        deploy)
            check_prerequisites
            deploy_application
            ;;
            
        test)
            run_tests
            ;;
            
        check)
            check_health
            ;;
            
        clean)
            cleanup_resources
            ;;
            
        help|*)
            show_usage
            ;;
    esac
}

# Run main function
main 