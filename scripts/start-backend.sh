#!/bin/bash

# Start FastAPI Backend Server
# This script starts the FastAPI backend with hot reload

cd "$(dirname "$0")/.."

# Activate Python 3.13.5 virtual environment
source .venv-py313/bin/activate

echo "🚀 Starting FastAPI backend server..."
echo "📊 Python version: $(python --version)"
echo "🌐 Server will be available at: http://localhost:8000"
echo "📚 API documentation: http://localhost:8000/docs"
echo ""

# Start the FastAPI server with hot reload
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000 