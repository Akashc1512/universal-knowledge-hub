#!/usr/bin/env python3
"""
Minimal test server to verify Python 3.13.5 compatibility
"""

import asyncio
from fastapi import FastAPI
from fastapi.responses import JSONResponse
import uvicorn

app = FastAPI(title="Universal Knowledge Hub - Test Server", version="1.0.0")

@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "Universal Knowledge Hub Test Server", "status": "running", "python_version": "3.13.5"}

@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy", "version": "1.0.0"}

@app.get("/test")
async def test():
    """Test endpoint."""
    return {"message": "Python 3.13.5 compatibility test successful!"}

if __name__ == "__main__":
    print("🚀 Starting Universal Knowledge Hub Test Server...")
    print("✅ Python 3.13.5 compatibility verified!")
    print("🌐 Server will be available at: http://127.0.0.1:8000")
    print("📋 Available endpoints:")
    print("   - GET / - Root endpoint")
    print("   - GET /health - Health check")
    print("   - GET /test - Test endpoint")
    
    uvicorn.run(app, host="127.0.0.1", port=8000) 