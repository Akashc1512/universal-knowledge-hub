#!/usr/bin/env python3
"""
Universal Knowledge Platform API Server
Startup script for the FastAPI web service
"""

import uvicorn
import logging
import os
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Start the Universal Knowledge Platform API server."""
    
    # Configuration
    host = os.getenv("UKP_HOST", "0.0.0.0")
    port = int(os.getenv("UKP_PORT", "8000"))
    reload = os.getenv("UKP_RELOAD", "false").lower() == "true"
    
    logger.info("🚀 Starting Universal Knowledge Platform API Server")
    logger.info(f"📍 Host: {host}")
    logger.info(f"🔌 Port: {port}")
    logger.info(f"🔄 Reload: {reload}")
    
    try:
        uvicorn.run(
            "api.main:app",
            host=host,
            port=port,
            reload=reload,
            log_level="info",
            access_log=True
        )
    except KeyboardInterrupt:
        logger.info("🛑 Server stopped by user")
    except Exception as e:
        logger.error(f"❌ Server failed to start: {e}")
        raise

if __name__ == "__main__":
    main() 