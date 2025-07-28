"""
Universal Knowledge Platform
A sophisticated AI-powered knowledge processing and retrieval system.

This package provides:
- Multi-agent architecture for complex query processing
- Advanced retrieval and synthesis capabilities
- Real-time knowledge integration
- Scalable API infrastructure
"""

__version__ = "1.0.0"
__author__ = "Universal Knowledge Platform Team"

# Core exports for easy importing
from api.main import app

__all__ = ["app"]

# Package metadata
PACKAGE_INFO = {
    "name": "universal-knowledge-hub",
    "version": __version__,
    "description": "AI-powered knowledge platform with multi-agent architecture",
    "author": __author__,
    "license": "MIT",
    "python_requires": ">=3.9",
    "keywords": [
        "ai",
        "knowledge",
        "retrieval",
        "synthesis",
        "agents",
        "nlp",
        "machine-learning",
        "api",
        "fastapi",
    ],
    "classifiers": [
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
}
