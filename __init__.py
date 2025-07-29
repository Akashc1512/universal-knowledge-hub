"""
Universal Knowledge Hub - AI-Powered Knowledge Platform
A production-ready multi-agent system for comprehensive, well-cited answers.
"""

__version__ = "1.0.0"
__author__ = "Universal Knowledge Hub Team"
__email__ = "support@sarvanom.com"

# Core package information
PACKAGE_INFO = {
    "name": "universal-knowledge-hub",
    "version": __version__,
    "description": "Universal Knowledge Platform with AI-powered search and synthesis",
    "author": __author__,
    "author_email": __email__,
    "url": "https://sarvanom.com",
    "license": "MIT",
    "classifiers": [
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.13",
        "Programming Language :: Python :: 3.13",
        "Programming Language :: Python :: 3.13",
        "Programming Language :: Python :: 3.13",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Text Processing :: Linguistic",
    ],
    "python_requires": ">=3.13.5",
    "install_requires": [
        "fastapi>=0.116.1",
        "uvicorn>=0.35.0",
        "pydantic>=2.11.7",
        "requests>=2.32.4",
        "python-dotenv>=1.1.1",
        "aiohttp>=3.12.14",
        "numpy>=2.0.2",
        "tenacity>=9.1.2",
    ],
    "extras_require": {
        "dev": [
            "pytest>=8.4.1",
            "pytest-cov>=5.0.0",
            "black>=25.1.1",
            "flake8>=7.2.1",
            "mypy>=1.12.0",
            "bandit>=1.8.1",
        ],
        "test": [
            "pytest>=8.4.1",
            "pytest-asyncio>=1.1.0",
            "pytest-cov>=5.0.0",
            "httpx>=0.28.1",
        ],
    },
}

# Export main components
from .agents.lead_orchestrator import LeadOrchestrator
from .agents.retrieval_agent import RetrievalAgent
from .agents.synthesis_agent import SynthesisAgent
from .agents.citation_agent import CitationAgent
from .agents.factcheck_agent import FactCheckAgent

__all__ = [
    "__version__",
    "__author__",
    "__email__",
    "PACKAGE_INFO",
    "LeadOrchestrator",
    "RetrievalAgent", 
    "SynthesisAgent",
    "CitationAgent",
    "FactCheckAgent",
]
