"""
API Package
===========

FastAPI-based REST API for the Universal Knowledge Platform.
"""

from .main import app
from .analytics import AnalyticsCollector
from .cache import QueryCache, SemanticCache
from .security import SecurityMonitor
from .recommendation_service import RecommendationService

__all__ = [
    'app',
    'AnalyticsCollector',
    'QueryCache',
    'SemanticCache',
    'SecurityMonitor',
    'RecommendationService'
] 