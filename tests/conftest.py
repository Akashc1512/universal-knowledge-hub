"""
Simplified Test Configuration for Universal Knowledge Hub Backend
Essential fixtures for unit testing main features
"""

import pytest
import asyncio
from typing import Any, Optional
from unittest.mock import Mock, patch
from fastapi.testclient import TestClient

# Import the main application
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.main_simple import app

# Create test client
@pytest.fixture
def client() -> TestClient:
    """Get test client for API testing."""
    return TestClient(app)

@pytest.fixture
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture
def mock_external_services():
    """Mock external services for testing."""
    with patch("api.main_simple.asyncio.sleep") as mock_sleep:
        mock_sleep.return_value = None
        yield {"sleep": mock_sleep}

@pytest.fixture
def test_data():
    """Provide test data for unit tests."""
    return {
        "valid_query": {
            "query": "What is Python 3.13.5?",
            "max_tokens": 1000,
            "confidence_threshold": 0.8
        },
        "valid_auth": {
            "username": "admin",
            "password": "password"
        },
        "invalid_auth": {
            "username": "wrong",
            "password": "wrong"
        },
        "empty_query": {
            "query": "",
            "max_tokens": 1000
        },
        "long_query": {
            "query": "x" * 10001,
            "max_tokens": 1000
        }
    }

# Test utilities
def assert_response_structure(response, expected_fields: list[str]):
    """Assert response has expected structure."""
    assert response.status_code == 200
    data = response.json()
    
    for field in expected_fields:
        assert field in data, f"Missing field: {field}"

def assert_error_response(response, status_code: int):
    """Assert error response structure."""
    assert response.status_code == status_code

def assert_performance(actual_time: float, expected_max: float):
    """Assert performance meets expectations."""
    assert actual_time <= expected_max, \
        f"Performance test failed: {actual_time:.3f}s > {expected_max:.3f}s"

# Export test utilities
__all__ = [
    'client',
    'event_loop',
    'mock_external_services',
    'test_data',
    'assert_response_structure',
    'assert_error_response',
    'assert_performance',
] 