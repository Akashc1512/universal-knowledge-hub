"""
Test Configuration - MAANG Standards.

This module provides comprehensive test configuration following MAANG
best practices for testing infrastructure.

Features:
    - Test database setup and teardown
    - Mock external services
    - Test data factories
    - Performance benchmarks
    - Security testing utilities
    - Coverage reporting
    - Parallel test execution

Testing Strategy:
    - Unit tests (fast, isolated)
    - Integration tests (external dependencies)
    - Performance tests (benchmarks)
    - Security tests (penetration testing)
    - Property-based tests (hypothesis)

Authors:
    - Universal Knowledge Platform Engineering Team
    
Version:
    2.0.0 (2024-12-28)
"""

import asyncio
import pytest
import pytest_asyncio
from typing import (
    Optional, Dict, Any, List, Union, Callable,
    TypeVar, AsyncGenerator, Generator
)
from unittest.mock import Mock, AsyncMock, patch
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
import tempfile
import shutil
import os
import json
import structlog

from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from api.main import app
from api.config import get_settings, Settings
from api.database.models import Base
from api.cache import get_cache_manager
from api.monitoring import start_monitoring, stop_monitoring
from api.security import get_threat_stats

logger = structlog.get_logger(__name__)

# Type definitions
T = TypeVar('T')

# Test configuration
@dataclass
class TestConfig:
    """Test configuration settings."""
    
    # Database
    database_url: str = "sqlite:///:memory:"
    create_tables: bool = True
    drop_tables: bool = True
    
    # Cache
    cache_enabled: bool = True
    cache_backend: str = "memory"
    
    # Monitoring
    monitoring_enabled: bool = False
    metrics_enabled: bool = False
    
    # Security
    security_enabled: bool = True
    threat_detection: bool = True
    
    # Performance
    performance_tests: bool = False
    benchmark_threshold: float = 1.0  # seconds
    
    # Coverage
    coverage_enabled: bool = True
    coverage_threshold: float = 95.0  # percentage
    
    # Parallel execution
    parallel_tests: bool = True
    max_workers: int = 4

# Test data factories
@dataclass
class UserFactory:
    """Factory for creating test user data."""
    
    def create_user(
        self,
        username: str = "testuser",
        email: str = "test@example.com",
        password: str = "TestPass123!",
        role: str = "user",
        **kwargs: Any
    ) -> Dict[str, Any]:
        """Create test user data."""
        return {
            "username": username,
            "email": email,
            "password": password,
            "role": role,
            "is_active": True,
            "created_at": datetime.now(timezone.utc),
            **kwargs
        }
    
    def create_admin_user(self, **kwargs: Any) -> Dict[str, Any]:
        """Create admin user data."""
        return self.create_user(
            username="admin",
            email="admin@example.com",
            role="admin",
            **kwargs
        )
    
    def create_bulk_users(self, count: int) -> List[Dict[str, Any]]:
        """Create bulk user data."""
        return [
            self.create_user(
                username=f"user{i}",
                email=f"user{i}@example.com"
            )
            for i in range(count)
        ]

@dataclass
class QueryFactory:
    """Factory for creating test query data."""
    
    def create_query(
        self,
        query: str = "What is machine learning?",
        user_id: Optional[str] = None,
        **kwargs: Any
    ) -> Dict[str, Any]:
        """Create test query data."""
        return {
            "query": query,
            "user_id": user_id,
            "timestamp": datetime.now(timezone.utc),
            "status": "pending",
            **kwargs
        }
    
    def create_malicious_query(self, attack_type: str = "sql_injection") -> Dict[str, Any]:
        """Create malicious query for security testing."""
        malicious_queries = {
            "sql_injection": "'; DROP TABLE users; --",
            "xss": "<script>alert('xss')</script>",
            "path_traversal": "../../../etc/passwd",
            "command_injection": "| cat /etc/passwd",
        }
        
        return self.create_query(
            query=malicious_queries.get(attack_type, "malicious query"),
            user_id="attacker"
        )
    
    def create_performance_query(self, complexity: str = "simple") -> Dict[str, Any]:
        """Create query for performance testing."""
        performance_queries = {
            "simple": "What is Python?",
            "medium": "Explain the differences between machine learning and deep learning with examples",
            "complex": "Provide a comprehensive analysis of the impact of artificial intelligence on modern society, including economic, social, and ethical considerations",
        }
        
        return self.create_query(
            query=performance_queries.get(complexity, "performance test query")
        )

# Test database setup
class TestDatabase:
    """Test database management."""
    
    def __init__(self, url: str = "sqlite:///:memory:"):
        """Initialize test database."""
        self.url = url
        self.engine = None
        self.SessionLocal = None
    
    def setup(self) -> None:
        """Set up test database."""
        self.engine = create_engine(
            self.url,
            connect_args={"check_same_thread": False},
            poolclass=StaticPool
        )
        self.SessionLocal = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=self.engine
        )
        
        # Create tables
        Base.metadata.create_all(bind=self.engine)
        logger.info("Test database setup complete")
    
    def teardown(self) -> None:
        """Tear down test database."""
        if self.engine:
            Base.metadata.drop_all(bind=self.engine)
            self.engine.dispose()
        logger.info("Test database teardown complete")
    
    def get_session(self):
        """Get database session."""
        if not self.SessionLocal:
            raise RuntimeError("Database not set up")
        return self.SessionLocal()

# Test cache setup
class TestCache:
    """Test cache management."""
    
    def __init__(self, backend: str = "memory"):
        """Initialize test cache."""
        self.backend = backend
        self.cache_manager = None
    
    async def setup(self) -> None:
        """Set up test cache."""
        self.cache_manager = get_cache_manager()
        await self.cache_manager.initialize()
        logger.info("Test cache setup complete")
    
    async def teardown(self) -> None:
        """Tear down test cache."""
        if self.cache_manager:
            await self.cache_manager.close()
        logger.info("Test cache teardown complete")
    
    async def clear(self) -> None:
        """Clear all cache data."""
        if self.cache_manager:
            await self.cache_manager.clear()

# Performance testing utilities
class PerformanceBenchmark:
    """Performance benchmarking utilities."""
    
    def __init__(self, threshold: float = 1.0):
        """Initialize benchmark."""
        self.threshold = threshold
        self.results: List[Dict[str, Any]] = []
    
    async def benchmark(
        self,
        name: str,
        func: Callable,
        *args: Any,
        **kwargs: Any
    ) -> Dict[str, Any]:
        """Run performance benchmark."""
        import time
        
        start_time = time.time()
        
        if asyncio.iscoroutinefunction(func):
            result = await func(*args, **kwargs)
        else:
            result = func(*args, **kwargs)
        
        duration = time.time() - start_time
        
        benchmark_result = {
            "name": name,
            "duration": duration,
            "threshold": self.threshold,
            "passed": duration <= self.threshold,
            "result": result
        }
        
        self.results.append(benchmark_result)
        
        if not benchmark_result["passed"]:
            pytest.fail(
                f"Performance benchmark failed: {name} took {duration:.3f}s "
                f"(threshold: {self.threshold:.3f}s)"
            )
        
        return benchmark_result
    
    def get_stats(self) -> Dict[str, Any]:
        """Get benchmark statistics."""
        if not self.results:
            return {}
        
        durations = [r["duration"] for r in self.results]
        return {
            "total_tests": len(self.results),
            "passed": sum(1 for r in self.results if r["passed"]),
            "failed": sum(1 for r in self.results if not r["passed"]),
            "avg_duration": sum(durations) / len(durations),
            "min_duration": min(durations),
            "max_duration": max(durations),
            "results": self.results
        }

# Security testing utilities
class SecurityTester:
    """Security testing utilities."""
    
    def __init__(self):
        """Initialize security tester."""
        self.attack_results: List[Dict[str, Any]] = []
    
    async def test_sql_injection(
        self,
        endpoint: str,
        client: TestClient,
        payloads: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Test for SQL injection vulnerabilities."""
        if payloads is None:
            payloads = [
                "'; DROP TABLE users; --",
                "' OR '1'='1",
                "' UNION SELECT * FROM users --",
                "admin'--",
            ]
        
        results = []
        for payload in payloads:
            response = client.post(
                endpoint,
                json={"query": payload}
            )
            
            result = {
                "payload": payload,
                "status_code": response.status_code,
                "response": response.text,
                "vulnerable": self._is_sql_injection_successful(response)
            }
            results.append(result)
            self.attack_results.append(result)
        
        return results
    
    async def test_xss(
        self,
        endpoint: str,
        client: TestClient,
        payloads: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Test for XSS vulnerabilities."""
        if payloads is None:
            payloads = [
                "<script>alert('xss')</script>",
                "javascript:alert('xss')",
                "<img src=x onerror=alert('xss')>",
                "<iframe src=javascript:alert('xss')>",
            ]
        
        results = []
        for payload in payloads:
            response = client.post(
                endpoint,
                json={"query": payload}
            )
            
            result = {
                "payload": payload,
                "status_code": response.status_code,
                "response": response.text,
                "vulnerable": self._is_xss_successful(response)
            }
            results.append(result)
            self.attack_results.append(result)
        
        return results
    
    def _is_sql_injection_successful(self, response) -> bool:
        """Check if SQL injection was successful."""
        # Check for database error messages
        error_indicators = [
            "sql syntax",
            "mysql error",
            "postgresql error",
            "sqlite error",
            "database error",
        ]
        
        response_text = response.text.lower()
        return any(indicator in response_text for indicator in error_indicators)
    
    def _is_xss_successful(self, response) -> bool:
        """Check if XSS was successful."""
        # Check if script tags are reflected
        xss_indicators = [
            "<script>",
            "javascript:",
            "onerror=",
            "onload=",
        ]
        
        response_text = response.text.lower()
        return any(indicator in response_text for indicator in xss_indicators)
    
    def get_security_report(self) -> Dict[str, Any]:
        """Get security testing report."""
        return {
            "total_attacks": len(self.attack_results),
            "successful_attacks": sum(1 for r in self.attack_results if r["vulnerable"]),
            "failed_attacks": sum(1 for r in self.attack_results if not r["vulnerable"]),
            "attack_results": self.attack_results
        }

# Pytest fixtures
@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture(scope="session")
def test_config() -> TestConfig:
    """Get test configuration."""
    return TestConfig()

@pytest.fixture(scope="session")
def test_database(test_config: TestConfig) -> TestDatabase:
    """Get test database instance."""
    return TestDatabase(test_config.database_url)

@pytest.fixture(scope="session")
def test_cache(test_config: TestConfig) -> TestCache:
    """Get test cache instance."""
    return TestCache(test_config.cache_backend)

@pytest.fixture
def client(test_database: TestDatabase, test_cache: TestCache) -> TestClient:
    """Get test client."""
    # Setup database
    test_database.setup()
    
    # Setup cache
    asyncio.run(test_cache.setup())
    
    # Create test client
    with TestClient(app) as test_client:
        yield test_client
    
    # Teardown
    test_database.teardown()
    asyncio.run(test_cache.teardown())

@pytest.fixture
def user_factory() -> UserFactory:
    """Get user factory."""
    return UserFactory()

@pytest.fixture
def query_factory() -> QueryFactory:
    """Get query factory."""
    return QueryFactory()

@pytest.fixture
def performance_benchmark(test_config: TestConfig) -> PerformanceBenchmark:
    """Get performance benchmark."""
    return PerformanceBenchmark(test_config.benchmark_threshold)

@pytest.fixture
def security_tester() -> SecurityTester:
    """Get security tester."""
    return SecurityTester()

@pytest.fixture
def mock_external_services():
    """Mock external services for testing."""
    with patch("api.health_checks.check_vector_db") as mock_vector_db, \
         patch("api.health_checks.check_elasticsearch") as mock_elasticsearch, \
         patch("api.health_checks.check_redis") as mock_redis:
        
        # Configure mocks
        mock_vector_db.return_value = {"status": "healthy", "response_time": 0.1}
        mock_elasticsearch.return_value = {"status": "healthy", "response_time": 0.1}
        mock_redis.return_value = {"status": "healthy", "response_time": 0.1}
        
        yield {
            "vector_db": mock_vector_db,
            "elasticsearch": mock_elasticsearch,
            "redis": mock_redis
        }

@pytest.fixture
def mock_ai_services():
    """Mock AI services for testing."""
    with patch("api.ai_services.openai.ChatCompletion.create") as mock_openai, \
         patch("api.ai_services.anthropic.Anthropic.messages.create") as mock_anthropic:
        
        # Configure OpenAI mock
        mock_openai.return_value = Mock(
            choices=[Mock(message=Mock(content="Test response"))],
            usage=Mock(total_tokens=10)
        )
        
        # Configure Anthropic mock
        mock_anthropic.return_value = Mock(
            content=[Mock(text="Test response")],
            usage=Mock(input_tokens=5, output_tokens=5)
        )
        
        yield {
            "openai": mock_openai,
            "anthropic": mock_anthropic
        }

@pytest.fixture
def test_settings() -> Settings:
    """Get test settings."""
    return Settings(
        environment="test",
        debug=True,
        database_url="sqlite:///:memory:",
        redis_url=None,
        cache_enabled=False,
        monitoring_enabled=False,
        security_enabled=False
    )

# Test markers
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "unit: Unit tests (fast, isolated)"
    )
    config.addinivalue_line(
        "markers", "integration: Integration tests (external dependencies)"
    )
    config.addinivalue_line(
        "markers", "performance: Performance tests (benchmarks)"
    )
    config.addinivalue_line(
        "markers", "security: Security tests (penetration testing)"
    )
    config.addinivalue_line(
        "markers", "slow: Slow tests (should be run separately)"
    )

# Test utilities
def assert_response_structure(response, expected_fields: List[str]):
    """Assert response has expected structure."""
    assert response.status_code == 200
    data = response.json()
    
    for field in expected_fields:
        assert field in data, f"Missing field: {field}"

def assert_error_response(response, status_code: int, error_type: str):
    """Assert error response structure."""
    assert response.status_code == status_code
    data = response.json()
    
    assert "error" in data
    assert "code" in data["error"]
    assert data["error"]["code"] == error_type

def assert_performance(actual_time: float, expected_max: float):
    """Assert performance meets expectations."""
    assert actual_time <= expected_max, \
        f"Performance test failed: {actual_time:.3f}s > {expected_max:.3f}s"

def assert_security(response, attack_type: str):
    """Assert security measures are working."""
    # Should not return sensitive information
    response_text = response.text.lower()
    
    sensitive_indicators = [
        "sql syntax",
        "database error",
        "stack trace",
        "internal error",
    ]
    
    for indicator in sensitive_indicators:
        assert indicator not in response_text, \
            f"Security test failed: {attack_type} exposed {indicator}"

# Coverage configuration
def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """Print coverage summary."""
    if config.getoption("--cov"):
        coverage = terminalreporter.stats.get("coverage", [])
        if coverage:
            print("\nCoverage Summary:")
            for line in coverage:
                print(f"  {line}")

# Export test utilities
__all__ = [
    # Classes
    'TestConfig',
    'UserFactory',
    'QueryFactory',
    'TestDatabase',
    'TestCache',
    'PerformanceBenchmark',
    'SecurityTester',
    
    # Fixtures
    'event_loop',
    'test_config',
    'test_database',
    'test_cache',
    'client',
    'user_factory',
    'query_factory',
    'performance_benchmark',
    'security_tester',
    'mock_external_services',
    'mock_ai_services',
    'test_settings',
    
    # Utilities
    'assert_response_structure',
    'assert_error_response',
    'assert_performance',
    'assert_security',
] 