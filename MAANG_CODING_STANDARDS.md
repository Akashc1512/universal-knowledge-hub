# 🚀 MAANG-Level Coding Standards

## Overview

This document defines the coding standards and best practices aligned with MAANG (Meta, Amazon, Apple, Netflix, Google) engineering excellence. Every line of code in this project adheres to these standards.

## 📋 Table of Contents

1. [Code Quality Standards](#code-quality-standards)
2. [Documentation Standards](#documentation-standards)
3. [Testing Standards](#testing-standards)
4. [Performance Standards](#performance-standards)
5. [Security Standards](#security-standards)
6. [Architecture Standards](#architecture-standards)
7. [Monitoring Standards](#monitoring-standards)

## 🎯 Code Quality Standards

### 1. Type Hints (PEP 484)
```python
# ❌ Bad
def process_data(data):
    return data.upper()

# ✅ Good
from typing import Optional, List, Dict, Union, TypeVar, Generic

T = TypeVar('T')

def process_data(data: str) -> str:
    """Process and return uppercase data."""
    return data.upper()
```

### 2. Docstrings (Google Style)
```python
# ✅ MAANG Standard
def calculate_similarity(
    vector_a: np.ndarray,
    vector_b: np.ndarray,
    metric: str = "cosine"
) -> float:
    """
    Calculate similarity between two vectors using specified metric.
    
    This function implements multiple similarity metrics optimized for
    high-dimensional vector spaces commonly used in ML applications.
    
    Args:
        vector_a: First vector of shape (n,) where n is dimension.
        vector_b: Second vector of shape (n,) must match vector_a dimension.
        metric: Similarity metric to use. Options:
            - 'cosine': Cosine similarity (default)
            - 'euclidean': Euclidean distance
            - 'manhattan': Manhattan distance
            
    Returns:
        Similarity score as float. Range depends on metric:
            - cosine: [-1, 1] where 1 is identical
            - euclidean: [0, ∞) where 0 is identical
            - manhattan: [0, ∞) where 0 is identical
            
    Raises:
        ValueError: If vectors have different dimensions.
        ValueError: If metric is not supported.
        TypeError: If inputs are not numpy arrays.
        
    Examples:
        >>> a = np.array([1, 2, 3])
        >>> b = np.array([4, 5, 6])
        >>> calculate_similarity(a, b, "cosine")
        0.9746318461970762
        
    Note:
        For large-scale operations, consider using batch processing
        with calculate_similarity_batch() for better performance.
    """
```

### 3. Error Handling
```python
# ✅ MAANG Standard
class VectorDimensionError(ValueError):
    """Raised when vector dimensions don't match."""
    pass

class MetricNotSupportedError(ValueError):
    """Raised when requested metric is not implemented."""
    pass

def safe_divide(a: float, b: float) -> float:
    """
    Safely divide two numbers with proper error handling.
    
    Args:
        a: Numerator
        b: Denominator
        
    Returns:
        Result of division
        
    Raises:
        ZeroDivisionError: When denominator is zero
    """
    if abs(b) < 1e-10:  # Use epsilon for float comparison
        raise ZeroDivisionError(f"Division by zero: {a} / {b}")
    return a / b
```

### 4. Constants and Configuration
```python
# ✅ MAANG Standard
from enum import Enum
from dataclasses import dataclass
from typing import Final

# Type-safe constants
MAX_RETRIES: Final[int] = 3
DEFAULT_TIMEOUT: Final[float] = 30.0
EPSILON: Final[float] = 1e-10

class HTTPMethod(str, Enum):
    """HTTP methods enum for type safety."""
    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    DELETE = "DELETE"
    PATCH = "PATCH"

@dataclass(frozen=True)
class ServerConfig:
    """Immutable server configuration."""
    host: str
    port: int
    workers: int
    timeout: float
    
    def __post_init__(self) -> None:
        """Validate configuration values."""
        if self.port < 1 or self.port > 65535:
            raise ValueError(f"Invalid port: {self.port}")
        if self.workers < 1:
            raise ValueError(f"Invalid workers: {self.workers}")
```

### 5. Logging Standards
```python
# ✅ MAANG Standard
import logging
import structlog
from typing import Any, Dict

# Structured logging setup
logger = structlog.get_logger()

class CorrelationLoggerAdapter:
    """Logger adapter that includes correlation ID in all logs."""
    
    def __init__(self, logger: structlog.BoundLogger, correlation_id: str):
        self.logger = logger
        self.correlation_id = correlation_id
    
    def info(self, message: str, **kwargs: Any) -> None:
        """Log info with correlation ID."""
        self.logger.info(
            message,
            correlation_id=self.correlation_id,
            **kwargs
        )

# Usage
logger.info(
    "Processing request",
    method="POST",
    path="/api/v1/query",
    user_id=user_id,
    request_size=len(request_body),
    duration_ms=duration * 1000
)
```

### 6. Performance Optimization
```python
# ✅ MAANG Standard
from functools import lru_cache, cached_property
import asyncio
from concurrent.futures import ThreadPoolExecutor

class OptimizedProcessor:
    """High-performance data processor with caching and parallelization."""
    
    def __init__(self, max_workers: int = 4):
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._cache: Dict[str, Any] = {}
    
    @lru_cache(maxsize=1000)
    def expensive_computation(self, key: str) -> float:
        """Cache expensive computations."""
        # Complex calculation here
        return result
    
    @cached_property
    def configuration(self) -> Dict[str, Any]:
        """Lazy-load and cache configuration."""
        return self._load_configuration()
    
    async def process_batch(self, items: List[str]) -> List[float]:
        """Process items in parallel for better performance."""
        tasks = [
            asyncio.create_task(self._process_item(item))
            for item in items
        ]
        return await asyncio.gather(*tasks)
```

### 7. Testing Standards
```python
# ✅ MAANG Standard
import pytest
from unittest.mock import Mock, patch, AsyncMock
from hypothesis import given, strategies as st

class TestUserService:
    """Test suite for UserService with comprehensive coverage."""
    
    @pytest.fixture
    def user_service(self):
        """Provide configured UserService instance."""
        return UserService(config=TestConfig())
    
    @pytest.mark.parametrize("username,expected", [
        ("admin", True),
        ("user123", True),
        ("", False),
        (None, False),
        ("a" * 256, False),  # Too long
    ])
    def test_validate_username(self, username: str, expected: bool):
        """Test username validation with edge cases."""
        assert UserService.validate_username(username) == expected
    
    @given(st.text(min_size=1, max_size=50))
    def test_username_property(self, username: str):
        """Property-based testing for username validation."""
        # Test invariants
        result = UserService.validate_username(username)
        assert isinstance(result, bool)
    
    @pytest.mark.asyncio
    async def test_async_operation(self, user_service: UserService):
        """Test asynchronous operations."""
        result = await user_service.fetch_user_data("123")
        assert result is not None
```

## 📚 Documentation Standards

### 1. Module Documentation
```python
"""
User Management Module for Universal Knowledge Platform.

This module implements secure user authentication and authorization
following OWASP security guidelines and industry best practices.

Architecture:
    The module follows a layered architecture:
    - API Layer: REST endpoints for user operations
    - Service Layer: Business logic and validation
    - Repository Layer: Data persistence abstraction
    - Security Layer: Authentication and encryption

Security Features:
    - Bcrypt password hashing with salt
    - JWT tokens with expiration
    - Rate limiting per user
    - SQL injection prevention
    - XSS protection

Performance:
    - Connection pooling for database
    - Redis caching for sessions
    - Async operations throughout

Example:
    >>> from api.user_management import UserService
    >>> service = UserService()
    >>> user = await service.create_user(
    ...     username="john_doe",
    ...     email="john@example.com",
    ...     password="SecurePass123!"
    ... )

Authors:
    - Engineering Team <engineering@company.com>

Version:
    1.0.0 (2024-01-01)
"""
```

### 2. Class Documentation
```python
class UserRepository:
    """
    Repository for user data persistence with caching.
    
    This class implements the repository pattern for user data access,
    providing an abstraction layer over the database with built-in
    caching and connection pooling.
    
    Attributes:
        _db_pool: Database connection pool
        _cache: Redis cache instance
        _metrics: Prometheus metrics collector
        
    Thread Safety:
        This class is thread-safe and can be used in async contexts.
        
    Performance Characteristics:
        - Read operations: O(1) with cache hit, O(log n) with cache miss
        - Write operations: O(log n)
        - Cache TTL: 5 minutes for user data
        
    Example:
        >>> repo = UserRepository(db_url="postgresql://...")
        >>> user = await repo.get_by_id("user123")
        >>> await repo.update(user)
    """
```

## 🧪 Testing Standards

### 1. Test Structure
```python
# tests/test_user_service.py
"""
Comprehensive test suite for UserService.

Test Categories:
    - Unit tests: Individual method testing
    - Integration tests: Component interaction
    - Performance tests: Load and stress testing
    - Security tests: Vulnerability testing
"""

import pytest
from freezegun import freeze_time
from faker import Faker

fake = Faker()

class TestUserService:
    """Test UserService with 100% coverage goal."""
    
    # Fixtures
    @pytest.fixture(autouse=True)
    def setup(self, db_session, redis_mock):
        """Set up test environment."""
        self.db = db_session
        self.redis = redis_mock
        yield
        # Cleanup
        self.db.rollback()
        self.redis.flushall()
    
    # Happy path tests
    def test_create_user_success(self):
        """Test successful user creation."""
        pass
    
    # Edge cases
    def test_create_user_duplicate_email(self):
        """Test user creation with duplicate email."""
        pass
    
    # Error cases
    def test_create_user_invalid_email(self):
        """Test user creation with invalid email."""
        pass
    
    # Performance tests
    @pytest.mark.performance
    def test_create_user_performance(self, benchmark):
        """Benchmark user creation performance."""
        result = benchmark(self.service.create_user, 
                         username=fake.user_name(),
                         email=fake.email())
        assert result is not None
```

## 🚀 Performance Standards

### 1. Profiling and Optimization
```python
import cProfile
import pstats
from memory_profiler import profile
from line_profiler import LineProfiler

class PerformanceMonitor:
    """Monitor and optimize performance."""
    
    @staticmethod
    def profile_function(func):
        """Decorator for profiling function performance."""
        def wrapper(*args, **kwargs):
            profiler = cProfile.Profile()
            profiler.enable()
            result = func(*args, **kwargs)
            profiler.disable()
            
            stats = pstats.Stats(profiler)
            stats.sort_stats('cumulative')
            stats.print_stats(10)  # Top 10 functions
            
            return result
        return wrapper
    
    @profile
    def memory_intensive_operation(self):
        """Monitor memory usage."""
        # Operation code
        pass
```

### 2. Caching Strategy
```python
from aiocache import cached, Cache
from aiocache.serializers import JsonSerializer

class CachedService:
    """Service with intelligent caching."""
    
    @cached(
        ttl=300,  # 5 minutes
        cache=Cache.REDIS,
        key_builder=lambda f, *args, **kwargs: f"{f.__name__}:{args[1]}",
        serializer=JsonSerializer()
    )
    async def get_user_data(self, user_id: str) -> Dict[str, Any]:
        """Get user data with caching."""
        return await self._fetch_from_db(user_id)
    
    async def invalidate_user_cache(self, user_id: str) -> None:
        """Invalidate cache when data changes."""
        cache = Cache(Cache.REDIS)
        await cache.delete(f"get_user_data:{user_id}")
```

## 🔒 Security Standards

### 1. Input Validation
```python
from pydantic import BaseModel, validator, EmailStr
import re

class SecureUserInput(BaseModel):
    """Secure user input with comprehensive validation."""
    
    username: str
    email: EmailStr
    password: str
    
    @validator('username')
    def validate_username(cls, v: str) -> str:
        """Validate username against security rules."""
        if not v or len(v) < 3 or len(v) > 50:
            raise ValueError("Username must be 3-50 characters")
        
        if not re.match(r'^[a-zA-Z0-9_-]+$', v):
            raise ValueError("Username contains invalid characters")
        
        # Check against reserved words
        reserved = {'admin', 'root', 'system', 'api'}
        if v.lower() in reserved:
            raise ValueError("Username is reserved")
        
        return v
    
    @validator('password')
    def validate_password(cls, v: str) -> str:
        """Validate password strength."""
        if len(v) < 12:
            raise ValueError("Password must be at least 12 characters")
        
        checks = [
            (r'[A-Z]', "uppercase letter"),
            (r'[a-z]', "lowercase letter"),
            (r'[0-9]', "digit"),
            (r'[!@#$%^&*(),.?":{}|<>]', "special character")
        ]
        
        for pattern, msg in checks:
            if not re.search(pattern, v):
                raise ValueError(f"Password must contain at least one {msg}")
        
        return v
```

### 2. SQL Injection Prevention
```python
from sqlalchemy import text
from sqlalchemy.orm import Session

class SecureRepository:
    """Repository with SQL injection prevention."""
    
    def get_user_by_email(self, db: Session, email: str) -> Optional[User]:
        """Get user by email with parameterized query."""
        # ❌ Bad - SQL injection vulnerable
        # query = f"SELECT * FROM users WHERE email = '{email}'"
        
        # ✅ Good - Parameterized query
        stmt = text("SELECT * FROM users WHERE email = :email")
        result = db.execute(stmt, {"email": email})
        return result.first()
```

## 🏗️ Architecture Standards

### 1. SOLID Principles
```python
from abc import ABC, abstractmethod

# Single Responsibility Principle
class UserValidator:
    """Handles only user validation logic."""
    def validate(self, user: User) -> bool:
        # Validation logic only
        pass

class UserRepository:
    """Handles only data persistence."""
    def save(self, user: User) -> None:
        # Persistence logic only
        pass

# Open/Closed Principle
class PaymentProcessor(ABC):
    """Abstract base for payment processing."""
    @abstractmethod
    def process(self, amount: float) -> bool:
        pass

class StripeProcessor(PaymentProcessor):
    """Stripe implementation."""
    def process(self, amount: float) -> bool:
        # Stripe-specific logic
        pass

# Liskov Substitution Principle
class Rectangle:
    def __init__(self, width: float, height: float):
        self._width = width
        self._height = height
    
    @property
    def area(self) -> float:
        return self._width * self._height

# Interface Segregation Principle
class Readable(ABC):
    @abstractmethod
    def read(self) -> str:
        pass

class Writable(ABC):
    @abstractmethod
    def write(self, data: str) -> None:
        pass

# Dependency Inversion Principle
class UserService:
    def __init__(self, repository: UserRepositoryInterface):
        self._repository = repository  # Depend on abstraction
```

### 2. Design Patterns
```python
# Singleton Pattern
class DatabaseConnection:
    """Thread-safe singleton for database connection."""
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

# Factory Pattern
class AgentFactory:
    """Factory for creating different types of agents."""
    
    @staticmethod
    def create_agent(agent_type: str, config: Dict) -> BaseAgent:
        """Create agent based on type."""
        agents = {
            'retrieval': RetrievalAgent,
            'synthesis': SynthesisAgent,
            'factcheck': FactCheckAgent
        }
        
        agent_class = agents.get(agent_type)
        if not agent_class:
            raise ValueError(f"Unknown agent type: {agent_type}")
        
        return agent_class(config)

# Observer Pattern
class EventBus:
    """Event bus for decoupled communication."""
    
    def __init__(self):
        self._observers: Dict[str, List[Callable]] = defaultdict(list)
    
    def subscribe(self, event: str, callback: Callable) -> None:
        """Subscribe to an event."""
        self._observers[event].append(callback)
    
    async def publish(self, event: str, data: Any) -> None:
        """Publish event to all subscribers."""
        for callback in self._observers[event]:
            if asyncio.iscoroutinefunction(callback):
                await callback(data)
            else:
                callback(data)
```

## 📊 Monitoring Standards

### 1. Metrics Collection
```python
from prometheus_client import Counter, Histogram, Gauge
import time

# Define metrics
request_count = Counter(
    'http_requests_total',
    'Total HTTP requests',
    ['method', 'endpoint', 'status']
)

request_duration = Histogram(
    'http_request_duration_seconds',
    'HTTP request duration',
    ['method', 'endpoint']
)

active_connections = Gauge(
    'active_connections',
    'Number of active connections'
)

class MetricsMiddleware:
    """Middleware for collecting metrics."""
    
    async def __call__(self, request, call_next):
        """Collect metrics for each request."""
        start_time = time.time()
        
        # Track active connections
        active_connections.inc()
        
        try:
            response = await call_next(request)
            
            # Record metrics
            request_count.labels(
                method=request.method,
                endpoint=request.url.path,
                status=response.status_code
            ).inc()
            
            duration = time.time() - start_time
            request_duration.labels(
                method=request.method,
                endpoint=request.url.path
            ).observe(duration)
            
            return response
        finally:
            active_connections.dec()
```

### 2. Health Checks
```python
from enum import Enum
from dataclasses import dataclass
from typing import Dict, List

class HealthStatus(Enum):
    """Health check status enum."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"

@dataclass
class HealthCheckResult:
    """Result of a health check."""
    name: str
    status: HealthStatus
    message: str
    latency_ms: float
    metadata: Dict[str, Any]

class HealthChecker:
    """Comprehensive health checking system."""
    
    async def check_all(self) -> Dict[str, Any]:
        """Run all health checks in parallel."""
        checks = [
            self._check_database(),
            self._check_redis(),
            self._check_external_apis(),
            self._check_disk_space(),
            self._check_memory_usage()
        ]
        
        results = await asyncio.gather(*checks, return_exceptions=True)
        
        overall_status = self._determine_overall_status(results)
        
        return {
            "status": overall_status.value,
            "timestamp": datetime.utcnow().isoformat(),
            "checks": [r.dict() for r in results if isinstance(r, HealthCheckResult)],
            "version": self._get_version()
        }
```

## 🎯 Code Review Checklist

Before any code is merged, it must pass this checklist:

- [ ] **Type Hints**: All functions have complete type annotations
- [ ] **Documentation**: All modules, classes, and functions have docstrings
- [ ] **Testing**: Unit tests with >90% coverage
- [ ] **Error Handling**: Proper exception handling with custom exceptions
- [ ] **Logging**: Structured logging with appropriate levels
- [ ] **Security**: Input validation and sanitization
- [ ] **Performance**: No N+1 queries, proper caching
- [ ] **Code Quality**: Passes linting (flake8, mypy, black)
- [ ] **Dependencies**: All dependencies are pinned versions
- [ ] **Monitoring**: Metrics and health checks implemented

## 📈 Continuous Improvement

1. **Code Metrics Dashboard**
   - Cyclomatic complexity < 10
   - Maintainability index > 80
   - Test coverage > 90%
   - Documentation coverage > 95%

2. **Performance Benchmarks**
   - API response time p99 < 200ms
   - Database query time p99 < 50ms
   - Memory usage < 512MB per instance
   - CPU usage < 70% under normal load

3. **Security Audits**
   - Weekly dependency vulnerability scans
   - Monthly penetration testing
   - Quarterly security review
   - Annual third-party audit

---

**Remember**: Every line of code represents the company's commitment to excellence. Write code as if the person maintaining it is a violent psychopath who knows where you live. 😊 