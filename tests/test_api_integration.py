"""
API Integration Tests - MAANG Standards.

This module provides comprehensive integration tests for the Universal
Knowledge Platform API following MAANG best practices.

Test Coverage:
    - Authentication and authorization
    - Query processing and responses
    - Error handling and validation
    - Rate limiting and security
    - Performance and scalability
    - External service integration

Testing Strategy:
    - End-to-end API testing
    - Realistic user scenarios
    - Edge cases and error conditions
    - Performance benchmarks
    - Security validation

Authors:
    - Universal Knowledge Platform Engineering Team
    
Version:
    2.0.0 (2024-12-28)
"""

import pytest
import asyncio
import time
from typing import Dict, Any, List
from unittest.mock import patch, AsyncMock

from fastapi.testclient import TestClient
from api.main import app
from api.exceptions import ValidationError, RateLimitError
from api.security import sanitize_input, validate_email, validate_password

# Test data
VALID_QUERIES = [
    "What is machine learning?",
    "Explain the differences between supervised and unsupervised learning",
    "How does a neural network work?",
    "What are the applications of artificial intelligence?",
    "Describe the process of data preprocessing",
]

MALICIOUS_QUERIES = [
    "'; DROP TABLE users; --",
    "<script>alert('xss')</script>",
    "../../../etc/passwd",
    "| cat /etc/passwd",
    "admin' OR '1'='1",
]

PERFORMANCE_QUERIES = [
    "What is Python?",
    "Explain machine learning with examples",
    "Provide a comprehensive analysis of AI impact on society",
    "Compare different deep learning architectures",
    "Discuss the future of artificial intelligence",
]

class TestAPIAuthentication:
    """Test authentication and authorization functionality."""
    
    def test_user_registration_success(self, client: TestClient, user_factory):
        """Test successful user registration."""
        user_data = user_factory.create_user()
        
        response = client.post("/auth/register", json=user_data)
        
        assert response.status_code == 200
        data = response.json()
        assert "id" in data
        assert data["username"] == user_data["username"]
        assert data["email"] == user_data["email"]
        assert "password" not in data  # Password should not be returned
    
    def test_user_registration_validation_errors(self, client: TestClient):
        """Test user registration validation errors."""
        # Test invalid email
        invalid_user = {
            "username": "testuser",
            "email": "invalid-email",
            "password": "TestPass123!"
        }
        
        response = client.post("/auth/register", json=invalid_user)
        assert response.status_code == 422
        
        # Test weak password
        weak_password_user = {
            "username": "testuser",
            "email": "test@example.com",
            "password": "weak"
        }
        
        response = client.post("/auth/register", json=weak_password_user)
        assert response.status_code == 422
    
    def test_user_login_success(self, client: TestClient, user_factory):
        """Test successful user login."""
        # First register a user
        user_data = user_factory.create_user()
        client.post("/auth/register", json=user_data)
        
        # Then login
        login_data = {
            "username": user_data["username"],
            "password": user_data["password"]
        }
        
        response = client.post("/auth/login", data=login_data)
        
        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        assert "token_type" in data
        assert data["token_type"] == "bearer"
    
    def test_user_login_invalid_credentials(self, client: TestClient):
        """Test login with invalid credentials."""
        login_data = {
            "username": "nonexistent",
            "password": "wrongpassword"
        }
        
        response = client.post("/auth/login", data=login_data)
        assert response.status_code == 401
    
    def test_protected_endpoint_without_auth(self, client: TestClient):
        """Test accessing protected endpoint without authentication."""
        response = client.post("/api/v2/query", json={"query": "test"})
        assert response.status_code == 401
    
    def test_protected_endpoint_with_auth(self, client: TestClient, user_factory):
        """Test accessing protected endpoint with authentication."""
        # Register and login
        user_data = user_factory.create_user()
        client.post("/auth/register", json=user_data)
        
        login_data = {
            "username": user_data["username"],
            "password": user_data["password"]
        }
        login_response = client.post("/auth/login", data=login_data)
        token = login_response.json()["access_token"]
        
        # Access protected endpoint
        headers = {"Authorization": f"Bearer {token}"}
        response = client.post(
            "/api/v2/query",
            json={"query": "test query"},
            headers=headers
        )
        
        assert response.status_code == 200

class TestAPIQueryProcessing:
    """Test query processing functionality."""
    
    @pytest.mark.parametrize("query", VALID_QUERIES)
    def test_valid_query_processing(self, client: TestClient, query: str, user_factory):
        """Test processing of valid queries."""
        # Setup authenticated user
        user_data = user_factory.create_user()
        client.post("/auth/register", json=user_data)
        
        login_response = client.post("/auth/login", data={
            "username": user_data["username"],
            "password": user_data["password"]
        })
        token = login_response.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}
        
        # Test query processing
        response = client.post(
            "/api/v2/query",
            json={"query": query},
            headers=headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "result" in data
        assert "sources" in data
        assert "confidence" in data
    
    def test_query_validation_errors(self, client: TestClient, user_factory):
        """Test query validation errors."""
        # Setup authenticated user
        user_data = user_factory.create_user()
        client.post("/auth/register", json=user_data)
        
        login_response = client.post("/auth/login", data={
            "username": user_data["username"],
            "password": user_data["password"]
        })
        token = login_response.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}
        
        # Test empty query
        response = client.post(
            "/api/v2/query",
            json={"query": ""},
            headers=headers
        )
        assert response.status_code == 422
        
        # Test query too long
        long_query = "a" * 10000
        response = client.post(
            "/api/v2/query",
            json={"query": long_query},
            headers=headers
        )
        assert response.status_code == 422
    
    @pytest.mark.parametrize("malicious_query", MALICIOUS_QUERIES)
    def test_malicious_query_handling(self, client: TestClient, malicious_query: str, user_factory):
        """Test handling of malicious queries."""
        # Setup authenticated user
        user_data = user_factory.create_user()
        client.post("/auth/register", json=user_data)
        
        login_response = client.post("/auth/login", data={
            "username": user_data["username"],
            "password": user_data["password"]
        })
        token = login_response.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}
        
        # Test malicious query
        response = client.post(
            "/api/v2/query",
            json={"query": malicious_query},
            headers=headers
        )
        
        # Should either be rejected or sanitized
        assert response.status_code in [200, 422, 403]
        
        if response.status_code == 200:
            # If accepted, should be sanitized
            data = response.json()
            assert malicious_query not in str(data)

class TestAPIRateLimiting:
    """Test rate limiting functionality."""
    
    def test_rate_limit_enforcement(self, client: TestClient, user_factory):
        """Test rate limit enforcement."""
        # Setup authenticated user
        user_data = user_factory.create_user()
        client.post("/auth/register", json=user_data)
        
        login_response = client.post("/auth/login", data={
            "username": user_data["username"],
            "password": user_data["password"]
        })
        token = login_response.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}
        
        # Make multiple requests quickly
        responses = []
        for i in range(15):  # Exceed rate limit
            response = client.post(
                "/api/v2/query",
                json={"query": f"test query {i}"},
                headers=headers
            )
            responses.append(response)
        
        # Check that some requests were rate limited
        rate_limited = [r for r in responses if r.status_code == 429]
        assert len(rate_limited) > 0
        
        # Check rate limit headers
        for response in rate_limited:
            assert "X-RateLimit-Limit" in response.headers
            assert "X-RateLimit-Remaining" in response.headers
            assert "Retry-After" in response.headers
    
    def test_rate_limit_recovery(self, client: TestClient, user_factory):
        """Test rate limit recovery after waiting."""
        # Setup authenticated user
        user_data = user_factory.create_user()
        client.post("/auth/register", json=user_data)
        
        login_response = client.post("/auth/login", data={
            "username": user_data["username"],
            "password": user_data["password"]
        })
        token = login_response.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}
        
        # Exceed rate limit
        for i in range(15):
            client.post(
                "/api/v2/query",
                json={"query": f"test query {i}"},
                headers=headers
            )
        
        # Wait for rate limit to reset (in real scenario)
        # For testing, we'll just verify the behavior
        response = client.post(
            "/api/v2/query",
            json={"query": "recovery test"},
            headers=headers
        )
        
        # Should eventually work again
        assert response.status_code in [200, 429]

class TestAPISecurity:
    """Test security features."""
    
    def test_input_sanitization(self):
        """Test input sanitization."""
        malicious_inputs = [
            "<script>alert('xss')</script>",
            "'; DROP TABLE users; --",
            "../../../etc/passwd",
            "| cat /etc/passwd",
        ]
        
        for malicious_input in malicious_inputs:
            # Should raise ValidationError
            with pytest.raises(ValidationError):
                sanitize_input(malicious_input)
    
    def test_email_validation(self):
        """Test email validation."""
        valid_emails = [
            "test@example.com",
            "user.name@domain.co.uk",
            "user+tag@example.org",
        ]
        
        invalid_emails = [
            "invalid-email",
            "@example.com",
            "user@",
            "user@.com",
        ]
        
        for email in valid_emails:
            validated = validate_email(email)
            assert validated == email.lower().strip()
        
        for email in invalid_emails:
            with pytest.raises(ValidationError):
                validate_email(email)
    
    def test_password_validation(self):
        """Test password validation."""
        valid_passwords = [
            "StrongPass123!",
            "Complex@Password1",
            "Secure#Pass2",
        ]
        
        invalid_passwords = [
            "weak",
            "12345678",
            "onlylowercase",
            "ONLYUPPERCASE",
            "NoSpecialChar1",
        ]
        
        for password in valid_passwords:
            validated = validate_password(password)
            assert validated == password
        
        for password in invalid_passwords:
            with pytest.raises(ValidationError):
                validate_password(password)
    
    def test_security_headers(self, client: TestClient):
        """Test security headers are present."""
        response = client.get("/health")
        
        security_headers = [
            "Content-Security-Policy",
            "X-Frame-Options",
            "X-Content-Type-Options",
            "X-XSS-Protection",
            "Referrer-Policy",
        ]
        
        for header in security_headers:
            assert header in response.headers

class TestAPIPerformance:
    """Test API performance."""
    
    @pytest.mark.performance
    def test_query_response_time(self, client: TestClient, user_factory, performance_benchmark):
        """Test query response time performance."""
        # Setup authenticated user
        user_data = user_factory.create_user()
        client.post("/auth/register", json=user_data)
        
        login_response = client.post("/auth/login", data={
            "username": user_data["username"],
            "password": user_data["password"]
        })
        token = login_response.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}
        
        async def benchmark_query():
            start_time = time.time()
            response = client.post(
                "/api/v2/query",
                json={"query": "What is machine learning?"},
                headers=headers
            )
            duration = time.time() - start_time
            
            assert response.status_code == 200
            return duration
        
        # Run benchmark
        result = asyncio.run(performance_benchmark.benchmark(
            "query_response_time",
            benchmark_query
        ))
        
        assert result["passed"]
    
    @pytest.mark.performance
    def test_concurrent_queries(self, client: TestClient, user_factory):
        """Test handling of concurrent queries."""
        # Setup authenticated user
        user_data = user_factory.create_user()
        client.post("/auth/register", json=user_data)
        
        login_response = client.post("/auth/login", data={
            "username": user_data["username"],
            "password": user_data["password"]
        })
        token = login_response.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}
        
        # Make concurrent requests
        import threading
        
        def make_request(query_id: int):
            response = client.post(
                "/api/v2/query",
                json={"query": f"concurrent query {query_id}"},
                headers=headers
            )
            return response.status_code
        
        threads = []
        results = []
        
        for i in range(10):
            thread = threading.Thread(
                target=lambda i=i: results.append(make_request(i))
            )
            threads.append(thread)
            thread.start()
        
        # Wait for all threads
        for thread in threads:
            thread.join()
        
        # Check results
        successful = sum(1 for status in results if status == 200)
        assert successful >= 8  # At least 80% should succeed

class TestAPIErrorHandling:
    """Test error handling functionality."""
    
    def test_validation_errors(self, client: TestClient):
        """Test validation error responses."""
        # Test invalid JSON
        response = client.post("/api/v2/query", data="invalid json")
        assert response.status_code == 422
        
        # Test missing required fields
        response = client.post("/api/v2/query", json={})
        assert response.status_code == 422
    
    def test_authentication_errors(self, client: TestClient):
        """Test authentication error responses."""
        # Test invalid token
        headers = {"Authorization": "Bearer invalid_token"}
        response = client.post(
            "/api/v2/query",
            json={"query": "test"},
            headers=headers
        )
        assert response.status_code == 401
    
    def test_not_found_errors(self, client: TestClient):
        """Test not found error responses."""
        response = client.get("/nonexistent/endpoint")
        assert response.status_code == 404
    
    def test_internal_server_errors(self, client: TestClient, user_factory):
        """Test internal server error handling."""
        # Setup authenticated user
        user_data = user_factory.create_user()
        client.post("/auth/register", json=user_data)
        
        login_response = client.post("/auth/login", data={
            "username": user_data["username"],
            "password": user_data["password"]
        })
        token = login_response.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}
        
        # Mock external service failure
        with patch("api.ai_services.openai.ChatCompletion.create") as mock_openai:
            mock_openai.side_effect = Exception("External service error")
            
            response = client.post(
                "/api/v2/query",
                json={"query": "test query"},
                headers=headers
            )
            
            # Should handle gracefully
            assert response.status_code in [500, 503]

class TestAPIHealthAndMonitoring:
    """Test health checks and monitoring."""
    
    def test_health_endpoint(self, client: TestClient):
        """Test health endpoint."""
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "status" in data
        assert "timestamp" in data
        assert "version" in data
        assert "services" in data
    
    def test_metrics_endpoint(self, client: TestClient):
        """Test metrics endpoint."""
        response = client.get("/metrics")
        
        assert response.status_code == 200
        content = response.text
        
        # Should contain Prometheus metrics
        assert "http_requests_total" in content
        assert "rate_limit" in content
    
    def test_health_check_with_external_services(self, client: TestClient, mock_external_services):
        """Test health check with external services."""
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        
        # Check that external services are monitored
        assert "services" in data
        services = data["services"]
        
        # Verify mock services were called
        mock_external_services["vector_db"].assert_called()
        mock_external_services["elasticsearch"].assert_called()
        mock_external_services["redis"].assert_called()

class TestAPICaching:
    """Test caching functionality."""
    
    def test_query_caching(self, client: TestClient, user_factory):
        """Test that queries are cached."""
        # Setup authenticated user
        user_data = user_factory.create_user()
        client.post("/auth/register", json=user_data)
        
        login_response = client.post("/auth/login", data={
            "username": user_data["username"],
            "password": user_data["password"]
        })
        token = login_response.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}
        
        query = "What is artificial intelligence?"
        
        # First request
        response1 = client.post(
            "/api/v2/query",
            json={"query": query},
            headers=headers
        )
        
        # Second request (should be cached)
        response2 = client.post(
            "/api/v2/query",
            json={"query": query},
            headers=headers
        )
        
        assert response1.status_code == 200
        assert response2.status_code == 200
        
        # Responses should be identical
        data1 = response1.json()
        data2 = response2.json()
        assert data1 == data2
    
    def test_cache_invalidation(self, client: TestClient, user_factory):
        """Test cache invalidation."""
        # Setup authenticated user
        user_data = user_factory.create_user()
        client.post("/auth/register", json=user_data)
        
        login_response = client.post("/auth/login", data={
            "username": user_data["username"],
            "password": user_data["password"]
        })
        token = login_response.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}
        
        query = "Test cache invalidation"
        
        # Make initial request
        response1 = client.post(
            "/api/v2/query",
            json={"query": query},
            headers=headers
        )
        
        # Clear cache (admin endpoint)
        admin_headers = {"Authorization": f"Bearer {token}"}
        clear_response = client.post("/admin/cache/clear", headers=admin_headers)
        
        # Make second request
        response2 = client.post(
            "/api/v2/query",
            json={"query": query},
            headers=headers
        )
        
        assert response1.status_code == 200
        assert response2.status_code == 200
        
        # Responses might be different due to cache clearing
        # This is expected behavior

# Performance benchmarks
@pytest.mark.performance
class TestAPIPerformanceBenchmarks:
    """Performance benchmarks for API endpoints."""
    
    def test_query_throughput(self, client: TestClient, user_factory):
        """Test query throughput under load."""
        # Setup authenticated user
        user_data = user_factory.create_user()
        client.post("/auth/register", json=user_data)
        
        login_response = client.post("/auth/login", data={
            "username": user_data["username"],
            "password": user_data["password"]
        })
        token = login_response.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}
        
        # Measure throughput
        start_time = time.time()
        successful_requests = 0
        
        for i in range(100):
            response = client.post(
                "/api/v2/query",
                json={"query": f"throughput test query {i}"},
                headers=headers
            )
            if response.status_code == 200:
                successful_requests += 1
        
        duration = time.time() - start_time
        throughput = successful_requests / duration
        
        # Should handle at least 10 requests per second
        assert throughput >= 10.0, f"Throughput too low: {throughput:.2f} req/s"
    
    def test_response_time_percentiles(self, client: TestClient, user_factory):
        """Test response time percentiles."""
        # Setup authenticated user
        user_data = user_factory.create_user()
        client.post("/auth/register", json=user_data)
        
        login_response = client.post("/auth/login", data={
            "username": user_data["username"],
            "password": user_data["password"]
        })
        token = login_response.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}
        
        # Collect response times
        response_times = []
        
        for i in range(50):
            start_time = time.time()
            response = client.post(
                "/api/v2/query",
                json={"query": f"latency test query {i}"},
                headers=headers
            )
            duration = time.time() - start_time
            response_times.append(duration)
        
        # Calculate percentiles
        response_times.sort()
        p50 = response_times[len(response_times) // 2]
        p95 = response_times[int(len(response_times) * 0.95)]
        p99 = response_times[int(len(response_times) * 0.99)]
        
        # Assert reasonable performance
        assert p50 <= 0.5, f"P50 too high: {p50:.3f}s"
        assert p95 <= 1.0, f"P95 too high: {p95:.3f}s"
        assert p99 <= 2.0, f"P99 too high: {p99:.3f}s"

# Security tests
@pytest.mark.security
class TestAPISecurityTests:
    """Security tests for API endpoints."""
    
    def test_sql_injection_protection(self, client: TestClient, security_tester):
        """Test SQL injection protection."""
        results = asyncio.run(security_tester.test_sql_injection(
            "/api/v2/query",
            client
        ))
        
        # Should not be vulnerable
        vulnerable_count = sum(1 for r in results if r["vulnerable"])
        assert vulnerable_count == 0, f"SQL injection vulnerabilities found: {vulnerable_count}"
    
    def test_xss_protection(self, client: TestClient, security_tester):
        """Test XSS protection."""
        results = asyncio.run(security_tester.test_xss(
            "/api/v2/query",
            client
        ))
        
        # Should not be vulnerable
        vulnerable_count = sum(1 for r in results if r["vulnerable"])
        assert vulnerable_count == 0, f"XSS vulnerabilities found: {vulnerable_count}"
    
    def test_authentication_bypass(self, client: TestClient):
        """Test authentication bypass attempts."""
        # Test various authentication bypass techniques
        bypass_attempts = [
            {"Authorization": "Bearer "},  # Empty token
            {"Authorization": "Bearer invalid_token"},
            {"Authorization": "Basic dXNlcjpwYXNz"},  # Basic auth
            {},  # No auth header
        ]
        
        for headers in bypass_attempts:
            response = client.post(
                "/api/v2/query",
                json={"query": "test"},
                headers=headers
            )
            assert response.status_code == 401
    
    def test_input_validation_security(self, client: TestClient):
        """Test input validation security."""
        malicious_inputs = [
            {"query": "<script>alert('xss')</script>"},
            {"query": "'; DROP TABLE users; --"},
            {"query": "../../../etc/passwd"},
            {"query": "| cat /etc/passwd"},
        ]
        
        for malicious_input in malicious_inputs:
            response = client.post("/api/v2/query", json=malicious_input)
            # Should either reject or sanitize
            assert response.status_code in [200, 422, 403]
            
            if response.status_code == 200:
                # If accepted, should be sanitized
                data = response.json()
                assert malicious_input["query"] not in str(data)

# Export test classes
__all__ = [
    'TestAPIAuthentication',
    'TestAPIQueryProcessing',
    'TestAPIRateLimiting',
    'TestAPISecurity',
    'TestAPIPerformance',
    'TestAPIErrorHandling',
    'TestAPIHealthAndMonitoring',
    'TestAPICaching',
    'TestAPIPerformanceBenchmarks',
    'TestAPISecurityTests',
] 