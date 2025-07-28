"""
Load Testing Suite for Universal Knowledge Platform
Tests performance under high load and stress conditions.
"""

import pytest
import asyncio
import time
import statistics
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any
import requests
from fastapi.testclient import TestClient

from api.main import app


class TestLoadTesting:
    """Load testing scenarios."""

    @pytest.fixture
    def client(self):
        return TestClient(app)

    def test_concurrent_users(self, client):
        """Test handling of many concurrent users."""

        def make_request(user_id: int) -> Dict[str, Any]:
            """Make a single request."""
            start_time = time.time()
            try:
                response = client.post(
                    "/query",
                    json={"query": f"Load test query from user {user_id}", "max_tokens": 500},
                )
                end_time = time.time()
                return {
                    "user_id": user_id,
                    "status_code": response.status_code,
                    "response_time": end_time - start_time,
                    "success": response.status_code == 200,
                }
            except Exception as e:
                return {
                    "user_id": user_id,
                    "status_code": 0,
                    "response_time": 0,
                    "success": False,
                    "error": str(e),
                }

        # Simulate 100 concurrent users
        num_users = 100
        with ThreadPoolExecutor(max_workers=20) as executor:
            futures = [executor.submit(make_request, i) for i in range(num_users)]
            results = [future.result() for future in futures]

        # Analyze results
        successful_requests = [r for r in results if r["success"]]
        failed_requests = [r for r in results if not r["success"]]
        response_times = [r["response_time"] for r in results if r["response_time"] > 0]

        # Assertions
        success_rate = len(successful_requests) / len(results)
        assert success_rate > 0.8, f"Success rate too low: {success_rate:.2%}"

        if response_times:
            avg_response_time = statistics.mean(response_times)
            max_response_time = max(response_times)
            assert (
                avg_response_time < 10.0
            ), f"Average response time too high: {avg_response_time:.2f}s"
            assert max_response_time < 30.0, f"Max response time too high: {max_response_time:.2f}s"

    def test_sustained_load(self, client):
        """Test sustained load over time."""
        duration = 60  # 1 minute test
        requests_per_second = 10
        total_requests = duration * requests_per_second

        results = []
        start_time = time.time()

        for i in range(total_requests):
            request_start = time.time()
            try:
                response = client.post(
                    "/query", json={"query": f"Sustained load test {i}", "max_tokens": 300}
                )
                request_time = time.time() - request_start
                results.append(
                    {
                        "request_id": i,
                        "status_code": response.status_code,
                        "response_time": request_time,
                        "success": response.status_code == 200,
                    }
                )
            except Exception as e:
                results.append(
                    {
                        "request_id": i,
                        "status_code": 0,
                        "response_time": 0,
                        "success": False,
                        "error": str(e),
                    }
                )

            # Rate limiting
            time.sleep(1.0 / requests_per_second)

        total_time = time.time() - start_time

        # Analyze sustained load results
        successful_requests = [r for r in results if r["success"]]
        response_times = [r["response_time"] for r in results if r["response_time"] > 0]

        success_rate = len(successful_requests) / len(results)
        avg_response_time = statistics.mean(response_times) if response_times else 0

        # Assertions for sustained load
        assert success_rate > 0.7, f"Sustained load success rate too low: {success_rate:.2%}"
        assert (
            avg_response_time < 5.0
        ), f"Sustained load avg response time too high: {avg_response_time:.2f}s"
        assert total_time < duration * 1.5, f"Test took too long: {total_time:.2f}s"

    def test_memory_leak_detection(self, client):
        """Test for memory leaks under load."""
        import psutil
        import os

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss

        # Make many requests to stress memory
        for i in range(1000):
            client.post("/query", json={"query": f"Memory leak test {i}", "max_tokens": 100})

            if i % 100 == 0:
                # Check memory every 100 requests
                current_memory = process.memory_info().rss
                memory_increase = current_memory - initial_memory

                # Memory should not grow excessively
                assert (
                    memory_increase < 200 * 1024 * 1024
                ), f"Memory leak detected: {memory_increase / 1024 / 1024:.1f}MB"

        final_memory = process.memory_info().rss
        total_memory_increase = final_memory - initial_memory

        # Final memory check
        assert (
            total_memory_increase < 100 * 1024 * 1024
        ), f"Excessive memory usage: {total_memory_increase / 1024 / 1024:.1f}MB"

    def test_error_rate_under_load(self, client):
        """Test error rate remains low under load."""
        num_requests = 500
        errors = 0

        for i in range(num_requests):
            try:
                response = client.post(
                    "/query", json={"query": f"Error rate test {i}", "max_tokens": 200}
                )
                if response.status_code >= 500:
                    errors += 1
            except Exception:
                errors += 1

        error_rate = errors / num_requests
        assert error_rate < 0.1, f"Error rate too high under load: {error_rate:.2%}"

    def test_concurrent_different_queries(self, client):
        """Test handling of many different concurrent queries."""
        queries = [
            "What is artificial intelligence?",
            "Explain machine learning algorithms",
            "How do neural networks work?",
            "What is deep learning?",
            "Explain natural language processing",
            "What is computer vision?",
            "How does reinforcement learning work?",
            "What is supervised learning?",
            "Explain unsupervised learning",
            "What is transfer learning?",
        ]

        def make_query_request(query: str, user_id: int) -> Dict[str, Any]:
            start_time = time.time()
            try:
                response = client.post(
                    "/query",
                    json={"query": query, "max_tokens": 400, "user_context": {"user_id": user_id}},
                )
                return {
                    "user_id": user_id,
                    "query": query,
                    "status_code": response.status_code,
                    "response_time": time.time() - start_time,
                    "success": response.status_code == 200,
                }
            except Exception as e:
                return {
                    "user_id": user_id,
                    "query": query,
                    "status_code": 0,
                    "response_time": 0,
                    "success": False,
                    "error": str(e),
                }

        # Make concurrent requests with different queries
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = []
            for i in range(50):  # 50 concurrent requests
                query = queries[i % len(queries)]
                futures.append(executor.submit(make_query_request, query, i))

            results = [future.result() for future in futures]

        # Analyze results
        successful_requests = [r for r in results if r["success"]]
        response_times = [r["response_time"] for r in results if r["response_time"] > 0]

        success_rate = len(successful_requests) / len(results)
        avg_response_time = statistics.mean(response_times) if response_times else 0

        assert success_rate > 0.8, f"Success rate too low: {success_rate:.2%}"
        assert avg_response_time < 8.0, f"Average response time too high: {avg_response_time:.2f}s"


class TestStressTesting:
    """Stress testing scenarios."""

    @pytest.fixture
    def client(self):
        return TestClient(app)

    def test_extreme_concurrency(self, client):
        """Test extreme concurrency conditions."""
        num_concurrent = 200

        def make_request():
            return client.post(
                "/query", json={"query": "Extreme concurrency test", "max_tokens": 100}
            )

        with ThreadPoolExecutor(max_workers=50) as executor:
            futures = [executor.submit(make_request) for _ in range(num_concurrent)]
            results = [future.result() for future in futures]

        # Should not crash completely
        successful_responses = [r for r in results if r.status_code == 200]
        assert len(successful_responses) > 0, "No successful responses under extreme load"

    def test_large_payload_stress(self, client):
        """Test handling of large payloads under stress."""
        large_query = "a" * 5000  # Large but not over limit

        def make_large_request():
            return client.post("/query", json={"query": large_query, "max_tokens": 1000})

        with ThreadPoolExecutor(max_workers=20) as executor:
            futures = [executor.submit(make_large_request) for _ in range(100)]
            results = [future.result() for future in futures]

        # Should handle large payloads gracefully
        successful_responses = [r for r in results if r.status_code == 200]
        assert len(successful_responses) > 0, "Failed to handle large payloads"

    def test_rapid_fire_requests(self, client):
        """Test rapid-fire requests without delays."""
        results = []

        for i in range(100):
            start_time = time.time()
            try:
                response = client.post(
                    "/query", json={"query": f"Rapid fire test {i}", "max_tokens": 50}
                )
                response_time = time.time() - start_time
                results.append(
                    {
                        "request_id": i,
                        "status_code": response.status_code,
                        "response_time": response_time,
                        "success": response.status_code == 200,
                    }
                )
            except Exception as e:
                results.append(
                    {
                        "request_id": i,
                        "status_code": 0,
                        "response_time": 0,
                        "success": False,
                        "error": str(e),
                    }
                )

        # Should handle rapid requests
        successful_requests = [r for r in results if r["success"]]
        assert len(successful_requests) > 0, "Failed to handle rapid-fire requests"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
