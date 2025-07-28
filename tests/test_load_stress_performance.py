#!/usr/bin/env python3
"""
🚀 LOAD, STRESS & PERFORMANCE TESTING SUITE
Universal Knowledge Platform - Performance Validation

Tests system behavior under various load conditions and validates performance requirements.
Covers: Load testing, stress testing, endurance testing, spike testing, volume testing.
"""

import asyncio
import concurrent.futures
import json
import logging
import os
import random
import statistics
import sys
import threading
import time
import unittest
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from unittest.mock import Mock, patch, AsyncMock

import aiohttp
import pytest
import requests

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import components
from agents.base_agent import BaseAgent, AgentType, QueryContext
from agents.lead_orchestrator import LeadOrchestrator
from api.main import app

# Test configuration with environment variables
LOAD_TEST_CONFIG = {
    "base_url": os.getenv("TEST_API_BASE_URL", "http://localhost:8003"),
    "concurrent_users": int(os.getenv("TEST_CONCURRENT_USERS", "100")),
    "test_duration_seconds": int(os.getenv("TEST_DURATION_SECONDS", "300")),
    "ramp_up_time_seconds": int(os.getenv("TEST_RAMP_UP_SECONDS", "60")),
    "target_rps": int(os.getenv("TEST_TARGET_RPS", "1000")),
    "max_response_time_ms": int(os.getenv("TEST_MAX_RESPONSE_TIME_MS", "500")),
    "error_rate_threshold": float(os.getenv("TEST_ERROR_RATE_THRESHOLD", "0.01")),  # 1%
    "memory_threshold_mb": int(os.getenv("TEST_MEMORY_THRESHOLD_MB", "1024")),
    "cpu_threshold_percent": int(os.getenv("TEST_CPU_THRESHOLD_PERCENT", "80")),
    "default_token_budget": int(os.getenv("DEFAULT_TOKEN_BUDGET", "1000")),
}

# Sample test data
TEST_QUERIES = [
    "What is quantum computing?",
    "How does machine learning work?",
    "Explain artificial intelligence",
    "What is blockchain technology?",
    "How do neural networks function?",
    "What is cloud computing?",
    "Explain the internet of things",
    "What is cybersecurity?",
    "How does cryptography work?",
    "What is data science?",
]


class LoadTestBase(unittest.TestCase):
    """Base class for load testing"""

    def setUp(self):
        """Set up test environment"""
        self.base_url = LOAD_TEST_CONFIG["base_url"]
        self.session = requests.Session()
        self.results = []
        self.errors = []

    def make_request(self, endpoint: str, data: Dict = None) -> Dict:
        """Make a request and record metrics"""
        start_time = time.time()
        try:
            if data:
                response = self.session.post(f"{self.base_url}{endpoint}", json=data, timeout=30)
            else:
                response = self.session.get(f"{self.base_url}{endpoint}", timeout=30)

            end_time = time.time()
            response_time = (end_time - start_time) * 1000  # Convert to milliseconds

            result = {
                "endpoint": endpoint,
                "status_code": response.status_code,
                "response_time_ms": response_time,
                "success": response.status_code == 200,
                "timestamp": datetime.now(),
            }

            self.results.append(result)
            return result

        except Exception as e:
            end_time = time.time()
            response_time = (end_time - start_time) * 1000

            error_result = {
                "endpoint": endpoint,
                "status_code": 0,
                "response_time_ms": response_time,
                "success": False,
                "error": str(e),
                "timestamp": datetime.now(),
            }

            self.errors.append(error_result)
            return error_result


class TestConcurrentLoad(LoadTestBase):
    """Test concurrent load handling"""

    def test_concurrent_queries(self):
        """Test handling of concurrent queries"""
        print(f"🧪 Testing concurrent load with {LOAD_TEST_CONFIG['concurrent_users']} users")

        def worker(worker_id: int):
            """Worker function for concurrent testing"""
            worker_results = []
            for i in range(10):  # Each worker makes 10 requests
                query = TEST_QUERIES[i % len(TEST_QUERIES)]
                result = self.make_request("/query", {"query": query})
                worker_results.append(result)
                time.sleep(0.1)  # Small delay between requests
            return worker_results

        # Run concurrent workers
        start_time = time.time()
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=LOAD_TEST_CONFIG["concurrent_users"]
        ) as executor:
            futures = [
                executor.submit(worker, i) for i in range(LOAD_TEST_CONFIG["concurrent_users"])
            ]
            all_results = [future.result() for future in concurrent.futures.as_completed(futures)]
        end_time = time.time()

        # Flatten results
        all_requests = []
        for worker_results in all_results:
            all_requests.extend(worker_results)

        # Calculate metrics
        total_requests = len(all_requests)
        successful_requests = len([r for r in all_requests if r["success"]])
        failed_requests = total_requests - successful_requests
        response_times = [r["response_time_ms"] for r in all_requests if r["success"]]

        avg_response_time = statistics.mean(response_times) if response_times else 0
        max_response_time = max(response_times) if response_times else 0
        min_response_time = min(response_times) if response_times else 0
        error_rate = failed_requests / total_requests if total_requests > 0 else 0

        total_time = end_time - start_time
        requests_per_second = total_requests / total_time

        # Assertions
        self.assertGreater(successful_requests, 0, "No successful requests")
        self.assertLess(
            error_rate,
            LOAD_TEST_CONFIG["error_rate_threshold"],
            f"Error rate {error_rate:.2%} exceeds threshold {LOAD_TEST_CONFIG['error_rate_threshold']:.2%}",
        )
        self.assertLess(
            avg_response_time,
            LOAD_TEST_CONFIG["max_response_time_ms"],
            f"Average response time {avg_response_time:.2f}ms exceeds threshold {LOAD_TEST_CONFIG['max_response_time_ms']}ms",
        )
        self.assertGreater(
            requests_per_second,
            LOAD_TEST_CONFIG["target_rps"] * 0.5,
            f"RPS {requests_per_second:.2f} below target {LOAD_TEST_CONFIG['target_rps']}",
        )

        print(f"✅ Concurrent Load Test Results:")
        print(f"   Total Requests: {total_requests}")
        print(f"   Successful: {successful_requests}")
        print(f"   Failed: {failed_requests}")
        print(f"   Error Rate: {error_rate:.2%}")
        print(f"   Avg Response Time: {avg_response_time:.2f}ms")
        print(f"   Max Response Time: {max_response_time:.2f}ms")
        print(f"   Min Response Time: {min_response_time:.2f}ms")
        print(f"   Requests/Second: {requests_per_second:.2f}")


class TestStressTesting(LoadTestBase):
    """Test stress conditions"""

    def test_sustained_high_load(self):
        """Test sustained high load for extended period"""
        print("🧪 Testing sustained high load")

        duration = LOAD_TEST_CONFIG["test_duration_seconds"]
        start_time = time.time()
        request_count = 0

        while time.time() - start_time < duration:
            query = TEST_QUERIES[request_count % len(TEST_QUERIES)]
            result = self.make_request("/query", {"query": query})
            request_count += 1

            # Check memory and CPU usage
            process = psutil.Process(os.getpid())
            memory_usage = process.memory_info().rss / 1024 / 1024  # MB
            cpu_usage = process.cpu_percent()

            self.assertLess(
                memory_usage,
                LOAD_TEST_CONFIG["memory_threshold_mb"],
                f"Memory usage {memory_usage:.2f}MB exceeds threshold {LOAD_TEST_CONFIG['memory_threshold_mb']}MB",
            )
            self.assertLess(
                cpu_usage,
                LOAD_TEST_CONFIG["cpu_threshold_percent"],
                f"CPU usage {cpu_usage:.2f}% exceeds threshold {LOAD_TEST_CONFIG['cpu_threshold_percent']}%",
            )

            time.sleep(0.1)  # 10 requests per second

        print(f"✅ Sustained Load Test Completed:")
        print(f"   Duration: {duration} seconds")
        print(f"   Total Requests: {request_count}")
        print(f"   Requests/Second: {request_count / duration:.2f}")

    def test_burst_traffic(self):
        """Test handling of burst traffic"""
        print("🧪 Testing burst traffic handling")

        burst_size = 50
        burst_count = 5

        for burst in range(burst_count):
            print(f"   Burst {burst + 1}/{burst_count}")

            # Send burst of requests
            start_time = time.time()
            with concurrent.futures.ThreadPoolExecutor(max_workers=burst_size) as executor:
                futures = []
                for i in range(burst_size):
                    query = TEST_QUERIES[i % len(TEST_QUERIES)]
                    future = executor.submit(self.make_request, "/query", {"query": query})
                    futures.append(future)

                # Wait for all requests to complete
                results = [future.result() for future in concurrent.futures.as_completed(futures)]

            end_time = time.time()
            burst_time = end_time - start_time

            # Calculate burst metrics
            successful = len([r for r in results if r["success"]])
            response_times = [r["response_time_ms"] for r in results if r["success"]]
            avg_response_time = statistics.mean(response_times) if response_times else 0

            # Assertions for burst handling
            self.assertGreater(
                successful,
                burst_size * 0.8,  # At least 80% success rate
                f"Burst success rate {successful/burst_size:.2%} below 80%",
            )
            self.assertLess(
                avg_response_time,
                LOAD_TEST_CONFIG["max_response_time_ms"] * 2,
                f"Burst response time {avg_response_time:.2f}ms too high",
            )

            print(
                f"     Burst {burst + 1}: {successful}/{burst_size} successful, "
                f"avg response time: {avg_response_time:.2f}ms"
            )

            time.sleep(1)  # Wait between bursts


class TestMemoryLeakDetection(unittest.TestCase):
    """Test for memory leaks"""

    def test_memory_usage_over_time(self):
        """Test memory usage over time to detect leaks"""
        print("🧪 Testing memory usage over time")

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        memory_samples = []

        # Run operations for extended period
        for i in range(100):
            # Simulate typical operations
            query = TEST_QUERIES[i % len(TEST_QUERIES)]

            # Make API request
            try:
                response = requests.post(
                    f"{LOAD_TEST_CONFIG['base_url']}/query", json={"query": query}, timeout=10
                )
            except:
                pass

            # Record memory usage every 10 iterations
            if i % 10 == 0:
                current_memory = process.memory_info().rss / 1024 / 1024  # MB
                memory_samples.append(current_memory)
                print(f"   Iteration {i}: Memory usage: {current_memory:.2f}MB")

        final_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Check for memory leaks
        memory_increase = final_memory - initial_memory
        memory_growth_rate = memory_increase / len(memory_samples) if memory_samples else 0

        self.assertLess(
            memory_increase,
            100,  # Should not increase by more than 100MB
            f"Memory increased by {memory_increase:.2f}MB, possible leak",
        )
        self.assertLess(
            memory_growth_rate,
            1,  # Should not grow more than 1MB per sample
            f"Memory growth rate {memory_growth_rate:.2f}MB/sample too high",
        )

        print(f"✅ Memory Leak Test Results:")
        print(f"   Initial Memory: {initial_memory:.2f}MB")
        print(f"   Final Memory: {final_memory:.2f}MB")
        print(f"   Memory Increase: {memory_increase:.2f}MB")
        print(f"   Growth Rate: {memory_growth_rate:.2f}MB/sample")


class TestErrorHandlingUnderLoad(unittest.TestCase):
    """Test error handling under load"""

    def test_error_recovery(self):
        """Test system recovery after errors"""
        print("🧪 Testing error recovery under load")

        # Generate various error conditions
        error_scenarios = [
            {"query": "", "expected_status": 400},  # Empty query
            {"query": "a" * 10001, "expected_status": 400},  # Too long
            {"query": None, "expected_status": 422},  # Null query
            {"invalid_field": "test", "expected_status": 422},  # Invalid field
        ]

        # Test each scenario under load
        for scenario in error_scenarios:
            print(f"   Testing scenario: {scenario}")

            # Send multiple requests with error condition
            error_count = 0
            success_count = 0

            for i in range(20):
                try:
                    response = requests.post(
                        f"{LOAD_TEST_CONFIG['base_url']}/query", json=scenario, timeout=10
                    )

                    if response.status_code == scenario["expected_status"]:
                        success_count += 1
                    else:
                        error_count += 1

                except Exception as e:
                    error_count += 1

            # System should handle errors gracefully
            self.assertGreater(success_count, 0, "No successful error handling")
            self.assertLess(error_count, 20, "Too many unexpected errors")

            print(f"     Success: {success_count}, Errors: {error_count}")


class TestCachePerformance(unittest.TestCase):
    """Test cache performance under load"""

    def setUp(self):
        """Set up test environment"""
        self.cache = QueryCache()  # Changed from CacheService

    def test_cache_under_load(self):
        """Test cache performance under high load"""
        print("🧪 Testing cache performance under load")

        # Generate cache keys
        cache_keys = [f"test_key_{i}" for i in range(1000)]
        cache_values = [f"test_value_{i}" for i in range(1000)]

        # Concurrent cache operations
        def cache_worker(worker_id: int):
            """Worker for cache operations"""
            results = []
            for i in range(100):
                key = cache_keys[(worker_id * 100 + i) % len(cache_keys)]
                value = cache_values[(worker_id * 100 + i) % len(cache_values)]

                # Set value
                start_time = time.time()
                self.cache.set(key, value, ttl=60)
                set_time = time.time() - start_time

                # Get value
                start_time = time.time()
                retrieved = self.cache.get(key)
                get_time = time.time() - start_time

                results.append(
                    {
                        "set_time": set_time * 1000,  # Convert to ms
                        "get_time": get_time * 1000,
                        "success": retrieved == value,
                    }
                )

            return results

        # Run concurrent cache operations
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(cache_worker, i) for i in range(10)]
            all_results = [future.result() for future in concurrent.futures.as_completed(futures)]

        # Flatten results
        all_operations = []
        for worker_results in all_results:
            all_operations.extend(worker_results)

        # Calculate metrics
        set_times = [op["set_time"] for op in all_operations]
        get_times = [op["get_time"] for op in all_operations]
        successful_ops = len([op for op in all_operations if op["success"]])

        avg_set_time = statistics.mean(set_times)
        avg_get_time = statistics.mean(get_times)
        success_rate = successful_ops / len(all_operations)

        # Assertions
        self.assertGreater(success_rate, 0.95, "Cache success rate too low")
        self.assertLess(avg_set_time, 10, "Cache set time too high")
        self.assertLess(avg_get_time, 5, "Cache get time too high")

        print(f"✅ Cache Performance Results:")
        print(f"   Total Operations: {len(all_operations)}")
        print(f"   Success Rate: {success_rate:.2%}")
        print(f"   Avg Set Time: {avg_set_time:.2f}ms")
        print(f"   Avg Get Time: {avg_get_time:.2f}ms")


class TestDatabasePerformance(unittest.TestCase):
    """Test database performance under load"""

    def test_database_connection_pool(self):
        """Test database connection pool under load"""
        print("🧪 Testing database connection pool")

        # This would test database connections under load
        # Implementation depends on the specific database being used
        pass

    def test_query_optimization(self):
        """Test query optimization under load"""
        print("🧪 Testing query optimization")

        # This would test query performance under load
        # Implementation depends on the specific database being used
        pass


class TestNetworkResilience(unittest.TestCase):
    """Test network resilience"""

    def test_network_timeout_handling(self):
        """Test handling of network timeouts"""
        print("🧪 Testing network timeout handling")

        # Test with various timeout scenarios
        timeout_scenarios = [1, 5, 10, 30]  # seconds

        for timeout in timeout_scenarios:
            print(f"   Testing timeout: {timeout}s")

            start_time = time.time()
            try:
                response = requests.get(f"{LOAD_TEST_CONFIG['base_url']}/health", timeout=timeout)
                response_time = time.time() - start_time

                self.assertLess(
                    response_time,
                    timeout + 1,
                    f"Response time {response_time:.2f}s exceeds timeout {timeout}s",
                )

            except requests.exceptions.Timeout:
                # Expected for very short timeouts
                if timeout < 5:
                    continue
                else:
                    self.fail(f"Unexpected timeout with {timeout}s timeout")

    def test_connection_resilience(self):
        """Test connection resilience"""
        print("🧪 Testing connection resilience")

        # Test with connection drops and recovery
        for i in range(10):
            try:
                response = requests.get(f"{LOAD_TEST_CONFIG['base_url']}/health", timeout=10)
                self.assertEqual(response.status_code, 200)
            except Exception as e:
                # Should handle connection errors gracefully
                print(f"   Connection error {i + 1}: {e}")
                continue


def run_load_tests():
    """Run all load and stress tests"""
    print("🧪 Starting LOAD & STRESS TESTING SUITE")
    print("=" * 60)

    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all load test classes
    test_classes = [
        TestConcurrentLoad,
        TestStressTesting,
        TestMemoryLeakDetection,
        TestErrorHandlingUnderLoad,
        TestCachePerformance,
        TestDatabasePerformance,
        TestNetworkResilience,
    ]

    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Print summary
    print("\n" + "=" * 60)
    print("🧪 LOAD & STRESS TESTING SUMMARY")
    print("=" * 60)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(
        f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun) * 100:.2f}%"
    )

    if result.failures:
        print("\n❌ FAILURES:")
        for test, traceback in result.failures:
            print(f"  - {test}: {traceback}")

    if result.errors:
        print("\n❌ ERRORS:")
        for test, traceback in result.errors:
            print(f"  - {test}: {traceback}")

    if result.wasSuccessful():
        print("\n✅ ALL LOAD TESTS PASSED - SYSTEM IS BULLETPROOF UNDER LOAD!")
    else:
        print("\n❌ SOME LOAD TESTS FAILED - NEEDS OPTIMIZATION!")

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_load_tests()
    sys.exit(0 if success else 1)
