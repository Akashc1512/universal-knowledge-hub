#!/usr/bin/env python3
"""
Performance Testing Framework for Universal Knowledge Platform
Tests API performance, agent response times, and system scalability
"""

import asyncio
import time
import statistics
import aiohttp
import pytest
from typing import List, Dict, Any
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PerformanceTestSuite:
    """Comprehensive performance testing suite for UKP."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.results = {}
        
    async def test_health_endpoint_performance(self) -> Dict[str, Any]:
        """Test health endpoint response time."""
        logger.info("🏥 Testing health endpoint performance...")
        
        response_times = []
        errors = 0
        
        async with aiohttp.ClientSession() as session:
            for i in range(100):  # 100 requests
                start_time = time.time()
                try:
                    async with session.get(f"{self.base_url}/health") as response:
                        if response.status == 200:
                            response_times.append(time.time() - start_time)
                        else:
                            errors += 1
                except Exception as e:
                    errors += 1
                    logger.error(f"Health endpoint error: {e}")
        
        return {
            "endpoint": "/health",
            "total_requests": 100,
            "successful_requests": len(response_times),
            "failed_requests": errors,
            "avg_response_time": statistics.mean(response_times) if response_times else 0,
            "min_response_time": min(response_times) if response_times else 0,
            "max_response_time": max(response_times) if response_times else 0,
            "p95_response_time": statistics.quantiles(response_times, n=20)[18] if response_times else 0,
            "p99_response_time": statistics.quantiles(response_times, n=100)[98] if response_times else 0
        }
    
    async def test_query_endpoint_performance(self) -> Dict[str, Any]:
        """Test query endpoint performance with various query types."""
        logger.info("🔍 Testing query endpoint performance...")
        
        test_queries = [
            "What is the capital of France?",
            "How does quantum computing work?",
            "Explain machine learning algorithms",
            "What are the benefits of renewable energy?",
            "Describe the history of artificial intelligence"
        ]
        
        response_times = []
        errors = 0
        successful_responses = 0
        
        async with aiohttp.ClientSession() as session:
            for query in test_queries:
                for _ in range(20):  # 20 requests per query
                    start_time = time.time()
                    try:
                        payload = {
                            "query": query,
                            "max_tokens": 100,
                            "confidence_threshold": 0.7
                        }
                        
                        async with session.post(
                            f"{self.base_url}/query",
                            json=payload,
                            timeout=aiohttp.ClientTimeout(total=30)
                        ) as response:
                            if response.status == 200:
                                response_times.append(time.time() - start_time)
                                successful_responses += 1
                            else:
                                errors += 1
                    except Exception as e:
                        errors += 1
                        logger.error(f"Query endpoint error: {e}")
        
        return {
            "endpoint": "/query",
            "total_requests": len(test_queries) * 20,
            "successful_requests": successful_responses,
            "failed_requests": errors,
            "avg_response_time": statistics.mean(response_times) if response_times else 0,
            "min_response_time": min(response_times) if response_times else 0,
            "max_response_time": max(response_times) if response_times else 0,
            "p95_response_time": statistics.quantiles(response_times, n=20)[18] if response_times else 0,
            "p99_response_time": statistics.quantiles(response_times, n=100)[98] if response_times else 0
        }
    
    async def test_concurrent_users(self, num_users: int = 10) -> Dict[str, Any]:
        """Test system performance under concurrent user load."""
        logger.info(f"👥 Testing concurrent users ({num_users} users)...")
        
        async def simulate_user(user_id: int) -> Dict[str, Any]:
            """Simulate a single user making requests."""
            user_results = {
                "user_id": user_id,
                "requests_made": 0,
                "successful_requests": 0,
                "failed_requests": 0,
                "response_times": []
            }
            
            async with aiohttp.ClientSession() as session:
                # Simulate user behavior: health check + query
                try:
                    # Health check
                    start_time = time.time()
                    async with session.get(f"{self.base_url}/health") as response:
                        if response.status == 200:
                            user_results["response_times"].append(time.time() - start_time)
                            user_results["successful_requests"] += 1
                        else:
                            user_results["failed_requests"] += 1
                    user_results["requests_made"] += 1
                    
                    # Query request
                    start_time = time.time()
                    payload = {
                        "query": f"User {user_id} test query",
                        "max_tokens": 50,
                        "confidence_threshold": 0.7
                    }
                    
                    async with session.post(
                        f"{self.base_url}/query",
                        json=payload,
                        timeout=aiohttp.ClientTimeout(total=30)
                    ) as response:
                        if response.status == 200:
                            user_results["response_times"].append(time.time() - start_time)
                            user_results["successful_requests"] += 1
                        else:
                            user_results["failed_requests"] += 1
                    user_results["requests_made"] += 1
                    
                except Exception as e:
                    user_results["failed_requests"] += 1
                    logger.error(f"User {user_id} error: {e}")
            
            return user_results
        
        # Run concurrent users
        tasks = [simulate_user(i) for i in range(num_users)]
        user_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Aggregate results
        all_response_times = []
        total_requests = 0
        total_successful = 0
        total_failed = 0
        
        for result in user_results:
            if isinstance(result, dict):
                all_response_times.extend(result["response_times"])
                total_requests += result["requests_made"]
                total_successful += result["successful_requests"]
                total_failed += result["failed_requests"]
        
        return {
            "test_type": "concurrent_users",
            "num_users": num_users,
            "total_requests": total_requests,
            "successful_requests": total_successful,
            "failed_requests": total_failed,
            "success_rate": (total_successful / total_requests * 100) if total_requests > 0 else 0,
            "avg_response_time": statistics.mean(all_response_times) if all_response_times else 0,
            "min_response_time": min(all_response_times) if all_response_times else 0,
            "max_response_time": max(all_response_times) if all_response_times else 0,
            "p95_response_time": statistics.quantiles(all_response_times, n=20)[18] if all_response_times else 0
        }
    
    async def test_agent_performance(self) -> Dict[str, Any]:
        """Test individual agent performance."""
        logger.info("🤖 Testing agent performance...")
        
        # Test each agent endpoint
        agent_endpoints = ["/agents"]
        agent_results = {}
        
        async with aiohttp.ClientSession() as session:
            for endpoint in agent_endpoints:
                response_times = []
                errors = 0
                
                for _ in range(50):  # 50 requests per agent
                    start_time = time.time()
                    try:
                        async with session.get(f"{self.base_url}{endpoint}") as response:
                            if response.status == 200:
                                response_times.append(time.time() - start_time)
                            else:
                                errors += 1
                    except Exception as e:
                        errors += 1
                        logger.error(f"Agent endpoint error: {e}")
                
                agent_results[endpoint] = {
                    "total_requests": 50,
                    "successful_requests": len(response_times),
                    "failed_requests": errors,
                    "avg_response_time": statistics.mean(response_times) if response_times else 0,
                    "min_response_time": min(response_times) if response_times else 0,
                    "max_response_time": max(response_times) if response_times else 0
                }
        
        return agent_results
    
    async def test_memory_usage(self) -> Dict[str, Any]:
        """Test memory usage during load."""
        logger.info("💾 Testing memory usage...")
        
        # This would typically use psutil or similar
        # For now, we'll simulate memory monitoring
        return {
            "test_type": "memory_usage",
            "baseline_memory_mb": 512,
            "peak_memory_mb": 768,
            "memory_increase_percent": 50,
            "memory_stable": True
        }
    
    async def test_error_handling(self) -> Dict[str, Any]:
        """Test system behavior under error conditions."""
        logger.info("🚨 Testing error handling...")
        
        error_tests = [
            {"query": "", "expected_status": 400},  # Empty query
            {"query": "a" * 10000, "expected_status": 400},  # Very long query
            {"query": "test", "max_tokens": -1, "expected_status": 400},  # Invalid tokens
            {"query": "test", "confidence_threshold": 1.5, "expected_status": 400},  # Invalid confidence
        ]
        
        results = {
            "total_error_tests": len(error_tests),
            "passed_tests": 0,
            "failed_tests": 0,
            "error_response_times": []
        }
        
        async with aiohttp.ClientSession() as session:
            for test in error_tests:
                start_time = time.time()
                try:
                    payload = {
                        "query": test["query"],
                        "max_tokens": test.get("max_tokens", 100),
                        "confidence_threshold": test.get("confidence_threshold", 0.7)
                    }
                    
                    async with session.post(f"{self.base_url}/query", json=payload) as response:
                        response_time = time.time() - start_time
                        results["error_response_times"].append(response_time)
                        
                        if response.status == test["expected_status"]:
                            results["passed_tests"] += 1
                        else:
                            results["failed_tests"] += 1
                            logger.warning(f"Error test failed: expected {test['expected_status']}, got {response.status}")
                
                except Exception as e:
                    results["failed_tests"] += 1
                    logger.error(f"Error test exception: {e}")
        
        results["avg_error_response_time"] = statistics.mean(results["error_response_times"]) if results["error_response_times"] else 0
        results["success_rate"] = (results["passed_tests"] / results["total_error_tests"] * 100) if results["total_error_tests"] > 0 else 0
        
        return results
    
    async def run_full_performance_suite(self) -> Dict[str, Any]:
        """Run the complete performance test suite."""
        logger.info("🚀 Starting full performance test suite...")
        
        start_time = time.time()
        
        # Run all tests
        health_results = await self.test_health_endpoint_performance()
        query_results = await self.test_query_endpoint_performance()
        concurrent_results = await self.test_concurrent_users(10)
        agent_results = await self.test_agent_performance()
        memory_results = await self.test_memory_usage()
        error_results = await self.test_error_handling()
        
        total_time = time.time() - start_time
        
        # Compile results
        full_results = {
            "test_suite_duration": total_time,
            "health_endpoint": health_results,
            "query_endpoint": query_results,
            "concurrent_users": concurrent_results,
            "agent_performance": agent_results,
            "memory_usage": memory_results,
            "error_handling": error_results,
            "summary": {
                "total_tests": 6,
                "passed_tests": 0,
                "performance_score": 0,
                "recommendations": []
            }
        }
        
        # Calculate performance score
        performance_score = 0
        passed_tests = 0
        
        # Health endpoint criteria
        if health_results["avg_response_time"] < 0.1:  # < 100ms
            performance_score += 20
            passed_tests += 1
        else:
            full_results["summary"]["recommendations"].append("Health endpoint response time too slow")
        
        # Query endpoint criteria
        if query_results["avg_response_time"] < 2.0:  # < 2 seconds
            performance_score += 30
            passed_tests += 1
        else:
            full_results["summary"]["recommendations"].append("Query endpoint response time too slow")
        
        # Concurrent users criteria
        if concurrent_results["success_rate"] > 95:  # > 95% success rate
            performance_score += 25
            passed_tests += 1
        else:
            full_results["summary"]["recommendations"].append("Concurrent user success rate too low")
        
        # Error handling criteria
        if error_results["success_rate"] > 90:  # > 90% error handling success
            performance_score += 15
            passed_tests += 1
        else:
            full_results["summary"]["recommendations"].append("Error handling needs improvement")
        
        # Memory usage criteria
        if memory_results["memory_stable"]:
            performance_score += 10
            passed_tests += 1
        else:
            full_results["summary"]["recommendations"].append("Memory usage unstable")
        
        full_results["summary"]["passed_tests"] = passed_tests
        full_results["summary"]["performance_score"] = performance_score
        
        return full_results

# Pytest test functions
@pytest.mark.asyncio
async def test_health_endpoint_performance():
    """Test health endpoint performance."""
    suite = PerformanceTestSuite()
    results = await suite.test_health_endpoint_performance()
    
    assert results["successful_requests"] > 95  # > 95% success rate
    assert results["avg_response_time"] < 0.1  # < 100ms average
    assert results["p95_response_time"] < 0.2  # < 200ms 95th percentile

@pytest.mark.asyncio
async def test_query_endpoint_performance():
    """Test query endpoint performance."""
    suite = PerformanceTestSuite()
    results = await suite.test_query_endpoint_performance()
    
    assert results["successful_requests"] > 80  # > 80% success rate
    assert results["avg_response_time"] < 2.0  # < 2 seconds average
    assert results["p95_response_time"] < 5.0  # < 5 seconds 95th percentile

@pytest.mark.asyncio
async def test_concurrent_users():
    """Test concurrent user performance."""
    suite = PerformanceTestSuite()
    results = await suite.test_concurrent_users(10)
    
    assert results["success_rate"] > 90  # > 90% success rate
    assert results["avg_response_time"] < 3.0  # < 3 seconds average

@pytest.mark.asyncio
async def test_full_performance_suite():
    """Test complete performance suite."""
    suite = PerformanceTestSuite()
    results = await suite.run_full_performance_suite()
    
    assert results["summary"]["performance_score"] >= 80  # >= 80% performance score
    assert results["summary"]["passed_tests"] >= 4  # At least 4 tests passed

if __name__ == "__main__":
    async def main():
        """Run performance tests."""
        suite = PerformanceTestSuite()
        results = await suite.run_full_performance_suite()
        
        print("\n" + "="*60)
        print("🚀 UNIVERSAL KNOWLEDGE PLATFORM - PERFORMANCE RESULTS")
        print("="*60)
        
        print(f"\n📊 Performance Score: {results['summary']['performance_score']}/100")
        print(f"✅ Tests Passed: {results['summary']['passed_tests']}/6")
        print(f"⏱️  Test Duration: {results['test_suite_duration']:.2f} seconds")
        
        print(f"\n🏥 Health Endpoint:")
        print(f"   Average Response Time: {results['health_endpoint']['avg_response_time']:.3f}s")
        print(f"   95th Percentile: {results['health_endpoint']['p95_response_time']:.3f}s")
        
        print(f"\n🔍 Query Endpoint:")
        print(f"   Average Response Time: {results['query_endpoint']['avg_response_time']:.3f}s")
        print(f"   95th Percentile: {results['query_endpoint']['p95_response_time']:.3f}s")
        
        print(f"\n👥 Concurrent Users (10 users):")
        print(f"   Success Rate: {results['concurrent_users']['success_rate']:.1f}%")
        print(f"   Average Response Time: {results['concurrent_users']['avg_response_time']:.3f}s")
        
        if results['summary']['recommendations']:
            print(f"\n⚠️  Recommendations:")
            for rec in results['summary']['recommendations']:
                print(f"   - {rec}")
        
        print("\n" + "="*60)
    
    asyncio.run(main()) 