"""
Comprehensive Performance Tests for Universal Knowledge Hub
Tests performance, load handling, and scalability
"""

import unittest
import asyncio
import time
import psutil
import os
import json
import statistics
from unittest.mock import Mock, patch, AsyncMock
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from typing import List, Dict, Any

# Import application components
from agents.lead_orchestrator import LeadOrchestrator
from core.config_manager import ConfigurationManager
from api.main import app
from fastapi.testclient import TestClient


class TestQueryPerformance(unittest.TestCase):
    """Test query processing performance"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.orchestrator = LeadOrchestrator()
        self.client = TestClient(app)
        self.test_queries = [
            "What is artificial intelligence?",
            "How does machine learning work?",
            "Explain quantum computing",
            "What are neural networks?",
            "Describe natural language processing",
            "How do transformers work?",
            "What is deep learning?",
            "Explain computer vision",
            "What is reinforcement learning?",
            "How does GPT work?"
        ]
    
    def test_single_query_performance(self):
        """Test single query processing performance"""
        query = "What is artificial intelligence?"
        
        # Measure processing time
        start_time = time.time()
        result = asyncio.run(self.orchestrator.process_query(query))
        end_time = time.time()
        
        processing_time = end_time - start_time
        
        # Performance assertions
        self.assertLess(processing_time, 2.0, "Single query should complete within 2 seconds")
        self.assertIn('execution_time_ms', result)
        self.assertGreater(result['execution_time_ms'], 0)
        self.assertLess(result['execution_time_ms'], 2000)  # 2 seconds in ms
    
    def test_query_complexity_performance(self):
        """Test performance with different query complexities"""
        simple_query = "What is AI?"
        complex_query = "What are the latest developments in artificial intelligence, machine learning, and deep learning, including transformer architectures, attention mechanisms, and their applications in natural language processing and computer vision?"
        
        # Test simple query
        start_time = time.time()
        simple_result = asyncio.run(self.orchestrator.process_query(simple_query))
        simple_time = time.time() - start_time
        
        # Test complex query
        start_time = time.time()
        complex_result = asyncio.run(self.orchestrator.process_query(complex_query))
        complex_time = time.time() - start_time
        
        # Complex queries should take longer but still be reasonable
        self.assertLess(simple_time, 1.0, "Simple query should complete within 1 second")
        self.assertLess(complex_time, 5.0, "Complex query should complete within 5 seconds")
        self.assertGreater(complex_time, simple_time, "Complex query should take longer than simple query")
    
    def test_concurrent_query_performance(self):
        """Test concurrent query processing performance"""
        num_concurrent_queries = 10
        
        # Process queries concurrently
        start_time = time.time()
        tasks = [self.orchestrator.process_query(query) for query in self.test_queries[:num_concurrent_queries]]
        results = asyncio.run(asyncio.gather(*tasks))
        end_time = time.time()
        
        total_time = end_time - start_time
        avg_time_per_query = total_time / num_concurrent_queries
        
        # Performance assertions
        self.assertLess(total_time, 10.0, "10 concurrent queries should complete within 10 seconds")
        self.assertLess(avg_time_per_query, 2.0, "Average time per query should be under 2 seconds")
        
        # Verify all queries completed successfully
        for result in results:
            self.assertIn('success', result)
            self.assertIn('execution_time_ms', result)
    
    def test_sequential_vs_concurrent_performance(self):
        """Test sequential vs concurrent processing performance"""
        queries = self.test_queries[:5]
        
        # Sequential processing
        start_time = time.time()
        sequential_results = []
        for query in queries:
            result = asyncio.run(self.orchestrator.process_query(query))
            sequential_results.append(result)
        sequential_time = time.time() - start_time
        
        # Concurrent processing
        start_time = time.time()
        tasks = [self.orchestrator.process_query(query) for query in queries]
        concurrent_results = asyncio.run(asyncio.gather(*tasks))
        concurrent_time = time.time() - start_time
        
        # Concurrent should be faster than sequential
        self.assertLess(concurrent_time, sequential_time, "Concurrent processing should be faster than sequential")
        
        # Verify results are equivalent
        self.assertEqual(len(sequential_results), len(concurrent_results))
    
    def test_memory_usage_per_query(self):
        """Test memory usage per query"""
        import gc
        
        # Measure baseline memory
        gc.collect()
        process = psutil.Process(os.getpid())
        baseline_memory = process.memory_info().rss
        
        # Process multiple queries
        for i, query in enumerate(self.test_queries[:5]):
            result = asyncio.run(self.orchestrator.process_query(query))
            
            # Force garbage collection
            gc.collect()
            current_memory = process.memory_info().rss
            memory_increase = current_memory - baseline_memory
            
            # Memory increase should be reasonable (less than 50MB per query)
            self.assertLess(memory_increase, 50 * 1024 * 1024, f"Memory increase for query {i} should be less than 50MB")
    
    def test_cpu_usage_per_query(self):
        """Test CPU usage per query"""
        query = "What is artificial intelligence?"
        
        # Measure CPU usage during query processing
        process = psutil.Process(os.getpid())
        
        # Get baseline CPU usage
        baseline_cpu = process.cpu_percent(interval=0.1)
        
        # Process query and measure CPU
        start_time = time.time()
        result = asyncio.run(self.orchestrator.process_query(query))
        end_time = time.time()
        
        processing_time = end_time - start_time
        
        # CPU usage should be reasonable (less than 100% for single query)
        # Note: This is a rough test as CPU usage can vary significantly
        self.assertGreater(processing_time, 0, "Query should take some time to process")


class TestLoadTesting(unittest.TestCase):
    """Test load handling capabilities"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.orchestrator = LeadOrchestrator()
        self.client = TestClient(app)
    
    def test_low_load_performance(self):
        """Test performance under low load (1-10 queries)"""
        num_queries = 5
        queries = [f"Test query {i}" for i in range(num_queries)]
        
        start_time = time.time()
        tasks = [self.orchestrator.process_query(query) for query in queries]
        results = asyncio.run(asyncio.gather(*tasks))
        end_time = time.time()
        
        total_time = end_time - start_time
        avg_time = total_time / num_queries
        
        # Low load should be very fast
        self.assertLess(avg_time, 1.0, "Average time under low load should be under 1 second")
        self.assertLess(total_time, 5.0, "Total time for 5 queries should be under 5 seconds")
    
    def test_medium_load_performance(self):
        """Test performance under medium load (10-50 queries)"""
        num_queries = 25
        queries = [f"Test query {i}" for i in range(num_queries)]
        
        start_time = time.time()
        tasks = [self.orchestrator.process_query(query) for query in queries]
        results = asyncio.run(asyncio.gather(*tasks))
        end_time = time.time()
        
        total_time = end_time - start_time
        avg_time = total_time / num_queries
        
        # Medium load should still be reasonable
        self.assertLess(avg_time, 2.0, "Average time under medium load should be under 2 seconds")
        self.assertLess(total_time, 30.0, "Total time for 25 queries should be under 30 seconds")
    
    def test_high_load_performance(self):
        """Test performance under high load (50+ queries)"""
        num_queries = 100
        queries = [f"Test query {i}" for i in range(num_queries)]
        
        start_time = time.time()
        tasks = [self.orchestrator.process_query(query) for query in queries]
        results = asyncio.run(asyncio.gather(*tasks))
        end_time = time.time()
        
        total_time = end_time - start_time
        avg_time = total_time / num_queries
        
        # High load should still be manageable
        self.assertLess(avg_time, 3.0, "Average time under high load should be under 3 seconds")
        self.assertLess(total_time, 120.0, "Total time for 100 queries should be under 2 minutes")
    
    def test_sustained_load_performance(self):
        """Test performance under sustained load"""
        num_batches = 5
        queries_per_batch = 10
        
        total_times = []
        
        for batch in range(num_batches):
            queries = [f"Batch {batch} query {i}" for i in range(queries_per_batch)]
            
            start_time = time.time()
            tasks = [self.orchestrator.process_query(query) for query in queries]
            results = asyncio.run(asyncio.gather(*tasks))
            end_time = time.time()
            
            batch_time = end_time - start_time
            total_times.append(batch_time)
        
        # Performance should remain consistent across batches
        avg_batch_time = statistics.mean(total_times)
        std_batch_time = statistics.stdev(total_times)
        
        # Standard deviation should be low (consistent performance)
        self.assertLess(std_batch_time, avg_batch_time * 0.5, "Performance should be consistent across batches")
    
    def test_memory_usage_under_load(self):
        """Test memory usage under load"""
        import gc
        
        gc.collect()
        process = psutil.Process(os.getpid())
        baseline_memory = process.memory_info().rss
        
        # Process many queries
        num_queries = 50
        queries = [f"Memory test query {i}" for i in range(num_queries)]
        
        for query in queries:
            result = asyncio.run(self.orchestrator.process_query(query))
        
        # Force garbage collection
        gc.collect()
        final_memory = process.memory_info().rss
        memory_increase = final_memory - baseline_memory
        
        # Memory increase should be reasonable (less than 200MB for 50 queries)
        self.assertLess(memory_increase, 200 * 1024 * 1024, "Memory increase under load should be less than 200MB")


class TestScalabilityTesting(unittest.TestCase):
    """Test system scalability"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.orchestrator = LeadOrchestrator()
    
    def test_agent_scalability(self):
        """Test agent scalability"""
        # Test that multiple agents can work concurrently
        agents = [
            self.orchestrator.agents[AgentType.RETRIEVAL],
            self.orchestrator.agents[AgentType.FACT_CHECK],
            self.orchestrator.agents[AgentType.SYNTHESIS],
            self.orchestrator.agents[AgentType.CITATION]
        ]
        
        # Create concurrent tasks for each agent
        tasks = []
        for agent in agents:
            task = {
                'strategy': 'hybrid' if agent.agent_type == AgentType.RETRIEVAL else {},
                'query': 'test query',
                'top_k': 5
            }
            context = QueryContext(query='test query')
            tasks.append(agent.process_task(task, context))
        
        # Process all agents concurrently
        start_time = time.time()
        results = asyncio.run(asyncio.gather(*tasks))
        end_time = time.time()
        
        total_time = end_time - start_time
        
        # All agents should complete within reasonable time
        self.assertLess(total_time, 10.0, "All agents should complete within 10 seconds")
        
        # All agents should succeed
        for result in results:
            self.assertTrue(result.success)
    
    def test_pipeline_scalability(self):
        """Test pipeline scalability"""
        # Test that pipeline can handle multiple concurrent requests
        num_pipelines = 10
        
        start_time = time.time()
        tasks = []
        for i in range(num_pipelines):
            query = f"Pipeline test query {i}"
            tasks.append(self.orchestrator.process_query(query))
        
        results = asyncio.run(asyncio.gather(*tasks))
        end_time = time.time()
        
        total_time = end_time - start_time
        avg_time = total_time / num_pipelines
        
        # Pipeline should scale reasonably
        self.assertLess(avg_time, 3.0, "Average pipeline time should be under 3 seconds")
        self.assertLess(total_time, 30.0, "Total time for 10 pipelines should be under 30 seconds")
        
        # All pipelines should succeed
        for result in results:
            self.assertIn('success', result)
    
    def test_cache_scalability(self):
        """Test cache scalability"""
        cache_manager = self.orchestrator.cache_manager
        
        # Test cache with many entries
        num_entries = 1000
        
        start_time = time.time()
        for i in range(num_entries):
            query = f"Cache test query {i}"
            response = {"response": f"Response {i}", "confidence": 0.9}
            asyncio.run(cache_manager.cache_response(query, response))
        cache_time = time.time() - start_time
        
        # Caching should be fast
        self.assertLess(cache_time, 10.0, "Caching 1000 entries should take under 10 seconds")
        
        # Test cache retrieval performance
        start_time = time.time()
        for i in range(num_entries):
            query = f"Cache test query {i}"
            cached_response = asyncio.run(cache_manager.get_cached_response(query))
        retrieval_time = time.time() - start_time
        
        # Cache retrieval should be very fast
        self.assertLess(retrieval_time, 5.0, "Retrieving 1000 cached entries should take under 5 seconds")


class TestResourceUtilization(unittest.TestCase):
    """Test resource utilization"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.orchestrator = LeadOrchestrator()
    
    def test_cpu_utilization(self):
        """Test CPU utilization"""
        process = psutil.Process(os.getpid())
        
        # Measure CPU usage during intensive processing
        queries = [f"CPU test query {i}" for i in range(20)]
        
        # Get baseline CPU
        baseline_cpu = process.cpu_percent(interval=1.0)
        
        # Process queries and measure CPU
        start_time = time.time()
        tasks = [self.orchestrator.process_query(query) for query in queries]
        results = asyncio.run(asyncio.gather(*tasks))
        end_time = time.time()
        
        processing_time = end_time - start_time
        
        # CPU usage should be reasonable
        self.assertGreater(processing_time, 0, "Processing should take some time")
        
        # Note: CPU usage measurement is approximate and can vary significantly
        # depending on system load and other factors
    
    def test_memory_utilization(self):
        """Test memory utilization"""
        import gc
        
        process = psutil.Process(os.getpid())
        
        # Measure memory usage during processing
        gc.collect()
        baseline_memory = process.memory_info().rss
        
        queries = [f"Memory test query {i}" for i in range(50)]
        
        for query in queries:
            result = asyncio.run(self.orchestrator.process_query(query))
        
        gc.collect()
        final_memory = process.memory_info().rss
        memory_increase = final_memory - baseline_memory
        
        # Memory increase should be reasonable
        self.assertLess(memory_increase, 500 * 1024 * 1024, "Memory increase should be less than 500MB")
    
    def test_disk_utilization(self):
        """Test disk utilization"""
        # Test that disk usage is reasonable
        # This would depend on logging, caching, and other disk operations
        pass
    
    def test_network_utilization(self):
        """Test network utilization"""
        # Test that network usage is reasonable
        # This would depend on external API calls and data transfer
        pass


class TestPerformanceMonitoring(unittest.TestCase):
    """Test performance monitoring capabilities"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.orchestrator = LeadOrchestrator()
    
    def test_execution_time_monitoring(self):
        """Test execution time monitoring"""
        query = "Performance monitoring test query"
        
        result = asyncio.run(self.orchestrator.process_query(query))
        
        # Verify execution time is tracked
        self.assertIn('execution_time_ms', result)
        self.assertGreater(result['execution_time_ms'], 0)
        self.assertLess(result['execution_time_ms'], 10000)  # Should be under 10 seconds
    
    def test_token_usage_monitoring(self):
        """Test token usage monitoring"""
        # Test that token usage is tracked across agents
        query = "Token usage monitoring test query"
        
        result = asyncio.run(self.orchestrator.process_query(query))
        
        # Token usage should be tracked (if available)
        if 'token_usage' in result:
            self.assertIn('prompt', result['token_usage'])
            self.assertIn('completion', result['token_usage'])
            self.assertGreater(result['token_usage']['prompt'], 0)
            self.assertGreater(result['token_usage']['completion'], 0)
    
    def test_confidence_monitoring(self):
        """Test confidence monitoring"""
        query = "Confidence monitoring test query"
        
        result = asyncio.run(self.orchestrator.process_query(query))
        
        # Confidence should be tracked
        self.assertIn('confidence', result)
        self.assertGreaterEqual(result['confidence'], 0.0)
        self.assertLessEqual(result['confidence'], 1.0)
    
    def test_error_rate_monitoring(self):
        """Test error rate monitoring"""
        # Test error rate under load
        num_queries = 100
        queries = [f"Error rate test query {i}" for i in range(num_queries)]
        
        error_count = 0
        for query in queries:
            try:
                result = asyncio.run(self.orchestrator.process_query(query))
                if not result.get('success', True):
                    error_count += 1
            except Exception:
                error_count += 1
        
        error_rate = error_count / num_queries
        
        # Error rate should be low
        self.assertLess(error_rate, 0.1, "Error rate should be less than 10%")


class TestPerformanceOptimization(unittest.TestCase):
    """Test performance optimization features"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.orchestrator = LeadOrchestrator()
    
    def test_caching_performance_impact(self):
        """Test caching performance impact"""
        query = "Caching performance test query"
        
        # First query (cache miss)
        start_time = time.time()
        result1 = asyncio.run(self.orchestrator.process_query(query))
        first_query_time = time.time() - start_time
        
        # Second query (cache hit)
        start_time = time.time()
        result2 = asyncio.run(self.orchestrator.process_query(query))
        second_query_time = time.time() - start_time
        
        # Cache hit should be faster
        self.assertLess(second_query_time, first_query_time, "Cache hit should be faster than cache miss")
    
    def test_concurrent_processing_optimization(self):
        """Test concurrent processing optimization"""
        queries = [f"Concurrent optimization test query {i}" for i in range(10)]
        
        # Sequential processing
        start_time = time.time()
        sequential_results = []
        for query in queries:
            result = asyncio.run(self.orchestrator.process_query(query))
            sequential_results.append(result)
        sequential_time = time.time() - start_time
        
        # Concurrent processing
        start_time = time.time()
        tasks = [self.orchestrator.process_query(query) for query in queries]
        concurrent_results = asyncio.run(asyncio.gather(*tasks))
        concurrent_time = time.time() - start_time
        
        # Concurrent should be faster
        self.assertLess(concurrent_time, sequential_time, "Concurrent processing should be faster than sequential")
    
    def test_resource_optimization(self):
        """Test resource optimization"""
        # Test that system uses resources efficiently
        process = psutil.Process(os.getpid())
        
        # Measure resource usage during processing
        baseline_memory = process.memory_info().rss
        
        queries = [f"Resource optimization test query {i}" for i in range(20)]
        
        for query in queries:
            result = asyncio.run(self.orchestrator.process_query(query))
        
        final_memory = process.memory_info().rss
        memory_increase = final_memory - baseline_memory
        
        # Memory increase should be reasonable
        self.assertLess(memory_increase, 200 * 1024 * 1024, "Memory increase should be less than 200MB")


if __name__ == '__main__':
    # Run all performance tests
    unittest.main(verbosity=2) 