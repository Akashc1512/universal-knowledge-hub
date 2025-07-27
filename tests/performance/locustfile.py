"""
Performance testing for Universal Knowledge Platform using Locust.
Tests API endpoints under various load conditions.
"""

import time
import random
import json
from locust import HttpUser, task, between, events
from typing import Dict, Any


class UniversalKnowledgeHubUser(HttpUser):
    """
    Locust user class for testing Universal Knowledge Platform.
    Simulates real user behavior with various API interactions.
    """
    
    wait_time = between(1, 3)  # Wait 1-3 seconds between requests
    
    def on_start(self):
        """Initialize user session."""
        self.api_key = None
        self.session_data = {}
        
        # Try to authenticate (if credentials are available)
        self._authenticate()
    
    def _authenticate(self):
        """Authenticate user and get API key."""
        try:
            # This would be replaced with actual authentication
            # For now, we'll use a mock API key
            self.api_key = "test-api-key-12345"
            self.client.headers.update({
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            })
        except Exception as e:
            print(f"Authentication failed: {e}")
    
    @task(3)
    def health_check(self):
        """Test health endpoint (high frequency)."""
        with self.client.get("/health", catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Health check failed: {response.status_code}")
    
    @task(2)
    def get_agents(self):
        """Test agents listing endpoint."""
        with self.client.get("/agents", catch_response=True) as response:
            if response.status_code == 200:
                data = response.json()
                if "agents" in data and len(data["agents"]) > 0:
                    response.success()
                else:
                    response.failure("No agents found in response")
            else:
                response.failure(f"Agents endpoint failed: {response.status_code}")
    
    @task(5)
    def simple_query(self):
        """Test simple query processing (most common)."""
        queries = [
            "What is machine learning?",
            "How does AI work?",
            "Explain neural networks",
            "What is deep learning?",
            "How do computers learn?",
            "What is artificial intelligence?",
            "Explain data science",
            "What is natural language processing?",
            "How does computer vision work?",
            "What is reinforcement learning?"
        ]
        
        query_data = {
            "query": random.choice(queries),
            "max_tokens": 500,
            "confidence_threshold": 0.7,
            "user_context": {"domain": "technology"}
        }
        
        with self.client.post("/query", json=query_data, catch_response=True) as response:
            if response.status_code == 200:
                data = response.json()
                if "answer" in data and "confidence" in data:
                    response.success()
                    # Store response time for analysis
                    self.session_data["last_query_time"] = response.elapsed.total_seconds()
                else:
                    response.failure("Invalid response format")
            else:
                response.failure(f"Query failed: {response.status_code}")
    
    @task(2)
    def complex_query(self):
        """Test complex query processing."""
        complex_queries = [
            "Explain the relationship between machine learning, deep learning, and artificial intelligence, including their applications in modern technology and future prospects.",
            "Compare and contrast supervised learning, unsupervised learning, and reinforcement learning with real-world examples and use cases.",
            "Discuss the ethical implications of artificial intelligence in healthcare, including privacy concerns, bias in algorithms, and the future of medical diagnosis.",
            "Analyze the impact of natural language processing on modern communication, including chatbots, translation services, and content generation.",
            "Explore the challenges and opportunities of implementing AI in autonomous vehicles, covering safety, regulation, and technological limitations."
        ]
        
        query_data = {
            "query": random.choice(complex_queries),
            "max_tokens": 1000,
            "confidence_threshold": 0.8,
            "user_context": {
                "domain": "technology",
                "complexity": "high",
                "detail_level": "comprehensive"
            }
        }
        
        with self.client.post("/query", json=query_data, catch_response=True) as response:
            if response.status_code == 200:
                data = response.json()
                if "answer" in data and len(data["answer"]) > 100:
                    response.success()
                else:
                    response.failure("Complex query response too short")
            else:
                response.failure(f"Complex query failed: {response.status_code}")
    
    @task(1)
    def bulk_queries(self):
        """Test bulk query processing."""
        bulk_queries = [
            "What is AI?",
            "What is ML?",
            "What is DL?",
            "What is NLP?",
            "What is CV?"
        ]
        
        for query in bulk_queries:
            query_data = {
                "query": query,
                "max_tokens": 200,
                "confidence_threshold": 0.6
            }
            
            with self.client.post("/query", json=query_data, catch_response=True) as response:
                if response.status_code == 200:
                    response.success()
                else:
                    response.failure(f"Bulk query failed: {response.status_code}")
            
            # Small delay between bulk queries
            time.sleep(0.1)
    
    @task(1)
    def edge_case_queries(self):
        """Test edge cases and error conditions."""
        edge_queries = [
            "",  # Empty query
            "a" * 1000,  # Very long query
            "🤖🚀💻",  # Emoji query
            "SELECT * FROM users",  # SQL injection attempt
            "<script>alert('xss')</script>",  # XSS attempt
            "file:///etc/passwd",  # Path traversal attempt
            "admin' OR '1'='1",  # SQL injection
            "javascript:alert('xss')",  # JavaScript injection
        ]
        
        query_data = {
            "query": random.choice(edge_queries),
            "max_tokens": 100,
            "confidence_threshold": 0.5
        }
        
        with self.client.post("/query", json=query_data, catch_response=True) as response:
            # Edge cases should be handled gracefully
            if response.status_code in [200, 400, 422]:
                response.success()
            else:
                response.failure(f"Edge case not handled properly: {response.status_code}")


class AdminUser(UniversalKnowledgeHubUser):
    """
    Admin user class for testing administrative functions.
    """
    
    wait_time = between(2, 5)  # Slower pace for admin actions
    
    @task(1)
    def system_health_check(self):
        """Test detailed system health check."""
        with self.client.get("/health", catch_response=True) as response:
            if response.status_code == 200:
                data = response.json()
                # Check for detailed health information
                if "status" in data and "timestamp" in data:
                    response.success()
                else:
                    response.failure("Incomplete health information")
            else:
                response.failure(f"Health check failed: {response.status_code}")
    
    @task(1)
    def get_system_stats(self):
        """Test system statistics endpoint (if available)."""
        with self.client.get("/admin/stats", catch_response=True) as response:
            if response.status_code in [200, 404]:  # 404 if endpoint doesn't exist
                response.success()
            else:
                response.failure(f"Stats endpoint failed: {response.status_code}")


class ContentManagerUser(UniversalKnowledgeHubUser):
    """
    Content manager user class for testing content management functions.
    """
    
    wait_time = between(3, 8)  # Slower pace for content management
    
    @task(1)
    def upload_content(self):
        """Test content upload functionality."""
        # Simulate file upload (mock data)
        upload_data = {
            "title": f"Test Document {random.randint(1, 1000)}",
            "content": "This is a test document for performance testing.",
            "category": "test",
            "tags": ["performance", "test", "document"]
        }
        
        with self.client.post("/content/upload", json=upload_data, catch_response=True) as response:
            if response.status_code in [200, 201]:
                response.success()
            else:
                response.failure(f"Content upload failed: {response.status_code}")
    
    @task(2)
    def search_content(self):
        """Test content search functionality."""
        search_terms = ["test", "document", "performance", "upload"]
        
        search_data = {
            "query": random.choice(search_terms),
            "filters": {
                "category": "test",
                "date_range": "last_7_days"
            }
        }
        
        with self.client.post("/content/search", json=search_data, catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Content search failed: {response.status_code}")


# Custom event handlers for detailed monitoring
@events.request.add_listener
def my_request_handler(request_type, name, response_time, response_length, response, context, exception, start_time, url, **kwargs):
    """Custom request handler for detailed monitoring."""
    if exception:
        print(f"Request failed: {name} - {exception}")
    else:
        print(f"Request successful: {name} - {response_time}ms")


@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    """Called when a test is starting."""
    print("Performance test starting...")


@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    """Called when a test is ending."""
    print("Performance test ending...")


# Custom metrics collection
class CustomMetrics:
    """Custom metrics collection for detailed performance analysis."""
    
    def __init__(self):
        self.query_times = []
        self.error_counts = {}
        self.response_sizes = []
    
    def record_query_time(self, query_type: str, response_time: float):
        """Record query response time."""
        self.query_times.append({
            "type": query_type,
            "time": response_time,
            "timestamp": time.time()
        })
    
    def record_error(self, endpoint: str, error_type: str):
        """Record error occurrence."""
        if endpoint not in self.error_counts:
            self.error_counts[endpoint] = {}
        if error_type not in self.error_counts[endpoint]:
            self.error_counts[endpoint][error_type] = 0
        self.error_counts[endpoint][error_type] += 1
    
    def record_response_size(self, endpoint: str, size: int):
        """Record response size."""
        self.response_sizes.append({
            "endpoint": endpoint,
            "size": size,
            "timestamp": time.time()
        })


# Global metrics instance
metrics = CustomMetrics()


# Configuration for different test scenarios
class TestScenarios:
    """Predefined test scenarios for different load conditions."""
    
    @staticmethod
    def light_load():
        """Light load scenario - few users, simple queries."""
        return {
            "users": 10,
            "spawn_rate": 1,
            "run_time": "5m"
        }
    
    @staticmethod
    def normal_load():
        """Normal load scenario - typical usage patterns."""
        return {
            "users": 50,
            "spawn_rate": 5,
            "run_time": "10m"
        }
    
    @staticmethod
    def heavy_load():
        """Heavy load scenario - high concurrent users."""
        return {
            "users": 200,
            "spawn_rate": 10,
            "run_time": "15m"
        }
    
    @staticmethod
    def stress_test():
        """Stress test scenario - maximum load."""
        return {
            "users": 500,
            "spawn_rate": 20,
            "run_time": "20m"
        }
    
    @staticmethod
    def spike_test():
        """Spike test scenario - sudden load increase."""
        return {
            "users": 1000,
            "spawn_rate": 100,
            "run_time": "5m"
        }


# Performance thresholds
PERFORMANCE_THRESHOLDS = {
    "health_check": {
        "p95_response_time": 100,  # ms
        "error_rate": 0.01  # 1%
    },
    "simple_query": {
        "p95_response_time": 2000,  # ms
        "error_rate": 0.05  # 5%
    },
    "complex_query": {
        "p95_response_time": 5000,  # ms
        "error_rate": 0.10  # 10%
    },
    "bulk_queries": {
        "p95_response_time": 3000,  # ms
        "error_rate": 0.08  # 8%
    }
}


# Helper functions for test execution
def run_performance_test(scenario: str):
    """Run a specific performance test scenario."""
    scenarios = {
        "light": TestScenarios.light_load(),
        "normal": TestScenarios.normal_load(),
        "heavy": TestScenarios.heavy_load(),
        "stress": TestScenarios.stress_test(),
        "spike": TestScenarios.spike_test()
    }
    
    if scenario not in scenarios:
        raise ValueError(f"Unknown scenario: {scenario}")
    
    config = scenarios[scenario]
    print(f"Running {scenario} load test with {config['users']} users")
    
    # This would be executed via command line:
    # locust -f locustfile.py --host=https://api.universal-knowledge-hub.com
    # --users={config['users']} --spawn-rate={config['spawn_rate']}
    # --run-time={config['run_time']}


if __name__ == "__main__":
    # Example usage
    print("Universal Knowledge Platform Performance Test")
    print("Available scenarios: light, normal, heavy, stress, spike")
    print("Run with: locust -f locustfile.py --host=https://api.universal-knowledge-hub.com") 