#!/usr/bin/env python3
"""
🧪 SIMPLE TEST RUNNER
Universal Knowledge Platform - Basic Functionality Tests

Runs essential tests to verify core functionality and basic health checks.
"""

import asyncio
import json
import logging
import os
import sys
import time
import unittest
from typing import Dict, Any
from unittest.mock import Mock, patch

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
TEST_CONFIG = {
    "api_base_url": os.getenv("TEST_API_BASE_URL", "http://localhost:8003"),
    "test_timeout": int(os.getenv("TEST_TIMEOUT", "30")),
    "response_time_limit": int(os.getenv("TEST_RESPONSE_TIME_LIMIT", "1000")),
    "default_token_budget": int(os.getenv("DEFAULT_TOKEN_BUDGET", "1000")),
}


class TestBasicFunctionality(unittest.TestCase):
    """Test basic functionality after package reorganization"""

    def setUp(self):
        """Set up test environment"""
        self.agent_id = "test_agent_001"
        self.agent_type = AgentType.RETRIEVAL

    def test_base_agent_creation(self):
        """Test that we can create a base agent"""
        agent = TestAgent(self.agent_id, self.agent_type)

        self.assertEqual(agent.agent_id, self.agent_id)
        self.assertEqual(agent.agent_type, self.agent_type)
        self.assertFalse(agent.is_running)
        self.assertIsNotNone(agent.stats)

    def test_agent_start_stop(self):
        """Test agent start and stop functionality"""
        agent = TestAgent(self.agent_id, self.agent_type)

        # Start agent
        agent.start()
        self.assertTrue(agent.is_running)
        self.assertIsNotNone(agent.start_time)

        # Stop agent
        agent.stop()
        self.assertFalse(agent.is_running)

    def test_agent_heartbeat(self):
        """Test agent heartbeat functionality"""
        agent = TestAgent(self.agent_id, self.agent_type)
        agent.start()

        heartbeat = agent.heartbeat()

        self.assertIn("agent_id", heartbeat)
        self.assertIn("agent_type", heartbeat)
        self.assertIn("is_running", heartbeat)
        self.assertIn("uptime", heartbeat)
        self.assertIn("stats", heartbeat)

        self.assertEqual(heartbeat["agent_id"], self.agent_id)
        self.assertEqual(heartbeat["agent_type"], self.agent_type.value)
        self.assertTrue(heartbeat["is_running"])

    def test_message_creation(self):
        """Test message creation"""
        agent = TestAgent(self.agent_id, self.agent_type)

        message = agent.create_message(
            content="test content", sender_id="sender_001", recipient_id="recipient_001"
        )

        self.assertIsInstance(message, AgentMessage)
        self.assertEqual(message.content, "test content")
        self.assertEqual(message.sender_id, "sender_001")
        self.assertEqual(message.recipient_id, "recipient_001")

    def test_result_creation(self):
        """Test result creation"""
        agent = TestAgent(self.agent_id, self.agent_type)

        result = agent.create_result(
            success=True, data="test data", execution_time=0.5, confidence=0.8
        )

        self.assertIsInstance(result, AgentResult)
        self.assertTrue(result.success)
        self.assertEqual(result.data, "test data")
        self.assertEqual(result.execution_time, 0.5)
        self.assertEqual(result.confidence, 0.8)

    def test_stats_update(self):
        """Test statistics update"""
        agent = TestAgent(self.agent_id, self.agent_type)

        # Initial stats
        self.assertEqual(agent.stats["tasks_processed"], 0)
        self.assertEqual(agent.stats["successful_tasks"], 0)
        self.assertEqual(agent.stats["failed_tasks"], 0)

        # Update stats for successful task
        agent.update_stats(True, 0.5)
        self.assertEqual(agent.stats["tasks_processed"], 1)
        self.assertEqual(agent.stats["successful_tasks"], 1)
        self.assertEqual(agent.stats["failed_tasks"], 0)
        self.assertEqual(agent.stats["total_execution_time"], 0.5)
        self.assertEqual(agent.stats["average_execution_time"], 0.5)

        # Update stats for failed task
        agent.update_stats(False, 0.3)
        self.assertEqual(agent.stats["tasks_processed"], 2)
        self.assertEqual(agent.stats["successful_tasks"], 1)
        self.assertEqual(agent.stats["failed_tasks"], 1)
        self.assertEqual(agent.stats["total_execution_time"], 0.8)
        self.assertEqual(agent.stats["average_execution_time"], 0.4)


class TestAPIEndpoints(unittest.TestCase):
    """Test API endpoints"""

    def setUp(self):
        """Set up test environment"""
        self.base_url = "http://localhost:8003"

    def test_health_endpoint(self):
        """Test health endpoint"""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=10)

            self.assertEqual(response.status_code, 200)

            data = response.json()
            self.assertIn("status", data)
            self.assertEqual(data["status"], "healthy")

        except requests.exceptions.RequestException as e:
            # API might not be running, skip test
            self.skipTest(f"API not available: {e}")

    def test_root_endpoint(self):
        """Test root endpoint"""
        try:
            response = requests.get(f"{self.base_url}/", timeout=10)

            self.assertEqual(response.status_code, 200)

            data = response.json()
            self.assertIn("message", data)

        except requests.exceptions.RequestException as e:
            # API might not be running, skip test
            self.skipTest(f"API not available: {e}")

    def test_agents_endpoint(self):
        """Test agents endpoint"""
        try:
            response = requests.get(f"{self.base_url}/agents", timeout=10)

            self.assertEqual(response.status_code, 200)

            data = response.json()
            self.assertIn("agents", data)

        except requests.exceptions.RequestException as e:
            # API might not be running, skip test
            self.skipTest(f"API not available: {e}")


class TestPerformanceBasic(unittest.TestCase):
    """Test basic performance"""

    def test_response_time(self):
        """Test API response time"""
        try:
            start_time = time.time()
            response = requests.get("http://localhost:8003/health", timeout=10)
            end_time = time.time()

            response_time = (end_time - start_time) * 1000  # Convert to milliseconds

            self.assertEqual(response.status_code, 200)
            self.assertLess(response_time, 1000)  # Should respond in less than 1 second

        except requests.exceptions.RequestException as e:
            # API might not be running, skip test
            self.skipTest(f"API not available: {e}")


def run_simple_tests():
    """Run all simple tests"""
    print("🧪 Starting SIMPLE FUNCTIONALITY TESTS")
    print("=" * 60)

    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all test classes
    test_classes = [TestBasicFunctionality, TestAPIEndpoints, TestPerformanceBasic]

    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Print summary
    print("\n" + "=" * 60)
    print("🧪 SIMPLE FUNCTIONALITY TESTING SUMMARY")
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
        print("\n✅ ALL SIMPLE TESTS PASSED - CORE FUNCTIONALITY IS WORKING!")
    else:
        print("\n❌ SOME SIMPLE TESTS FAILED - NEEDS FIXING!")

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_simple_tests()
    sys.exit(0 if success else 1)
