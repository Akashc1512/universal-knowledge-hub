#!/usr/bin/env python3
"""
🧪 MINIMAL CORE TEST
Universal Knowledge Platform - Core Functionality Test

Tests only the core functionality without heavy dependencies.
"""

import unittest
import sys
import os
import time
from datetime import datetime
import asyncio

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import only core components
from agents.base_agent import BaseAgent, AgentType, AgentMessage, AgentResult, QueryContext


class TestAgent(BaseAgent):
    """Test agent for testing purposes"""

    def __init__(self, agent_id: str, agent_type: AgentType):
        super().__init__(agent_id, agent_type)

    async def process_task(self, task: AgentMessage) -> AgentResult:
        """Process a test task"""
        start_time = time.time()

        try:
            # Simulate processing
            await asyncio.sleep(0.1)

            result = self.create_result(
                success=True,
                data=f"Processed: {task.content}",
                execution_time=time.time() - start_time,
                confidence=0.8,
            )

            self.update_stats(True, result.execution_time)
            return result

        except Exception as e:
            result = self.handle_error(e, "process_task")
            self.update_stats(False, result.execution_time)
            return result

    def get_capabilities(self) -> list:
        """Get test agent capabilities"""
        return ["test_processing", "basic_validation"]


class TestMinimalCore(unittest.TestCase):
    """Test minimal core functionality"""

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

    def test_error_handling(self):
        """Test error handling"""
        agent = TestAgent(self.agent_id, self.agent_type)

        # Test error handling
        result = agent.handle_error(Exception("Test error"), "test_context")

        self.assertIsInstance(result, AgentResult)
        self.assertFalse(result.success)
        self.assertIn("Test error", result.error_message)
        self.assertEqual(result.execution_time, 0.0)
        self.assertEqual(result.confidence, 0.0)

    def test_capabilities(self):
        """Test agent capabilities"""
        agent = TestAgent(self.agent_id, self.agent_type)

        capabilities = agent.get_capabilities()

        self.assertIsInstance(capabilities, list)
        self.assertIn("test_processing", capabilities)
        self.assertIn("basic_validation", capabilities)

    def test_input_validation(self):
        """Test input validation"""
        agent = TestAgent(self.agent_id, self.agent_type)

        # Test default validation (should always return True)
        self.assertTrue(agent.validate_input("test input"))
        self.assertTrue(agent.validate_input(""))
        self.assertTrue(agent.validate_input(None))


def run_minimal_tests():
    """Run minimal tests"""
    print("🧪 Starting MINIMAL CORE FUNCTIONALITY TESTS")
    print("=" * 60)

    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add test class
    tests = loader.loadTestsFromTestCase(TestMinimalCore)
    suite.addTests(tests)

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Print summary
    print("\n" + "=" * 60)
    print("🧪 MINIMAL CORE FUNCTIONALITY TESTING SUMMARY")
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
        print("\n✅ ALL MINIMAL TESTS PASSED - CORE FUNCTIONALITY IS WORKING!")
    else:
        print("\n❌ SOME MINIMAL TESTS FAILED - NEEDS FIXING!")

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_minimal_tests()
    sys.exit(0 if success else 1)
