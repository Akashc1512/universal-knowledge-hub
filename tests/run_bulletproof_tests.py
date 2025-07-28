#!/usr/bin/env python3
"""
🚀 BULLETPROOF TEST RUNNER
Universal Knowledge Platform - Complete Test Suite

Runs all comprehensive tests to ensure the application is bulletproof,
bug-free, and performs optimally under all conditions.
"""

import unittest
import sys
import os
import time
import subprocess
import threading
import concurrent.futures
from datetime import datetime
from typing import List, Dict, Any

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Test configuration
TEST_CONFIG = {
    "test_timeout": 1800,  # 30 minutes
    "parallel_tests": 4,
    "required_success_rate": 95.0,  # 95% success rate required
    "performance_thresholds": {
        "max_response_time_ms": 500,
        "max_memory_usage_mb": 1024,
        "max_cpu_usage_percent": 80,
    },
}

# Test categories
TEST_CATEGORIES = {
    "unit_tests": [
        "test_bulletproof_comprehensive.py",
        "test_prompts_comprehensive.py",
        "test_load_stress_performance.py",
        "test_security_comprehensive.py",
        "test_agents.py",
        "test_integration.py",
        "test_complete_system.py",
        "test_core_functionality.py",
        "test_configuration.py",
        "test_frontend_components.py",
        "test_performance.py",
        "test_recommendation_system.py",
        "test_security.py",
        "test_load.py",
        "test_lead_orchestrator.py",
        "test_retrieval.py",
        "test_factcheck.py",
    ],
    "integration_tests": ["test_integration.py", "test_complete_system.py"],
    "performance_tests": ["test_load_stress_performance.py", "test_performance.py", "test_load.py"],
    "security_tests": ["test_security_comprehensive.py", "test_security.py"],
    "frontend_tests": ["test_frontend_components.py"],
}


class BulletproofTestRunner:
    """Master test runner for bulletproof testing"""

    def __init__(self):
        """Initialize test runner"""
        self.start_time = None
        self.results = {}
        self.summary = {
            "total_tests": 0,
            "passed_tests": 0,
            "failed_tests": 0,
            "error_tests": 0,
            "skipped_tests": 0,
            "categories": {},
        }

    def print_banner(self):
        """Print test banner"""
        print("🚀" + "=" * 58)
        print("🚀 BULLETPROOF TESTING SUITE - UNIVERSAL KNOWLEDGE PLATFORM")
        print("🚀" + "=" * 58)
        print(f"🚀 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"🚀 Target Success Rate: {TEST_CONFIG['required_success_rate']}%")
        print(f"🚀 Timeout: {TEST_CONFIG['test_timeout']} seconds")
        print(f"🚀 Parallel Tests: {TEST_CONFIG['parallel_tests']}")
        print("🚀" + "=" * 58)
        print()

    def run_test_file(self, test_file: str) -> Dict[str, Any]:
        """Run a single test file"""
        print(f"🧪 Running: {test_file}")

        start_time = time.time()
        result = {
            "file": test_file,
            "success": False,
            "tests_run": 0,
            "failures": 0,
            "errors": 0,
            "skipped": 0,
            "execution_time": 0,
            "error_message": "",
        }

        try:
            # Run test file
            test_path = os.path.join(os.path.dirname(__file__), test_file)

            if not os.path.exists(test_path):
                result["error_message"] = f"Test file not found: {test_file}"
                return result

            # Run with pytest for better output
            cmd = [sys.executable, "-m", "pytest", test_path, "-v", "--tb=short"]
            process = subprocess.run(
                cmd, capture_output=True, text=True, timeout=TEST_CONFIG["test_timeout"]
            )

            result["execution_time"] = time.time() - start_time

            # Parse pytest output
            if process.returncode == 0:
                result["success"] = True
                # Extract test counts from output
                lines = process.stdout.split("\n")
                for line in lines:
                    if "passed" in line and "failed" in line:
                        # Parse pytest summary line
                        parts = line.split()
                        for part in parts:
                            if part.isdigit():
                                if "passed" in line:
                                    result["tests_run"] = int(part)
                                elif "failed" in line:
                                    result["failures"] = int(part)
                                elif "error" in line:
                                    result["errors"] = int(part)
                                elif "skipped" in line:
                                    result["skipped"] = int(part)
                        break
            else:
                result["error_message"] = process.stderr
                # Try to parse error output
                lines = process.stderr.split("\n")
                for line in lines:
                    if "failed" in line or "error" in line:
                        result["error_message"] = line
                        break

            print(f"   ✅ Completed: {test_file} ({result['execution_time']:.2f}s)")
            if not result["success"]:
                print(f"   ❌ Failed: {result['error_message']}")

        except subprocess.TimeoutExpired:
            result["error_message"] = f"Test timeout after {TEST_CONFIG['test_timeout']} seconds"
            print(f"   ⏰ Timeout: {test_file}")
        except Exception as e:
            result["error_message"] = str(e)
            print(f"   ❌ Error: {test_file} - {e}")

        return result

    def run_category_tests(self, category: str, test_files: List[str]) -> Dict[str, Any]:
        """Run tests for a specific category"""
        print(f"\n📋 Running {category.upper()} Tests")
        print("-" * 50)

        category_results = {
            "category": category,
            "total_files": len(test_files),
            "successful_files": 0,
            "failed_files": 0,
            "total_tests": 0,
            "passed_tests": 0,
            "failed_tests": 0,
            "error_tests": 0,
            "skipped_tests": 0,
            "execution_time": 0,
            "files": [],
        }

        start_time = time.time()

        # Run tests in parallel
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=TEST_CONFIG["parallel_tests"]
        ) as executor:
            future_to_file = {
                executor.submit(self.run_test_file, test_file): test_file
                for test_file in test_files
            }

            for future in concurrent.futures.as_completed(future_to_file):
                result = future.result()
                category_results["files"].append(result)

                if result["success"]:
                    category_results["successful_files"] += 1
                    category_results["total_tests"] += result["tests_run"]
                    category_results["passed_tests"] += (
                        result["tests_run"]
                        - result["failures"]
                        - result["errors"]
                        - result["skipped"]
                    )
                    category_results["failed_tests"] += result["failures"]
                    category_results["error_tests"] += result["errors"]
                    category_results["skipped_tests"] += result["skipped"]
                else:
                    category_results["failed_files"] += 1

        category_results["execution_time"] = time.time() - start_time

        # Print category summary
        success_rate = (
            (category_results["successful_files"] / category_results["total_files"]) * 100
            if category_results["total_files"] > 0
            else 0
        )
        test_success_rate = (
            (category_results["passed_tests"] / category_results["total_tests"]) * 100
            if category_results["total_tests"] > 0
            else 0
        )

        print(f"\n📊 {category.upper()} Summary:")
        print(
            f"   Files: {category_results['successful_files']}/{category_results['total_files']} successful ({success_rate:.1f}%)"
        )
        print(
            f"   Tests: {category_results['passed_tests']}/{category_results['total_tests']} passed ({test_success_rate:.1f}%)"
        )
        print(f"   Time: {category_results['execution_time']:.2f}s")

        return category_results

    def run_all_tests(self) -> bool:
        """Run all test categories"""
        self.start_time = time.time()
        self.print_banner()

        # Run each category
        for category, test_files in TEST_CATEGORIES.items():
            # Filter existing test files
            existing_files = [
                f for f in test_files if os.path.exists(os.path.join(os.path.dirname(__file__), f))
            ]

            if existing_files:
                category_result = self.run_category_tests(category, existing_files)
                self.results[category] = category_result
                self.summary["categories"][category] = category_result
            else:
                print(f"\n⚠️  No test files found for category: {category}")

        # Calculate overall summary
        self.calculate_summary()

        # Print final results
        self.print_final_results()

        # Check if overall success rate meets requirements
        overall_success_rate = (
            (self.summary["passed_tests"] / self.summary["total_tests"]) * 100
            if self.summary["total_tests"] > 0
            else 0
        )

        return overall_success_rate >= TEST_CONFIG["required_success_rate"]

    def calculate_summary(self):
        """Calculate overall test summary"""
        for category_result in self.summary["categories"].values():
            self.summary["total_tests"] += category_result["total_tests"]
            self.summary["passed_tests"] += category_result["passed_tests"]
            self.summary["failed_tests"] += category_result["failed_tests"]
            self.summary["error_tests"] += category_result["error_tests"]
            self.summary["skipped_tests"] += category_result["skipped_tests"]

    def print_final_results(self):
        """Print final test results"""
        total_time = time.time() - self.start_time
        overall_success_rate = (
            (self.summary["passed_tests"] / self.summary["total_tests"]) * 100
            if self.summary["total_tests"] > 0
            else 0
        )

        print("\n" + "🚀" + "=" * 58)
        print("🚀 BULLETPROOF TESTING RESULTS")
        print("🚀" + "=" * 58)
        print(f"🚀 Total Execution Time: {total_time:.2f} seconds")
        print(f"🚀 Total Tests: {self.summary['total_tests']}")
        print(f"🚀 Passed: {self.summary['passed_tests']}")
        print(f"🚀 Failed: {self.summary['failed_tests']}")
        print(f"🚀 Errors: {self.summary['error_tests']}")
        print(f"🚀 Skipped: {self.summary['skipped_tests']}")
        print(f"🚀 Success Rate: {overall_success_rate:.2f}%")
        print(f"🚀 Required Rate: {TEST_CONFIG['required_success_rate']}%")

        # Category breakdown
        print("\n📊 Category Breakdown:")
        for category, result in self.summary["categories"].items():
            category_success_rate = (
                (result["passed_tests"] / result["total_tests"]) * 100
                if result["total_tests"] > 0
                else 0
            )
            status = "✅" if category_success_rate >= TEST_CONFIG["required_success_rate"] else "❌"
            print(
                f"   {status} {category.upper()}: {result['passed_tests']}/{result['total_tests']} ({category_success_rate:.1f}%)"
            )

        # Final verdict
        if overall_success_rate >= TEST_CONFIG["required_success_rate"]:
            print(f"\n🎉 BULLETPROOF SUCCESS! System is ready for production!")
            print(
                f"🎉 Success Rate: {overall_success_rate:.2f}% >= {TEST_CONFIG['required_success_rate']}%"
            )
        else:
            print(f"\n❌ BULLETPROOF FAILED! System needs fixing!")
            print(
                f"❌ Success Rate: {overall_success_rate:.2f}% < {TEST_CONFIG['required_success_rate']}%"
            )

        print("🚀" + "=" * 58)

    def generate_test_report(self):
        """Generate detailed test report"""
        report = {
            "timestamp": datetime.now().isoformat(),
            "summary": self.summary,
            "results": self.results,
            "config": TEST_CONFIG,
        }

        # Save report to file
        report_file = os.path.join(os.path.dirname(__file__), "bulletproof_test_report.json")
        with open(report_file, "w") as f:
            import json

            json.dump(report, f, indent=2, default=str)

        print(f"\n📄 Test report saved to: {report_file}")


def main():
    """Main test runner"""
    runner = BulletproofTestRunner()

    try:
        success = runner.run_all_tests()
        runner.generate_test_report()

        if success:
            print("\n🎯 BULLETPROOF TESTING COMPLETED SUCCESSFULLY!")
            print("🎯 Your Universal Knowledge Platform is bulletproof and ready for production!")
            sys.exit(0)
        else:
            print("\n💥 BULLETPROOF TESTING FAILED!")
            print("💥 Please fix the issues before deploying to production!")
            sys.exit(1)

    except KeyboardInterrupt:
        print("\n⚠️  Testing interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Testing failed with error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
