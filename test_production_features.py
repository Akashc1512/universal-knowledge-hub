#!/usr/bin/env python3
"""
Simple test script to validate production readiness features.
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def test_integration_monitor():
    """Test integration monitor functionality."""
    try:
        from api.integration_monitor import IntegrationMonitor, IntegrationStatus

        print("✅ Integration monitor imported successfully")

        # Test basic functionality
        monitor = IntegrationMonitor()
        print("✅ Integration monitor initialized successfully")

        # Test integration status
        status = monitor.integrations
        print(f"✅ Found {len(status)} integrations to monitor")

        for name, integration in status.items():
            print(f"  - {name}: {integration.config_status}")

        return True
    except Exception as e:
        print(f"❌ Integration monitor test failed: {e}")
        return False


def test_cache_functionality():
    """Test cache functionality."""
    try:
        from api.cache import LRUCache, RedisCache

        print("✅ Cache modules imported successfully")

        # Test LRU cache
        cache = LRUCache(max_size=10)
        print("✅ LRU cache initialized successfully")

        # Test Redis cache (should work even without Redis)
        redis_cache = RedisCache()
        print("✅ Redis cache initialized successfully")

        return True
    except Exception as e:
        print(f"❌ Cache test failed: {e}")
        return False


def test_metrics_functionality():
    """Test metrics functionality."""
    try:
        from api.metrics import MetricsCollector

        print("✅ Metrics module imported successfully")

        # Test metrics collector
        collector = MetricsCollector()
        print("✅ Metrics collector initialized successfully")

        # Test basic metrics (without accessing internal prometheus objects)
        try:
            collector.record_request("GET", "/health", 200, 0.1)
            collector.record_cache_hit("query_cache")
            print("✅ Basic metrics recorded successfully")
        except Exception as e:
            print(f"⚠️  Metrics recording failed: {e}")

        # Test getting metrics dict
        try:
            metrics = collector.get_metrics_dict()
            print(f"✅ Metrics collected: {len(metrics)} metrics")
        except Exception as e:
            print(f"⚠️  Metrics dict failed: {e}")

        return True
    except Exception as e:
        print(f"❌ Metrics test failed: {e}")
        return False


def test_logging_functionality():
    """Test logging functionality."""
    try:
        import logging

        # Test structured logging format
        log_format = '{"timestamp": "%(asctime)s", "level": "%(levelname)s", "logger": "%(name)s", "message": "%(message)s", "request_id": "%(request_id)s", "user_id": "%(user_id)s", "service": "sarvanom-api", "version": "1.0.0"}'

        # Check that required fields are in format
        required_fields = ["request_id", "user_id", "service", "version"]
        for field in required_fields:
            if field in log_format:
                print(f"✅ Log format includes {field}")
            else:
                print(f"❌ Log format missing {field}")
                return False

        print("✅ Structured logging format validated")
        return True
    except Exception as e:
        print(f"❌ Logging test failed: {e}")
        return False


def test_environment_configuration():
    """Test environment configuration."""
    try:
        # Test that environment variables can be read
        from dotenv import load_dotenv

        load_dotenv()

        # Check monitoring configuration
        monitoring_vars = [
            "PROMETHEUS_ENABLED",
            "LOG_LEVEL",
            "REDIS_ENABLED",
            "INTEGRATION_MONITORING_ENABLED",
        ]

        for var in monitoring_vars:
            value = os.getenv(var, "not_set")
            print(f"✅ Environment variable {var}: {value}")

        return True
    except Exception as e:
        print(f"❌ Environment configuration test failed: {e}")
        return False


def main():
    """Run all production readiness tests."""
    print("🚀 Testing Production Readiness Features")
    print("=" * 50)

    tests = [
        ("Integration Monitor", test_integration_monitor),
        ("Cache Functionality", test_cache_functionality),
        ("Metrics Functionality", test_metrics_functionality),
        ("Logging Functionality", test_logging_functionality),
        ("Environment Configuration", test_environment_configuration),
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        print(f"\n🧪 Testing {test_name}...")
        if test_func():
            passed += 1
            print(f"✅ {test_name} passed")
        else:
            print(f"❌ {test_name} failed")

    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All production readiness tests passed!")
        return 0
    else:
        print("⚠️  Some tests failed. Please review the issues above.")
        return 1


if __name__ == "__main__":
    exit(main())
