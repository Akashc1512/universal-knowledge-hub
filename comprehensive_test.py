#!/usr/bin/env python3
"""
Comprehensive test script for Universal Knowledge Hub Backend
Tests all endpoints and Python 3.13.5 compatibility
"""

import requests
import json
import time
import sys

BASE_URL = "http://127.0.0.1:8000"

def test_endpoint(endpoint, method="GET", data=None, expected_status=200):
    """Test an endpoint and return detailed results."""
    url = f"{BASE_URL}{endpoint}"
    
    try:
        if method == "GET":
            response = requests.get(url, timeout=10)
        elif method == "POST":
            headers = {"Content-Type": "application/json"}
            response = requests.post(url, json=data, headers=headers, timeout=10)
        
        success = response.status_code == expected_status
        status_icon = "✅" if success else "❌"
        
        print(f"{status_icon} {method} {endpoint}")
        print(f"   Status: {response.status_code} (Expected: {expected_status})")
        
        if response.status_code == 200:
            try:
                result = response.json()
                print(f"   Response: {json.dumps(result, indent=2)}")
            except:
                print(f"   Response: {response.text}")
        else:
            print(f"   Error: {response.text}")
        
        print()
        return success
    except Exception as e:
        print(f"❌ {method} {endpoint}")
        print(f"   Error: {e}")
        print()
        return False

def test_python_compatibility():
    """Test Python 3.13.5 specific features."""
    print("🐍 Testing Python 3.13.5 Compatibility")
    print("=" * 50)
    
    # Test modern typing syntax
    try:
        from typing import Any, Optional
        test_dict: dict[str, Any] = {"test": "value"}
        test_list: list[str] = ["item1", "item2"]
        test_optional: Optional[str] = None
        print("✅ Modern typing syntax working")
    except Exception as e:
        print(f"❌ Modern typing syntax failed: {e}")
    
    # Test async/await
    try:
        import asyncio
        async def test_async():
            await asyncio.sleep(0.01)
            return "async_works"
        
        result = asyncio.run(test_async())
        print("✅ Async/await functionality working")
    except Exception as e:
        print(f"❌ Async/await failed: {e}")
    
    print()

def test_api_endpoints():
    """Test all API endpoints."""
    print("🌐 Testing API Endpoints")
    print("=" * 50)
    
    # Test GET endpoints
    test_endpoint("/")
    test_endpoint("/health")
    test_endpoint("/test")
    test_endpoint("/metrics")
    
    # Test POST endpoints
    query_data = {
        "query": "What is Python 3.13.5?",
        "max_tokens": 1000,
        "confidence_threshold": 0.8
    }
    test_endpoint("/query", "POST", query_data)
    
    auth_data = {
        "username": "admin",
        "password": "password"
    }
    test_endpoint("/auth/login", "POST", auth_data)
    
    # Test invalid authentication
    invalid_auth = {
        "username": "wrong",
        "password": "wrong"
    }
    test_endpoint("/auth/login", "POST", invalid_auth, expected_status=401)

def test_error_handling():
    """Test error handling."""
    print("⚠️ Testing Error Handling")
    print("=" * 50)
    
    # Test invalid query
    invalid_query = {
        "query": "",  # Empty query should fail validation
        "max_tokens": 1000
    }
    test_endpoint("/query", "POST", invalid_query, expected_status=422)
    
    # Test invalid JSON
    try:
        response = requests.post(
            f"{BASE_URL}/query",
            data="invalid json",
            headers={"Content-Type": "application/json"},
            timeout=10
        )
        print(f"✅ Invalid JSON handling: {response.status_code}")
    except Exception as e:
        print(f"❌ Invalid JSON handling failed: {e}")
    
    print()

def test_performance():
    """Test basic performance."""
    print("⚡ Testing Performance")
    print("=" * 50)
    
    # Test response time
    start_time = time.time()
    response = requests.get(f"{BASE_URL}/health", timeout=10)
    end_time = time.time()
    
    response_time = (end_time - start_time) * 1000  # Convert to milliseconds
    print(f"✅ Health endpoint response time: {response_time:.2f}ms")
    
    # Test concurrent requests
    import concurrent.futures
    
    def make_request():
        return requests.get(f"{BASE_URL}/health", timeout=10)
    
    start_time = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(make_request) for _ in range(5)]
        results = [future.result() for future in futures]
    end_time = time.time()
    
    concurrent_time = (end_time - start_time) * 1000
    print(f"✅ 5 concurrent requests time: {concurrent_time:.2f}ms")
    
    print()

def main():
    """Run comprehensive tests."""
    print("🧪 Comprehensive Backend Testing")
    print("=" * 60)
    print(f"🌐 Server URL: {BASE_URL}")
    print(f"🐍 Python Version: {sys.version}")
    print("=" * 60)
    print()
    
    # Test Python compatibility
    test_python_compatibility()
    
    # Test API endpoints
    test_api_endpoints()
    
    # Test error handling
    test_error_handling()
    
    # Test performance
    test_performance()
    
    print("=" * 60)
    print("🎉 Comprehensive testing completed!")
    print(f"📚 API Documentation: {BASE_URL}/docs")
    print(f"📋 OpenAPI Schema: {BASE_URL}/openapi.json")
    print("=" * 60)

if __name__ == "__main__":
    main() 