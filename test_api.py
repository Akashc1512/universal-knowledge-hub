#!/usr/bin/env python3
"""
Test script for Universal Knowledge Platform API
"""

import requests
import json
import time
from typing import Dict, Any

BASE_URL = "http://localhost:8000"

def test_health():
    """Test the health endpoint."""
    print("🏥 Testing health endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/health")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Health check passed: {data['status']}")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False

def test_root():
    """Test the root endpoint."""
    print("🏠 Testing root endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Root endpoint: {data['message']}")
            return True
        else:
            print(f"❌ Root endpoint failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Root endpoint error: {e}")
        return False

def test_agents():
    """Test the agents endpoint."""
    print("🤖 Testing agents endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/agents")
        if response.status_code == 200:
            data = response.json()
            agents = data.get('agents', {})
            print(f"✅ Found {len(agents)} agents:")
            for name, info in agents.items():
                print(f"   - {name}: {info['description']}")
            return True
        else:
            print(f"❌ Agents endpoint failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Agents endpoint error: {e}")
        return False

def test_query():
    """Test the query endpoint."""
    print("🔍 Testing query endpoint...")
    
    test_query = {
        "query": "What is the capital of France?",
        "user_context": {"domain": "geography"},
        "max_tokens": 500,
        "confidence_threshold": 0.7
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/query",
            json=test_query,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Query processed successfully!")
            print(f"   Answer: {data['answer'][:100]}...")
            print(f"   Confidence: {data['confidence']}")
            print(f"   Execution time: {data['execution_time']:.2f}s")
            print(f"   Citations: {len(data['citations'])}")
            return True
        else:
            print(f"❌ Query failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Query error: {e}")
        return False

def main():
    """Run all API tests."""
    print("🧪 Starting Universal Knowledge Platform API Tests")
    print("=" * 50)
    
    # Wait a moment for server to start
    print("⏳ Waiting for server to start...")
    time.sleep(2)
    
    tests = [
        ("Health Check", test_health),
        ("Root Endpoint", test_root),
        ("Agents Endpoint", test_agents),
        ("Query Processing", test_query)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n📋 Running: {test_name}")
        if test_func():
            passed += 1
        print("-" * 30)
    
    print(f"\n📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! API is working correctly.")
    else:
        print("⚠️  Some tests failed. Check the server logs.")
    
    return passed == total

if __name__ == "__main__":
    main() 