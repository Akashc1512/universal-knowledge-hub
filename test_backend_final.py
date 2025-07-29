#!/usr/bin/env python3
"""
Simple test script to verify the backend is working.
"""

import requests
import time
import json

def test_backend():
    """Test the backend endpoints."""
    base_url = "http://127.0.0.1:8000"
    
    print("🧪 Testing Universal Knowledge Hub Backend...")
    print("=" * 50)
    
    # Test 1: Health check
    print("\n1. Testing health endpoint...")
    try:
        response = requests.get(f"{base_url}/health", timeout=5)
        if response.status_code == 200:
            print("✅ Health endpoint working")
            print(f"   Response: {response.json()}")
        else:
            print(f"❌ Health endpoint failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Health endpoint error: {e}")
    
    # Test 2: Root endpoint
    print("\n2. Testing root endpoint...")
    try:
        response = requests.get(f"{base_url}/", timeout=5)
        if response.status_code == 200:
            print("✅ Root endpoint working")
            print(f"   Response: {response.json()}")
        else:
            print(f"❌ Root endpoint failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Root endpoint error: {e}")
    
    # Test 3: Metrics endpoint (requires auth)
    print("\n3. Testing metrics endpoint (should require auth)...")
    try:
        response = requests.get(f"{base_url}/metrics", timeout=5)
        if response.status_code == 401:
            print("✅ Metrics endpoint properly protected (401 Unauthorized)")
        else:
            print(f"⚠️  Metrics endpoint returned: {response.status_code}")
    except Exception as e:
        print(f"❌ Metrics endpoint error: {e}")
    
    # Test 4: Query endpoint (requires auth)
    print("\n4. Testing query endpoint (should require auth)...")
    try:
        response = requests.post(f"{base_url}/query", 
                               json={"query": "test query"}, 
                               timeout=5)
        if response.status_code == 401:
            print("✅ Query endpoint properly protected (401 Unauthorized)")
        else:
            print(f"⚠️  Query endpoint returned: {response.status_code}")
    except Exception as e:
        print(f"❌ Query endpoint error: {e}")
    
    print("\n" + "=" * 50)
    print("🎉 Backend test completed!")
    print("\nThe backend is running successfully if you see:")
    print("- ✅ Health endpoint working")
    print("- ✅ Root endpoint working") 
    print("- ✅ Protected endpoints returning 401 (as expected)")

if __name__ == "__main__":
    # Wait a moment for backend to fully start
    print("Waiting 3 seconds for backend to start...")
    time.sleep(3)
    test_backend() 