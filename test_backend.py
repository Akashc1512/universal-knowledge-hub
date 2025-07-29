#!/usr/bin/env python3
"""
Test script for Universal Knowledge Hub Backend
"""

import requests
import json
import time

BASE_URL = "http://127.0.0.1:8000"

def test_endpoint(endpoint, method="GET", data=None):
    """Test an endpoint and return the response."""
    url = f"{BASE_URL}{endpoint}"
    
    try:
        if method == "GET":
            response = requests.get(url)
        elif method == "POST":
            headers = {"Content-Type": "application/json"}
            response = requests.post(url, json=data, headers=headers)
        
        print(f"✅ {method} {endpoint} - Status: {response.status_code}")
        if response.status_code == 200:
            print(f"   Response: {response.json()}")
        else:
            print(f"   Error: {response.text}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ {method} {endpoint} - Error: {e}")
        return False

def main():
    """Test all endpoints."""
    print("🧪 Testing Universal Knowledge Hub Backend")
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
    
    print("\n" + "=" * 50)
    print("🎉 Backend testing completed!")
    print(f"🌐 API Documentation: {BASE_URL}/docs")
    print(f"📋 OpenAPI Schema: {BASE_URL}/openapi.json")

if __name__ == "__main__":
    main() 