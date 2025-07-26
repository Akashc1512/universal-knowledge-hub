#!/usr/bin/env python3
"""
Simple API test for Universal Knowledge Platform
"""

import requests
import time
import json

def test_api():
    """Test the API endpoints."""
    base_url = "http://localhost:8000"
    
    print("🧪 Testing Universal Knowledge Platform API")
    print("=" * 50)
    
    # Test health endpoint
    try:
        response = requests.get(f"{base_url}/health", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Health check passed: {data['status']}")
            print(f"   Version: {data['version']}")
            print(f"   Agents: {data['agents_status']}")
        else:
            print(f"❌ Health check failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Health check error: {e}")
    
    # Test root endpoint
    try:
        response = requests.get(f"{base_url}/", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Root endpoint: {data['message']}")
        else:
            print(f"❌ Root endpoint failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Root endpoint error: {e}")
    
    # Test agents endpoint
    try:
        response = requests.get(f"{base_url}/agents", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Agents endpoint: {data['total_agents']} agents available")
            for agent_name, agent_info in data['agents'].items():
                print(f"   - {agent_info['name']}: {agent_info['description']}")
        else:
            print(f"❌ Agents endpoint failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Agents endpoint error: {e}")
    
    # Test query endpoint
    try:
        query_data = {
            "query": "What is the capital of France?",
            "max_tokens": 100,
            "confidence_threshold": 0.7
        }
        response = requests.post(f"{base_url}/query", json=query_data, timeout=30)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Query endpoint: Answer generated")
            print(f"   Query: {data['query']}")
            print(f"   Answer: {data['answer'][:100]}...")
            print(f"   Confidence: {data['confidence']}")
        elif response.status_code == 503:
            print("⚠️  Query endpoint: Service temporarily unavailable (orchestrator not ready)")
        else:
            print(f"❌ Query endpoint failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Query endpoint error: {e}")
    
    print("\n🎉 API test completed!")

if __name__ == "__main__":
    test_api() 