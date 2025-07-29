#!/usr/bin/env python3
"""
Simple test to check if the backend is working.
"""

import requests
import time

def test_backend_simple():
    """Test if the backend is responding at all."""
    print("🧪 Testing backend connectivity...")
    
    try:
        # Test if server is responding
        response = requests.get("http://127.0.0.1:8000/", timeout=5)
        print(f"✅ Backend is responding (status: {response.status_code})")
        print(f"   Response: {response.text[:200]}...")
        return True
    except requests.exceptions.ConnectionError:
        print("❌ Backend is not responding (connection refused)")
        return False
    except Exception as e:
        print(f"❌ Backend error: {e}")
        return False

if __name__ == "__main__":
    test_backend_simple() 