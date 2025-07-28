#!/usr/bin/env python3
"""
Integration Test Script - Frontend-Backend Communication
Tests the fixes for integration issues identified.
"""

import asyncio
import json
import sys
import time
from typing import Dict, Any

import httpx


class IntegrationTester:
    def __init__(self, base_url: str = "http://localhost:8002"):
        self.base_url = base_url
        self.client = httpx.AsyncClient(timeout=30.0)
        self.headers = {
            "Content-Type": "application/json",
            "X-API-Key": "user-key-456"
        }
        
    async def test_health_endpoint(self) -> Dict[str, Any]:
        """Test the health endpoint."""
        print("🔍 Testing /health endpoint...")
        try:
            response = await self.client.get(f"{self.base_url}/health")
            if response.status_code == 200:
                print("✅ Health endpoint working")
                return {"status": "success", "data": response.json()}
            else:
                print(f"❌ Health endpoint failed: {response.status_code}")
                return {"status": "error", "error": f"HTTP {response.status_code}"}
        except Exception as e:
            print(f"❌ Health endpoint error: {e}")
            return {"status": "error", "error": str(e)}
    
    async def test_query_endpoint(self) -> Dict[str, Any]:
        """Test the query endpoint with the expected request format."""
        print("🔍 Testing /query endpoint...")
        try:
            query_data = {
                "query": "What is artificial intelligence?",
                "max_tokens": 1000,
                "confidence_threshold": 0.8
            }
            
            response = await self.client.post(
                f"{self.base_url}/query",
                headers=self.headers,
                json=query_data
            )
            
            if response.status_code == 200:
                data = response.json()
                
                # Check if response has expected fields
                required_fields = ["answer", "confidence", "citations", "query_id"]
                missing_fields = [field for field in required_fields if field not in data]
                
                if missing_fields:
                    print(f"❌ Query response missing fields: {missing_fields}")
                    return {"status": "error", "error": f"Missing fields: {missing_fields}"}
                
                # Check if processing_time is included (new requirement)
                if "processing_time" in data:
                    print("✅ Query endpoint working with processing_time")
                else:
                    print("⚠️ Query endpoint working but missing processing_time")
                
                print(f"📊 Response: answer={len(data['answer'])} chars, confidence={data['confidence']}")
                return {"status": "success", "data": data, "query_id": data.get("query_id")}
            else:
                print(f"❌ Query endpoint failed: {response.status_code}")
                error_detail = response.json() if response.content else "No error details"
                return {"status": "error", "error": f"HTTP {response.status_code}: {error_detail}"}
                
        except Exception as e:
            print(f"❌ Query endpoint error: {e}")
            return {"status": "error", "error": str(e)}
    
    async def test_feedback_endpoint(self, query_id: str) -> Dict[str, Any]:
        """Test the feedback endpoint (the main fix)."""
        print("🔍 Testing /feedback endpoint...")
        try:
            feedback_data = {
                "query_id": query_id,
                "feedback_type": "helpful",
                "details": "This is a test feedback"
            }
            
            response = await self.client.post(
                f"{self.base_url}/feedback",
                headers=self.headers,
                json=feedback_data
            )
            
            if response.status_code == 200:
                data = response.json()
                print("✅ Feedback endpoint working")
                print(f"📊 Feedback response: {data}")
                return {"status": "success", "data": data}
            else:
                print(f"❌ Feedback endpoint failed: {response.status_code}")
                error_detail = response.json() if response.content else "No error details"
                return {"status": "error", "error": f"HTTP {response.status_code}: {error_detail}"}
                
        except Exception as e:
            print(f"❌ Feedback endpoint error: {e}")
            return {"status": "error", "error": str(e)}
    
    async def test_cors_headers(self) -> Dict[str, Any]:
        """Test CORS configuration."""
        print("🔍 Testing CORS headers...")
        try:
            response = await self.client.options(
                f"{self.base_url}/query",
                headers={
                    "Origin": "http://localhost:3000",
                    "Access-Control-Request-Method": "POST",
                    "Access-Control-Request-Headers": "Content-Type,X-API-Key"
                }
            )
            
            cors_headers = {
                "Access-Control-Allow-Origin": response.headers.get("Access-Control-Allow-Origin"),
                "Access-Control-Allow-Methods": response.headers.get("Access-Control-Allow-Methods"),
                "Access-Control-Allow-Headers": response.headers.get("Access-Control-Allow-Headers"),
            }
            
            if response.status_code in [200, 204]:
                print("✅ CORS preflight working")
                print(f"📊 CORS headers: {cors_headers}")
                return {"status": "success", "headers": cors_headers}
            else:
                print(f"❌ CORS preflight failed: {response.status_code}")
                return {"status": "error", "error": f"HTTP {response.status_code}"}
                
        except Exception as e:
            print(f"❌ CORS test error: {e}")
            return {"status": "error", "error": str(e)}
    
    async def run_integration_tests(self):
        """Run all integration tests."""
        print("🚀 Starting Frontend-Backend Integration Tests")
        print("=" * 60)
        
        results = {}
        
        # Test 1: Health check
        results["health"] = await self.test_health_endpoint()
        
        # Test 2: Query endpoint
        results["query"] = await self.test_query_endpoint()
        query_id = results["query"].get("query_id", "test_query_id")
        
        # Test 3: Feedback endpoint (the main fix)
        results["feedback"] = await self.test_feedback_endpoint(query_id)
        
        # Test 4: CORS configuration
        results["cors"] = await self.test_cors_headers()
        
        print("\n" + "=" * 60)
        print("📊 INTEGRATION TEST RESULTS")
        print("=" * 60)
        
        total_tests = len(results)
        passed_tests = sum(1 for result in results.values() if result["status"] == "success")
        
        for test_name, result in results.items():
            status_icon = "✅" if result["status"] == "success" else "❌"
            print(f"{status_icon} {test_name.upper()}: {result['status']}")
            if result["status"] == "error":
                print(f"   Error: {result['error']}")
        
        print(f"\n📈 SUMMARY: {passed_tests}/{total_tests} tests passed")
        
        if passed_tests == total_tests:
            print("🎉 ALL INTEGRATION TESTS PASSED!")
            print("✅ Frontend-Backend integration is working correctly")
        else:
            print("⚠️ Some integration tests failed")
            print("❌ Frontend-Backend integration needs attention")
        
        await self.client.aclose()
        return results


async def main():
    """Main test function."""
    if len(sys.argv) > 1:
        base_url = sys.argv[1]
    else:
        base_url = "http://localhost:8002"
    
    print(f"🔗 Testing integration with backend at: {base_url}")
    
    tester = IntegrationTester(base_url)
    results = await tester.run_integration_tests()
    
    # Exit with error code if any tests failed
    failed_tests = [name for name, result in results.items() if result["status"] == "error"]
    if failed_tests:
        print(f"\n❌ Failed tests: {', '.join(failed_tests)}")
        sys.exit(1)
    else:
        print("\n✅ All integration tests passed!")
        sys.exit(0)


if __name__ == "__main__":
    asyncio.run(main()) 