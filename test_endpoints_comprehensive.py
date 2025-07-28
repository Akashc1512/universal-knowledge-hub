#!/usr/bin/env python3
"""
Comprehensive Endpoint Testing Script
Tests each frontend-backend endpoint individually for integration, CORS, and authentication.
"""

import asyncio
import json
import sys
import time
from typing import Dict, Any, List

import httpx


class ComprehensiveEndpointTester:
    def __init__(self, backend_url: str = "http://localhost:8002", frontend_url: str = "http://localhost:3000"):
        self.backend_url = backend_url
        self.frontend_url = frontend_url
        self.client = httpx.AsyncClient(timeout=30.0)
        
        # Test different authentication scenarios
        self.auth_scenarios = {
            "valid_user": {"X-API-Key": "user-key-456"},
            "valid_admin": {"X-API-Key": "admin-key-123"},
            "invalid_key": {"X-API-Key": "invalid-key"},
            "no_auth": {},
            "wrong_header": {"Authorization": "Bearer user-key-456"}
        }
        
        # CORS test origins
        self.cors_origins = [
            "http://localhost:3000",
            "http://localhost:3001", 
            "https://app.example.com",
            "https://malicious.com"
        ]

    async def test_endpoint_basic(self, method: str, endpoint: str, auth_key: str = "valid_user", data: Dict = None) -> Dict[str, Any]:
        """Test basic endpoint functionality."""
        headers = {
            "Content-Type": "application/json",
            **self.auth_scenarios[auth_key]
        }
        
        try:
            if method.upper() == "GET":
                response = await self.client.get(f"{self.backend_url}{endpoint}", headers=headers)
            elif method.upper() == "POST":
                response = await self.client.post(f"{self.backend_url}{endpoint}", headers=headers, json=data)
            else:
                return {"status": "error", "error": f"Unsupported method: {method}"}
                
            return {
                "status": "success" if response.status_code < 400 else "error",
                "status_code": response.status_code,
                "response_data": response.json() if response.content else None,
                "headers": dict(response.headers)
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}

    async def test_cors_preflight(self, endpoint: str, origin: str) -> Dict[str, Any]:
        """Test CORS preflight requests."""
        headers = {
            "Origin": origin,
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "Content-Type,X-API-Key"
        }
        
        try:
            response = await self.client.options(f"{self.backend_url}{endpoint}", headers=headers)
            
            cors_headers = {
                "access_control_allow_origin": response.headers.get("access-control-allow-origin"),
                "access_control_allow_methods": response.headers.get("access-control-allow-methods"),
                "access_control_allow_headers": response.headers.get("access-control-allow-headers"),
                "access_control_allow_credentials": response.headers.get("access-control-allow-credentials")
            }
            
            return {
                "status": "success" if response.status_code in [200, 204] else "error",
                "status_code": response.status_code,
                "cors_headers": cors_headers,
                "origin_allowed": cors_headers["access_control_allow_origin"] in ["*", origin]
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}

    async def test_auth_scenarios(self, endpoint: str, method: str = "GET", data: Dict = None) -> Dict[str, Any]:
        """Test different authentication scenarios."""
        results = {}
        
        for scenario_name, headers in self.auth_scenarios.items():
            full_headers = {"Content-Type": "application/json", **headers}
            
            try:
                if method.upper() == "GET":
                    response = await self.client.get(f"{self.backend_url}{endpoint}", headers=full_headers)
                else:
                    response = await self.client.post(f"{self.backend_url}{endpoint}", headers=full_headers, json=data)
                
                results[scenario_name] = {
                    "status_code": response.status_code,
                    "success": response.status_code < 400,
                    "response": response.json() if response.content else None
                }
            except Exception as e:
                results[scenario_name] = {
                    "status_code": 0,
                    "success": False,
                    "error": str(e)
                }
        
        return results

    async def test_root_endpoint(self):
        """Test / endpoint."""
        print("🔍 Testing ROOT (/) endpoint...")
        
        # Basic functionality
        result = await self.test_endpoint_basic("GET", "/")
        print(f"   Basic Test: {'✅ PASS' if result['status'] == 'success' else '❌ FAIL'} (Status: {result.get('status_code', 'N/A')})")
        
        # CORS test
        cors_result = await self.test_cors_preflight("/", "http://localhost:3000")
        print(f"   CORS Test: {'✅ PASS' if cors_result['status'] == 'success' else '❌ FAIL'} (Origin allowed: {cors_result.get('origin_allowed', False)})")
        
        # Auth test
        auth_results = await self.test_auth_scenarios("/")
        valid_auth = auth_results["valid_user"]["success"]
        print(f"   Auth Test: {'✅ PASS' if valid_auth else '❌ FAIL'}")
        
        return {"basic": result, "cors": cors_result, "auth": auth_results}

    async def test_health_endpoint(self):
        """Test /health endpoint."""
        print("🔍 Testing HEALTH (/health) endpoint...")
        
        # Basic functionality
        result = await self.test_endpoint_basic("GET", "/health")
        print(f"   Basic Test: {'✅ PASS' if result['status'] == 'success' else '❌ FAIL'} (Status: {result.get('status_code', 'N/A')})")
        
        # Check response structure
        if result.get('response_data'):
            expected_fields = ['status', 'version', 'timestamp', 'uptime']
            has_fields = all(field in result['response_data'] for field in expected_fields)
            print(f"   Response Structure: {'✅ PASS' if has_fields else '❌ FAIL'}")
        
        # CORS test
        cors_result = await self.test_cors_preflight("/health", "http://localhost:3000")
        print(f"   CORS Test: {'✅ PASS' if cors_result['status'] == 'success' else '❌ FAIL'}")
        
        return {"basic": result, "cors": cors_result}

    async def test_query_endpoint(self):
        """Test /query endpoint."""
        print("🔍 Testing QUERY (/query) endpoint...")
        
        # Test data
        query_data = {
            "query": "What is artificial intelligence?",
            "max_tokens": 1000,
            "confidence_threshold": 0.8,
            "user_context": {"test": "endpoint_testing"}
        }
        
        # Basic functionality
        result = await self.test_endpoint_basic("POST", "/query", data=query_data)
        print(f"   Basic Test: {'✅ PASS' if result['status'] == 'success' else '❌ FAIL'} (Status: {result.get('status_code', 'N/A')})")
        
        # Check response structure
        if result.get('response_data'):
            expected_fields = ['answer', 'confidence', 'citations', 'query_id']
            has_fields = all(field in result['response_data'] for field in expected_fields)
            print(f"   Response Structure: {'✅ PASS' if has_fields else '❌ FAIL'}")
            
            # Check if processing_time is included (new requirement)
            has_processing_time = 'processing_time' in result['response_data']
            print(f"   Processing Time Field: {'✅ PASS' if has_processing_time else '⚠️  MISSING'}")
        
        # CORS test
        cors_result = await self.test_cors_preflight("/query", "http://localhost:3000")
        print(f"   CORS Test: {'✅ PASS' if cors_result['status'] == 'success' else '❌ FAIL'}")
        
        # Auth scenarios
        auth_results = await self.test_auth_scenarios("/query", "POST", query_data)
        valid_auth = auth_results["valid_user"]["success"]
        invalid_auth = not auth_results["invalid_key"]["success"]
        print(f"   Auth Valid Key: {'✅ PASS' if valid_auth else '❌ FAIL'}")
        print(f"   Auth Invalid Key: {'✅ PASS' if invalid_auth else '❌ FAIL'}")
        
        return {"basic": result, "cors": cors_result, "auth": auth_results, "query_data": query_data}

    async def test_feedback_endpoint(self):
        """Test /feedback endpoint."""
        print("🔍 Testing FEEDBACK (/feedback) endpoint...")
        
        # Test data
        feedback_data = {
            "query_id": "test-query-123",
            "feedback_type": "helpful",
            "details": "This is a test feedback submission"
        }
        
        # Basic functionality
        result = await self.test_endpoint_basic("POST", "/feedback", data=feedback_data)
        print(f"   Basic Test: {'✅ PASS' if result['status'] == 'success' else '❌ FAIL'} (Status: {result.get('status_code', 'N/A')})")
        
        # Check response structure
        if result.get('response_data'):
            expected_fields = ['success', 'message']
            has_fields = all(field in result['response_data'] for field in expected_fields)
            print(f"   Response Structure: {'✅ PASS' if has_fields else '❌ FAIL'}")
        
        # CORS test
        cors_result = await self.test_cors_preflight("/feedback", "http://localhost:3000")
        print(f"   CORS Test: {'✅ PASS' if cors_result['status'] == 'success' else '❌ FAIL'}")
        
        # Auth scenarios
        auth_results = await self.test_auth_scenarios("/feedback", "POST", feedback_data)
        valid_auth = auth_results["valid_user"]["success"]
        invalid_auth = not auth_results["invalid_key"]["success"]
        print(f"   Auth Valid Key: {'✅ PASS' if valid_auth else '❌ FAIL'}")
        print(f"   Auth Invalid Key: {'✅ PASS' if invalid_auth else '❌ FAIL'}")
        
        return {"basic": result, "cors": cors_result, "auth": auth_results}

    async def test_analytics_endpoint(self):
        """Test /analytics endpoint."""
        print("🔍 Testing ANALYTICS (/analytics) endpoint...")
        
        # Basic functionality
        result = await self.test_endpoint_basic("GET", "/analytics")
        print(f"   Basic Test: {'✅ PASS' if result['status'] == 'success' else '❌ FAIL'} (Status: {result.get('status_code', 'N/A')})")
        
        # Check response structure
        if result.get('response_data'):
            expected_fields = ['total_queries', 'successful_queries', 'failed_queries']
            has_fields = all(field in result['response_data'] for field in expected_fields)
            print(f"   Response Structure: {'✅ PASS' if has_fields else '❌ FAIL'}")
        
        # CORS test
        cors_result = await self.test_cors_preflight("/analytics", "http://localhost:3000")
        print(f"   CORS Test: {'✅ PASS' if cors_result['status'] == 'success' else '❌ FAIL'}")
        
        # Auth scenarios
        auth_results = await self.test_auth_scenarios("/analytics")
        valid_auth = auth_results["valid_user"]["success"]
        print(f"   Auth Test: {'✅ PASS' if valid_auth else '❌ FAIL'}")
        
        return {"basic": result, "cors": cors_result, "auth": auth_results}

    async def test_metrics_endpoint(self):
        """Test /metrics endpoint."""
        print("🔍 Testing METRICS (/metrics) endpoint...")
        
        # Basic functionality
        result = await self.test_endpoint_basic("GET", "/metrics")
        print(f"   Basic Test: {'✅ PASS' if result['status'] == 'success' else '❌ FAIL'} (Status: {result.get('status_code', 'N/A')})")
        
        # CORS test
        cors_result = await self.test_cors_preflight("/metrics", "http://localhost:3000")
        print(f"   CORS Test: {'✅ PASS' if cors_result['status'] == 'success' else '❌ FAIL'}")
        
        return {"basic": result, "cors": cors_result}

    async def test_comprehensive_cors(self):
        """Test CORS with different origins."""
        print("🔍 Testing COMPREHENSIVE CORS...")
        
        results = {}
        for origin in self.cors_origins:
            cors_result = await self.test_cors_preflight("/query", origin)
            allowed = cors_result.get('origin_allowed', False)
            expected_allowed = origin in ["http://localhost:3000", "http://localhost:3001"]
            
            if expected_allowed:
                status = "✅ PASS" if allowed else "❌ FAIL"
                print(f"   {origin}: {status} (Should be allowed)")
            else:
                status = "✅ PASS" if not allowed else "⚠️  SECURITY RISK"
                print(f"   {origin}: {status} (Should be blocked)")
            
            results[origin] = cors_result
        
        return results

    async def test_frontend_integration(self):
        """Test frontend endpoints if available."""
        print("🔍 Testing FRONTEND Integration...")
        
        try:
            # Test if frontend is accessible
            response = await self.client.get(self.frontend_url)
            if response.status_code == 200:
                print(f"   Frontend Accessible: ✅ PASS (Status: {response.status_code})")
                
                # Check if frontend loads Universal Knowledge Platform
                content = response.text
                has_title = "SarvanOM" in content or "Universal Knowledge Platform" in content
                print(f"   Frontend Content: {'✅ PASS' if has_title else '❌ FAIL'} (Title check)")
                
                # Check if frontend has API integration
                has_api_call = "localhost:8002" in content or "api" in content.lower()
                print(f"   API Integration: {'✅ PASS' if has_api_call else '⚠️  UNKNOWN'}")
                
                return {"status": "success", "accessible": True, "has_title": has_title}
            else:
                print(f"   Frontend Accessible: ❌ FAIL (Status: {response.status_code})")
                return {"status": "error", "accessible": False}
        except Exception as e:
            print(f"   Frontend Accessible: ❌ FAIL (Error: {str(e)})")
            return {"status": "error", "accessible": False, "error": str(e)}

    async def run_comprehensive_tests(self):
        """Run all comprehensive endpoint tests."""
        print("🚀 COMPREHENSIVE ENDPOINT TESTING")
        print("=" * 80)
        
        results = {}
        
        # Test each endpoint individually
        results["root"] = await self.test_root_endpoint()
        print()
        
        results["health"] = await self.test_health_endpoint()
        print()
        
        results["query"] = await self.test_query_endpoint()
        print()
        
        results["feedback"] = await self.test_feedback_endpoint()
        print()
        
        results["analytics"] = await self.test_analytics_endpoint()
        print()
        
        results["metrics"] = await self.test_metrics_endpoint()
        print()
        
        # Comprehensive CORS testing
        results["cors_comprehensive"] = await self.test_comprehensive_cors()
        print()
        
        # Frontend integration testing
        results["frontend"] = await self.test_frontend_integration()
        print()
        
        # Generate summary
        await self.generate_test_summary(results)
        
        await self.client.aclose()
        return results

    async def generate_test_summary(self, results: Dict):
        """Generate comprehensive test summary."""
        print("=" * 80)
        print("📊 COMPREHENSIVE TEST SUMMARY")
        print("=" * 80)
        
        total_tests = 0
        passed_tests = 0
        
        # Count endpoint tests
        endpoint_tests = ["root", "health", "query", "feedback", "analytics", "metrics"]
        for endpoint in endpoint_tests:
            if endpoint in results:
                total_tests += 1
                basic_success = results[endpoint]["basic"]["status"] == "success"
                if basic_success:
                    passed_tests += 1
                print(f"✅ {endpoint.upper()}: {'PASS' if basic_success else 'FAIL'}")
        
        # CORS summary
        cors_working = any(
            results.get("cors_comprehensive", {}).get(origin, {}).get("origin_allowed", False)
            for origin in ["http://localhost:3000", "http://localhost:3001"]
        )
        print(f"✅ CORS: {'PASS' if cors_working else 'FAIL'}")
        
        # Frontend summary
        frontend_working = results.get("frontend", {}).get("accessible", False)
        print(f"✅ FRONTEND: {'PASS' if frontend_working else 'FAIL'}")
        
        print(f"\n📈 OVERALL RESULTS: {passed_tests}/{total_tests} endpoints working")
        
        if passed_tests == total_tests and cors_working:
            print("🎉 ALL INTEGRATION TESTS PASSED!")
            print("✅ Frontend-Backend integration is FULLY OPERATIONAL")
        else:
            print("⚠️ Some tests failed - review individual results above")


async def main():
    """Main test execution."""
    if len(sys.argv) > 1:
        backend_url = sys.argv[1]
    else:
        backend_url = "http://localhost:8002"
    
    if len(sys.argv) > 2:
        frontend_url = sys.argv[2]
    else:
        frontend_url = "http://localhost:3000"
    
    print(f"🔗 Testing integration:")
    print(f"   Backend: {backend_url}")
    print(f"   Frontend: {frontend_url}")
    print()
    
    tester = ComprehensiveEndpointTester(backend_url, frontend_url)
    results = await tester.run_comprehensive_tests()
    
    # Return appropriate exit code
    endpoint_tests = ["root", "health", "query", "feedback", "analytics", "metrics"]
    failed_endpoints = [
        endpoint for endpoint in endpoint_tests 
        if endpoint in results and results[endpoint]["basic"]["status"] != "success"
    ]
    
    if failed_endpoints:
        print(f"\n❌ Failed endpoints: {', '.join(failed_endpoints)}")
        sys.exit(1)
    else:
        print("\n✅ All critical endpoints operational!")
        sys.exit(0)


if __name__ == "__main__":
    asyncio.run(main()) 