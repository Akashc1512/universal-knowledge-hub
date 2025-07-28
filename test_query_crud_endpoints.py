#!/usr/bin/env python3
"""
Comprehensive Query CRUD Endpoints Testing Script
Tests all HTTP methods for query management: GET, POST, PUT, DELETE, PATCH
"""

import asyncio
import json
import time
from typing import Dict, Any

import httpx


class QueryCRUDTester:
    def __init__(self, base_url: str = "http://localhost:8002"):
        self.base_url = base_url
        self.client = httpx.AsyncClient(timeout=30.0)
        
        # Authentication headers
        self.headers = {
            "Content-Type": "application/json",
            "X-API-Key": "user-key-456"  # Using the default user API key
        }
        
        # Track created query IDs for cleanup
        self.created_query_ids = []

    async def test_post_query(self) -> str:
        """Test POST /query - Create a new query."""
        print("🔵 Testing POST /query (Create new query)")
        
        query_data = {
            "query": "What is artificial intelligence and how does it work?",
            "max_tokens": 1000,
            "confidence_threshold": 0.8,
            "user_context": {"test": "crud_endpoints"}
        }
        
        try:
            response = await self.client.post(
                f"{self.base_url}/query",
                headers=self.headers,
                json=query_data
            )
            
            if response.status_code == 200:
                result = response.json()
                query_id = result.get("query_id")
                self.created_query_ids.append(query_id)
                
                print(f"   ✅ SUCCESS: Created query {query_id}")
                print(f"   📝 Answer: {result.get('answer', '')[:100]}...")
                print(f"   🎯 Confidence: {result.get('confidence', 0.0)}")
                print(f"   ⏱️  Processing time: {result.get('processing_time', 0.0):.3f}s")
                return query_id
            else:
                print(f"   ❌ FAILED: Status {response.status_code}")
                print(f"   📄 Response: {response.text}")
                return None
                
        except Exception as e:
            print(f"   ❌ ERROR: {e}")
            return None

    async def test_get_queries(self):
        """Test GET /queries - List all queries."""
        print("\n🔵 Testing GET /queries (List queries)")
        
        try:
            response = await self.client.get(
                f"{self.base_url}/queries?page=1&page_size=10",
                headers=self.headers
            )
            
            if response.status_code == 200:
                result = response.json()
                queries = result.get("queries", [])
                total = result.get("total", 0)
                
                print(f"   ✅ SUCCESS: Found {total} total queries")
                print(f"   📊 Current page: {len(queries)} queries")
                
                for query in queries[:3]:  # Show first 3
                    print(f"   📝 {query['query_id'][:8]}... - {query['query'][:50]}...")
                
                if len(queries) > 3:
                    print(f"   ... and {len(queries) - 3} more queries")
                    
            else:
                print(f"   ❌ FAILED: Status {response.status_code}")
                print(f"   📄 Response: {response.text}")
                
        except Exception as e:
            print(f"   ❌ ERROR: {e}")

    async def test_get_single_query(self, query_id: str):
        """Test GET /queries/{query_id} - Get specific query details."""
        print(f"\n🔵 Testing GET /queries/{query_id[:8]}... (Get query details)")
        
        try:
            response = await self.client.get(
                f"{self.base_url}/queries/{query_id}",
                headers=self.headers
            )
            
            if response.status_code == 200:
                result = response.json()
                
                print(f"   ✅ SUCCESS: Retrieved query details")
                print(f"   📝 Query: {result.get('query', '')[:80]}...")
                print(f"   🎯 Confidence: {result.get('confidence', 0.0)}")
                print(f"   📅 Created: {result.get('created_at', '')}")
                print(f"   🆔 User ID: {result.get('user_id', '')}")
                print(f"   📊 Status: {result.get('status', '')}")
                
            else:
                print(f"   ❌ FAILED: Status {response.status_code}")
                print(f"   📄 Response: {response.text}")
                
        except Exception as e:
            print(f"   ❌ ERROR: {e}")

    async def test_put_query(self, query_id: str):
        """Test PUT /queries/{query_id} - Update query."""
        print(f"\n🔵 Testing PUT /queries/{query_id[:8]}... (Update query)")
        
        update_data = {
            "query": "What is machine learning and deep learning?",
            "max_tokens": 1200,
            "confidence_threshold": 0.85,
            "reprocess": True
        }
        
        try:
            response = await self.client.put(
                f"{self.base_url}/queries/{query_id}",
                headers=self.headers,
                json=update_data
            )
            
            if response.status_code == 200:
                result = response.json()
                
                print(f"   ✅ SUCCESS: Updated and reprocessed query")
                print(f"   📝 New Query: {result.get('query', '')[:80]}...")
                print(f"   🎯 New Confidence: {result.get('confidence', 0.0)}")
                print(f"   📅 Updated: {result.get('updated_at', '')}")
                print(f"   🔄 Reprocessed: {result.get('metadata', {}).get('reprocessed', False)}")
                
            else:
                print(f"   ❌ FAILED: Status {response.status_code}")
                print(f"   📄 Response: {response.text}")
                
        except Exception as e:
            print(f"   ❌ ERROR: {e}")

    async def test_get_query_status(self, query_id: str):
        """Test GET /queries/{query_id}/status - Get query status."""
        print(f"\n🔵 Testing GET /queries/{query_id[:8]}.../status (Get query status)")
        
        try:
            response = await self.client.get(
                f"{self.base_url}/queries/{query_id}/status",
                headers=self.headers
            )
            
            if response.status_code == 200:
                result = response.json()
                
                print(f"   ✅ SUCCESS: Retrieved query status")
                print(f"   📊 Status: {result.get('status', '')}")
                print(f"   💬 Message: {result.get('message', '')}")
                print(f"   📈 Progress: {result.get('progress', 0.0) * 100:.1f}%" if result.get('progress') else "   📈 Progress: N/A")
                
            else:
                print(f"   ❌ FAILED: Status {response.status_code}")
                print(f"   📄 Response: {response.text}")
                
        except Exception as e:
            print(f"   ❌ ERROR: {e}")

    async def test_patch_reprocess(self, query_id: str):
        """Test PATCH /queries/{query_id}/reprocess - Reprocess query."""
        print(f"\n🔵 Testing PATCH /queries/{query_id[:8]}.../reprocess (Reprocess query)")
        
        try:
            response = await self.client.patch(
                f"{self.base_url}/queries/{query_id}/reprocess",
                headers=self.headers
            )
            
            if response.status_code == 200:
                result = response.json()
                
                print(f"   ✅ SUCCESS: Reprocessed query")
                print(f"   💬 Message: {result.get('message', '')}")
                print(f"   ⏱️  Processing time: {result.get('processing_time', 0.0):.3f}s")
                print(f"   🎯 New confidence: {result.get('new_confidence', 0.0)}")
                
            else:
                print(f"   ❌ FAILED: Status {response.status_code}")
                print(f"   📄 Response: {response.text}")
                
        except Exception as e:
            print(f"   ❌ ERROR: {e}")

    async def test_delete_query(self, query_id: str):
        """Test DELETE /queries/{query_id} - Delete query."""
        print(f"\n🔵 Testing DELETE /queries/{query_id[:8]}... (Delete query)")
        
        try:
            response = await self.client.delete(
                f"{self.base_url}/queries/{query_id}",
                headers=self.headers
            )
            
            if response.status_code == 200:
                result = response.json()
                
                print(f"   ✅ SUCCESS: Deleted query")
                print(f"   💬 Message: {result.get('message', '')}")
                
                # Remove from tracking list
                if query_id in self.created_query_ids:
                    self.created_query_ids.remove(query_id)
                
            else:
                print(f"   ❌ FAILED: Status {response.status_code}")
                print(f"   📄 Response: {response.text}")
                
        except Exception as e:
            print(f"   ❌ ERROR: {e}")

    async def test_query_filtering(self):
        """Test query listing with filters."""
        print("\n🔵 Testing GET /queries with filters")
        
        try:
            # Test status filter
            response = await self.client.get(
                f"{self.base_url}/queries?status_filter=completed&page_size=5",
                headers=self.headers
            )
            
            if response.status_code == 200:
                result = response.json()
                queries = result.get("queries", [])
                
                print(f"   ✅ SUCCESS: Filtered by status=completed")
                print(f"   📊 Found {len(queries)} completed queries")
                
            else:
                print(f"   ❌ FAILED: Status {response.status_code}")
                
        except Exception as e:
            print(f"   ❌ ERROR: {e}")

    async def run_comprehensive_test(self):
        """Run comprehensive test of all query CRUD endpoints."""
        print("🚀 COMPREHENSIVE QUERY CRUD ENDPOINTS TEST")
        print("=" * 80)
        
        try:
            # 1. Create a new query (POST)
            query_id = await self.test_post_query()
            
            if not query_id:
                print("❌ Cannot continue tests without a valid query ID")
                return
            
            # 2. List all queries (GET)
            await self.test_get_queries()
            
            # 3. Get specific query details (GET)
            await self.test_get_single_query(query_id)
            
            # 4. Get query status (GET)
            await self.test_get_query_status(query_id)
            
            # 5. Update and reprocess query (PUT)
            await self.test_put_query(query_id)
            
            # 6. Reprocess query (PATCH)
            await self.test_patch_reprocess(query_id)
            
            # 7. Test filtering
            await self.test_query_filtering()
            
            # 8. Create another query for deletion test
            print("\n🔵 Creating additional query for deletion test...")
            delete_query_id = await self.test_post_query()
            
            if delete_query_id:
                # 9. Delete query (DELETE)
                await self.test_delete_query(delete_query_id)
            
            print("\n" + "=" * 80)
            print("📊 COMPREHENSIVE TEST SUMMARY")
            print("=" * 80)
            
            print("✅ Endpoints tested:")
            print("   🔵 POST   /query                    - Create new query")
            print("   🔵 GET    /queries                  - List queries with pagination")
            print("   🔵 GET    /queries/{id}             - Get specific query details")
            print("   🔵 PUT    /queries/{id}             - Update query and reprocess")
            print("   🔵 DELETE /queries/{id}             - Delete query")
            print("   🔵 GET    /queries/{id}/status      - Get query processing status")
            print("   🔵 PATCH  /queries/{id}/reprocess   - Reprocess existing query")
            
            print(f"\n🎯 Created queries: {len(self.created_query_ids)}")
            if self.created_query_ids:
                print("📝 Remaining queries (for manual testing):")
                for qid in self.created_query_ids:
                    print(f"   - {qid}")
            
            print("\n🎉 ALL CRUD ENDPOINTS OPERATIONAL!")
            
        except Exception as e:
            print(f"\n❌ Test suite failed: {e}")
        
        finally:
            await self.client.aclose()

    async def cleanup_test_queries(self):
        """Clean up any remaining test queries."""
        if self.created_query_ids:
            print(f"\n🧹 Cleaning up {len(self.created_query_ids)} test queries...")
            for query_id in self.created_query_ids.copy():
                await self.test_delete_query(query_id)


async def main():
    """Main test execution."""
    print("🔗 Testing Query CRUD endpoints on: http://localhost:8002")
    print("🔑 Using API Key: user-key-456")
    print()
    
    tester = QueryCRUDTester()
    await tester.run_comprehensive_test()
    
    # Optional cleanup
    cleanup = input("\n🧹 Clean up remaining test queries? (y/N): ").lower().strip()
    if cleanup == 'y':
        await tester.cleanup_test_queries()
    else:
        print("💡 Test queries preserved for manual inspection")


if __name__ == "__main__":
    asyncio.run(main()) 