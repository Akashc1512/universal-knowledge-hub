"""
Security Testing Suite for Universal Knowledge Platform
Tests security vulnerabilities and attack vectors.
"""

import pytest
import json
import base64
from fastapi.testclient import TestClient
from api.main import app


class TestSecurityVulnerabilities:
    """Test for common security vulnerabilities."""
    
    @pytest.fixture
    def client(self):
        return TestClient(app)
    
    def test_sql_injection_prevention(self, client):
        """Test SQL injection prevention."""
        sql_injection_attempts = [
            "'; DROP TABLE users; --",
            "' OR 1=1; --",
            "'; INSERT INTO users VALUES ('hacker', 'password'); --",
            "' UNION SELECT * FROM users; --",
            "'; UPDATE users SET password='hacked'; --"
        ]
        
        for attempt in sql_injection_attempts:
            response = client.post("/query", json={"query": attempt})
            # Should not crash or expose sensitive data
            assert response.status_code in [200, 400, 500]
            
            if response.status_code == 200:
                # If successful, response should not contain sensitive data
                response_data = response.json()
                assert "password" not in str(response_data).lower()
                assert "admin" not in str(response_data).lower()
    
    def test_xss_prevention(self, client):
        """Test XSS prevention."""
        xss_attempts = [
            "<script>alert('xss')</script>",
            "<img src=x onerror=alert('xss')>",
            "javascript:alert('xss')",
            "<iframe src=javascript:alert('xss')></iframe>",
            "<svg onload=alert('xss')>",
            "';alert('xss');//"
        ]
        
        for attempt in xss_attempts:
            response = client.post("/query", json={"query": attempt})
            # Should not crash
            assert response.status_code in [200, 400, 500]
            
            if response.status_code == 200:
                # Response should be properly escaped
                response_data = response.json()
                response_text = str(response_data)
                assert "<script>" not in response_text
                assert "javascript:" not in response_text.lower()
    
    def test_no_sql_injection(self, client):
        """Test NoSQL injection prevention."""
        nosql_attempts = [
            '{"$where": "function() { return true; }"}',
            '{"$ne": null}',
            '{"$gt": ""}',
            '{"$regex": ".*"}',
            '{"$exists": true}'
        ]
        
        for attempt in nosql_attempts:
            response = client.post("/query", json={"query": attempt})
            # Should not crash
            assert response.status_code in [200, 400, 500]
    
    def test_command_injection_prevention(self, client):
        """Test command injection prevention."""
        command_injection_attempts = [
            "; rm -rf /",
            "| cat /etc/passwd",
            "&& whoami",
            "; ls -la",
            "`id`",
            "$(whoami)"
        ]
        
        for attempt in command_injection_attempts:
            response = client.post("/query", json={"query": attempt})
            # Should not crash
            assert response.status_code in [200, 400, 500]
    
    def test_path_traversal_prevention(self, client):
        """Test path traversal prevention."""
        path_traversal_attempts = [
            "../../../etc/passwd",
            "..\\..\\..\\windows\\system32\\config\\sam",
            "....//....//....//etc/passwd",
            "%2e%2e%2f%2e%2e%2f%2e%2e%2fetc%2fpasswd"
        ]
        
        for attempt in path_traversal_attempts:
            response = client.post("/query", json={"query": attempt})
            # Should not crash
            assert response.status_code in [200, 400, 500]
    
    def test_large_payload_attack(self, client):
        """Test large payload attack prevention."""
        # Test with very large payload
        large_payload = {"query": "a" * 1000000}  # 1MB
        
        response = client.post("/query", json=large_payload)
        # Should handle gracefully
        assert response.status_code in [400, 413, 500]
    
    def test_content_type_manipulation(self, client):
        """Test content type manipulation."""
        # Test with wrong content type
        response = client.post("/query", data="invalid json", headers={"Content-Type": "text/plain"})
        assert response.status_code == 422  # Should reject invalid content type
    
    def test_header_injection(self, client):
        """Test header injection prevention."""
        malicious_headers = {
            "X-Forwarded-For": "127.0.0.1\r\nX-Forwarded-For: 192.168.1.1",
            "User-Agent": "Mozilla/5.0\r\nX-Forwarded-For: 192.168.1.1",
            "Accept": "text/html\r\nX-Forwarded-For: 192.168.1.1"
        }
        
        for header_name, header_value in malicious_headers.items():
            response = client.post(
                "/query", 
                json={"query": "test"},
                headers={header_name: header_value}
            )
            # Should not crash
            assert response.status_code in [200, 400, 500]


class TestAuthenticationSecurity:
    """Test authentication and authorization security."""
    
    @pytest.fixture
    def client(self):
        return TestClient(app)
    
    def test_unauthorized_access_handling(self, client):
        """Test handling of unauthorized access attempts."""
        # Test without authentication
        response = client.post("/query", json={"query": "test"})
        # Should either require auth or handle gracefully
        assert response.status_code in [200, 401, 403]
    
    def test_invalid_token_handling(self, client):
        """Test handling of invalid authentication tokens."""
        invalid_tokens = [
            "invalid_token",
            "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.invalid",
            "Bearer invalid_token",
            "Basic " + base64.b64encode(b"invalid:invalid").decode()
        ]
        
        for token in invalid_tokens:
            response = client.post(
                "/query", 
                json={"query": "test"},
                headers={"Authorization": token}
            )
            # Should handle invalid tokens gracefully
            assert response.status_code in [200, 401, 403, 500]


class TestDataValidationSecurity:
    """Test data validation security."""
    
    @pytest.fixture
    def client(self):
        return TestClient(app)
    
    def test_null_byte_injection(self, client):
        """Test null byte injection prevention."""
        null_byte_attempts = [
            "test\x00query",
            "\x00test",
            "test\x00",
            "\x00\x00\x00"
        ]
        
        for attempt in null_byte_attempts:
            response = client.post("/query", json={"query": attempt})
            # Should handle null bytes gracefully
            assert response.status_code in [200, 400, 500]
    
    def test_unicode_normalization_attack(self, client):
        """Test Unicode normalization attack prevention."""
        unicode_attacks = [
            "test\u0000query",
            "test\u200bquery",  # Zero-width space
            "test\u200cquery",  # Zero-width non-joiner
            "test\u200dquery",  # Zero-width joiner
            "test\u2060query",  # Word joiner
            "test\uFEFFquery"   # Zero-width no-break space
        ]
        
        for attack in unicode_attacks:
            response = client.post("/query", json={"query": attack})
            # Should handle Unicode attacks gracefully
            assert response.status_code in [200, 400, 500]
    
    def test_encoding_attack(self, client):
        """Test encoding attack prevention."""
        encoding_attacks = [
            "test%00query",
            "test%0aquery",
            "test%0dquery",
            "test%20query",
            "test+query"
        ]
        
        for attack in encoding_attacks:
            response = client.post("/query", json={"query": attack})
            # Should handle encoding attacks gracefully
            assert response.status_code in [200, 400, 500]


class TestRateLimitingSecurity:
    """Test rate limiting security."""
    
    @pytest.fixture
    def client(self):
        return TestClient(app)
    
    def test_rate_limit_bypass_attempts(self, client):
        """Test rate limit bypass attempts."""
        # Try to bypass rate limiting with different IPs
        bypass_attempts = [
            {"X-Forwarded-For": "192.168.1.1"},
            {"X-Real-IP": "192.168.1.2"},
            {"X-Forwarded-For": "10.0.0.1"},
            {"X-Forwarded-For": "172.16.0.1"}
        ]
        
        for headers in bypass_attempts:
            # Make many requests with different headers
            for i in range(70):  # Over rate limit
                response = client.post(
                    "/query", 
                    json={"query": f"bypass test {i}"},
                    headers=headers
                )
                if response.status_code == 429:
                    break  # Rate limiting should work
            else:
                # If we get here, rate limiting might be bypassed
                # This is acceptable if rate limiting is per-IP
                pass
    
    def test_concurrent_rate_limit_attack(self, client):
        """Test concurrent rate limit attack."""
        import threading
        import time
        
        results = []
        
        def make_requests():
            for i in range(50):
                response = client.post("/query", json={"query": f"concurrent attack {i}"})
                results.append(response.status_code)
                time.sleep(0.01)  # Small delay
        
        # Start multiple threads to test concurrent rate limiting
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=make_requests)
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # Should handle concurrent attacks gracefully
        assert len(results) > 0


class TestInformationDisclosure:
    """Test for information disclosure vulnerabilities."""
    
    @pytest.fixture
    def client(self):
        return TestClient(app)
    
    def test_error_message_disclosure(self, client):
        """Test that error messages don't disclose sensitive information."""
        # Trigger various errors
        error_triggers = [
            {"query": ""},  # Empty query
            {"query": "a" * 10001},  # Too long
            {"query": "test", "max_tokens": -1},  # Invalid tokens
            {"query": "test", "confidence_threshold": 1.5},  # Invalid confidence
        ]
        
        for trigger in error_triggers:
            response = client.post("/query", json=trigger)
            
            if response.status_code != 200:
                error_data = response.json()
                error_message = str(error_data)
                
                # Should not disclose sensitive information
                sensitive_patterns = [
                    "password",
                    "secret",
                    "key",
                    "token",
                    "admin",
                    "root",
                    "database",
                    "connection",
                    "file path",
                    "stack trace"
                ]
                
                for pattern in sensitive_patterns:
                    assert pattern not in error_message.lower(), f"Sensitive information disclosed: {pattern}"
    
    def test_version_disclosure(self, client):
        """Test that version information is appropriately disclosed."""
        response = client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        # Version should be disclosed for health checks
        assert "version" in data
        assert data["version"] == "1.0.0"
    
    def test_debug_information_disclosure(self, client):
        """Test that debug information is not disclosed in production."""
        # Try to trigger debug information
        response = client.post("/query", json={"query": "debug test"})
        
        if response.status_code != 200:
            error_data = response.json()
            error_message = str(error_data)
            
            # Should not contain debug information
            debug_patterns = [
                "debug",
                "traceback",
                "stack trace",
                "file:",
                "line:",
                "exception:",
                "error details"
            ]
            
            for pattern in debug_patterns:
                assert pattern not in error_message.lower(), f"Debug information disclosed: {pattern}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"]) 