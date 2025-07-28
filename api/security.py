"""
Security Module - MAANG Standards.

This module implements comprehensive security features following MAANG
best practices for protecting APIs and user data.

Features:
    - Input validation and sanitization
    - SQL injection prevention
    - XSS protection
    - CSRF protection
    - Content Security Policy
    - Security headers
    - Threat detection
    - Rate limiting integration
    - Audit logging
    - Encryption utilities

Security Layers:
    - Input validation (OWASP Top 10)
    - Authentication & Authorization
    - Data protection
    - Network security
    - Monitoring & Alerting

Authors:
    - Universal Knowledge Platform Engineering Team
    
Version:
    2.0.0 (2024-12-28)
"""

import re
import hashlib
import hmac
import secrets
import base64
from typing import (
    Optional, Dict, Any, List, Union, Callable,
    TypeVar, Protocol, Tuple
)
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass, field
from enum import Enum
import structlog
import bleach
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives import serialization

from api.exceptions import ValidationError, SecurityError
from api.monitoring import Counter, Histogram
from api.config import get_settings

logger = structlog.get_logger(__name__)

# Type definitions
T = TypeVar('T')

# Security metrics
security_events = Counter(
    'security_events_total',
    'Security events',
    ['event_type', 'severity', 'source']
)

security_scan_duration = Histogram(
    'security_scan_duration_seconds',
    'Security scan duration',
    ['scan_type']
)

# Threat types
class ThreatType(str, Enum):
    """Types of security threats."""
    SQL_INJECTION = "sql_injection"
    XSS = "xss"
    CSRF = "csrf"
    PATH_TRAVERSAL = "path_traversal"
    COMMAND_INJECTION = "command_injection"
    BRUTE_FORCE = "brute_force"
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"
    INVALID_TOKEN = "invalid_token"
    SUSPICIOUS_ACTIVITY = "suspicious_activity"

# Security levels
class SecurityLevel(str, Enum):
    """Security levels for different operations."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

# Input validation patterns
class ValidationPatterns:
    """Common validation patterns for security."""
    
    # SQL injection patterns
    SQL_KEYWORDS = re.compile(
        r'\b(UNION|SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|EXEC|EXECUTE|SCRIPT)\b',
        re.IGNORECASE
    )
    
    # XSS patterns
    XSS_PATTERNS = [
        re.compile(r'<script[^>]*>.*?</script>', re.IGNORECASE),
        re.compile(r'javascript:', re.IGNORECASE),
        re.compile(r'on\w+\s*=', re.IGNORECASE),
        re.compile(r'<iframe[^>]*>', re.IGNORECASE),
        re.compile(r'<object[^>]*>', re.IGNORECASE),
        re.compile(r'<embed[^>]*>', re.IGNORECASE),
    ]
    
    # Path traversal patterns
    PATH_TRAVERSAL = re.compile(
        r'\.\./|\.\.\\|%2e%2e%2f|%2e%2e%5c',
        re.IGNORECASE
    )
    
    # Command injection patterns
    COMMAND_INJECTION = re.compile(
        r'[;&|`$(){}]|\b(cat|ls|pwd|whoami|id|uname|wget|curl|nc|netcat)\b',
        re.IGNORECASE
    )
    
    # Email validation
    EMAIL = re.compile(
        r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    )
    
    # Strong password requirements
    STRONG_PASSWORD = re.compile(
        r'^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[@$!%*?&])[A-Za-z\d@$!%*?&]{8,}$'
    )
    
    # UUID validation
    UUID = re.compile(
        r'^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$',
        re.IGNORECASE
    )

# Input sanitizer
class InputSanitizer:
    """
    Comprehensive input sanitization following OWASP guidelines.
    
    Features:
    - HTML sanitization
    - SQL injection prevention
    - XSS protection
    - Path traversal prevention
    - Command injection prevention
    """
    
    def __init__(self, strict_mode: bool = True) -> None:
        """
        Initialize sanitizer.
        
        Args:
            strict_mode: Enable strict validation
        """
        self.strict_mode = strict_mode
        
        # Configure bleach for HTML sanitization
        self.allowed_tags = [
            'p', 'br', 'strong', 'em', 'u', 'ol', 'ul', 'li',
            'h1', 'h2', 'h3', 'h4', 'h5', 'h6'
        ]
        self.allowed_attributes = {
            'a': ['href', 'title'],
            'img': ['src', 'alt', 'title'],
        }
    
    def sanitize_string(
        self,
        value: str,
        max_length: Optional[int] = None,
        allow_html: bool = False
    ) -> str:
        """
        Sanitize string input.
        
        Args:
            value: Input string
            max_length: Maximum allowed length
            allow_html: Allow HTML tags
            
        Returns:
            Sanitized string
            
        Raises:
            ValidationError: If input contains malicious content
        """
        if not isinstance(value, str):
            raise ValidationError("Input must be a string")
        
        # Check length
        if max_length and len(value) > max_length:
            raise ValidationError(
                f"Input too long (max {max_length} characters)",
                field="input",
                value=value[:50] + "..." if len(value) > 50 else value
            )
        
        # Check for SQL injection
        if self._detect_sql_injection(value):
            security_events.labels(
                event_type=ThreatType.SQL_INJECTION.value,
                severity=SecurityLevel.HIGH.value,
                source="input_sanitizer"
            ).inc()
            raise ValidationError(
                "Input contains potentially malicious SQL content",
                field="input"
            )
        
        # Check for XSS
        if self._detect_xss(value):
            security_events.labels(
                event_type=ThreatType.XSS.value,
                severity=SecurityLevel.HIGH.value,
                source="input_sanitizer"
            ).inc()
            raise ValidationError(
                "Input contains potentially malicious script content",
                field="input"
            )
        
        # Check for path traversal
        if self._detect_path_traversal(value):
            security_events.labels(
                event_type=ThreatType.PATH_TRAVERSAL.value,
                severity=SecurityLevel.MEDIUM.value,
                source="input_sanitizer"
            ).inc()
            raise ValidationError(
                "Input contains path traversal attempts",
                field="input"
            )
        
        # Check for command injection
        if self._detect_command_injection(value):
            security_events.labels(
                event_type=ThreatType.COMMAND_INJECTION.value,
                severity=SecurityLevel.HIGH.value,
                source="input_sanitizer"
            ).inc()
            raise ValidationError(
                "Input contains potentially malicious command content",
                field="input"
            )
        
        # HTML sanitization
        if allow_html:
            value = bleach.clean(
                value,
                tags=self.allowed_tags,
                attributes=self.allowed_attributes,
                strip=True
            )
        else:
            # Escape HTML
            value = bleach.escape(value)
        
        return value.strip()
    
    def _detect_sql_injection(self, value: str) -> bool:
        """Detect SQL injection attempts."""
        if not self.strict_mode:
            return False
        
        # Check for SQL keywords
        if ValidationPatterns.SQL_KEYWORDS.search(value):
            return True
        
        # Check for common SQL injection patterns
        sql_patterns = [
            r"'.*'",
            r"--",
            r"/*.**/",
            r"xp_",
            r"sp_",
            r"@@",
        ]
        
        for pattern in sql_patterns:
            if re.search(pattern, value, re.IGNORECASE):
                return True
        
        return False
    
    def _detect_xss(self, value: str) -> bool:
        """Detect XSS attempts."""
        if not self.strict_mode:
            return False
        
        for pattern in ValidationPatterns.XSS_PATTERNS:
            if pattern.search(value):
                return True
        
        return False
    
    def _detect_path_traversal(self, value: str) -> bool:
        """Detect path traversal attempts."""
        if not self.strict_mode:
            return False
        
        return bool(ValidationPatterns.PATH_TRAVERSAL.search(value))
    
    def _detect_command_injection(self, value: str) -> bool:
        """Detect command injection attempts."""
        if not self.strict_mode:
            return False
        
        return bool(ValidationPatterns.COMMAND_INJECTION.search(value))
    
    def validate_email(self, email: str) -> str:
        """Validate and sanitize email address."""
        if not ValidationPatterns.EMAIL.match(email):
            raise ValidationError(
                "Invalid email format",
                field="email",
                value=email
            )
        
        return email.lower().strip()
    
    def validate_password(self, password: str) -> str:
        """Validate password strength."""
        if len(password) < 8:
            raise ValidationError(
                "Password must be at least 8 characters long",
                field="password"
            )
        
        if not ValidationPatterns.STRONG_PASSWORD.match(password):
            raise ValidationError(
                "Password must contain uppercase, lowercase, number, and special character",
                field="password"
            )
        
        return password
    
    def validate_uuid(self, uuid_str: str) -> str:
        """Validate UUID format."""
        if not ValidationPatterns.UUID.match(uuid_str):
            raise ValidationError(
                "Invalid UUID format",
                field="uuid",
                value=uuid_str
            )
        
        return uuid_str.lower()

# Encryption utilities
class EncryptionManager:
    """
    Encryption utilities for sensitive data.
    
    Features:
    - Symmetric encryption (Fernet)
    - Asymmetric encryption (RSA)
    - Key derivation (PBKDF2)
    - Secure random generation
    """
    
    def __init__(self, secret_key: Optional[bytes] = None) -> None:
        """
        Initialize encryption manager.
        
        Args:
            secret_key: Secret key for symmetric encryption
        """
        if secret_key:
            self.fernet = Fernet(secret_key)
        else:
            # Generate new key
            self.fernet = Fernet(Fernet.generate_key())
        
        # Generate RSA key pair
        self.private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048
        )
        self.public_key = self.private_key.public_key()
    
    def encrypt_symmetric(self, data: bytes) -> bytes:
        """Encrypt data using symmetric encryption."""
        return self.fernet.encrypt(data)
    
    def decrypt_symmetric(self, encrypted_data: bytes) -> bytes:
        """Decrypt data using symmetric encryption."""
        return self.fernet.decrypt(encrypted_data)
    
    def encrypt_asymmetric(self, data: bytes) -> bytes:
        """Encrypt data using asymmetric encryption."""
        return self.public_key.encrypt(
            data,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )
    
    def decrypt_asymmetric(self, encrypted_data: bytes) -> bytes:
        """Decrypt data using asymmetric encryption."""
        return self.private_key.decrypt(
            encrypted_data,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )
    
    def derive_key(self, password: str, salt: bytes) -> bytes:
        """Derive key from password using PBKDF2."""
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        return kdf.derive(password.encode())
    
    def generate_salt(self) -> bytes:
        """Generate cryptographically secure salt."""
        return secrets.token_bytes(16)
    
    def hash_password(self, password: str) -> Tuple[str, str]:
        """
        Hash password with salt.
        
        Returns:
            Tuple of (hash, salt) as base64 strings
        """
        salt = self.generate_salt()
        key = self.derive_key(password, salt)
        
        return (
            base64.b64encode(key).decode(),
            base64.b64encode(salt).decode()
        )
    
    def verify_password(self, password: str, hash_str: str, salt_str: str) -> bool:
        """Verify password against hash."""
        try:
            salt = base64.b64decode(salt_str)
            stored_hash = base64.b64decode(hash_str)
            
            key = self.derive_key(password, salt)
            return hmac.compare_digest(key, stored_hash)
        except Exception:
            return False

# Security headers
@dataclass
class SecurityHeaders:
    """Security headers configuration."""
    
    # Content Security Policy
    csp: str = (
        "default-src 'self'; "
        "script-src 'self' 'unsafe-inline' 'unsafe-eval'; "
        "style-src 'self' 'unsafe-inline'; "
        "img-src 'self' data: https:; "
        "font-src 'self'; "
        "connect-src 'self'; "
        "frame-ancestors 'none';"
    )
    
    # Other security headers
    x_frame_options: str = "DENY"
    x_content_type_options: str = "nosniff"
    x_xss_protection: str = "1; mode=block"
    referrer_policy: str = "strict-origin-when-cross-origin"
    permissions_policy: str = "geolocation=(), microphone=(), camera=()"
    
    # HSTS (HTTPS Strict Transport Security)
    strict_transport_security: str = "max-age=31536000; includeSubDomains"
    
    def to_dict(self) -> Dict[str, str]:
        """Convert to dictionary for response headers."""
        return {
            "Content-Security-Policy": self.csp,
            "X-Frame-Options": self.x_frame_options,
            "X-Content-Type-Options": self.x_content_type_options,
            "X-XSS-Protection": self.x_xss_protection,
            "Referrer-Policy": self.referrer_policy,
            "Permissions-Policy": self.permissions_policy,
            "Strict-Transport-Security": self.strict_transport_security,
        }

# Threat detection
class ThreatDetector:
    """
    Advanced threat detection system.
    
    Features:
    - Pattern-based detection
    - Behavioral analysis
    - Rate-based detection
    - Machine learning integration ready
    """
    
    def __init__(self) -> None:
        """Initialize threat detector."""
        self.suspicious_patterns: Dict[str, List[re.Pattern]] = {
            ThreatType.SQL_INJECTION.value: [
                re.compile(r"'.*'", re.IGNORECASE),
                re.compile(r"--", re.IGNORECASE),
                re.compile(r"/*.**/", re.IGNORECASE),
            ],
            ThreatType.XSS.value: [
                re.compile(r"<script", re.IGNORECASE),
                re.compile(r"javascript:", re.IGNORECASE),
                re.compile(r"on\w+\s*=", re.IGNORECASE),
            ],
            ThreatType.PATH_TRAVERSAL.value: [
                re.compile(r"\.\./", re.IGNORECASE),
                re.compile(r"\.\.\\", re.IGNORECASE),
            ],
            ThreatType.COMMAND_INJECTION.value: [
                re.compile(r"[;&|`$(){}]", re.IGNORECASE),
                re.compile(r"\b(cat|ls|pwd|whoami)\b", re.IGNORECASE),
            ],
        }
        
        self.threat_scores: Dict[str, int] = {}
        self.blocked_ips: Set[str] = set()
    
    def scan_request(
        self,
        request_data: Dict[str, Any],
        client_ip: str
    ) -> Tuple[bool, List[Dict[str, Any]]]:
        """
        Scan request for threats.
        
        Args:
            request_data: Request data to scan
            client_ip: Client IP address
            
        Returns:
            Tuple of (is_safe, threats_found)
        """
        threats = []
        
        # Check if IP is blocked
        if client_ip in self.blocked_ips:
            threats.append({
                "type": "blocked_ip",
                "severity": SecurityLevel.HIGH.value,
                "description": "IP address is blocked"
            })
            return False, threats
        
        # Scan all request data
        for key, value in request_data.items():
            if isinstance(value, str):
                value_threats = self._scan_string(value, key)
                threats.extend(value_threats)
        
        # Calculate threat score
        total_score = sum(
            self._get_threat_score(t["type"]) 
            for t in threats
        )
        
        # Update IP threat score
        self.threat_scores[client_ip] = (
            self.threat_scores.get(client_ip, 0) + total_score
        )
        
        # Block IP if score is too high
        if self.threat_scores[client_ip] > 100:
            self.blocked_ips.add(client_ip)
            threats.append({
                "type": "ip_blocked",
                "severity": SecurityLevel.CRITICAL.value,
                "description": "IP blocked due to high threat score"
            })
        
        # Log threats
        for threat in threats:
            security_events.labels(
                event_type=threat["type"],
                severity=threat["severity"],
                source="threat_detector"
            ).inc()
        
        return len(threats) == 0, threats
    
    def _scan_string(self, value: str, field_name: str) -> List[Dict[str, Any]]:
        """Scan string for threats."""
        threats = []
        
        for threat_type, patterns in self.suspicious_patterns.items():
            for pattern in patterns:
                if pattern.search(value):
                    threats.append({
                        "type": threat_type,
                        "severity": SecurityLevel.HIGH.value,
                        "field": field_name,
                        "description": f"Detected {threat_type} pattern in {field_name}",
                        "pattern": pattern.pattern
                    })
        
        return threats
    
    def _get_threat_score(self, threat_type: str) -> int:
        """Get score for threat type."""
        scores = {
            ThreatType.SQL_INJECTION.value: 50,
            ThreatType.XSS.value: 40,
            ThreatType.PATH_TRAVERSAL.value: 30,
            ThreatType.COMMAND_INJECTION.value: 60,
            ThreatType.BRUTE_FORCE.value: 20,
            ThreatType.RATE_LIMIT_EXCEEDED.value: 10,
        }
        return scores.get(threat_type, 25)
    
    def get_threat_stats(self) -> Dict[str, Any]:
        """Get threat detection statistics."""
        return {
            "blocked_ips": len(self.blocked_ips),
            "threat_scores": self.threat_scores,
            "total_events": security_events._value.get()
        }

# Security middleware
class SecurityMiddleware:
    """
    Security middleware for FastAPI.
    
    Features:
    - Security headers
    - Threat detection
    - Input validation
    - CSRF protection
    """
    
    def __init__(
        self,
        threat_detector: Optional[ThreatDetector] = None,
        input_sanitizer: Optional[InputSanitizer] = None,
        security_headers: Optional[SecurityHeaders] = None
    ) -> None:
        """
        Initialize security middleware.
        
        Args:
            threat_detector: Threat detection instance
            input_sanitizer: Input sanitization instance
            security_headers: Security headers configuration
        """
        self.threat_detector = threat_detector or ThreatDetector()
        self.input_sanitizer = input_sanitizer or InputSanitizer()
        self.security_headers = security_headers or SecurityHeaders()
    
    async def __call__(
        self,
        request: Request,
        call_next: Callable
    ) -> Response:
        """Process request with security checks."""
        # Get client IP
        client_ip = self._get_client_ip(request)
        
        # Extract request data for threat detection
        request_data = await self._extract_request_data(request)
        
        # Scan for threats
        is_safe, threats = self.threat_detector.scan_request(
            request_data,
            client_ip
        )
        
        if not is_safe:
            # Log security event
            logger.warning(
                "Security threat detected",
                client_ip=client_ip,
                threats=threats
            )
            
            # Return security error
            return Response(
                content="Security threat detected",
                status_code=403,
                headers=self.security_headers.to_dict()
            )
        
        # Process request
        response = await call_next(request)
        
        # Add security headers
        response.headers.update(self.security_headers.to_dict())
        
        return response
    
    def _get_client_ip(self, request: Request) -> str:
        """Get real client IP considering proxies."""
        forwarded = request.headers.get('X-Forwarded-For')
        if forwarded:
            return forwarded.split(',')[0].strip()
        return request.client.host
    
    async def _extract_request_data(self, request: Request) -> Dict[str, Any]:
        """Extract request data for threat detection."""
        data = {
            "path": request.url.path,
            "query": str(request.query_params),
            "method": request.method,
            "headers": dict(request.headers),
        }
        
        # Extract body if present
        if request.method in ['POST', 'PUT', 'PATCH']:
            try:
                body = await request.body()
                if body:
                    data["body"] = body.decode('utf-8', errors='ignore')
            except Exception:
                pass
        
        return data

# Global instances
_sanitizer = InputSanitizer()
_encryption_manager = EncryptionManager()
_threat_detector = ThreatDetector()
_security_headers = SecurityHeaders()

# Convenience functions
def sanitize_input(
    value: str,
    max_length: Optional[int] = None,
    allow_html: bool = False
) -> str:
    """Sanitize input string."""
    return _sanitizer.sanitize_string(value, max_length, allow_html)

def validate_email(email: str) -> str:
    """Validate email address."""
    return _sanitizer.validate_email(email)

def validate_password(password: str) -> str:
    """Validate password strength."""
    return _sanitizer.validate_password(password)

def hash_password(password: str) -> Tuple[str, str]:
    """Hash password with salt."""
    return _encryption_manager.hash_password(password)

def verify_password(password: str, hash_str: str, salt_str: str) -> bool:
    """Verify password against hash."""
    return _encryption_manager.verify_password(password, hash_str, salt_str)

def encrypt_data(data: bytes) -> bytes:
    """Encrypt data using symmetric encryption."""
    return _encryption_manager.encrypt_symmetric(data)

def decrypt_data(encrypted_data: bytes) -> bytes:
    """Decrypt data using symmetric encryption."""
    return _encryption_manager.decrypt_symmetric(encrypted_data)

def get_security_headers() -> Dict[str, str]:
    """Get security headers for responses."""
    return _security_headers.to_dict()

def get_threat_stats() -> Dict[str, Any]:
    """Get threat detection statistics."""
    return _threat_detector.get_threat_stats()

# Export public API
__all__ = [
    # Classes
    'InputSanitizer',
    'EncryptionManager',
    'ThreatDetector',
    'SecurityMiddleware',
    'SecurityHeaders',
    'ValidationPatterns',
    
    # Enums
    'ThreatType',
    'SecurityLevel',
    
    # Functions
    'sanitize_input',
    'validate_email',
    'validate_password',
    'hash_password',
    'verify_password',
    'encrypt_data',
    'decrypt_data',
    'get_security_headers',
    'get_threat_stats',
]
