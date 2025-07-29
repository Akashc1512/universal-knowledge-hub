"""
Authentication and Authorization System for Universal Knowledge Platform
Provides secure access control with API keys, OAuth, and role-based permissions.
"""

import os
import time
import hashlib
import hmac
import secrets
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
import jwt
from fastapi import HTTPException, Depends, Request, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials, OAuth2PasswordBearer
import logging

from api.exceptions import ConfigurationError

logger = logging.getLogger(__name__)

# Security configuration
SECRET_KEY = os.getenv("SECRET_KEY", secrets.token_urlsafe(32))
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))

# Environment-based API Key configuration
def _load_api_keys():
    """Load API keys from environment or generate secure defaults for development."""
    api_keys = {}
    
    # Only use hardcoded keys in development if no environment keys are set
    if os.getenv("ENVIRONMENT") == "development" and not os.getenv("ADMIN_API_KEY"):
        logger.warning("Using development API keys - NOT SECURE FOR PRODUCTION")
        api_keys.update({
            "admin-dev-key": {
                "role": "admin",
                "permissions": ["read", "write", "admin"],
                "rate_limit": 1000,
                "description": "Development Admin Key - INSECURE",
            },
            "user-dev-key": {
                "role": "user",
                "permissions": ["read", "write"],
                "rate_limit": 100,
                "description": "Development User Key - INSECURE",
            },
            "readonly-dev-key": {
                "role": "readonly",
                "permissions": ["read"],
                "rate_limit": 50,
                "description": "Development Readonly Key - INSECURE",
            },
        })
    
    # Load from environment variables (production)
    admin_key = os.getenv("ADMIN_API_KEY")
    if admin_key:
        api_keys[admin_key] = {
            "role": "admin",
            "permissions": ["read", "write", "admin"],
            "rate_limit": int(os.getenv("ADMIN_RATE_LIMIT", "1000")),
            "description": "Production Admin API Key",
        }
    
    user_key = os.getenv("USER_API_KEY")
    if user_key:
        api_keys[user_key] = {
            "role": "user",
            "permissions": ["read", "write"],
            "rate_limit": int(os.getenv("USER_RATE_LIMIT", "100")),
            "description": "Production User API Key",
        }
    
    readonly_key = os.getenv("READONLY_API_KEY")
    if readonly_key:
        api_keys[readonly_key] = {
            "role": "readonly",
            "permissions": ["read"],
            "rate_limit": int(os.getenv("READONLY_RATE_LIMIT", "50")),
            "description": "Production Readonly API Key",
        }
    
    # Warn if no API keys are configured
    if not api_keys:
        logger.error("No API keys configured! Set ADMIN_API_KEY, USER_API_KEY, or READONLY_API_KEY environment variables.")
        raise ConfigurationError("authentication", "No API keys configured", "Set API key environment variables")
    
    return api_keys

# Load API keys at startup
API_KEYS = _load_api_keys()

# User database (in production, use a real database)
USERS_DB = {
    "admin": {
        "password_hash": "hashed_password_here",  # In production, use proper hashing
        "role": "admin",
        "permissions": ["read", "write", "admin"],
        "api_keys": ["admin-key-123"],
    },
    "user1": {
        "password_hash": "hashed_password_here",
        "role": "user", 
        "permissions": ["read", "write"],
        "api_keys": ["user-key-456"],
    },
    "readonly_user": {
        "password_hash": "hashed_password_here",
        "role": "readonly",
        "permissions": ["read"],
        "api_keys": ["readonly-key-789"],
    },
}

# API Key management
API_KEY_REGISTRY = {}

def generate_api_key(user_id: str, role: str, permissions: List[str]) -> str:
    """Generate a new API key for a user."""
    api_key = f"{user_id}-{secrets.token_urlsafe(16)}"
    API_KEY_REGISTRY[api_key] = {
        "user_id": user_id,
        "role": role,
        "permissions": permissions,
        "rate_limit": 100 if role == "user" else 50 if role == "readonly" else 1000,
        "description": f"API Key for {user_id}",
        "created_at": datetime.now(),
    }
    return api_key

def revoke_api_key(api_key: str) -> bool:
    """Revoke an API key."""
    if api_key in API_KEY_REGISTRY:
        del API_KEY_REGISTRY[api_key]
        return True
    return False

def get_user_api_keys(user_id: str) -> List[str]:
    """Get all API keys for a user."""
    return [
        key for key, data in API_KEY_REGISTRY.items() 
        if data["user_id"] == user_id
    ]

# OAuth configuration (for future use)
OAUTH_CONFIG = {
    "google": {
        "client_id": os.getenv("GOOGLE_CLIENT_ID"),
        "client_secret": os.getenv("GOOGLE_CLIENT_SECRET"),
        "redirect_uri": os.getenv("GOOGLE_REDIRECT_URI"),
    },
    "github": {
        "client_id": os.getenv("GITHUB_CLIENT_ID"),
        "client_secret": os.getenv("GITHUB_CLIENT_SECRET"),
        "redirect_uri": os.getenv("GITHUB_REDIRECT_URI"),
    },
}

# Password hashing utilities
def hash_password(password: str) -> str:
    """Hash a password using bcrypt."""
    import bcrypt
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
    return hashed.decode('utf-8')

def verify_password(password: str, hashed: str) -> bool:
    """Verify a password against its hash."""
    import bcrypt
    return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))

# Login and registration functions
async def authenticate_user(username: str, password: str) -> Optional[Dict[str, Any]]:
    """Authenticate a user with username and password."""
    if username not in USERS_DB:
        return None
    
    user_data = USERS_DB[username]
    if verify_password(password, user_data["password_hash"]):
        return {
            "user_id": username,
            "role": user_data["role"],
            "permissions": user_data["permissions"],
        }
    return None

async def register_user(username: str, password: str, role: str = "user") -> Optional[str]:
    """Register a new user and return their API key."""
    if username in USERS_DB:
        return None  # User already exists
    
    # Hash password
    hashed_password = hash_password(password)
    
    # Determine permissions based on role
    permissions = ["read", "write"] if role == "user" else ["read"] if role == "readonly" else ["read", "write", "admin"]
    
    # Add user to database
    USERS_DB[username] = {
        "password_hash": hashed_password,
        "role": role,
        "permissions": permissions,
        "api_keys": [],
    }
    
    # Generate API key
    api_key = generate_api_key(username, role, permissions)
    USERS_DB[username]["api_keys"].append(api_key)
    
    return api_key

async def login_user(username: str, password: str) -> Optional[Dict[str, Any]]:
    """Login a user and return access token and API key."""
    user_data = await authenticate_user(username, password)
    if not user_data:
        return None
    
    # Generate access token
    access_token = JWTAuth.create_access_token(
        data={"sub": user_data["user_id"], "role": user_data["role"]}
    )
    
    # Get or generate API key
    api_keys = get_user_api_keys(user_data["user_id"])
    if not api_keys:
        api_key = generate_api_key(user_data["user_id"], user_data["role"], user_data["permissions"])
    else:
        api_key = api_keys[0]
    
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "api_key": api_key,
        "user_id": user_data["user_id"],
        "role": user_data["role"],
        "permissions": user_data["permissions"],
    }

# User sessions (in production, use Redis or database)
user_sessions = {}

security = HTTPBearer(auto_error=False)


class AuthException(HTTPException):
    """Custom authentication exception."""

    def __init__(self, detail: str):
        super().__init__(status_code=status.HTTP_401_UNAUTHORIZED, detail=detail)


class PermissionException(HTTPException):
    """Custom permission exception."""

    def __init__(self, detail: str):
        super().__init__(status_code=status.HTTP_403_FORBIDDEN, detail=detail)


class User:
    """User model with authentication and authorization."""

    def __init__(
        self, user_id: str, role: str, permissions: List[str], api_key: Optional[str] = None
    ):
        self.user_id = user_id
        self.role = role
        self.permissions = permissions
        self.api_key = api_key
        self.created_at = datetime.now()
        self.last_activity = datetime.now()

    def has_permission(self, permission: str) -> bool:
        """Check if user has specific permission."""
        return permission in self.permissions

    def has_role(self, role: str) -> bool:
        """Check if user has specific role."""
        return self.role == role

    def update_activity(self):
        """Update last activity timestamp."""
        self.last_activity = datetime.now()


class APIKeyAuth:
    """API Key authentication handler."""

    @staticmethod
    def verify_api_key(api_key: str) -> Optional[Dict[str, Any]]:
        """Verify API key and return user info."""
        # Check static API keys first
        if api_key in API_KEYS:
            user_info = API_KEYS[api_key]
            logger.debug(f"Found API key: {api_key}")  # Proper debug logging
            user = User(
                user_id=f"user_{api_key[-8:]}",
                role=user_info["role"],
                permissions=user_info["permissions"],
                api_key=api_key,
            )
            logger.debug(f"User created: {user.user_id}, role: {user.role}")  # Proper debug logging
            return user
        
        # Check dynamic API key registry
        if api_key in API_KEY_REGISTRY:
            key_info = API_KEY_REGISTRY[api_key]
            user = User(
                user_id=key_info["user_id"],
                role=key_info["role"],
                permissions=key_info["permissions"],
                api_key=api_key,
            )
            return user
        
        logger.debug(f"Failed to create user from API key: {api_key}")  # Proper debug logging
        return None

    @staticmethod
    def create_user_from_api_key(api_key: str) -> Optional[User]:
        """Create user object from API key."""
        user_info = APIKeyAuth.verify_api_key(api_key)
        if not user_info:
            return None

        return User(
            user_id=f"api_user_{api_key[:8]}",
            role=user_info["role"],
            permissions=user_info["permissions"],
            api_key=api_key,
        )


class JWTAuth:
    """JWT authentication handler."""

    @staticmethod
    def create_access_token(data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
        """Create JWT access token."""
        to_encode = data.copy()
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)

        to_encode.update({"exp": expire})
        encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
        return encoded_jwt

    @staticmethod
    def verify_token(token: str) -> Optional[Dict[str, Any]]:
        """Verify JWT token and return payload."""
        try:
            payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
            return payload
        except jwt.PyJWTError:
            return None

    @staticmethod
    def create_user_from_token(token: str) -> Optional[User]:
        """Create user object from JWT token."""
        payload = JWTAuth.verify_token(token)
        if not payload:
            return None

        return User(
            user_id=payload.get("sub"),
            role=payload.get("role", "user"),
            permissions=payload.get("permissions", ["read"]),
        )


class RateLimiter:
    """Rate limiting with different limits per user type."""

    def __init__(self):
        self.requests = {}

    def check_rate_limit(self, user_id: str, limit: int) -> bool:
        """Check if user has exceeded rate limit."""
        now = time.time()

        if user_id not in self.requests:
            self.requests[user_id] = []

        # Clean old requests
        self.requests[user_id] = [
            req_time
            for req_time in self.requests[user_id]
            if now - req_time < 60  # 1 minute window
        ]

        # Check limit
        if len(self.requests[user_id]) >= limit:
            return False

        # Add current request
        self.requests[user_id].append(now)
        return True


# Global rate limiter
rate_limiter = RateLimiter()


async def get_current_user(
    request: Request, credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
) -> User:
    """Get current authenticated user."""
    
    # Import here to avoid circular import
    try:
        from api.user_management_v2 import get_user_manager
    except ImportError:
        # Fallback to legacy user management
        from api.user_management import get_user_manager

    # Check for API key in headers
    api_key = request.headers.get("X-API-Key")
    if api_key:
        user = APIKeyAuth.create_user_from_api_key(api_key)
        if user:
            # Check rate limit for API key user
            limit = API_KEYS.get(api_key, {}).get("rate_limit", 100)
            if not rate_limiter.check_rate_limit(user.user_id, limit):
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail=f"Rate limit exceeded. Limit: {limit} requests/minute",
                )
            user.update_activity()
            return user

    # Check for Bearer token
    if credentials:
        # First try the new user management system
        user_manager = get_user_manager()
        user_from_db = user_manager.get_user_by_token(credentials.credentials)
        if user_from_db:
            # Create User object from database user
            return User(
                user_id=user_from_db.username,
                role=user_from_db.role,
                permissions=["read", "write"] if user_from_db.role in ["admin", "user"] else ["read"]
            )
        
        # Fall back to legacy JWT auth
        user = JWTAuth.create_user_from_token(credentials.credentials)
        if user:
            user.update_activity()
            return user

    # Check for session token (for web interface)
    session_token = request.cookies.get("session_token")
    if session_token and session_token in user_sessions:
        user = user_sessions[session_token]
        user.update_activity()
        return user

    # No authentication found
    raise AuthException("Authentication required. Please provide API key or valid token.")


async def require_permission(permission: str):
    """Dependency to require specific permission."""

    def permission_checker(current_user: User = Depends(get_current_user)) -> User:
        if not current_user.has_permission(permission):
            raise PermissionException(f"Permission '{permission}' required")
        return current_user

    return permission_checker


async def require_role(role: str):
    """Dependency to require specific role."""

    def role_checker(current_user: User = Depends(get_current_user)) -> User:
        if not current_user.has_role(role):
            raise PermissionException(f"Role '{role}' required")
        return current_user

    return role_checker


def require_read():
    """Dependency to require read permission."""

    async def permission_checker(current_user: User = Depends(get_current_user)) -> User:
        if not current_user.has_permission("read"):
            raise PermissionException("Read permission required")
        return current_user

    return permission_checker


def require_write():
    """Dependency to require write permission."""

    async def permission_checker(current_user: User = Depends(get_current_user)) -> User:
        if not current_user.has_permission("write"):
            raise PermissionException("Write permission required")
        return current_user

    return permission_checker


def require_admin():
    """Dependency to require admin permission."""

    async def permission_checker(current_user: User = Depends(get_current_user)) -> User:
        if not current_user.has_permission("admin"):
            raise PermissionException("Admin permission required")
        return current_user

    return permission_checker


def require_admin_role():
    """Dependency to require admin role."""

    async def role_checker(current_user: User = Depends(get_current_user)) -> User:
        if not current_user.has_role("admin"):
            raise PermissionException("Admin role required")
        return current_user

    return role_checker


def sanitize_log_data(data: str, max_length: int = 100) -> str:
    """Sanitize data for logging to prevent sensitive information exposure."""
    if not data:
        return ""

    # Truncate long data
    if len(data) > max_length:
        return data[:max_length] + "..."

    # Remove potential sensitive patterns
    sensitive_patterns = [
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",  # Email
        r"\b\d{3}-\d{2}-\d{4}\b",  # SSN
        r"\b\d{4}-\d{4}-\d{4}-\d{4}\b",  # Credit card
        r"\b\d{10,}\b",  # Long numbers (phone, etc.)
    ]

    import re

    sanitized = data
    for pattern in sensitive_patterns:
        sanitized = re.sub(pattern, "[REDACTED]", sanitized)

    return sanitized


def log_request_safely(request: Request, user: Optional[User] = None):
    """Log request information safely without exposing sensitive data."""
    log_data = {
        "method": request.method,
        "path": request.url.path,
        "client_ip": get_client_ip(request),
        "user_agent": request.headers.get("User-Agent", "Unknown"),
        "timestamp": datetime.now().isoformat(),
    }

    if user:
        log_data["user_id"] = user.user_id
        log_data["role"] = user.role

    # Sanitize any query parameters
    if request.query_params:
        sanitized_params = {}
        for key, value in request.query_params.items():
            sanitized_params[key] = sanitize_log_data(value, 50)
        log_data["query_params"] = sanitized_params

    logger.info(f"Request: {log_data}")


def get_client_ip(request: Request) -> str:
    """Get client IP address safely."""
    forwarded_for = request.headers.get("X-Forwarded-For")
    if forwarded_for:
        return forwarded_for.split(",")[0].strip()
    return request.client.host if request.client else "unknown"
