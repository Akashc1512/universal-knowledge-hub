"""
Authentication and authorization module for Universal Knowledge Platform.
Includes JWT token management, user models, and authentication logic.
"""

import asyncio
import logging
import hashlib
import secrets
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from enum import Enum
import jwt
from pydantic import BaseModel, EmailStr

logger = logging.getLogger(__name__)


class UserRole(str, Enum):
    """User roles for authorization."""
    ADMIN = "admin"
    MODERATOR = "moderator"
    USER = "user"
    GUEST = "guest"


class Permission(str, Enum):
    """System permissions."""
    READ_CONTENT = "read_content"
    WRITE_CONTENT = "write_content"
    DELETE_CONTENT = "delete_content"
    MANAGE_USERS = "manage_users"
    MANAGE_SYSTEM = "manage_system"
    VIEW_ANALYTICS = "view_analytics"


@dataclass
class User:
    """User model for authentication and authorization."""
    id: str
    username: str
    email: str
    password_hash: str
    role: UserRole = UserRole.USER
    permissions: List[Permission] = field(default_factory=list)
    is_active: bool = True
    is_verified: bool = False
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    last_login: Optional[datetime] = None
    failed_login_attempts: int = 0
    locked_until: Optional[datetime] = None
    
    def has_permission(self, permission: Permission) -> bool:
        """Check if user has specific permission."""
        if self.role == UserRole.ADMIN:
            return True
        return permission in self.permissions
    
    def has_role(self, role: UserRole) -> bool:
        """Check if user has specific role."""
        role_hierarchy = {
            UserRole.ADMIN: 4,
            UserRole.MODERATOR: 3,
            UserRole.USER: 2,
            UserRole.GUEST: 1,
        }
        return role_hierarchy.get(self.role, 0) >= role_hierarchy.get(role, 0)


class UserCreate(BaseModel):
    """Model for user creation."""
    username: str
    email: EmailStr
    password: str
    role: UserRole = UserRole.USER


class UserLogin(BaseModel):
    """Model for user login."""
    username: str
    password: str


class UserUpdate(BaseModel):
    """Model for user updates."""
    email: Optional[EmailStr] = None
    role: Optional[UserRole] = None
    permissions: Optional[List[Permission]] = None
    is_active: Optional[bool] = None


class TokenData(BaseModel):
    """Model for JWT token data."""
    user_id: str
    username: str
    role: UserRole
    permissions: List[Permission]
    exp: datetime


class AuthenticationService:
    """
    Authentication service for user management and JWT token handling.
    """
    
    def __init__(self, secret_key: str, algorithm: str = "HS256"):
        self.secret_key = secret_key
        self.algorithm = algorithm
        self.access_token_expire_minutes = 30
        self.refresh_token_expire_days = 7
        self.max_failed_attempts = 5
        self.lockout_duration_minutes = 15
        
        # In-memory user storage (replace with database in production)
        self.users: Dict[str, User] = {}
        self.refresh_tokens: Dict[str, str] = {}  # token -> user_id
    
    def hash_password(self, password: str) -> str:
        """Hash password using SHA-256 with salt."""
        salt = secrets.token_hex(16)
        hash_obj = hashlib.sha256()
        hash_obj.update((password + salt).encode())
        return f"{salt}${hash_obj.hexdigest()}"
    
    def verify_password(self, password: str, password_hash: str) -> bool:
        """Verify password against hash."""
        try:
            salt, hash_value = password_hash.split('$', 1)
            hash_obj = hashlib.sha256()
            hash_obj.update((password + salt).encode())
            return hash_obj.hexdigest() == hash_value
        except ValueError:
            return False
    
    def create_access_token(self, user: User) -> str:
        """Create JWT access token."""
        expire = datetime.utcnow() + timedelta(minutes=self.access_token_expire_minutes)
        to_encode = {
            "sub": user.id,
            "username": user.username,
            "role": user.role.value,
            "permissions": [p.value for p in user.permissions],
            "exp": expire,
            "type": "access"
        }
        return jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
    
    def create_refresh_token(self, user: User) -> str:
        """Create JWT refresh token."""
        expire = datetime.utcnow() + timedelta(days=self.refresh_token_expire_days)
        to_encode = {
            "sub": user.id,
            "exp": expire,
            "type": "refresh"
        }
        token = jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
        self.refresh_tokens[token] = user.id
        return token
    
    def verify_token(self, token: str) -> Optional[TokenData]:
        """Verify and decode JWT token."""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            
            if payload.get("type") != "access":
                return None
            
            user_id = payload.get("sub")
            if user_id is None:
                return None
            
            return TokenData(
                user_id=user_id,
                username=payload.get("username"),
                role=UserRole(payload.get("role")),
                permissions=[Permission(p) for p in payload.get("permissions", [])],
                exp=datetime.fromtimestamp(payload.get("exp"))
            )
        except jwt.ExpiredSignatureError:
            logger.warning("Token expired")
            return None
        except jwt.JWTError as e:
            logger.warning(f"Invalid token: {e}")
            return None
    
    def verify_refresh_token(self, token: str) -> Optional[str]:
        """Verify refresh token and return user ID."""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            
            if payload.get("type") != "refresh":
                return None
            
            user_id = payload.get("sub")
            if user_id is None or token not in self.refresh_tokens:
                return None
            
            return user_id
        except jwt.ExpiredSignatureError:
            logger.warning("Refresh token expired")
            return None
        except jwt.JWTError as e:
            logger.warning(f"Invalid refresh token: {e}")
            return None
    
    async def create_user(self, user_data: UserCreate) -> User:
        """Create a new user."""
        # Check if username or email already exists
        for user in self.users.values():
            if user.username == user_data.username:
                raise ValueError("Username already exists")
            if user.email == user_data.email:
                raise ValueError("Email already exists")
        
        # Create user
        user_id = secrets.token_hex(16)
        password_hash = self.hash_password(user_data.password)
        
        user = User(
            id=user_id,
            username=user_data.username,
            email=user_data.email,
            password_hash=password_hash,
            role=user_data.role,
            permissions=self._get_default_permissions(user_data.role)
        )
        
        self.users[user_id] = user
        logger.info(f"Created user: {user.username}")
        return user
    
    async def authenticate_user(self, username: str, password: str) -> Optional[User]:
        """Authenticate user with username and password."""
        # Find user by username or email
        user = None
        for u in self.users.values():
            if u.username == username or u.email == username:
                user = u
                break
        
        if not user:
            return None
        
        # Check if account is locked
        if user.locked_until and user.locked_until > datetime.utcnow():
            logger.warning(f"Account locked for user: {user.username}")
            return None
        
        # Verify password
        if not self.verify_password(password, user.password_hash):
            # Increment failed attempts
            user.failed_login_attempts += 1
            
            # Lock account if too many failed attempts
            if user.failed_login_attempts >= self.max_failed_attempts:
                user.locked_until = datetime.utcnow() + timedelta(minutes=self.lockout_duration_minutes)
                logger.warning(f"Account locked for user: {user.username}")
            
            return None
        
        # Reset failed attempts on successful login
        user.failed_login_attempts = 0
        user.locked_until = None
        user.last_login = datetime.utcnow()
        user.updated_at = datetime.utcnow()
        
        logger.info(f"User authenticated: {user.username}")
        return user
    
    async def get_user_by_id(self, user_id: str) -> Optional[User]:
        """Get user by ID."""
        return self.users.get(user_id)
    
    async def get_user_by_username(self, username: str) -> Optional[User]:
        """Get user by username."""
        for user in self.users.values():
            if user.username == username:
                return user
        return None
    
    async def update_user(self, user_id: str, user_data: UserUpdate) -> Optional[User]:
        """Update user information."""
        user = self.users.get(user_id)
        if not user:
            return None
        
        if user_data.email is not None:
            # Check if email is already taken
            for u in self.users.values():
                if u.id != user_id and u.email == user_data.email:
                    raise ValueError("Email already exists")
            user.email = user_data.email
        
        if user_data.role is not None:
            user.role = user_data.role
            user.permissions = self._get_default_permissions(user_data.role)
        
        if user_data.permissions is not None:
            user.permissions = user_data.permissions
        
        if user_data.is_active is not None:
            user.is_active = user_data.is_active
        
        user.updated_at = datetime.utcnow()
        
        logger.info(f"Updated user: {user.username}")
        return user
    
    async def delete_user(self, user_id: str) -> bool:
        """Delete user."""
        if user_id in self.users:
            username = self.users[user_id].username
            del self.users[user_id]
            logger.info(f"Deleted user: {username}")
            return True
        return False
    
    async def revoke_refresh_token(self, token: str) -> bool:
        """Revoke a refresh token."""
        if token in self.refresh_tokens:
            del self.refresh_tokens[token]
            return True
        return False
    
    def _get_default_permissions(self, role: UserRole) -> List[Permission]:
        """Get default permissions for a role."""
        permissions = {
            UserRole.ADMIN: [
                Permission.READ_CONTENT,
                Permission.WRITE_CONTENT,
                Permission.DELETE_CONTENT,
                Permission.MANAGE_USERS,
                Permission.MANAGE_SYSTEM,
                Permission.VIEW_ANALYTICS
            ],
            UserRole.MODERATOR: [
                Permission.READ_CONTENT,
                Permission.WRITE_CONTENT,
                Permission.DELETE_CONTENT,
                Permission.VIEW_ANALYTICS
            ],
            UserRole.USER: [
                Permission.READ_CONTENT,
                Permission.WRITE_CONTENT
            ],
            UserRole.GUEST: [
                Permission.READ_CONTENT
            ]
        }
        return permissions.get(role, [])
    
    async def create_default_admin(self) -> User:
        """Create default admin user if no users exist."""
        if not self.users:
            admin_data = UserCreate(
                username="admin",
                email="admin@universal-knowledge-hub.com",
                password="admin123",  # Change in production
                role=UserRole.ADMIN
            )
            return await self.create_user(admin_data)
        return None


# Global authentication service instance
# Use a default secret key for development - should be overridden in production
DEFAULT_SECRET_KEY = "your-secret-key-change-in-production"
auth_service = AuthenticationService(
    secret_key=DEFAULT_SECRET_KEY,
    algorithm="HS256"
)


# FastAPI dependency for authentication
async def get_current_user(token: str) -> Optional[User]:
    """Get current user from JWT token."""
    token_data = auth_service.verify_token(token)
    if token_data is None:
        return None
    
    user = await auth_service.get_user_by_id(token_data.user_id)
    if user is None or not user.is_active:
        return None
    
    return user


async def require_permission(permission: Permission):
    """Decorator to require specific permission."""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            # This would be implemented in FastAPI dependency injection
            # For now, it's a placeholder
            return await func(*args, **kwargs)
        return wrapper
    return decorator


# Example usage functions
async def login_user(username: str, password: str) -> Optional[Dict[str, str]]:
    """Login user and return tokens."""
    user = await auth_service.authenticate_user(username, password)
    if not user:
        return None
    
    access_token = auth_service.create_access_token(user)
    refresh_token = auth_service.create_refresh_token(user)
    
    return {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "token_type": "bearer"
    }


async def refresh_access_token(refresh_token: str) -> Optional[str]:
    """Refresh access token using refresh token."""
    user_id = auth_service.verify_refresh_token(refresh_token)
    if not user_id:
        return None
    
    user = await auth_service.get_user_by_id(user_id)
    if not user or not user.is_active:
        return None
    
    return auth_service.create_access_token(user)


async def logout_user(refresh_token: str) -> bool:
    """Logout user by revoking refresh token."""
    return await auth_service.revoke_refresh_token(refresh_token) 