"""
User Management Module for Universal Knowledge Platform
Handles user creation, authentication, and role management.
"""

import os
import logging
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
from passlib.context import CryptContext
import jwt
from pydantic import BaseModel, EmailStr, Field
import json
from pathlib import Path

logger = logging.getLogger(__name__)

# Password hashing configuration
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# JWT configuration
SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-here-change-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))

# User roles
class UserRole:
    ADMIN = "admin"
    USER = "user"
    READONLY = "readonly"

# User models
class UserCreate(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    email: EmailStr
    password: str = Field(..., min_length=8)
    role: str = Field(default=UserRole.USER)
    full_name: Optional[str] = None

class UserInDB(BaseModel):
    username: str
    email: str
    hashed_password: str
    role: str
    full_name: Optional[str]
    is_active: bool = True
    created_at: datetime
    last_login: Optional[datetime] = None

class UserResponse(BaseModel):
    username: str
    email: str
    role: str
    full_name: Optional[str]
    is_active: bool
    created_at: datetime
    last_login: Optional[datetime]

class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int

class TokenData(BaseModel):
    username: Optional[str] = None
    role: Optional[str] = None

class UserManager:
    """Manages user operations including creation, authentication, and storage."""
    
    def __init__(self, users_file: str = "data/users.json"):
        self.users_file = Path(users_file)
        self.users_file.parent.mkdir(parents=True, exist_ok=True)
        self._users_cache: Dict[str, UserInDB] = {}
        self._load_users()
        
        # Create default users if none exist
        if not self._users_cache:
            self._create_default_users()
    
    def _load_users(self):
        """Load users from JSON file."""
        if self.users_file.exists():
            try:
                with open(self.users_file, 'r') as f:
                    users_data = json.load(f)
                    for username, user_data in users_data.items():
                        # Convert datetime strings back to datetime objects
                        if 'created_at' in user_data:
                            user_data['created_at'] = datetime.fromisoformat(user_data['created_at'])
                        if 'last_login' in user_data and user_data['last_login']:
                            user_data['last_login'] = datetime.fromisoformat(user_data['last_login'])
                        self._users_cache[username] = UserInDB(**user_data)
                logger.info(f"Loaded {len(self._users_cache)} users")
            except Exception as e:
                logger.error(f"Error loading users: {e}")
    
    def _save_users(self):
        """Save users to JSON file."""
        try:
            users_data = {}
            for username, user in self._users_cache.items():
                user_dict = user.dict()
                # Convert datetime to ISO format strings
                user_dict['created_at'] = user_dict['created_at'].isoformat()
                if user_dict['last_login']:
                    user_dict['last_login'] = user_dict['last_login'].isoformat()
                users_data[username] = user_dict
            
            with open(self.users_file, 'w') as f:
                json.dump(users_data, f, indent=2)
            logger.info(f"Saved {len(users_data)} users")
        except Exception as e:
            logger.error(f"Error saving users: {e}")
    
    def _create_default_users(self):
        """Create default admin and user accounts."""
        logger.info("Creating default users...")
        
        # Create admin user
        admin = UserCreate(
            username="admin",
            email="admin@sarvanom.ai",
            password="AdminPass123!",
            role=UserRole.ADMIN,
            full_name="System Administrator"
        )
        self.create_user(admin)
        
        # Create regular user
        user = UserCreate(
            username="user",
            email="user@sarvanom.ai",
            password="UserPass123!",
            role=UserRole.USER,
            full_name="Default User"
        )
        self.create_user(user)
        
        logger.info("Default users created successfully")
        # Log the creation of default credentials with security warning
        logger.warning("Default user credentials created - CHANGE IMMEDIATELY IN PRODUCTION")
        logger.info("Admin account created with username: admin")
        logger.info("User account created with username: user")
        logger.warning("Default passwords are insecure - change immediately!")
        
        # Print to console for development convenience (remove in production)
        if os.getenv("ENVIRONMENT") != "production":
            print("\n" + "="*60)
            print("🔐 DEFAULT USER CREDENTIALS CREATED")
            print("="*60)
            print("Admin Account:")
            print("  Username: admin")
            print("  Password: AdminPass123!")
            print("\nUser Account:")
            print("  Username: user")
            print("  Password: UserPass123!")
            print("\n⚠️  IMPORTANT: Change these passwords immediately!")
            print("="*60 + "\n")
    
    def get_password_hash(self, password: str) -> str:
        """Hash a password."""
        return pwd_context.hash(password)
    
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify a password against its hash."""
        return pwd_context.verify(plain_password, hashed_password)
    
    def create_user(self, user_create: UserCreate) -> UserResponse:
        """Create a new user."""
        # Check if user already exists
        if user_create.username in self._users_cache:
            raise ValueError(f"User {user_create.username} already exists")
        
        # Check if email already exists
        for existing_user in self._users_cache.values():
            if existing_user.email == user_create.email:
                raise ValueError(f"Email {user_create.email} already registered")
        
        # Create user
        user_in_db = UserInDB(
            username=user_create.username,
            email=user_create.email,
            hashed_password=self.get_password_hash(user_create.password),
            role=user_create.role,
            full_name=user_create.full_name,
            is_active=True,
            created_at=datetime.utcnow()
        )
        
        # Save to cache and file
        self._users_cache[user_create.username] = user_in_db
        self._save_users()
        
        logger.info(f"Created user: {user_create.username} with role: {user_create.role}")
        
        return UserResponse(**user_in_db.dict())
    
    def get_user(self, username: str) -> Optional[UserInDB]:
        """Get user by username."""
        return self._users_cache.get(username)
    
    def authenticate_user(self, username: str, password: str) -> Optional[UserInDB]:
        """Authenticate a user."""
        user = self.get_user(username)
        if not user:
            return None
        if not self.verify_password(password, user.hashed_password):
            return None
        if not user.is_active:
            return None
        
        # Update last login
        user.last_login = datetime.utcnow()
        self._save_users()
        
        return user
    
    def create_access_token(self, data: dict, expires_delta: Optional[timedelta] = None) -> str:
        """Create a JWT access token."""
        to_encode = data.copy()
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        
        to_encode.update({"exp": expire})
        encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
        return encoded_jwt
    
    def verify_token(self, token: str) -> Optional[TokenData]:
        """Verify and decode a JWT token."""
        try:
            payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
            username: str = payload.get("sub")
            role: str = payload.get("role")
            if username is None:
                return None
            return TokenData(username=username, role=role)
        except jwt.ExpiredSignatureError:
            logger.warning("Token expired")
            return None
        except jwt.JWTError as e:
            logger.warning(f"JWT error: {e}")
            return None
    
    def login(self, username: str, password: str) -> Optional[Token]:
        """Login a user and return access token."""
        user = self.authenticate_user(username, password)
        if not user:
            return None
        
        # Create access token
        access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = self.create_access_token(
            data={"sub": user.username, "role": user.role},
            expires_delta=access_token_expires
        )
        
        logger.info(f"User {username} logged in successfully")
        
        return Token(
            access_token=access_token,
            expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60  # Convert to seconds
        )
    
    def change_password(self, username: str, current_password: str, new_password: str) -> bool:
        """Change user password."""
        user = self.authenticate_user(username, current_password)
        if not user:
            return False
        
        # Update password
        user.hashed_password = self.get_password_hash(new_password)
        self._save_users()
        
        logger.info(f"Password changed for user: {username}")
        return True
    
    def update_user_role(self, username: str, new_role: str) -> bool:
        """Update user role (admin only)."""
        user = self.get_user(username)
        if not user:
            return False
        
        user.role = new_role
        self._save_users()
        
        logger.info(f"Role updated for user {username}: {new_role}")
        return True
    
    def deactivate_user(self, username: str) -> bool:
        """Deactivate a user."""
        user = self.get_user(username)
        if not user:
            return False
        
        user.is_active = False
        self._save_users()
        
        logger.info(f"User deactivated: {username}")
        return True
    
    def list_users(self) -> List[UserResponse]:
        """List all users."""
        return [UserResponse(**user.dict()) for user in self._users_cache.values()]
    
    def get_user_by_token(self, token: str) -> Optional[UserInDB]:
        """Get user from JWT token."""
        token_data = self.verify_token(token)
        if not token_data:
            return None
        
        return self.get_user(token_data.username)


# Global user manager instance
_user_manager: Optional[UserManager] = None

def get_user_manager() -> UserManager:
    """Get or create the global user manager instance."""
    global _user_manager
    
    if _user_manager is None:
        _user_manager = UserManager()
    
    return _user_manager 