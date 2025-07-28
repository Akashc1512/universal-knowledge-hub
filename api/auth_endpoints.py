"""
Authentication Endpoints for Universal Knowledge Platform
Handles login, logout, and user management endpoints.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from typing import List
import logging

from api.user_management import (
    UserManager, get_user_manager,
    UserCreate, UserResponse, Token,
    UserRole
)
from api.auth import get_current_user

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/auth", tags=["authentication"])

# OAuth2 scheme for token authentication
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")


@router.post("/login", response_model=Token)
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    """
    Login endpoint that accepts username and password.
    Returns JWT access token.
    """
    user_manager = get_user_manager()
    
    # Authenticate user
    token = user_manager.login(form_data.username, form_data.password)
    
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    return token


@router.post("/register", response_model=UserResponse)
async def register(user_create: UserCreate):
    """
    Register a new user (open registration).
    New users get 'user' role by default.
    """
    user_manager = get_user_manager()
    
    # Force role to 'user' for open registration
    user_create.role = UserRole.USER
    
    try:
        user = user_manager.create_user(user_create)
        return user
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )


@router.get("/me", response_model=UserResponse)
async def get_current_user_info(current_user=Depends(get_current_user)):
    """Get current user information."""
    user_manager = get_user_manager()
    user = user_manager.get_user(current_user.user_id)
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    return UserResponse(**user.dict())


@router.post("/change-password")
async def change_password(
    current_password: str,
    new_password: str,
    current_user=Depends(get_current_user)
):
    """Change current user's password."""
    user_manager = get_user_manager()
    
    success = user_manager.change_password(
        current_user.user_id,
        current_password,
        new_password
    )
    
    if not success:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Current password is incorrect"
        )
    
    return {"message": "Password changed successfully"}


# Admin endpoints
@router.get("/users", response_model=List[UserResponse])
async def list_users(current_user=Depends(get_current_user)):
    """
    List all users (admin only).
    """
    # Check if user is admin
    user_manager = get_user_manager()
    user = user_manager.get_user(current_user.user_id)
    
    if not user or user.role != UserRole.ADMIN:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )
    
    return user_manager.list_users()


@router.post("/users", response_model=UserResponse)
async def create_user(
    user_create: UserCreate,
    current_user=Depends(get_current_user)
):
    """
    Create a new user with any role (admin only).
    """
    # Check if user is admin
    user_manager = get_user_manager()
    user = user_manager.get_user(current_user.user_id)
    
    if not user or user.role != UserRole.ADMIN:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )
    
    try:
        new_user = user_manager.create_user(user_create)
        return new_user
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )


@router.put("/users/{username}/role")
async def update_user_role(
    username: str,
    new_role: str,
    current_user=Depends(get_current_user)
):
    """
    Update user role (admin only).
    """
    # Check if user is admin
    user_manager = get_user_manager()
    user = user_manager.get_user(current_user.user_id)
    
    if not user or user.role != UserRole.ADMIN:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )
    
    # Validate role
    valid_roles = [UserRole.ADMIN, UserRole.USER, UserRole.READONLY]
    if new_role not in valid_roles:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid role. Must be one of: {valid_roles}"
        )
    
    success = user_manager.update_user_role(username, new_role)
    
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    return {"message": f"Role updated successfully to {new_role}"}


@router.delete("/users/{username}")
async def deactivate_user(
    username: str,
    current_user=Depends(get_current_user)
):
    """
    Deactivate a user (admin only).
    """
    # Check if user is admin
    user_manager = get_user_manager()
    user = user_manager.get_user(current_user.user_id)
    
    if not user or user.role != UserRole.ADMIN:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )
    
    # Prevent self-deactivation
    if username == current_user.user_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot deactivate your own account"
        )
    
    success = user_manager.deactivate_user(username)
    
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    return {"message": "User deactivated successfully"} 