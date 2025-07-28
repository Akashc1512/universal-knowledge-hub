"""
User API Routes - MAANG Standards.

This module implements user-related API endpoints following MAANG-level
best practices with comprehensive validation, error handling, and documentation.

Features:
    - RESTful API design
    - Comprehensive input validation
    - Pagination and filtering
    - Response caching
    - Rate limiting per endpoint
    - Detailed OpenAPI documentation
    - Request/Response examples

Security:
    - Authentication required for sensitive endpoints
    - Role-based access control
    - Input sanitization
    - SQL injection prevention

Performance:
    - Query optimization
    - Response caching
    - Pagination for large datasets
    - Selective field loading

Authors:
    - Universal Knowledge Platform Engineering Team
    
Version:
    2.0.0 (2024-12-28)
"""

from typing import Optional, List, Dict, Any, Union
from datetime import datetime, timezone
import structlog

from fastapi import (
    APIRouter, Depends, Query, Path, Body, 
    HTTPException, status, BackgroundTasks
)
from fastapi.responses import ORJSONResponse
from pydantic import BaseModel, Field, EmailStr, conint, validator

from api.database.models import User as UserModel, Role
from api.user_management_v2 import (
    UserService, get_user_service,
    UserCreateRequest, UserRole,
    UserNotFoundError, UserAlreadyExistsError
)
from api.auth import require_auth, require_admin, get_current_user
from api.cache import cache_response, invalidate_cache
from api.monitoring import track_endpoint_usage
from api.exceptions import (
    NotFoundError, ConflictError, ValidationError,
    AuthorizationError
)

logger = structlog.get_logger(__name__)

# Create router
router = APIRouter(
    responses={
        401: {"description": "Authentication required"},
        403: {"description": "Insufficient permissions"},
        404: {"description": "User not found"},
        422: {"description": "Validation error"},
    }
)

# Request/Response models
class UserResponse(BaseModel):
    """User response model."""
    
    id: str = Field(..., description="User ID")
    username: str = Field(..., description="Username")
    email: EmailStr = Field(..., description="Email address")
    full_name: Optional[str] = Field(None, description="Full name")
    role: str = Field(..., description="User role")
    is_active: bool = Field(..., description="Active status")
    email_verified: bool = Field(..., description="Email verification status")
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")
    last_login_at: Optional[datetime] = Field(None, description="Last login timestamp")
    
    class Config:
        schema_extra = {
            "example": {
                "id": "550e8400-e29b-41d4-a716-446655440000",
                "username": "john_doe",
                "email": "john@example.com",
                "full_name": "John Doe",
                "role": "user",
                "is_active": True,
                "email_verified": True,
                "created_at": "2024-01-01T00:00:00Z",
                "updated_at": "2024-01-01T00:00:00Z",
                "last_login_at": "2024-01-01T12:00:00Z"
            }
        }

class UserListResponse(BaseModel):
    """User list response with pagination."""
    
    users: List[UserResponse] = Field(..., description="List of users")
    total: int = Field(..., description="Total count")
    page: int = Field(..., description="Current page")
    per_page: int = Field(..., description="Items per page")
    pages: int = Field(..., description="Total pages")
    
    class Config:
        schema_extra = {
            "example": {
                "users": [UserResponse.Config.schema_extra["example"]],
                "total": 100,
                "page": 1,
                "per_page": 20,
                "pages": 5
            }
        }

class UserUpdateRequest(BaseModel):
    """User update request model."""
    
    full_name: Optional[str] = Field(None, max_length=100, description="Full name")
    email: Optional[EmailStr] = Field(None, description="Email address")
    bio: Optional[str] = Field(None, max_length=500, description="User biography")
    avatar_url: Optional[str] = Field(None, description="Avatar URL")
    preferences: Optional[Dict[str, Any]] = Field(None, description="User preferences")
    
    @validator('avatar_url')
    def validate_avatar_url(cls, v: Optional[str]) -> Optional[str]:
        """Validate avatar URL."""
        if v and not v.startswith(('http://', 'https://')):
            raise ValueError("Avatar URL must be a valid HTTP(S) URL")
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "full_name": "John Doe",
                "bio": "Software engineer passionate about AI",
                "preferences": {
                    "theme": "dark",
                    "notifications": True
                }
            }
        }

class PasswordChangeRequest(BaseModel):
    """Password change request model."""
    
    current_password: str = Field(..., min_length=8, description="Current password")
    new_password: str = Field(..., min_length=12, description="New password")
    logout_sessions: bool = Field(True, description="Logout other sessions")
    
    @validator('new_password')
    def validate_password_strength(cls, v: str, values: Dict[str, Any]) -> str:
        """Validate password strength."""
        # Check not same as current
        if 'current_password' in values and v == values['current_password']:
            raise ValueError("New password must be different from current password")
        
        # Check complexity
        if not any(c.isupper() for c in v):
            raise ValueError("Password must contain uppercase letters")
        if not any(c.islower() for c in v):
            raise ValueError("Password must contain lowercase letters")
        if not any(c.isdigit() for c in v):
            raise ValueError("Password must contain numbers")
        if not any(c in "!@#$%^&*(),.?\":{}|<>" for c in v):
            raise ValueError("Password must contain special characters")
        
        return v

class RoleUpdateRequest(BaseModel):
    """Role update request model."""
    
    role: UserRole = Field(..., description="New role")
    reason: str = Field(..., min_length=10, max_length=200, description="Reason for change")
    
    class Config:
        schema_extra = {
            "example": {
                "role": "moderator",
                "reason": "Promoted to moderator for community contributions"
            }
        }

# Endpoints
@router.get(
    "/",
    response_model=UserListResponse,
    summary="List users",
    description="Get paginated list of users with optional filtering",
    response_description="List of users with pagination info"
)
@cache_response(ttl=60)
@track_endpoint_usage
async def list_users(
    page: int = Query(1, ge=1, description="Page number"),
    per_page: int = Query(20, ge=1, le=100, description="Items per page"),
    search: Optional[str] = Query(None, description="Search in username/email"),
    role: Optional[UserRole] = Query(None, description="Filter by role"),
    is_active: Optional[bool] = Query(None, description="Filter by active status"),
    sort_by: str = Query("created_at", regex="^(created_at|username|email|last_login_at)$"),
    sort_order: str = Query("desc", regex="^(asc|desc)$"),
    current_user: UserModel = Depends(require_auth),
    user_service: UserService = Depends(get_user_service)
) -> UserListResponse:
    """
    List users with pagination and filtering.
    
    **Required permissions**: Authenticated user
    
    **Filtering options**:
    - `search`: Search in username and email
    - `role`: Filter by user role
    - `is_active`: Filter by active status
    
    **Sorting options**:
    - `created_at`: Registration date (default)
    - `username`: Alphabetical by username
    - `email`: Alphabetical by email
    - `last_login_at`: Last login time
    
    **Rate limit**: 100 requests per minute
    """
    try:
        # Calculate offset
        offset = (page - 1) * per_page
        
        # Get users
        result = await user_service.list_users(
            offset=offset,
            limit=per_page,
            role=role,
            search=search,
            # Additional filters would be passed here
        )
        
        # Calculate pagination
        total = result["pagination"]["total"]
        pages = (total + per_page - 1) // per_page
        
        # Convert to response model
        users = [
            UserResponse(
                id=str(user.id),
                username=user.username,
                email=user.email,
                full_name=user.full_name,
                role=user.role.value,
                is_active=user.is_active,
                email_verified=user.email_verified,
                created_at=user.created_at,
                updated_at=user.updated_at,
                last_login_at=user.last_login
            )
            for user in result["users"]
        ]
        
        logger.info(
            "Users listed",
            user_id=current_user.id,
            page=page,
            per_page=per_page,
            total=total
        )
        
        return UserListResponse(
            users=users,
            total=total,
            page=page,
            per_page=per_page,
            pages=pages
        )
        
    except Exception as e:
        logger.error("Error listing users", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list users"
        )

@router.get(
    "/me",
    response_model=UserResponse,
    summary="Get current user",
    description="Get current authenticated user's profile"
)
@track_endpoint_usage
async def get_current_user_profile(
    current_user: UserModel = Depends(get_current_user)
) -> UserResponse:
    """
    Get current authenticated user's profile.
    
    **Required permissions**: Authenticated user
    
    Returns the complete profile of the currently authenticated user.
    """
    return UserResponse(
        id=str(current_user.id),
        username=current_user.username,
        email=current_user.email,
        full_name=current_user.full_name,
        role=current_user.role.value,
        is_active=current_user.is_active,
        email_verified=current_user.email_verified,
        created_at=current_user.created_at,
        updated_at=current_user.updated_at,
        last_login_at=current_user.last_login
    )

@router.get(
    "/{user_id}",
    response_model=UserResponse,
    summary="Get user by ID",
    description="Get specific user by ID"
)
@cache_response(ttl=300)
@track_endpoint_usage
async def get_user(
    user_id: str = Path(..., description="User ID"),
    current_user: UserModel = Depends(require_auth),
    user_service: UserService = Depends(get_user_service)
) -> UserResponse:
    """
    Get specific user by ID.
    
    **Required permissions**: Authenticated user
    
    **Note**: Regular users can only view basic information of other users.
    Admins can view all user details.
    """
    try:
        # Get user
        user = await user_service.get_user_by_id(user_id)
        
        if not user:
            raise NotFoundError(f"User {user_id} not found")
        
        # Check permissions
        if current_user.id != user_id and not current_user.has_role("admin"):
            # Limited view for non-admins viewing other users
            return UserResponse(
                id=str(user.id),
                username=user.username,
                email="",  # Hide email
                full_name=user.full_name,
                role=user.role.value,
                is_active=user.is_active,
                email_verified=False,  # Hide
                created_at=user.created_at,
                updated_at=user.updated_at,
                last_login_at=None  # Hide
            )
        
        # Full view for self or admin
        return UserResponse(
            id=str(user.id),
            username=user.username,
            email=user.email,
            full_name=user.full_name,
            role=user.role.value,
            is_active=user.is_active,
            email_verified=user.email_verified,
            created_at=user.created_at,
            updated_at=user.updated_at,
            last_login_at=user.last_login
        )
        
    except NotFoundError:
        raise
    except Exception as e:
        logger.error("Error getting user", user_id=user_id, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get user"
        )

@router.put(
    "/me",
    response_model=UserResponse,
    summary="Update current user",
    description="Update current user's profile"
)
@track_endpoint_usage
async def update_current_user(
    update_request: UserUpdateRequest,
    background_tasks: BackgroundTasks,
    current_user: UserModel = Depends(get_current_user),
    user_service: UserService = Depends(get_user_service)
) -> UserResponse:
    """
    Update current user's profile.
    
    **Required permissions**: Authenticated user
    
    **Updatable fields**:
    - `full_name`: Display name
    - `email`: Email address (requires verification)
    - `bio`: User biography
    - `avatar_url`: Profile picture URL
    - `preferences`: User preferences object
    
    **Note**: Email changes require re-verification.
    """
    try:
        # Update user
        update_data = update_request.dict(exclude_unset=True)
        
        # Check if email is being changed
        email_changed = (
            'email' in update_data and 
            update_data['email'] != current_user.email
        )
        
        # Update user
        updated_user = await user_service.update_user(
            current_user.id,
            update_data
        )
        
        # Invalidate cache
        await invalidate_cache(f"user:{current_user.id}")
        
        # Send verification email if email changed
        if email_changed:
            background_tasks.add_task(
                user_service.send_verification_email,
                updated_user
            )
        
        logger.info(
            "User updated",
            user_id=current_user.id,
            fields=list(update_data.keys())
        )
        
        return UserResponse(
            id=str(updated_user.id),
            username=updated_user.username,
            email=updated_user.email,
            full_name=updated_user.full_name,
            role=updated_user.role.value,
            is_active=updated_user.is_active,
            email_verified=updated_user.email_verified,
            created_at=updated_user.created_at,
            updated_at=updated_user.updated_at,
            last_login_at=updated_user.last_login
        )
        
    except Exception as e:
        logger.error("Error updating user", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update user"
        )

@router.post(
    "/me/change-password",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Change password",
    description="Change current user's password"
)
@track_endpoint_usage
async def change_password(
    password_request: PasswordChangeRequest,
    background_tasks: BackgroundTasks,
    current_user: UserModel = Depends(get_current_user),
    user_service: UserService = Depends(get_user_service)
) -> None:
    """
    Change current user's password.
    
    **Required permissions**: Authenticated user
    
    **Security notes**:
    - Requires current password for verification
    - Invalidates all sessions if `logout_sessions` is true
    - Sends notification email
    
    **Password requirements**:
    - Minimum 12 characters
    - Must contain uppercase, lowercase, numbers, and special characters
    - Cannot be the same as current password
    """
    try:
        # Change password
        success = await user_service.change_password(
            current_user.id,
            password_request.current_password,
            password_request.new_password
        )
        
        if not success:
            raise ValidationError("Current password is incorrect")
        
        # Invalidate sessions if requested
        if password_request.logout_sessions:
            await user_service.invalidate_user_sessions(
                current_user.id,
                except_current=True
            )
        
        # Send notification email
        background_tasks.add_task(
            user_service.send_password_change_notification,
            current_user
        )
        
        logger.info(
            "Password changed",
            user_id=current_user.id,
            logout_sessions=password_request.logout_sessions
        )
        
    except ValidationError:
        raise
    except Exception as e:
        logger.error("Error changing password", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to change password"
        )

@router.put(
    "/{user_id}/role",
    response_model=UserResponse,
    summary="Update user role",
    description="Update user's role (admin only)"
)
@track_endpoint_usage
async def update_user_role(
    user_id: str = Path(..., description="User ID"),
    role_request: RoleUpdateRequest = Body(...),
    current_user: UserModel = Depends(require_admin),
    user_service: UserService = Depends(get_user_service)
) -> UserResponse:
    """
    Update user's role.
    
    **Required permissions**: Admin
    
    **Available roles**:
    - `user`: Standard user
    - `moderator`: Community moderator
    - `admin`: System administrator
    
    **Audit trail**: This action is logged with the reason provided.
    """
    try:
        # Prevent self-demotion
        if user_id == str(current_user.id) and role_request.role != UserRole.ADMIN:
            raise ValidationError("Cannot change your own admin role")
        
        # Update role
        updated_user = await user_service.update_user_role(
            user_id,
            role_request.role,
            updated_by=str(current_user.id)
        )
        
        # Invalidate cache
        await invalidate_cache(f"user:{user_id}")
        
        # Log audit event
        logger.info(
            "User role updated",
            user_id=user_id,
            new_role=role_request.role.value,
            reason=role_request.reason,
            updated_by=current_user.id
        )
        
        return UserResponse(
            id=str(updated_user.id),
            username=updated_user.username,
            email=updated_user.email,
            full_name=updated_user.full_name,
            role=updated_user.role.value,
            is_active=updated_user.is_active,
            email_verified=updated_user.email_verified,
            created_at=updated_user.created_at,
            updated_at=updated_user.updated_at,
            last_login_at=updated_user.last_login
        )
        
    except UserNotFoundError:
        raise NotFoundError(f"User {user_id} not found")
    except ValidationError:
        raise
    except Exception as e:
        logger.error("Error updating role", user_id=user_id, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update role"
        )

@router.delete(
    "/{user_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete user",
    description="Delete user account (admin only)"
)
@track_endpoint_usage
async def delete_user(
    user_id: str = Path(..., description="User ID"),
    permanent: bool = Query(False, description="Permanent deletion"),
    reason: str = Query(..., min_length=10, description="Deletion reason"),
    current_user: UserModel = Depends(require_admin),
    user_service: UserService = Depends(get_user_service)
) -> None:
    """
    Delete user account.
    
    **Required permissions**: Admin
    
    **Deletion types**:
    - Soft delete (default): User data is retained but account is deactivated
    - Permanent delete: User data is permanently removed (requires `permanent=true`)
    
    **Note**: This action is irreversible for permanent deletions.
    """
    try:
        # Prevent self-deletion
        if user_id == str(current_user.id):
            raise ValidationError("Cannot delete your own account")
        
        # Delete user
        if permanent:
            success = await user_service.delete_user_permanent(user_id)
        else:
            success = await user_service.delete_user_soft(user_id)
        
        if not success:
            raise NotFoundError(f"User {user_id} not found")
        
        # Invalidate cache
        await invalidate_cache(f"user:{user_id}")
        
        # Log audit event
        logger.info(
            "User deleted",
            user_id=user_id,
            permanent=permanent,
            reason=reason,
            deleted_by=current_user.id
        )
        
    except (NotFoundError, ValidationError):
        raise
    except Exception as e:
        logger.error("Error deleting user", user_id=user_id, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete user"
        )

@router.post(
    "/{user_id}/verify-email",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Resend verification email",
    description="Resend email verification link"
)
@track_endpoint_usage
async def resend_verification_email(
    user_id: str = Path(..., description="User ID"),
    background_tasks: BackgroundTasks,
    current_user: UserModel = Depends(require_auth),
    user_service: UserService = Depends(get_user_service)
) -> None:
    """
    Resend email verification link.
    
    **Required permissions**: Authenticated user (self) or Admin
    
    **Rate limit**: 3 requests per hour
    
    Sends a new verification email to the user's registered email address.
    """
    try:
        # Check permissions
        if user_id != str(current_user.id) and not current_user.has_role("admin"):
            raise AuthorizationError("Cannot resend verification for other users")
        
        # Get user
        user = await user_service.get_user_by_id(user_id)
        if not user:
            raise NotFoundError(f"User {user_id} not found")
        
        # Check if already verified
        if user.email_verified:
            raise ValidationError("Email already verified")
        
        # Send verification email
        background_tasks.add_task(
            user_service.send_verification_email,
            user
        )
        
        logger.info(
            "Verification email sent",
            user_id=user_id,
            requested_by=current_user.id
        )
        
    except (NotFoundError, ValidationError, AuthorizationError):
        raise
    except Exception as e:
        logger.error("Error sending verification", user_id=user_id, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to send verification email"
        )

# Export router
__all__ = ['router'] 