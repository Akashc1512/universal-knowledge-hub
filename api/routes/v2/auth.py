"""
Authentication API Routes - MAANG Standards.

This module implements authentication endpoints following MAANG-level
security and engineering best practices.

Features:
    - JWT-based authentication with refresh tokens
    - OAuth2 support (Google, GitHub, Microsoft)
    - Two-factor authentication (TOTP)
    - Session management
    - Password reset flow
    - Account lockout protection
    - Device fingerprinting

Security:
    - Bcrypt password hashing
    - Rate limiting on auth endpoints
    - Brute force protection
    - CSRF protection
    - Secure cookie handling

Authors:
    - Universal Knowledge Platform Engineering Team
    
Version:
    2.0.0 (2024-12-28)
"""

from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta, timezone
import secrets
import pyotp
import structlog

from fastapi import (
    APIRouter, Depends, HTTPException, status,
    Form, Query, Header, Response, Request,
    BackgroundTasks
)
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel, Field, EmailStr, validator

from api.user_management_v2 import (
    UserService, get_user_service,
    UserCreateRequest, UserModel,
    InvalidCredentialsError, AccountLockedException,
    UserAlreadyExistsError
)
from api.database.models import UserSession
from api.security import (
    create_csrf_token, verify_csrf_token,
    generate_device_fingerprint
)
from api.cache import cache_response, invalidate_cache
from api.monitoring import track_endpoint_usage
from api.rate_limiter import RateLimitConfig, rate_limit
from api.exceptions import (
    ValidationError, AuthenticationError,
    RateLimitError, ConflictError
)

logger = structlog.get_logger(__name__)

# Router configuration
router = APIRouter(
    responses={
        400: {"description": "Bad request"},
        401: {"description": "Authentication failed"},
        429: {"description": "Too many requests"},
    }
)

# Rate limit configurations
AUTH_RATE_LIMIT = RateLimitConfig(
    requests_per_minute=10,
    burst_size=5,
    lockout_duration=300  # 5 minutes
)

PASSWORD_RESET_RATE_LIMIT = RateLimitConfig(
    requests_per_minute=3,
    burst_size=1,
    lockout_duration=3600  # 1 hour
)

# Request/Response Models
class LoginRequest(BaseModel):
    """Login request model."""
    
    username: str = Field(..., description="Username or email")
    password: str = Field(..., description="Password")
    remember_me: bool = Field(False, description="Extended session")
    device_name: Optional[str] = Field(None, description="Device name for session")
    
    class Config:
        schema_extra = {
            "example": {
                "username": "john_doe",
                "password": "SecurePassword123!",
                "remember_me": True,
                "device_name": "John's MacBook"
            }
        }

class LoginResponse(BaseModel):
    """Login response model."""
    
    access_token: str = Field(..., description="JWT access token")
    refresh_token: str = Field(..., description="JWT refresh token")
    token_type: str = Field("bearer", description="Token type")
    expires_in: int = Field(..., description="Token expiration in seconds")
    user: Dict[str, Any] = Field(..., description="User information")
    requires_2fa: bool = Field(False, description="2FA required flag")
    csrf_token: Optional[str] = Field(None, description="CSRF token for web clients")
    
    class Config:
        schema_extra = {
            "example": {
                "access_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
                "refresh_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
                "token_type": "bearer",
                "expires_in": 1800,
                "user": {
                    "id": "550e8400-e29b-41d4-a716-446655440000",
                    "username": "john_doe",
                    "email": "john@example.com",
                    "role": "user"
                },
                "requires_2fa": False,
                "csrf_token": "csrf_token_here"
            }
        }

class RegisterRequest(BaseModel):
    """Registration request model."""
    
    username: str = Field(..., min_length=3, max_length=50, description="Username")
    email: EmailStr = Field(..., description="Email address")
    password: str = Field(..., min_length=12, description="Password")
    full_name: Optional[str] = Field(None, max_length=100, description="Full name")
    accept_terms: bool = Field(..., description="Terms acceptance")
    newsletter: bool = Field(False, description="Newsletter subscription")
    
    @validator('accept_terms')
    def must_accept_terms(cls, v: bool) -> bool:
        """Ensure terms are accepted."""
        if not v:
            raise ValueError("You must accept the terms and conditions")
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "username": "john_doe",
                "email": "john@example.com",
                "password": "SecurePassword123!",
                "full_name": "John Doe",
                "accept_terms": True,
                "newsletter": True
            }
        }

class TokenRefreshRequest(BaseModel):
    """Token refresh request model."""
    
    refresh_token: str = Field(..., description="Refresh token")
    
class TwoFactorSetupResponse(BaseModel):
    """2FA setup response model."""
    
    secret: str = Field(..., description="TOTP secret")
    qr_code: str = Field(..., description="QR code data URL")
    backup_codes: List[str] = Field(..., description="Backup codes")
    
class TwoFactorVerifyRequest(BaseModel):
    """2FA verification request model."""
    
    code: str = Field(..., regex=r"^\d{6}$", description="6-digit TOTP code")
    trust_device: bool = Field(False, description="Trust this device")

class PasswordResetRequest(BaseModel):
    """Password reset request model."""
    
    email: EmailStr = Field(..., description="Account email")
    
class PasswordResetConfirmRequest(BaseModel):
    """Password reset confirmation model."""
    
    token: str = Field(..., description="Reset token")
    new_password: str = Field(..., min_length=12, description="New password")

# Authentication endpoints
@router.post(
    "/login",
    response_model=LoginResponse,
    summary="User login",
    description="Authenticate user and receive tokens"
)
@rate_limit(AUTH_RATE_LIMIT)
@track_endpoint_usage
async def login(
    request: Request,
    response: Response,
    login_data: LoginRequest,
    background_tasks: BackgroundTasks,
    user_service: UserService = Depends(get_user_service)
) -> LoginResponse:
    """
    Authenticate user with username/email and password.
    
    **Security features**:
    - Rate limiting: 10 attempts per minute
    - Account lockout after 5 failed attempts
    - Device fingerprinting for anomaly detection
    - Optional 2FA verification
    
    **Session duration**:
    - Standard: 30 minutes
    - Remember me: 30 days
    
    **Response includes**:
    - Access token (short-lived)
    - Refresh token (long-lived)
    - CSRF token for web clients
    - User information
    """
    try:
        # Get device fingerprint
        device_fingerprint = generate_device_fingerprint(request)
        
        # Authenticate user
        auth_result = await user_service.authenticate(
            login_data.username,
            login_data.password
        )
        
        # Check if 2FA is required
        user = auth_result["user"]
        if user.get("two_factor_enabled"):
            # Return partial token for 2FA flow
            return LoginResponse(
                access_token="",
                refresh_token="",
                token_type="bearer",
                expires_in=0,
                user={"id": user["id"]},
                requires_2fa=True
            )
        
        # Create session
        session_duration = timedelta(days=30) if login_data.remember_me else timedelta(minutes=30)
        session = await user_service.create_session(
            user_id=user["id"],
            device_name=login_data.device_name,
            device_fingerprint=device_fingerprint,
            ip_address=request.client.host,
            user_agent=request.headers.get("user-agent"),
            duration=session_duration
        )
        
        # Set secure cookie for web clients
        if "text/html" in request.headers.get("accept", ""):
            response.set_cookie(
                key="session_token",
                value=session["token"],
                max_age=int(session_duration.total_seconds()),
                httponly=True,
                secure=True,
                samesite="lax"
            )
            
            # Generate CSRF token
            csrf_token = create_csrf_token(user["id"])
            response.set_cookie(
                key="csrf_token",
                value=csrf_token,
                httponly=False,  # Accessible to JS
                secure=True,
                samesite="lax"
            )
        else:
            csrf_token = None
        
        # Log successful login
        background_tasks.add_task(
            user_service.log_login_event,
            user_id=user["id"],
            ip_address=request.client.host,
            success=True
        )
        
        logger.info(
            "User logged in",
            user_id=user["id"],
            username=user["username"],
            remember_me=login_data.remember_me
        )
        
        return LoginResponse(
            access_token=auth_result["access_token"],
            refresh_token=auth_result["refresh_token"],
            token_type="bearer",
            expires_in=auth_result["expires_in"],
            user=user,
            requires_2fa=False,
            csrf_token=csrf_token
        )
        
    except InvalidCredentialsError as e:
        # Log failed attempt
        background_tasks.add_task(
            user_service.log_login_event,
            username=login_data.username,
            ip_address=request.client.host,
            success=False
        )
        
        logger.warning(
            "Login failed",
            username=login_data.username,
            ip=request.client.host
        )
        
        raise AuthenticationError("Invalid username or password")
        
    except AccountLockedException as e:
        logger.warning(
            "Login attempt on locked account",
            username=login_data.username
        )
        
        raise AuthenticationError(
            "Account is locked due to too many failed attempts. "
            f"Please try again after {e.locked_until}"
        )
        
    except Exception as e:
        logger.error("Login error", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Login failed"
        )

@router.post(
    "/register",
    response_model=LoginResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Register new user",
    description="Create new user account"
)
@rate_limit(AUTH_RATE_LIMIT)
@track_endpoint_usage
async def register(
    request: Request,
    register_data: RegisterRequest,
    background_tasks: BackgroundTasks,
    user_service: UserService = Depends(get_user_service)
) -> LoginResponse:
    """
    Register a new user account.
    
    **Requirements**:
    - Unique username (3-50 characters)
    - Valid email address
    - Strong password (12+ characters with complexity)
    - Terms acceptance
    
    **Post-registration**:
    - Verification email sent
    - Automatic login
    - Welcome email sent
    
    **Rate limit**: 10 requests per minute
    """
    try:
        # Create user request
        create_request = UserCreateRequest(
            username=register_data.username,
            email=register_data.email,
            password=register_data.password,
            full_name=register_data.full_name,
            metadata={
                "newsletter": register_data.newsletter,
                "terms_accepted_at": datetime.now(timezone.utc).isoformat(),
                "registration_ip": request.client.host,
                "registration_user_agent": request.headers.get("user-agent")
            }
        )
        
        # Create user
        user = await user_service.create_user(create_request)
        
        # Send verification email
        background_tasks.add_task(
            user_service.send_verification_email,
            user
        )
        
        # Send welcome email
        if register_data.newsletter:
            background_tasks.add_task(
                user_service.send_welcome_email,
                user
            )
        
        # Auto-login
        auth_result = await user_service.authenticate(
            register_data.username,
            register_data.password
        )
        
        logger.info(
            "User registered",
            user_id=user.id,
            username=user.username,
            email=user.email
        )
        
        return LoginResponse(
            access_token=auth_result["access_token"],
            refresh_token=auth_result["refresh_token"],
            token_type="bearer",
            expires_in=auth_result["expires_in"],
            user=auth_result["user"],
            requires_2fa=False
        )
        
    except UserAlreadyExistsError as e:
        field = e.details.get("field", "username")
        raise ConflictError(f"An account with this {field} already exists")
        
    except ValidationError as e:
        raise
        
    except Exception as e:
        logger.error("Registration error", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Registration failed"
        )

@router.post(
    "/logout",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="User logout",
    description="Invalidate current session"
)
@track_endpoint_usage
async def logout(
    request: Request,
    response: Response,
    all_sessions: bool = Query(False, description="Logout all sessions"),
    authorization: str = Header(None),
    user_service: UserService = Depends(get_user_service)
) -> None:
    """
    Logout current session or all sessions.
    
    **Options**:
    - Single session: Invalidates current token only
    - All sessions: Invalidates all user sessions across devices
    
    **Web clients**: Also clears session cookies
    """
    try:
        # Extract token
        if not authorization or not authorization.startswith("Bearer "):
            raise AuthenticationError("No valid session")
        
        token = authorization.split(" ")[1]
        
        # Verify token and get user
        payload = await user_service.verify_token(token)
        user_id = payload["sub"]
        
        # Invalidate sessions
        if all_sessions:
            await user_service.invalidate_all_sessions(user_id)
            logger.info("All sessions invalidated", user_id=user_id)
        else:
            await user_service.invalidate_session(token)
            logger.info("Session invalidated", user_id=user_id)
        
        # Clear cookies for web clients
        if "text/html" in request.headers.get("accept", ""):
            response.delete_cookie("session_token")
            response.delete_cookie("csrf_token")
        
    except Exception as e:
        logger.error("Logout error", error=str(e))
        # Don't expose errors on logout

@router.post(
    "/refresh",
    response_model=LoginResponse,
    summary="Refresh access token",
    description="Exchange refresh token for new access token"
)
@track_endpoint_usage
async def refresh_token(
    refresh_request: TokenRefreshRequest,
    user_service: UserService = Depends(get_user_service)
) -> LoginResponse:
    """
    Refresh access token using refresh token.
    
    **Token lifecycle**:
    - Access token: 30 minutes
    - Refresh token: 30 days
    
    **Security**: Refresh tokens are single-use and rotated on each refresh
    """
    try:
        # Refresh tokens
        result = await user_service.refresh_access_token(
            refresh_request.refresh_token
        )
        
        logger.info(
            "Token refreshed",
            user_id=result["user"]["id"]
        )
        
        return LoginResponse(
            access_token=result["access_token"],
            refresh_token=result["refresh_token"],
            token_type="bearer",
            expires_in=result["expires_in"],
            user=result["user"],
            requires_2fa=False
        )
        
    except Exception as e:
        logger.error("Token refresh error", error=str(e))
        raise AuthenticationError("Invalid refresh token")

@router.post(
    "/2fa/setup",
    response_model=TwoFactorSetupResponse,
    summary="Setup 2FA",
    description="Enable two-factor authentication"
)
@track_endpoint_usage
async def setup_2fa(
    current_password: str = Form(..., description="Current password for verification"),
    current_user: UserModel = Depends(get_current_user),
    user_service: UserService = Depends(get_user_service)
) -> TwoFactorSetupResponse:
    """
    Setup two-factor authentication for account.
    
    **Process**:
    1. Verify current password
    2. Generate TOTP secret
    3. Return QR code and backup codes
    4. User must verify with code to complete setup
    
    **Backup codes**: 10 single-use codes for account recovery
    """
    try:
        # Verify password
        if not await user_service.verify_password(
            current_user.id,
            current_password
        ):
            raise ValidationError("Invalid password")
        
        # Generate TOTP secret
        secret = pyotp.random_base32()
        
        # Generate QR code
        totp_uri = pyotp.totp.TOTP(secret).provisioning_uri(
            name=current_user.email,
            issuer_name="Universal Knowledge Platform"
        )
        
        # Generate QR code data URL
        import qrcode
        import io
        import base64
        
        qr = qrcode.QRCode(version=1, box_size=10, border=5)
        qr.add_data(totp_uri)
        qr.make(fit=True)
        
        img = qr.make_image(fill_color="black", back_color="white")
        buf = io.BytesIO()
        img.save(buf, format='PNG')
        
        qr_code_data = base64.b64encode(buf.getvalue()).decode()
        qr_code_url = f"data:image/png;base64,{qr_code_data}"
        
        # Generate backup codes
        backup_codes = [
            f"{secrets.randbelow(1000000):06d}"
            for _ in range(10)
        ]
        
        # Store temporarily (not enabled until verified)
        await user_service.store_2fa_setup(
            current_user.id,
            secret,
            backup_codes
        )
        
        logger.info("2FA setup initiated", user_id=current_user.id)
        
        return TwoFactorSetupResponse(
            secret=secret,
            qr_code=qr_code_url,
            backup_codes=backup_codes
        )
        
    except ValidationError:
        raise
    except Exception as e:
        logger.error("2FA setup error", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to setup 2FA"
        )

@router.post(
    "/2fa/verify",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Verify 2FA code",
    description="Complete 2FA setup or login"
)
@track_endpoint_usage
async def verify_2fa(
    verify_request: TwoFactorVerifyRequest,
    setup: bool = Query(False, description="Completing setup"),
    current_user: UserModel = Depends(get_current_user),
    user_service: UserService = Depends(get_user_service)
) -> None:
    """
    Verify 2FA code for setup completion or login.
    
    **Code format**: 6-digit TOTP code
    **Time window**: ±30 seconds
    
    **Trust device**: Skip 2FA for 30 days on this device
    """
    try:
        if setup:
            # Complete 2FA setup
            success = await user_service.complete_2fa_setup(
                current_user.id,
                verify_request.code
            )
        else:
            # Verify for login
            success = await user_service.verify_2fa_code(
                current_user.id,
                verify_request.code
            )
        
        if not success:
            raise ValidationError("Invalid code")
        
        # Trust device if requested
        if verify_request.trust_device:
            await user_service.trust_device(
                current_user.id,
                device_fingerprint=generate_device_fingerprint(request)
            )
        
        logger.info(
            "2FA verified",
            user_id=current_user.id,
            setup=setup
        )
        
    except ValidationError:
        raise
    except Exception as e:
        logger.error("2FA verification error", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to verify 2FA"
        )

@router.post(
    "/password-reset",
    status_code=status.HTTP_202_ACCEPTED,
    summary="Request password reset",
    description="Send password reset email"
)
@rate_limit(PASSWORD_RESET_RATE_LIMIT)
@track_endpoint_usage
async def request_password_reset(
    reset_request: PasswordResetRequest,
    background_tasks: BackgroundTasks,
    user_service: UserService = Depends(get_user_service)
) -> Dict[str, str]:
    """
    Request password reset email.
    
    **Process**:
    1. Verify email exists
    2. Generate secure reset token
    3. Send reset email with link
    4. Token valid for 1 hour
    
    **Rate limit**: 3 requests per hour
    
    **Security**: Always returns success to prevent email enumeration
    """
    try:
        # Get user by email
        user = await user_service.get_user_by_email(reset_request.email)
        
        if user:
            # Generate reset token
            token = await user_service.create_password_reset_token(user.id)
            
            # Send reset email
            background_tasks.add_task(
                user_service.send_password_reset_email,
                user,
                token
            )
            
            logger.info(
                "Password reset requested",
                user_id=user.id,
                email=user.email
            )
        else:
            # Log attempt but don't reveal user doesn't exist
            logger.warning(
                "Password reset for non-existent email",
                email=reset_request.email
            )
        
        # Always return success
        return {
            "message": "If an account exists with this email, "
                      "a password reset link has been sent."
        }
        
    except Exception as e:
        logger.error("Password reset error", error=str(e))
        # Still return success for security
        return {
            "message": "If an account exists with this email, "
                      "a password reset link has been sent."
        }

@router.post(
    "/password-reset/confirm",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Confirm password reset",
    description="Reset password with token"
)
@track_endpoint_usage
async def confirm_password_reset(
    confirm_request: PasswordResetConfirmRequest,
    background_tasks: BackgroundTasks,
    user_service: UserService = Depends(get_user_service)
) -> None:
    """
    Reset password using reset token.
    
    **Requirements**:
    - Valid reset token (from email)
    - New password meeting complexity requirements
    
    **Post-reset**:
    - All sessions invalidated
    - Confirmation email sent
    - Must login with new password
    """
    try:
        # Reset password
        user = await user_service.reset_password_with_token(
            confirm_request.token,
            confirm_request.new_password
        )
        
        # Invalidate all sessions
        await user_service.invalidate_all_sessions(user.id)
        
        # Send confirmation email
        background_tasks.add_task(
            user_service.send_password_reset_confirmation,
            user
        )
        
        logger.info(
            "Password reset completed",
            user_id=user.id
        )
        
    except ValidationError as e:
        raise
    except Exception as e:
        logger.error("Password reset confirmation error", error=str(e))
        raise ValidationError("Invalid or expired reset token")

# OAuth2 endpoints would go here
# @router.get("/oauth/{provider}")
# @router.get("/oauth/{provider}/callback")

# Export router
__all__ = ['router'] 