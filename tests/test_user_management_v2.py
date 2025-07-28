"""
Comprehensive Test Suite for User Management Module.

This test suite follows MAANG-level testing standards with:
- Unit tests for all public methods
- Integration tests for component interaction
- Property-based testing for edge cases
- Performance benchmarks
- Security vulnerability tests
- Mock and fixture best practices

Test Categories:
    - Unit Tests: Test individual components in isolation
    - Integration Tests: Test component interactions
    - Performance Tests: Benchmark critical operations
    - Security Tests: Test against common vulnerabilities
    - Property Tests: Test with generated data

Coverage Goal: >95% with mutation testing

Authors:
    - Universal Knowledge Platform QA Team
    
Version:
    2.0.0 (2024-12-28)
"""

import asyncio
import secrets
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from unittest.mock import Mock, AsyncMock, patch, MagicMock

import pytest
import pytest_asyncio
from freezegun import freeze_time
from hypothesis import given, strategies as st, settings, assume
from faker import Faker

# Import module under test
from api.user_management_v2 import (
    # Exceptions
    UserManagementError,
    UserNotFoundError,
    UserAlreadyExistsError,
    InvalidCredentialsError,
    AccountLockedException,
    TokenError,
    
    # Enums and Models
    UserRole,
    UserStatus,
    AuthenticationMethod,
    UserCreateRequest,
    UserModel,
    UserSession,
    PasswordStrength,
    
    # Service and Repository
    UserService,
    InMemoryUserRepository,
    UserRepositoryProtocol,
    
    # Constants
    MAX_LOGIN_ATTEMPTS,
    LOCKOUT_DURATION_MINUTES,
    PASSWORD_MIN_LENGTH
)

# Test fixtures
fake = Faker()
pytestmark = pytest.mark.asyncio

class TestConstants:
    """Test constants for consistency across tests."""
    
    VALID_PASSWORD = "SecurePass123!@#"
    WEAK_PASSWORD = "weak"
    VALID_USERNAME = "test_user_123"
    VALID_EMAIL = "test@example.com"
    ADMIN_USER_ID = "user_admin_test"
    USER_ID = "user_regular_test"

@pytest.fixture
def mock_repository() -> AsyncMock:
    """Create mock repository for unit testing."""
    repository = AsyncMock(spec=UserRepositoryProtocol)
    repository.get_by_id.return_value = None
    repository.get_by_username.return_value = None
    repository.get_by_email.return_value = None
    repository.create.side_effect = lambda user: user
    repository.update.side_effect = lambda user: user
    repository.delete.return_value = True
    repository.list_users.return_value = []
    return repository

@pytest.fixture
def in_memory_repository() -> InMemoryUserRepository:
    """Create in-memory repository for integration testing."""
    return InMemoryUserRepository()

@pytest.fixture
async def user_service(mock_repository: AsyncMock) -> UserService:
    """Create UserService with mock repository."""
    return UserService(repository=mock_repository)

@pytest.fixture
async def integration_user_service(
    in_memory_repository: InMemoryUserRepository
) -> UserService:
    """Create UserService with real repository for integration tests."""
    return UserService(repository=in_memory_repository)

@pytest.fixture
def valid_user_request() -> UserCreateRequest:
    """Create valid user creation request."""
    return UserCreateRequest(
        username=TestConstants.VALID_USERNAME,
        email=TestConstants.VALID_EMAIL,
        password=TestConstants.VALID_PASSWORD,
        full_name="Test User",
        role=UserRole.USER
    )

@pytest.fixture
def admin_user() -> UserModel:
    """Create admin user model."""
    return UserModel(
        id=TestConstants.ADMIN_USER_ID,
        username="admin",
        email="admin@example.com",
        hashed_password="$2b$12$hashed_password_here",
        role=UserRole.ADMIN,
        status=UserStatus.ACTIVE,
        full_name="Admin User",
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
        email_verified=True
    )

@pytest.fixture
def regular_user() -> UserModel:
    """Create regular user model."""
    return UserModel(
        id=TestConstants.USER_ID,
        username=TestConstants.VALID_USERNAME,
        email=TestConstants.VALID_EMAIL,
        hashed_password="$2b$12$hashed_password_here",
        role=UserRole.USER,
        status=UserStatus.ACTIVE,
        full_name="Regular User",
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
        email_verified=True
    )

# Unit Tests
class TestPasswordStrength:
    """Test password strength analyzer."""
    
    @pytest.mark.parametrize("password,expected_score", [
        ("", 0),
        ("password", 1),
        ("Password1", 3),
        ("Password123!", 5),
        ("P@ssw0rd!123#Complex", 5),
    ])
    def test_password_strength_scoring(
        self,
        password: str,
        expected_score: int
    ) -> None:
        """Test password strength scoring algorithm."""
        strength = PasswordStrength.analyze(password)
        assert strength.score == expected_score
    
    def test_password_feedback(self) -> None:
        """Test password strength feedback messages."""
        strength = PasswordStrength.analyze("weak")
        assert len(strength.feedback) > 0
        assert any("12 characters" in f for f in strength.feedback)
        assert any("uppercase" in f.lower() for f in strength.feedback)
    
    @given(st.text(min_size=0, max_size=200))
    def test_password_strength_properties(self, password: str) -> None:
        """Property test: password strength always returns valid score."""
        strength = PasswordStrength.analyze(password)
        assert 0 <= strength.score <= 5
        assert isinstance(strength.feedback, list)
        assert isinstance(strength.crack_time_display, str)

class TestUserCreateRequest:
    """Test user creation request validation."""
    
    def test_valid_request(self) -> None:
        """Test creating valid user request."""
        request = UserCreateRequest(
            username="valid_user",
            email="valid@example.com",
            password=TestConstants.VALID_PASSWORD
        )
        assert request.username == "valid_user"
        assert request.role == UserRole.USER  # Default
    
    @pytest.mark.parametrize("username", [
        "ab",  # Too short
        "a" * 51,  # Too long
        "user@name",  # Invalid character
        "user name",  # Space
        "admin",  # Reserved
        "root",  # Reserved
        "system",  # Reserved
    ])
    def test_invalid_username(self, username: str) -> None:
        """Test username validation rules."""
        with pytest.raises(ValueError):
            UserCreateRequest(
                username=username,
                email="test@example.com",
                password=TestConstants.VALID_PASSWORD
            )
    
    @pytest.mark.parametrize("email", [
        "invalid",  # Not an email
        "@example.com",  # Missing local part
        "user@",  # Missing domain
        "user@tempmail.com",  # Disposable domain
    ])
    def test_invalid_email(self, email: str) -> None:
        """Test email validation rules."""
        with pytest.raises(ValueError):
            UserCreateRequest(
                username="testuser",
                email=email,
                password=TestConstants.VALID_PASSWORD
            )
    
    @pytest.mark.parametrize("password", [
        "short",  # Too short
        "a" * 129,  # Too long
        "password123",  # No special chars or uppercase
        "PASSWORD123!",  # No lowercase
        "Password!",  # No numbers
        "password123!",  # Common password
    ])
    def test_invalid_password(self, password: str) -> None:
        """Test password validation rules."""
        with pytest.raises(ValueError):
            UserCreateRequest(
                username="testuser",
                email="test@example.com",
                password=password
            )
    
    def test_username_normalization(self) -> None:
        """Test username is normalized to lowercase."""
        request = UserCreateRequest(
            username="TestUser",
            email="test@example.com",
            password=TestConstants.VALID_PASSWORD
        )
        assert request.username == "testuser"
    
    def test_email_normalization(self) -> None:
        """Test email is normalized to lowercase."""
        request = UserCreateRequest(
            username="testuser",
            email="Test@Example.COM",
            password=TestConstants.VALID_PASSWORD
        )
        assert request.email == "test@example.com"

class TestUserModel:
    """Test user model functionality."""
    
    def test_is_locked_when_locked_until_future(self) -> None:
        """Test account lock status when locked until future."""
        user = UserModel(
            id="test_id",
            username="testuser",
            email="test@example.com",
            hashed_password="hash",
            role=UserRole.USER,
            status=UserStatus.ACTIVE,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            locked_until=datetime.utcnow() + timedelta(hours=1)
        )
        assert user.is_locked is True
    
    def test_is_locked_when_lock_expired(self) -> None:
        """Test account lock status when lock has expired."""
        user = UserModel(
            id="test_id",
            username="testuser",
            email="test@example.com",
            hashed_password="hash",
            role=UserRole.USER,
            status=UserStatus.ACTIVE,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            locked_until=datetime.utcnow() - timedelta(hours=1)
        )
        assert user.is_locked is False
    
    @pytest.mark.parametrize("status,email_verified,locked,expected", [
        (UserStatus.ACTIVE, True, False, True),
        (UserStatus.ACTIVE, False, False, False),  # Email not verified
        (UserStatus.INACTIVE, True, False, False),
        (UserStatus.SUSPENDED, True, False, False),
        (UserStatus.ACTIVE, True, True, False),  # Locked
    ])
    def test_is_active(
        self,
        status: UserStatus,
        email_verified: bool,
        locked: bool,
        expected: bool
    ) -> None:
        """Test is_active property with various conditions."""
        user = UserModel(
            id="test_id",
            username="testuser",
            email="test@example.com",
            hashed_password="hash",
            role=UserRole.USER,
            status=status,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            email_verified=email_verified,
            locked_until=datetime.utcnow() + timedelta(hours=1) if locked else None
        )
        assert user.is_active == expected
    
    def test_can_login(self) -> None:
        """Test can_login method."""
        user = UserModel(
            id="test_id",
            username="testuser",
            email="test@example.com",
            hashed_password="hash",
            role=UserRole.USER,
            status=UserStatus.ACTIVE,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            email_verified=True
        )
        assert user.can_login() is True
        
        user.status = UserStatus.SUSPENDED
        assert user.can_login() is False

class TestUserRole:
    """Test user role functionality."""
    
    def test_role_from_string(self) -> None:
        """Test creating role from string."""
        assert UserRole.from_string("admin") == UserRole.ADMIN
        assert UserRole.from_string("ADMIN") == UserRole.ADMIN
        assert UserRole.from_string("user") == UserRole.USER
    
    def test_role_from_invalid_string(self) -> None:
        """Test error on invalid role string."""
        with pytest.raises(ValueError, match="Invalid role"):
            UserRole.from_string("invalid_role")
    
    @pytest.mark.parametrize("user_role,required_role,expected", [
        (UserRole.ADMIN, UserRole.USER, True),
        (UserRole.ADMIN, UserRole.ADMIN, True),
        (UserRole.USER, UserRole.ADMIN, False),
        (UserRole.USER, UserRole.USER, True),
        (UserRole.READONLY, UserRole.USER, False),
        (UserRole.MODERATOR, UserRole.USER, True),
        (UserRole.MODERATOR, UserRole.ADMIN, False),
    ])
    def test_has_permission(
        self,
        user_role: UserRole,
        required_role: UserRole,
        expected: bool
    ) -> None:
        """Test role permission hierarchy."""
        assert user_role.has_permission(required_role) == expected

class TestInMemoryUserRepository:
    """Test in-memory repository implementation."""
    
    @pytest.fixture
    def repository(self) -> InMemoryUserRepository:
        """Create repository instance."""
        return InMemoryUserRepository()
    
    async def test_create_user(
        self,
        repository: InMemoryUserRepository,
        regular_user: UserModel
    ) -> None:
        """Test creating a user."""
        created = await repository.create(regular_user)
        assert created.id == regular_user.id
        
        # Verify user can be retrieved
        retrieved = await repository.get_by_id(regular_user.id)
        assert retrieved == regular_user
    
    async def test_create_duplicate_user(
        self,
        repository: InMemoryUserRepository,
        regular_user: UserModel
    ) -> None:
        """Test error on creating duplicate user."""
        await repository.create(regular_user)
        
        with pytest.raises(UserAlreadyExistsError):
            await repository.create(regular_user)
    
    async def test_get_by_username(
        self,
        repository: InMemoryUserRepository,
        regular_user: UserModel
    ) -> None:
        """Test retrieving user by username."""
        await repository.create(regular_user)
        
        # Test case-insensitive lookup
        retrieved = await repository.get_by_username(regular_user.username.upper())
        assert retrieved == regular_user
    
    async def test_get_by_email(
        self,
        repository: InMemoryUserRepository,
        regular_user: UserModel
    ) -> None:
        """Test retrieving user by email."""
        await repository.create(regular_user)
        
        # Test case-insensitive lookup
        retrieved = await repository.get_by_email(regular_user.email.upper())
        assert retrieved == regular_user
    
    async def test_update_user(
        self,
        repository: InMemoryUserRepository,
        regular_user: UserModel
    ) -> None:
        """Test updating user."""
        await repository.create(regular_user)
        
        # Update user
        regular_user.full_name = "Updated Name"
        updated = await repository.update(regular_user)
        
        assert updated.full_name == "Updated Name"
        
        # Verify persistence
        retrieved = await repository.get_by_id(regular_user.id)
        assert retrieved.full_name == "Updated Name"
    
    async def test_update_nonexistent_user(
        self,
        repository: InMemoryUserRepository,
        regular_user: UserModel
    ) -> None:
        """Test error on updating nonexistent user."""
        with pytest.raises(UserNotFoundError):
            await repository.update(regular_user)
    
    async def test_delete_user(
        self,
        repository: InMemoryUserRepository,
        regular_user: UserModel
    ) -> None:
        """Test deleting user."""
        await repository.create(regular_user)
        
        # Delete user
        result = await repository.delete(regular_user.id)
        assert result is True
        
        # Verify user is gone
        retrieved = await repository.get_by_id(regular_user.id)
        assert retrieved is None
    
    async def test_delete_nonexistent_user(
        self,
        repository: InMemoryUserRepository
    ) -> None:
        """Test deleting nonexistent user returns False."""
        result = await repository.delete("nonexistent_id")
        assert result is False
    
    async def test_list_users_with_filters(
        self,
        repository: InMemoryUserRepository,
        regular_user: UserModel,
        admin_user: UserModel
    ) -> None:
        """Test listing users with filters."""
        await repository.create(regular_user)
        await repository.create(admin_user)
        
        # List all users
        all_users = await repository.list_users()
        assert len(all_users) == 2
        
        # Filter by role
        admins = await repository.list_users(filters={"role": UserRole.ADMIN})
        assert len(admins) == 1
        assert admins[0].role == UserRole.ADMIN
        
        # Test pagination
        page1 = await repository.list_users(offset=0, limit=1)
        assert len(page1) == 1
        
        page2 = await repository.list_users(offset=1, limit=1)
        assert len(page2) == 1
        assert page1[0].id != page2[0].id

class TestUserService:
    """Test UserService business logic."""
    
    async def test_create_user_success(
        self,
        user_service: UserService,
        mock_repository: AsyncMock,
        valid_user_request: UserCreateRequest
    ) -> None:
        """Test successful user creation."""
        # Setup mock
        mock_repository.get_by_username.return_value = None
        mock_repository.get_by_email.return_value = None
        
        # Create user
        user = await user_service.create_user(valid_user_request)
        
        # Verify
        assert user.username == valid_user_request.username.lower()
        assert user.email == valid_user_request.email.lower()
        assert user.role == valid_user_request.role
        assert user.status == UserStatus.PENDING_VERIFICATION
        
        # Verify repository calls
        mock_repository.create.assert_called_once()
        created_user = mock_repository.create.call_args[0][0]
        assert created_user.username == valid_user_request.username.lower()
    
    async def test_create_user_duplicate_username(
        self,
        user_service: UserService,
        mock_repository: AsyncMock,
        valid_user_request: UserCreateRequest,
        regular_user: UserModel
    ) -> None:
        """Test error on duplicate username."""
        # Setup mock
        mock_repository.get_by_username.return_value = regular_user
        
        # Attempt to create duplicate
        with pytest.raises(UserAlreadyExistsError) as exc_info:
            await user_service.create_user(valid_user_request)
        
        assert "username" in str(exc_info.value)
    
    async def test_create_user_duplicate_email(
        self,
        user_service: UserService,
        mock_repository: AsyncMock,
        valid_user_request: UserCreateRequest,
        regular_user: UserModel
    ) -> None:
        """Test error on duplicate email."""
        # Setup mock
        mock_repository.get_by_username.return_value = None
        mock_repository.get_by_email.return_value = regular_user
        
        # Attempt to create duplicate
        with pytest.raises(UserAlreadyExistsError) as exc_info:
            await user_service.create_user(valid_user_request)
        
        assert "email" in str(exc_info.value)
    
    async def test_authenticate_success(
        self,
        user_service: UserService,
        mock_repository: AsyncMock,
        regular_user: UserModel
    ) -> None:
        """Test successful authentication."""
        # Setup mock
        mock_repository.get_by_username.return_value = regular_user
        
        # Mock password verification
        with patch.object(user_service, '_verify_password', return_value=True):
            result = await user_service.authenticate(
                regular_user.username,
                TestConstants.VALID_PASSWORD
            )
        
        # Verify response
        assert "access_token" in result
        assert "refresh_token" in result
        assert result["token_type"] == "bearer"
        assert result["user"]["id"] == regular_user.id
        
        # Verify user was updated
        mock_repository.update.assert_called_once()
        updated_user = mock_repository.update.call_args[0][0]
        assert updated_user.failed_login_attempts == 0
        assert updated_user.last_login is not None
    
    async def test_authenticate_invalid_username(
        self,
        user_service: UserService,
        mock_repository: AsyncMock
    ) -> None:
        """Test authentication with invalid username."""
        # Setup mock
        mock_repository.get_by_username.return_value = None
        mock_repository.get_by_email.return_value = None
        
        # Attempt authentication
        with pytest.raises(InvalidCredentialsError):
            await user_service.authenticate("invalid_user", "password")
    
    async def test_authenticate_wrong_password(
        self,
        user_service: UserService,
        mock_repository: AsyncMock,
        regular_user: UserModel
    ) -> None:
        """Test authentication with wrong password."""
        # Setup mock
        mock_repository.get_by_username.return_value = regular_user
        
        # Mock password verification to fail
        with patch.object(user_service, '_verify_password', return_value=False):
            with pytest.raises(InvalidCredentialsError):
                await user_service.authenticate(
                    regular_user.username,
                    "wrong_password"
                )
        
        # Verify failed attempts incremented
        mock_repository.update.assert_called_once()
        updated_user = mock_repository.update.call_args[0][0]
        assert updated_user.failed_login_attempts == 1
    
    async def test_authenticate_account_lockout(
        self,
        user_service: UserService,
        mock_repository: AsyncMock,
        regular_user: UserModel
    ) -> None:
        """Test account lockout after max failed attempts."""
        # Setup user with max failed attempts - 1
        regular_user.failed_login_attempts = MAX_LOGIN_ATTEMPTS - 1
        mock_repository.get_by_username.return_value = regular_user
        
        # Mock password verification to fail
        with patch.object(user_service, '_verify_password', return_value=False):
            with pytest.raises(InvalidCredentialsError):
                await user_service.authenticate(
                    regular_user.username,
                    "wrong_password"
                )
        
        # Verify account is locked
        mock_repository.update.assert_called_once()
        updated_user = mock_repository.update.call_args[0][0]
        assert updated_user.failed_login_attempts == MAX_LOGIN_ATTEMPTS
        assert updated_user.locked_until is not None
        assert updated_user.status == UserStatus.SUSPENDED
    
    async def test_authenticate_locked_account(
        self,
        user_service: UserService,
        mock_repository: AsyncMock,
        regular_user: UserModel
    ) -> None:
        """Test authentication with locked account."""
        # Setup locked user
        regular_user.locked_until = datetime.utcnow() + timedelta(hours=1)
        mock_repository.get_by_username.return_value = regular_user
        
        # Attempt authentication
        with pytest.raises(AccountLockedException) as exc_info:
            await user_service.authenticate(
                regular_user.username,
                TestConstants.VALID_PASSWORD
            )
        
        assert regular_user.username in str(exc_info.value)
    
    async def test_verify_token_valid(
        self,
        user_service: UserService,
        regular_user: UserModel
    ) -> None:
        """Test verifying valid token."""
        # Generate token
        token = user_service._generate_access_token(regular_user)
        
        # Verify token
        payload = await user_service.verify_token(token)
        
        assert payload["sub"] == regular_user.id
        assert payload["username"] == regular_user.username
        assert payload["role"] == regular_user.role.value
        assert payload["type"] == "access"
    
    async def test_verify_token_expired(
        self,
        user_service: UserService,
        regular_user: UserModel
    ) -> None:
        """Test verifying expired token."""
        # Generate token with past expiration
        with freeze_time(datetime.utcnow() - timedelta(hours=2)):
            token = user_service._generate_access_token(regular_user)
        
        # Verify token fails
        with pytest.raises(TokenError, match="expired"):
            await user_service.verify_token(token)
    
    async def test_verify_token_invalid(
        self,
        user_service: UserService
    ) -> None:
        """Test verifying invalid token."""
        with pytest.raises(TokenError, match="Invalid token"):
            await user_service.verify_token("invalid.token.here")
    
    async def test_refresh_access_token(
        self,
        user_service: UserService,
        mock_repository: AsyncMock,
        regular_user: UserModel
    ) -> None:
        """Test refreshing access token."""
        # Setup mock
        mock_repository.get_by_id.return_value = regular_user
        
        # Generate refresh token
        refresh_token = user_service._generate_refresh_token(regular_user)
        
        # Refresh access token
        result = await user_service.refresh_access_token(refresh_token)
        
        assert "access_token" in result
        assert result["token_type"] == "bearer"
        assert result["expires_in"] > 0
    
    async def test_refresh_with_access_token_fails(
        self,
        user_service: UserService,
        regular_user: UserModel
    ) -> None:
        """Test refresh fails with access token."""
        # Generate access token (not refresh)
        access_token = user_service._generate_access_token(regular_user)
        
        with pytest.raises(TokenError, match="Not a refresh token"):
            await user_service.refresh_access_token(access_token)
    
    async def test_change_password_success(
        self,
        user_service: UserService,
        mock_repository: AsyncMock,
        regular_user: UserModel
    ) -> None:
        """Test successful password change."""
        # Setup mock
        mock_repository.get_by_id.return_value = regular_user
        
        # Mock password verification
        with patch.object(user_service, '_verify_password', return_value=True):
            with patch.object(user_service, '_invalidate_user_sessions'):
                result = await user_service.change_password(
                    regular_user.id,
                    "current_password",
                    TestConstants.VALID_PASSWORD
                )
        
        assert result is True
        
        # Verify user was updated
        mock_repository.update.assert_called_once()
        updated_user = mock_repository.update.call_args[0][0]
        assert updated_user.updated_at > regular_user.updated_at
    
    async def test_change_password_wrong_current(
        self,
        user_service: UserService,
        mock_repository: AsyncMock,
        regular_user: UserModel
    ) -> None:
        """Test password change with wrong current password."""
        # Setup mock
        mock_repository.get_by_id.return_value = regular_user
        
        # Mock password verification to fail
        with patch.object(user_service, '_verify_password', return_value=False):
            with pytest.raises(InvalidCredentialsError):
                await user_service.change_password(
                    regular_user.id,
                    "wrong_password",
                    TestConstants.VALID_PASSWORD
                )
    
    async def test_change_password_weak_new(
        self,
        user_service: UserService,
        mock_repository: AsyncMock,
        regular_user: UserModel
    ) -> None:
        """Test password change with weak new password."""
        # Setup mock
        mock_repository.get_by_id.return_value = regular_user
        
        # Mock password verification
        with patch.object(user_service, '_verify_password', return_value=True):
            with pytest.raises(ValueError, match="password too weak"):
                await user_service.change_password(
                    regular_user.id,
                    "current_password",
                    "weak"
                )
    
    async def test_update_user_role(
        self,
        user_service: UserService,
        mock_repository: AsyncMock,
        regular_user: UserModel,
        admin_user: UserModel
    ) -> None:
        """Test updating user role."""
        # Setup mock
        mock_repository.get_by_id.return_value = regular_user
        
        # Update role
        updated = await user_service.update_user_role(
            regular_user.id,
            UserRole.MODERATOR,
            admin_user.id
        )
        
        assert updated.role == UserRole.MODERATOR
        assert "last_role_change" in updated.metadata
        assert updated.metadata["last_role_change"]["changed_by"] == admin_user.id
    
    async def test_list_users_with_search(
        self,
        user_service: UserService,
        mock_repository: AsyncMock,
        regular_user: UserModel,
        admin_user: UserModel
    ) -> None:
        """Test listing users with search filter."""
        # Setup mock
        mock_repository.list_users.return_value = [regular_user, admin_user]
        
        # Search for "admin"
        result = await user_service.list_users(search="admin")
        
        assert len(result["users"]) == 1
        assert result["users"][0].username == "admin"
        
        # Verify pagination info
        assert result["pagination"]["offset"] == 0
        assert result["pagination"]["limit"] == 100
        assert result["pagination"]["total"] == 1

# Integration Tests
class TestUserServiceIntegration:
    """Integration tests with real repository."""
    
    async def test_full_user_lifecycle(
        self,
        integration_user_service: UserService
    ) -> None:
        """Test complete user lifecycle: create, auth, update, delete."""
        # Create user
        create_request = UserCreateRequest(
            username="lifecycle_user",
            email="lifecycle@example.com",
            password=TestConstants.VALID_PASSWORD,
            full_name="Lifecycle Test User"
        )
        
        created_user = await integration_user_service.create_user(create_request)
        assert created_user.id is not None
        
        # Authenticate user
        auth_result = await integration_user_service.authenticate(
            "lifecycle_user",
            TestConstants.VALID_PASSWORD
        )
        assert auth_result["access_token"] is not None
        
        # Verify token
        token_payload = await integration_user_service.verify_token(
            auth_result["access_token"]
        )
        assert token_payload["sub"] == created_user.id
        
        # Change password
        new_password = "NewSecurePass123!@#"
        success = await integration_user_service.change_password(
            created_user.id,
            TestConstants.VALID_PASSWORD,
            new_password
        )
        assert success is True
        
        # Verify new password works
        auth_result2 = await integration_user_service.authenticate(
            "lifecycle_user",
            new_password
        )
        assert auth_result2["access_token"] is not None
        
        # Update role
        updated_user = await integration_user_service.update_user_role(
            created_user.id,
            UserRole.MODERATOR,
            created_user.id  # Self-update for testing
        )
        assert updated_user.role == UserRole.MODERATOR
    
    async def test_concurrent_user_creation(
        self,
        integration_user_service: UserService
    ) -> None:
        """Test concurrent user creation doesn't cause race conditions."""
        # Create multiple users concurrently
        tasks = []
        for i in range(10):
            request = UserCreateRequest(
                username=f"concurrent_user_{i}",
                email=f"concurrent{i}@example.com",
                password=TestConstants.VALID_PASSWORD
            )
            tasks.append(integration_user_service.create_user(request))
        
        # Execute concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Verify all succeeded
        successful = [r for r in results if not isinstance(r, Exception)]
        assert len(successful) == 10
        
        # Verify all have unique IDs
        ids = [u.id for u in successful]
        assert len(set(ids)) == 10
    
    async def test_authentication_rate_limiting(
        self,
        integration_user_service: UserService
    ) -> None:
        """Test authentication rate limiting behavior."""
        # Create user
        create_request = UserCreateRequest(
            username="rate_limit_user",
            email="ratelimit@example.com",
            password=TestConstants.VALID_PASSWORD
        )
        user = await integration_user_service.create_user(create_request)
        
        # Attempt multiple failed logins
        for i in range(MAX_LOGIN_ATTEMPTS - 1):
            with pytest.raises(InvalidCredentialsError):
                await integration_user_service.authenticate(
                    "rate_limit_user",
                    "wrong_password"
                )
        
        # Last attempt should lock the account
        with pytest.raises(InvalidCredentialsError):
            await integration_user_service.authenticate(
                "rate_limit_user",
                "wrong_password"
            )
        
        # Verify account is now locked
        with pytest.raises(AccountLockedException):
            await integration_user_service.authenticate(
                "rate_limit_user",
                TestConstants.VALID_PASSWORD  # Even correct password fails
            )

# Performance Tests
class TestUserServicePerformance:
    """Performance benchmarks for critical operations."""
    
    @pytest.mark.benchmark
    async def test_password_hashing_performance(
        self,
        user_service: UserService,
        benchmark
    ) -> None:
        """Benchmark password hashing performance."""
        password = TestConstants.VALID_PASSWORD
        
        # Benchmark hashing
        result = benchmark(
            user_service._hash_password,
            password
        )
        
        assert result is not None
        assert result.startswith("$2b$")
    
    @pytest.mark.benchmark
    async def test_token_generation_performance(
        self,
        user_service: UserService,
        regular_user: UserModel,
        benchmark
    ) -> None:
        """Benchmark JWT token generation."""
        result = benchmark(
            user_service._generate_access_token,
            regular_user
        )
        
        assert result is not None
        assert len(result) > 50
    
    @pytest.mark.benchmark
    async def test_user_creation_performance(
        self,
        integration_user_service: UserService,
        benchmark
    ) -> None:
        """Benchmark user creation end-to-end."""
        async def create_user():
            request = UserCreateRequest(
                username=f"perf_user_{secrets.token_hex(8)}",
                email=f"perf_{secrets.token_hex(8)}@example.com",
                password=TestConstants.VALID_PASSWORD
            )
            return await integration_user_service.create_user(request)
        
        result = benchmark(asyncio.run, create_user())
        assert result.id is not None

# Security Tests
class TestUserServiceSecurity:
    """Security-focused tests."""
    
    @pytest.mark.parametrize("malicious_input", [
        "admin' OR '1'='1",  # SQL injection
        "<script>alert('xss')</script>",  # XSS
        "../../etc/passwd",  # Path traversal
        "\x00null\x00byte",  # Null byte injection
        "A" * 10000,  # Buffer overflow attempt
    ])
    async def test_username_injection_protection(
        self,
        user_service: UserService,
        mock_repository: AsyncMock,
        malicious_input: str
    ) -> None:
        """Test protection against various injection attacks in username."""
        # Attempt to create user with malicious username
        with pytest.raises(ValueError):
            request = UserCreateRequest(
                username=malicious_input,
                email="test@example.com",
                password=TestConstants.VALID_PASSWORD
            )
    
    async def test_timing_attack_resistance(
        self,
        user_service: UserService,
        mock_repository: AsyncMock,
        regular_user: UserModel
    ) -> None:
        """Test resistance to timing attacks on authentication."""
        import time
        
        # Setup mock
        mock_repository.get_by_username.return_value = regular_user
        
        # Time valid username with wrong password
        start1 = time.perf_counter()
        with pytest.raises(InvalidCredentialsError):
            await user_service.authenticate("valid_user", "wrong_pass")
        time1 = time.perf_counter() - start1
        
        # Time invalid username
        mock_repository.get_by_username.return_value = None
        start2 = time.perf_counter()
        with pytest.raises(InvalidCredentialsError):
            await user_service.authenticate("invalid_user", "wrong_pass")
        time2 = time.perf_counter() - start2
        
        # Times should be similar (within 50ms)
        assert abs(time1 - time2) < 0.05
    
    async def test_password_not_logged(
        self,
        user_service: UserService,
        mock_repository: AsyncMock,
        valid_user_request: UserCreateRequest,
        caplog
    ) -> None:
        """Test that passwords are never logged."""
        # Create user
        await user_service.create_user(valid_user_request)
        
        # Check logs don't contain password
        for record in caplog.records:
            assert TestConstants.VALID_PASSWORD not in record.getMessage()
            assert "password" not in record.getMessage().lower()

# Property-Based Tests
class TestUserServiceProperties:
    """Property-based tests using Hypothesis."""
    
    @given(
        username=st.text(
            alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd"), 
            whitelist_characters="_-"),
            min_size=3,
            max_size=50
        ),
        email=st.emails(),
        password=st.text(min_size=12, max_size=128)
    )
    @settings(max_examples=50)
    async def test_user_creation_properties(
        self,
        integration_user_service: UserService,
        username: str,
        email: str,
        password: str
    ) -> None:
        """Property test: valid inputs should create user or raise expected errors."""
        # Assume password meets strength requirements
        assume(any(c.isupper() for c in password))
        assume(any(c.islower() for c in password))
        assume(any(c.isdigit() for c in password))
        assume(any(c in "!@#$%^&*(),.?\":{}|<>" for c in password))
        
        # Skip reserved usernames
        assume(username.lower() not in {
            'admin', 'root', 'system', 'api', 'null', 'undefined',
            'test', 'user', 'guest', 'anonymous', 'nobody'
        })
        
        try:
            request = UserCreateRequest(
                username=username,
                email=email,
                password=password
            )
            user = await integration_user_service.create_user(request)
            
            # Verify invariants
            assert user.id is not None
            assert user.username == username.lower()
            assert user.email == email.lower()
            assert user.role == UserRole.USER
            assert user.status == UserStatus.PENDING_VERIFICATION
            
        except (ValueError, UserAlreadyExistsError) as e:
            # Expected exceptions are fine
            pass
    
    @given(st.text())
    @settings(max_examples=100)
    def test_password_strength_never_crashes(self, password: str) -> None:
        """Property test: password strength analyzer never crashes."""
        strength = PasswordStrength.analyze(password)
        
        # Verify invariants
        assert 0 <= strength.score <= 5
        assert isinstance(strength.feedback, list)
        assert all(isinstance(f, str) for f in strength.feedback)
        assert isinstance(strength.crack_time_display, str)

# Fixture cleanup
@pytest.fixture(autouse=True)
async def cleanup_metrics():
    """Reset metrics after each test."""
    yield
    # Reset Prometheus metrics
    # In real implementation, would reset counters/gauges

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--cov=api.user_management_v2", "--cov-report=html"]) 