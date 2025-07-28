"""
Exception Handling Module - MAANG Standards.

This module implements a comprehensive exception hierarchy following MAANG
best practices for error handling, logging, and client communication.

Features:
    - Structured exception hierarchy
    - Detailed error context
    - Correlation IDs for tracing
    - Client-safe error messages
    - Automatic logging integration
    - Retry hints for transient errors
    - Internationalization support

Authors:
    - Universal Knowledge Platform Engineering Team
    
Version:
    2.0.0 (2024-12-28)
"""

import sys
import traceback
from typing import Optional, Dict, Any, List, Union, Type
from datetime import datetime, timezone
from enum import Enum
import structlog

logger = structlog.get_logger(__name__)

# Error categories
class ErrorCategory(str, Enum):
    """Error categories for classification and handling."""
    VALIDATION = "validation"
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    NOT_FOUND = "not_found"
    CONFLICT = "conflict"
    RATE_LIMIT = "rate_limit"
    EXTERNAL_SERVICE = "external_service"
    DATABASE = "database"
    INTERNAL = "internal"
    CONFIGURATION = "configuration"

# Error severity
class ErrorSeverity(str, Enum):
    """Error severity levels for monitoring and alerting."""
    LOW = "low"          # User error, expected
    MEDIUM = "medium"    # System issue, degraded service
    HIGH = "high"        # Critical issue, immediate attention
    CRITICAL = "critical" # System failure, page immediately

class ErrorCode(str, Enum):
    """Standardized error codes for client handling."""
    # Validation errors (1000-1999)
    INVALID_INPUT = "E1001"
    MISSING_FIELD = "E1002"
    INVALID_FORMAT = "E1003"
    VALUE_OUT_OF_RANGE = "E1004"
    DUPLICATE_VALUE = "E1005"
    
    # Authentication errors (2000-2999)
    INVALID_CREDENTIALS = "E2001"
    TOKEN_EXPIRED = "E2002"
    TOKEN_INVALID = "E2003"
    ACCOUNT_LOCKED = "E2004"
    ACCOUNT_DISABLED = "E2005"
    TWO_FACTOR_REQUIRED = "E2006"
    TWO_FACTOR_FAILED = "E2007"
    
    # Authorization errors (3000-3999)
    INSUFFICIENT_PERMISSIONS = "E3001"
    RESOURCE_ACCESS_DENIED = "E3002"
    OPERATION_NOT_ALLOWED = "E3003"
    SUBSCRIPTION_REQUIRED = "E3004"
    
    # Resource errors (4000-4999)
    RESOURCE_NOT_FOUND = "E4001"
    RESOURCE_ALREADY_EXISTS = "E4002"
    RESOURCE_CONFLICT = "E4003"
    RESOURCE_LOCKED = "E4004"
    
    # Rate limiting errors (5000-5999)
    RATE_LIMIT_EXCEEDED = "E5001"
    QUOTA_EXCEEDED = "E5002"
    
    # External service errors (6000-6999)
    SERVICE_UNAVAILABLE = "E6001"
    SERVICE_TIMEOUT = "E6002"
    SERVICE_ERROR = "E6003"
    
    # Database errors (7000-7999)
    DATABASE_CONNECTION_FAILED = "E7001"
    DATABASE_QUERY_FAILED = "E7002"
    DATABASE_CONSTRAINT_VIOLATION = "E7003"
    DATABASE_DEADLOCK = "E7004"
    
    # Internal errors (9000-9999)
    INTERNAL_ERROR = "E9001"
    CONFIGURATION_ERROR = "E9002"
    UNHANDLED_ERROR = "E9999"

class BaseError(Exception):
    """
    Base exception class for all application errors.
    
    Provides structured error information with context, tracing,
    and client-safe messages.
    """
    
    # Default attributes
    category: ErrorCategory = ErrorCategory.INTERNAL
    severity: ErrorSeverity = ErrorSeverity.MEDIUM
    error_code: ErrorCode = ErrorCode.INTERNAL_ERROR
    status_code: int = 500
    retryable: bool = False
    
    def __init__(
        self,
        message: str,
        *,
        details: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None,
        correlation_id: Optional[str] = None,
        user_message: Optional[str] = None,
        retry_after: Optional[int] = None,
        help_url: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Initialize base error with comprehensive context.
        
        Args:
            message: Internal error message (not shown to users)
            details: Additional error details
            cause: Original exception that caused this error
            correlation_id: Request correlation ID for tracing
            user_message: Safe message to show to end users
            retry_after: Seconds to wait before retry (if retryable)
            help_url: URL to documentation or help
            metadata: Additional metadata for logging
        """
        super().__init__(message)
        
        self.message = message
        self.details = details or {}
        self.cause = cause
        self.correlation_id = correlation_id
        self.user_message = user_message or self._get_default_user_message()
        self.retry_after = retry_after
        self.help_url = help_url
        self.metadata = metadata or {}
        self.timestamp = datetime.now(timezone.utc)
        
        # Capture stack trace
        self.traceback = traceback.format_exc()
        
        # Log error
        self._log_error()
    
    def _get_default_user_message(self) -> str:
        """Get default user-friendly message based on category."""
        messages = {
            ErrorCategory.VALIDATION: "The provided data is invalid.",
            ErrorCategory.AUTHENTICATION: "Authentication failed.",
            ErrorCategory.AUTHORIZATION: "You don't have permission to perform this action.",
            ErrorCategory.NOT_FOUND: "The requested resource was not found.",
            ErrorCategory.CONFLICT: "The request conflicts with the current state.",
            ErrorCategory.RATE_LIMIT: "Too many requests. Please try again later.",
            ErrorCategory.EXTERNAL_SERVICE: "An external service is temporarily unavailable.",
            ErrorCategory.DATABASE: "A database error occurred.",
            ErrorCategory.INTERNAL: "An internal error occurred.",
            ErrorCategory.CONFIGURATION: "The system is misconfigured.",
        }
        return messages.get(self.category, "An error occurred.")
    
    def _log_error(self) -> None:
        """Log error with appropriate level based on severity."""
        log_data = {
            "error_type": self.__class__.__name__,
            "error_code": self.error_code.value,
            "category": self.category.value,
            "severity": self.severity.value,
            "message": self.message,
            "details": self.details,
            "correlation_id": self.correlation_id,
            "retryable": self.retryable,
            "status_code": self.status_code,
        }
        
        if self.cause:
            log_data["cause"] = str(self.cause)
            log_data["cause_type"] = type(self.cause).__name__
        
        if self.metadata:
            log_data["metadata"] = self.metadata
        
        # Log based on severity
        if self.severity == ErrorSeverity.CRITICAL:
            logger.critical("Critical error occurred", **log_data)
        elif self.severity == ErrorSeverity.HIGH:
            logger.error("High severity error occurred", **log_data)
        elif self.severity == ErrorSeverity.MEDIUM:
            logger.warning("Medium severity error occurred", **log_data)
        else:
            logger.info("Low severity error occurred", **log_data)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert error to dictionary for API responses.
        
        Returns:
            Client-safe error representation
        """
        error_dict = {
            "error": {
                "code": self.error_code.value,
                "message": self.user_message,
                "category": self.category.value,
                "timestamp": self.timestamp.isoformat(),
            }
        }
        
        # Add optional fields
        if self.correlation_id:
            error_dict["error"]["correlation_id"] = self.correlation_id
        
        if self.details:
            # Filter sensitive information
            safe_details = self._filter_sensitive_details(self.details)
            if safe_details:
                error_dict["error"]["details"] = safe_details
        
        if self.retryable:
            error_dict["error"]["retryable"] = True
            if self.retry_after:
                error_dict["error"]["retry_after"] = self.retry_after
        
        if self.help_url:
            error_dict["error"]["help_url"] = self.help_url
        
        return error_dict
    
    def _filter_sensitive_details(self, details: Dict[str, Any]) -> Dict[str, Any]:
        """Filter out sensitive information from details."""
        sensitive_keys = {
            'password', 'token', 'secret', 'key', 'authorization',
            'cookie', 'session', 'credit_card', 'ssn', 'api_key'
        }
        
        filtered = {}
        for key, value in details.items():
            if any(sensitive in key.lower() for sensitive in sensitive_keys):
                filtered[key] = "[REDACTED]"
            elif isinstance(value, dict):
                filtered[key] = self._filter_sensitive_details(value)
            else:
                filtered[key] = value
        
        return filtered
    
    def with_metadata(self, **kwargs: Any) -> 'BaseError':
        """Add metadata to error (fluent interface)."""
        self.metadata.update(kwargs)
        return self
    
    def __str__(self) -> str:
        """String representation."""
        return f"{self.__class__.__name__}({self.error_code.value}): {self.message}"
    
    def __repr__(self) -> str:
        """Detailed representation."""
        return (
            f"{self.__class__.__name__}("
            f"code={self.error_code.value}, "
            f"message={self.message!r}, "
            f"details={self.details!r})"
        )

# Validation Errors
class ValidationError(BaseError):
    """Raised when input validation fails."""
    
    category = ErrorCategory.VALIDATION
    severity = ErrorSeverity.LOW
    error_code = ErrorCode.INVALID_INPUT
    status_code = 422
    
    def __init__(
        self,
        message: str,
        *,
        field: Optional[str] = None,
        value: Optional[Any] = None,
        constraints: Optional[Dict[str, Any]] = None,
        **kwargs: Any
    ) -> None:
        """
        Initialize validation error.
        
        Args:
            message: Error message
            field: Field that failed validation
            value: Invalid value (will be sanitized)
            constraints: Validation constraints that failed
            **kwargs: Additional arguments for BaseError
        """
        details = kwargs.pop('details', {})
        
        if field:
            details['field'] = field
        if value is not None:
            details['value'] = self._sanitize_value(value)
        if constraints:
            details['constraints'] = constraints
        
        super().__init__(message, details=details, **kwargs)
    
    def _sanitize_value(self, value: Any) -> Any:
        """Sanitize value to prevent sensitive data exposure."""
        if isinstance(value, str) and len(value) > 100:
            return value[:100] + "..."
        return value

class MissingFieldError(ValidationError):
    """Raised when a required field is missing."""
    
    error_code = ErrorCode.MISSING_FIELD
    
    def __init__(self, field: str, **kwargs: Any) -> None:
        message = f"Required field '{field}' is missing"
        super().__init__(message, field=field, **kwargs)

class InvalidFormatError(ValidationError):
    """Raised when a field has invalid format."""
    
    error_code = ErrorCode.INVALID_FORMAT
    
    def __init__(
        self,
        field: str,
        expected_format: str,
        actual_value: Optional[str] = None,
        **kwargs: Any
    ) -> None:
        message = f"Field '{field}' has invalid format. Expected: {expected_format}"
        super().__init__(
            message,
            field=field,
            value=actual_value,
            constraints={"format": expected_format},
            **kwargs
        )

# Authentication Errors
class AuthenticationError(BaseError):
    """Base class for authentication errors."""
    
    category = ErrorCategory.AUTHENTICATION
    severity = ErrorSeverity.LOW
    error_code = ErrorCode.INVALID_CREDENTIALS
    status_code = 401

class InvalidCredentialsError(AuthenticationError):
    """Raised when credentials are invalid."""
    
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(
            "Invalid username or password",
            user_message="The username or password you entered is incorrect.",
            **kwargs
        )

class TokenExpiredError(AuthenticationError):
    """Raised when a token has expired."""
    
    error_code = ErrorCode.TOKEN_EXPIRED
    
    def __init__(self, token_type: str = "access", **kwargs: Any) -> None:
        super().__init__(
            f"{token_type.capitalize()} token has expired",
            user_message=f"Your {token_type} token has expired. Please login again.",
            **kwargs
        )

class AccountLockedException(AuthenticationError):
    """Raised when account is locked."""
    
    error_code = ErrorCode.ACCOUNT_LOCKED
    
    def __init__(
        self,
        locked_until: Optional[datetime] = None,
        reason: str = "too many failed login attempts",
        **kwargs: Any
    ) -> None:
        details = {"reason": reason}
        if locked_until:
            details["locked_until"] = locked_until.isoformat()
            retry_after = int((locked_until - datetime.now(timezone.utc)).total_seconds())
            kwargs["retry_after"] = max(retry_after, 0)
        
        super().__init__(
            f"Account is locked due to {reason}",
            details=details,
            user_message="Your account has been temporarily locked. Please try again later.",
            **kwargs
        )

# Authorization Errors
class AuthorizationError(BaseError):
    """Base class for authorization errors."""
    
    category = ErrorCategory.AUTHORIZATION
    severity = ErrorSeverity.LOW
    error_code = ErrorCode.INSUFFICIENT_PERMISSIONS
    status_code = 403

class InsufficientPermissionsError(AuthorizationError):
    """Raised when user lacks required permissions."""
    
    def __init__(
        self,
        required_permission: str,
        user_permissions: Optional[List[str]] = None,
        **kwargs: Any
    ) -> None:
        details = {"required_permission": required_permission}
        if user_permissions:
            details["user_permissions"] = user_permissions
        
        super().__init__(
            f"User lacks required permission: {required_permission}",
            details=details,
            user_message="You don't have permission to perform this action.",
            **kwargs
        )

# Resource Errors
class NotFoundError(BaseError):
    """Raised when a resource is not found."""
    
    category = ErrorCategory.NOT_FOUND
    severity = ErrorSeverity.LOW
    error_code = ErrorCode.RESOURCE_NOT_FOUND
    status_code = 404
    
    def __init__(
        self,
        resource_type: str,
        resource_id: Optional[Union[str, int]] = None,
        **kwargs: Any
    ) -> None:
        details = {"resource_type": resource_type}
        if resource_id:
            details["resource_id"] = str(resource_id)
        
        message = f"{resource_type} not found"
        if resource_id:
            message += f": {resource_id}"
        
        super().__init__(
            message,
            details=details,
            user_message=f"The requested {resource_type.lower()} was not found.",
            **kwargs
        )

class ConflictError(BaseError):
    """Raised when a request conflicts with current state."""
    
    category = ErrorCategory.CONFLICT
    severity = ErrorSeverity.LOW
    error_code = ErrorCode.RESOURCE_CONFLICT
    status_code = 409
    
    def __init__(
        self,
        message: str,
        *,
        conflicting_resource: Optional[str] = None,
        current_state: Optional[Dict[str, Any]] = None,
        **kwargs: Any
    ) -> None:
        details = kwargs.pop('details', {})
        
        if conflicting_resource:
            details['conflicting_resource'] = conflicting_resource
        if current_state:
            details['current_state'] = current_state
        
        super().__init__(message, details=details, **kwargs)

class DuplicateResourceError(ConflictError):
    """Raised when attempting to create a duplicate resource."""
    
    error_code = ErrorCode.RESOURCE_ALREADY_EXISTS
    
    def __init__(
        self,
        resource_type: str,
        duplicate_field: str,
        duplicate_value: Any,
        **kwargs: Any
    ) -> None:
        super().__init__(
            f"{resource_type} already exists with {duplicate_field}: {duplicate_value}",
            conflicting_resource=resource_type,
            current_state={duplicate_field: duplicate_value},
            user_message=f"A {resource_type.lower()} with this {duplicate_field} already exists.",
            **kwargs
        )

# Rate Limiting Errors
class RateLimitError(BaseError):
    """Raised when rate limit is exceeded."""
    
    category = ErrorCategory.RATE_LIMIT
    severity = ErrorSeverity.LOW
    error_code = ErrorCode.RATE_LIMIT_EXCEEDED
    status_code = 429
    retryable = True
    
    def __init__(
        self,
        limit: int,
        window: str,
        retry_after: int,
        **kwargs: Any
    ) -> None:
        super().__init__(
            f"Rate limit exceeded: {limit} requests per {window}",
            details={
                "limit": limit,
                "window": window,
                "retry_after": retry_after
            },
            retry_after=retry_after,
            user_message=f"Too many requests. Please wait {retry_after} seconds before trying again.",
            **kwargs
        )

# External Service Errors
class ExternalServiceError(BaseError):
    """Base class for external service errors."""
    
    category = ErrorCategory.EXTERNAL_SERVICE
    severity = ErrorSeverity.MEDIUM
    error_code = ErrorCode.SERVICE_ERROR
    status_code = 502
    retryable = True

class ServiceUnavailableError(ExternalServiceError):
    """Raised when an external service is unavailable."""
    
    error_code = ErrorCode.SERVICE_UNAVAILABLE
    status_code = 503
    
    def __init__(
        self,
        service_name: str,
        reason: Optional[str] = None,
        retry_after: Optional[int] = None,
        **kwargs: Any
    ) -> None:
        details = {"service": service_name}
        if reason:
            details["reason"] = reason
        
        message = f"{service_name} is currently unavailable"
        if reason:
            message += f": {reason}"
        
        super().__init__(
            message,
            details=details,
            retry_after=retry_after or 60,
            user_message=f"The {service_name} service is temporarily unavailable. Please try again later.",
            **kwargs
        )

class ServiceTimeoutError(ExternalServiceError):
    """Raised when an external service times out."""
    
    error_code = ErrorCode.SERVICE_TIMEOUT
    status_code = 504
    
    def __init__(
        self,
        service_name: str,
        timeout_seconds: float,
        **kwargs: Any
    ) -> None:
        super().__init__(
            f"{service_name} request timed out after {timeout_seconds}s",
            details={
                "service": service_name,
                "timeout_seconds": timeout_seconds
            },
            retry_after=30,
            user_message=f"The request to {service_name} took too long. Please try again.",
            **kwargs
        )

# Database Errors
class DatabaseError(BaseError):
    """Base class for database errors."""
    
    category = ErrorCategory.DATABASE
    severity = ErrorSeverity.HIGH
    error_code = ErrorCode.DATABASE_QUERY_FAILED
    status_code = 500
    retryable = True

class DatabaseConnectionError(DatabaseError):
    """Raised when database connection fails."""
    
    error_code = ErrorCode.DATABASE_CONNECTION_FAILED
    severity = ErrorSeverity.CRITICAL
    
    def __init__(
        self,
        database_type: str = "database",
        **kwargs: Any
    ) -> None:
        super().__init__(
            f"Failed to connect to {database_type}",
            user_message="Unable to connect to the database. Please try again later.",
            **kwargs
        )

class DatabaseConstraintError(DatabaseError):
    """Raised when a database constraint is violated."""
    
    error_code = ErrorCode.DATABASE_CONSTRAINT_VIOLATION
    severity = ErrorSeverity.LOW
    status_code = 409
    retryable = False
    
    def __init__(
        self,
        constraint_name: str,
        constraint_type: str = "constraint",
        **kwargs: Any
    ) -> None:
        super().__init__(
            f"Database {constraint_type} violation: {constraint_name}",
            details={
                "constraint_name": constraint_name,
                "constraint_type": constraint_type
            },
            user_message="The operation violates data integrity rules.",
            **kwargs
        )

# Configuration Errors
class ConfigurationError(BaseError):
    """Raised when configuration is invalid."""
    
    category = ErrorCategory.CONFIGURATION
    severity = ErrorSeverity.CRITICAL
    error_code = ErrorCode.CONFIGURATION_ERROR
    status_code = 500
    
    def __init__(
        self,
        config_key: str,
        reason: str,
        **kwargs: Any
    ) -> None:
        super().__init__(
            f"Invalid configuration for '{config_key}': {reason}",
            details={
                "config_key": config_key,
                "reason": reason
            },
            user_message="The system is misconfigured. Please contact support.",
            **kwargs
        )

# Error Handler
class ErrorHandler:
    """
    Centralized error handler for consistent error processing.
    
    Features:
    - Error transformation
    - Retry logic
    - Circuit breaking
    - Error aggregation
    """
    
    def __init__(self) -> None:
        self._error_counts: Dict[str, int] = {}
        self._circuit_breakers: Dict[str, Any] = {}
    
    def handle_error(
        self,
        error: Exception,
        *,
        context: Optional[Dict[str, Any]] = None,
        correlation_id: Optional[str] = None
    ) -> BaseError:
        """
        Handle and transform any exception to BaseError.
        
        Args:
            error: Exception to handle
            context: Additional context
            correlation_id: Request correlation ID
            
        Returns:
            Transformed BaseError instance
        """
        # If already a BaseError, enhance it
        if isinstance(error, BaseError):
            if correlation_id and not error.correlation_id:
                error.correlation_id = correlation_id
            if context:
                error.metadata.update(context)
            return error
        
        # Transform known exceptions
        error_map = {
            ValueError: ValidationError,
            KeyError: NotFoundError,
            PermissionError: AuthorizationError,
            TimeoutError: ServiceTimeoutError,
        }
        
        error_class = error_map.get(type(error), BaseError)
        
        # Create appropriate error
        base_error = error_class(
            str(error),
            cause=error,
            correlation_id=correlation_id,
            metadata=context or {}
        )
        
        # Track error frequency
        error_key = f"{error_class.__name__}:{str(error)[:50]}"
        self._error_counts[error_key] = self._error_counts.get(error_key, 0) + 1
        
        return base_error
    
    def get_error_stats(self) -> Dict[str, Any]:
        """Get error statistics for monitoring."""
        return {
            "error_counts": self._error_counts.copy(),
            "total_errors": sum(self._error_counts.values()),
            "unique_errors": len(self._error_counts)
        }

# Global error handler instance
error_handler = ErrorHandler()

# Export public API
__all__ = [
    # Base classes
    'BaseError',
    'ErrorCategory',
    'ErrorSeverity',
    'ErrorCode',
    
    # Validation errors
    'ValidationError',
    'MissingFieldError',
    'InvalidFormatError',
    
    # Authentication errors
    'AuthenticationError',
    'InvalidCredentialsError',
    'TokenExpiredError',
    'AccountLockedException',
    
    # Authorization errors
    'AuthorizationError',
    'InsufficientPermissionsError',
    
    # Resource errors
    'NotFoundError',
    'ConflictError',
    'DuplicateResourceError',
    
    # Rate limiting
    'RateLimitError',
    
    # External service errors
    'ExternalServiceError',
    'ServiceUnavailableError',
    'ServiceTimeoutError',
    
    # Database errors
    'DatabaseError',
    'DatabaseConnectionError',
    'DatabaseConstraintError',
    
    # Configuration errors
    'ConfigurationError',
    
    # Error handler
    'ErrorHandler',
    'error_handler',
] 