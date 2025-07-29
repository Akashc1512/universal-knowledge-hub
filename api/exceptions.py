"""
Custom exceptions for the Universal Knowledge Platform.
Provides granular error handling and secure error messages.
"""

import logging
from typing import Optional, Dict, Any
from fastapi import HTTPException, status


logger = logging.getLogger(__name__)


# Base exceptions
class UKPException(Exception):
    """Base exception for Universal Knowledge Platform."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        self.message = message
        self.details = details or {}
        super().__init__(self.message)


class UKPHTTPException(HTTPException):
    """Base HTTP exception with logging and secure error messages."""
    
    def __init__(
        self,
        status_code: int,
        detail: str,
        internal_message: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None
    ):
        self.internal_message = internal_message or detail
        super().__init__(status_code=status_code, detail=detail, headers=headers)
        
        # Log internal details
        logger.error(f"HTTP {status_code}: {self.internal_message}")


# Authentication and Authorization exceptions
class AuthenticationError(UKPHTTPException):
    """Authentication failed."""
    
    def __init__(self, internal_message: Optional[str] = None):
        super().__init__(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            internal_message=internal_message or "Authentication failed"
        )


class AuthorizationError(UKPHTTPException):
    """Insufficient permissions."""
    
    def __init__(self, internal_message: Optional[str] = None):
        super().__init__(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Insufficient permissions",
            internal_message=internal_message or "Authorization failed"
        )


class InvalidAPIKeyError(AuthenticationError):
    """Invalid API key provided."""
    
    def __init__(self, api_key_hint: Optional[str] = None):
        internal_msg = f"Invalid API key: {api_key_hint}" if api_key_hint else "Invalid API key"
        super().__init__(internal_message=internal_msg)


class RateLimitExceededError(UKPHTTPException):
    """Rate limit exceeded."""
    
    def __init__(self, limit: int, window: str, retry_after: Optional[int] = None):
        headers = {"Retry-After": str(retry_after)} if retry_after else None
        super().__init__(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit exceeded: {limit} requests per {window}",
            internal_message=f"Rate limit exceeded: {limit}/{window}",
            headers=headers
        )


# Agent and processing exceptions
class AgentError(UKPException):
    """Base exception for agent errors."""
    
    def __init__(self, agent_type: str, message: str, details: Optional[Dict[str, Any]] = None):
        self.agent_type = agent_type
        super().__init__(message, details)


class AgentTimeoutError(AgentError):
    """Agent processing timeout."""
    
    def __init__(self, agent_type: str, timeout_seconds: int):
        super().__init__(
            agent_type=agent_type,
            message=f"Agent {agent_type} timed out after {timeout_seconds} seconds",
            details={"timeout_seconds": timeout_seconds}
        )


class AgentProcessingError(AgentError):
    """Agent processing failed."""
    
    def __init__(self, agent_type: str, error_message: str, recoverable: bool = True):
        super().__init__(
            agent_type=agent_type,
            message=f"Agent {agent_type} processing failed: {error_message}",
            details={"recoverable": recoverable, "error": error_message}
        )


class QueryProcessingError(UKPHTTPException):
    """Query processing failed."""
    
    def __init__(self, query_id: str, internal_error: str, recoverable: bool = True):
        detail = "Query processing failed" if recoverable else "Query processing failed permanently"
        super().__init__(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR if not recoverable else status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=detail,
            internal_message=f"Query {query_id} failed: {internal_error}"
        )


# Resource and infrastructure exceptions
class ResourceNotFoundError(UKPHTTPException):
    """Requested resource not found."""
    
    def __init__(self, resource_type: str, resource_id: str):
        super().__init__(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"{resource_type} not found",
            internal_message=f"{resource_type} {resource_id} not found"
        )


class DatabaseError(UKPException):
    """Database operation failed."""
    
    def __init__(self, operation: str, error: str, database: str = "primary"):
        super().__init__(
            message=f"Database {operation} failed on {database}: {error}",
            details={"operation": operation, "database": database, "error": error}
        )


class CacheError(UKPException):
    """Cache operation failed."""
    
    def __init__(self, operation: str, cache_type: str, error: str):
        super().__init__(
            message=f"Cache {operation} failed on {cache_type}: {error}",
            details={"operation": operation, "cache_type": cache_type, "error": error}
        )


class ExternalServiceError(UKPException):
    """External service integration failed."""
    
    def __init__(self, service: str, operation: str, error: str, retryable: bool = True):
        super().__init__(
            message=f"External service {service} failed during {operation}: {error}",
            details={"service": service, "operation": operation, "retryable": retryable, "error": error}
        )


# Validation exceptions
class ValidationError(UKPHTTPException):
    """Request validation failed."""
    
    def __init__(self, field: str, message: str, value: Any = None):
        super().__init__(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Validation error: {message}",
            internal_message=f"Validation failed for {field}: {message} (value: {value})"
        )


class SecurityViolationError(UKPHTTPException):
    """Security policy violation."""
    
    def __init__(self, violation_type: str, details: Optional[str] = None):
        super().__init__(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Request blocked by security policy",
            internal_message=f"Security violation: {violation_type} - {details or 'No details'}"
        )


# Alias for backwards compatibility
SecurityError = SecurityViolationError
RateLimitError = RateLimitExceededError


# Configuration exceptions
class ConfigurationError(UKPException):
    """Configuration error."""
    
    def __init__(self, component: str, issue: str, suggestion: Optional[str] = None):
        message = f"Configuration error in {component}: {issue}"
        if suggestion:
            message += f". Suggestion: {suggestion}"
        
        super().__init__(
            message=message,
            details={"component": component, "issue": issue, "suggestion": suggestion}
        )


# Expert review exceptions
class ExpertReviewError(UKPHTTPException):
    """Expert review operation failed."""
    
    def __init__(self, operation: str, review_id: str, reason: str):
        super().__init__(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Expert review {operation} failed",
            internal_message=f"Expert review {operation} failed for {review_id}: {reason}"
        )


# Utility functions for error handling
def handle_agent_error(agent_type: str, error: Exception) -> AgentError:
    """Convert generic exceptions to agent-specific errors."""
    if isinstance(error, AgentError):
        return error
    elif isinstance(error, TimeoutError):
        return AgentTimeoutError(agent_type, 30)  # Default timeout
    else:
        return AgentProcessingError(agent_type, str(error), recoverable=True)


def handle_external_service_error(service: str, operation: str, error: Exception) -> ExternalServiceError:
    """Convert generic exceptions to external service errors."""
    if isinstance(error, ExternalServiceError):
        return error
    else:
        # Determine if error is retryable based on error type
        retryable = not isinstance(error, (ValueError, TypeError, KeyError))
        return ExternalServiceError(service, operation, str(error), retryable)


def sanitize_error_message(error: Exception, include_details: bool = False) -> str:
    """
    Sanitize error messages for safe client exposure.
    
    Args:
        error: The exception to sanitize
        include_details: Whether to include detailed error information
        
    Returns:
        Sanitized error message safe for client consumption
    """
    if isinstance(error, UKPHTTPException):
        return error.detail
    elif isinstance(error, UKPException):
        return error.message if include_details else "An error occurred"
    else:
        # Generic exceptions should not expose internal details
        return "An internal error occurred" 