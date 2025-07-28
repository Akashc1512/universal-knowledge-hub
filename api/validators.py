"""
Input Validation Module for Universal Knowledge Platform
Provides comprehensive validation for all API endpoints.
"""

import re
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, validator, conint
from typing import Annotated
from pydantic.types import StringConstraints
from datetime import datetime
import bleach

# Constants for validation
MAX_QUERY_LENGTH = 5000
MIN_QUERY_LENGTH = 3
MAX_TOKENS = 10000
MIN_TOKENS = 10
MAX_CONFIDENCE_THRESHOLD = 1.0
MIN_CONFIDENCE_THRESHOLD = 0.0
ALLOWED_SEARCH_TYPES = ["hybrid", "vector", "keyword", "graph"]
ALLOWED_LANGUAGES = ["en", "es", "fr", "de", "it", "pt", "zh", "ja", "ko"]
MAX_RESULTS = 100
MIN_RESULTS = 1

# Regex patterns for validation
SQL_INJECTION_PATTERN = re.compile(
    r"(\b(union|select|insert|update|delete|drop|create|alter|exec|execute)\b)",
    re.IGNORECASE
)
XSS_PATTERN = re.compile(
    r"(<script|javascript:|onerror=|onclick=|<iframe|<object|<embed)",
    re.IGNORECASE
)
PATH_TRAVERSAL_PATTERN = re.compile(r"(\.\./|\.\.\\|%2e%2e)")


class SanitizedStr(str):
    """Custom string type that sanitizes input."""
    
    @classmethod
    def __get_validators__(cls):
        yield cls.validate
    
    @classmethod
    def validate(cls, v):
        if not isinstance(v, str):
            raise TypeError("string required")
        
        # Remove any HTML tags
        v = bleach.clean(v, tags=[], strip=True)
        
        # Check for SQL injection attempts
        if SQL_INJECTION_PATTERN.search(v):
            raise ValueError("Potential SQL injection detected")
        
        # Check for XSS attempts
        if XSS_PATTERN.search(v):
            raise ValueError("Potential XSS attack detected")
        
        # Check for path traversal attempts
        if PATH_TRAVERSAL_PATTERN.search(v):
            raise ValueError("Potential path traversal detected")
        
        return cls(v)


class QueryRequestValidator(BaseModel):
    """Validated query request model."""
    
    query: SanitizedStr = Field(
        ...,
        min_length=MIN_QUERY_LENGTH,
        max_length=MAX_QUERY_LENGTH,
        description="The query to process"
    )
    
    max_tokens: conint(ge=MIN_TOKENS, le=MAX_TOKENS) = Field(
        default=1000,
        description="Maximum tokens for response"
    )
    
    confidence_threshold: float = Field(
        default=0.7,
        ge=MIN_CONFIDENCE_THRESHOLD,
        le=MAX_CONFIDENCE_THRESHOLD,
        description="Minimum confidence threshold"
    )
    
    search_type: Optional[str] = Field(
        default="hybrid",
        description="Type of search to perform"
    )
    
    language: Optional[str] = Field(
        default="en",
        description="Query language"
    )
    
    include_sources: bool = Field(
        default=True,
        description="Include source citations"
    )
    
    metadata: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Additional metadata"
    )
    
    @validator("search_type")
    def validate_search_type(cls, v):
        if v and v not in ALLOWED_SEARCH_TYPES:
            raise ValueError(f"Invalid search type. Must be one of: {ALLOWED_SEARCH_TYPES}")
        return v
    
    @validator("language")
    def validate_language(cls, v):
        if v and v not in ALLOWED_LANGUAGES:
            raise ValueError(f"Invalid language. Must be one of: {ALLOWED_LANGUAGES}")
        return v
    
    @validator("metadata")
    def validate_metadata(cls, v):
        if v and len(str(v)) > 1000:
            raise ValueError("Metadata too large (max 1000 characters)")
        return v


class FeedbackRequestValidator(BaseModel):
    """Validated feedback request model."""
    
    query_id: Annotated[str, StringConstraints(pattern=r"^[a-f0-9\-]{36}$")] = Field(
        ...,
        description="UUID of the query"
    )
    
    rating: conint(ge=1, le=5) = Field(
        ...,
        description="Rating from 1-5"
    )
    
    feedback: Optional[SanitizedStr] = Field(
        None,
        max_length=1000,
        description="Optional feedback text"
    )
    
    metadata: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Additional metadata"
    )


class SearchRequestValidator(BaseModel):
    """Validated search request model."""
    
    query: SanitizedStr = Field(
        ...,
        min_length=1,
        max_length=500,
        description="Search query"
    )
    
    filters: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Search filters"
    )
    
    limit: conint(ge=MIN_RESULTS, le=MAX_RESULTS) = Field(
        default=20,
        description="Number of results"
    )
    
    offset: conint(ge=0) = Field(
        default=0,
        description="Result offset for pagination"
    )
    
    sort_by: Optional[str] = Field(
        default="relevance",
        pattern=r"^(relevance|date|score)$",
        description="Sort order"
    )


class AnalyticsRequestValidator(BaseModel):
    """Validated analytics request model."""
    
    start_date: Optional[datetime] = Field(
        None,
        description="Start date for analytics"
    )
    
    end_date: Optional[datetime] = Field(
        None,
        description="End date for analytics"
    )
    
    metrics: Optional[List[str]] = Field(
        default_factory=list,
        description="Specific metrics to retrieve"
    )
    
    group_by: Optional[str] = Field(
        None,
        pattern=r"^(hour|day|week|month)$",
        description="Grouping period"
    )
    
    @validator("end_date")
    def validate_date_range(cls, v, values):
        if v and "start_date" in values and values["start_date"]:
            if v < values["start_date"]:
                raise ValueError("End date must be after start date")
            
            # Maximum 90 days range
            delta = v - values["start_date"]
            if delta.days > 90:
                raise ValueError("Date range cannot exceed 90 days")
        
        return v
    
    @validator("metrics")
    def validate_metrics(cls, v):
        allowed_metrics = [
            "total_queries", "unique_users", "avg_response_time",
            "error_rate", "cache_hit_rate", "confidence_scores"
        ]
        
        if v:
            invalid = [m for m in v if m not in allowed_metrics]
            if invalid:
                raise ValueError(f"Invalid metrics: {invalid}")
        
        return v


class ConfigUpdateValidator(BaseModel):
    """Validated configuration update model."""
    
    key: Annotated[str, StringConstraints(pattern=r"^[a-zA-Z][a-zA-Z0-9_]*$")] = Field(
        ...,
        max_length=50,
        description="Configuration key"
    )
    
    value: Any = Field(
        ...,
        description="Configuration value"
    )
    
    description: Optional[str] = Field(
        None,
        max_length=200,
        description="Description of the change"
    )
    
    @validator("key")
    def validate_key(cls, v):
        # Prevent modification of critical keys
        protected_keys = [
            "SECRET_KEY", "DATABASE_URL", "ADMIN_PASSWORD",
            "API_KEY_SECRET", "ENCRYPTION_KEY"
        ]
        
        if v in protected_keys:
            raise ValueError(f"Cannot modify protected configuration key: {v}")
        
        return v
    
    @validator("value")
    def validate_value(cls, v, values):
        # Validate based on key type
        if "key" in values:
            key = values["key"]
            
            # Boolean keys
            if key.endswith("_ENABLED") or key.startswith("IS_"):
                if not isinstance(v, bool):
                    raise ValueError(f"{key} must be a boolean")
            
            # Numeric keys
            elif key.endswith("_LIMIT") or key.endswith("_TIMEOUT"):
                if not isinstance(v, (int, float)) or v < 0:
                    raise ValueError(f"{key} must be a positive number")
            
            # URL keys
            elif key.endswith("_URL"):
                url_pattern = re.compile(
                    r"^https?://[a-zA-Z0-9\-._~:/?#[\]@!$&'()*+,;=]+$"
                )
                if not isinstance(v, str) or not url_pattern.match(v):
                    raise ValueError(f"{key} must be a valid URL")
        
        return v


def validate_api_key(api_key: str) -> bool:
    """
    Validate API key format.
    
    Args:
        api_key: API key to validate
        
    Returns:
        True if valid, False otherwise
    """
    if not api_key:
        return False
    
    # API key should be 32-64 characters, alphanumeric with hyphens
    pattern = re.compile(r"^[a-zA-Z0-9\-]{32,64}$")
    return bool(pattern.match(api_key))


def validate_email(email: str) -> bool:
    """
    Validate email format.
    
    Args:
        email: Email to validate
        
    Returns:
        True if valid, False otherwise
    """
    pattern = re.compile(
        r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
    )
    return bool(pattern.match(email))


def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename to prevent directory traversal.
    
    Args:
        filename: Filename to sanitize
        
    Returns:
        Sanitized filename
    """
    # Remove any path components
    filename = filename.replace("..", "").replace("/", "").replace("\\", "")
    
    # Allow only alphanumeric, dots, hyphens, and underscores
    filename = re.sub(r"[^a-zA-Z0-9._\-]", "", filename)
    
    # Limit length
    return filename[:255] 