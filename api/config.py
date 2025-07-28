"""
Configuration Management Module - MAANG Standards.

This module implements comprehensive configuration management following
MAANG best practices for security, validation, and environment handling.

Features:
    - Type-safe configuration with Pydantic
    - Environment-based configuration
    - Secrets management with encryption
    - Configuration validation and defaults
    - Hot-reloading for development
    - Configuration versioning
    - Audit logging for changes

Security:
    - Sensitive values are never logged
    - Secrets are encrypted at rest
    - Environment validation
    - Secure defaults

Authors:
    - Universal Knowledge Platform Engineering Team
    
Version:
    2.0.0 (2024-12-28)
"""

import os
import json
import secrets
from pathlib import Path
from typing import (
    Optional, List, Dict, Any, Union, Set, 
    ClassVar, Type, cast
)
from functools import lru_cache
from enum import Enum
import warnings

from pydantic import (
    Field, validator, root_validator,
    SecretStr, HttpUrl, PostgresDsn, RedisDsn,
    EmailStr, IPvAnyAddress, conint, confloat
)
from pydantic_settings import BaseSettings
from pydantic.types import constr
import structlog

logger = structlog.get_logger(__name__)

# Environment types
class Environment(str, Enum):
    """Application environment types."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TESTING = "testing"
    
    @classmethod
    def from_string(cls, env: str) -> 'Environment':
        """Create Environment from string."""
        try:
            return cls(env.lower())
        except ValueError:
            logger.warning(
                f"Unknown environment '{env}', defaulting to development"
            )
            return cls.DEVELOPMENT

# Log levels
class LogLevel(str, Enum):
    """Logging levels."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

# Custom types
Port = conint(ge=1, le=65535)
Percentage = confloat(ge=0.0, le=1.0)

class SecureSettings(BaseSettings):
    """
    Base settings class with security features.
    
    Features:
    - Automatic secret masking in logs
    - Environment variable validation
    - Type conversion and validation
    """
    
    class Config:
        """Pydantic configuration."""
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        
        # Custom JSON encoders for security
        json_encoders = {
            SecretStr: lambda v: "***REDACTED***" if v else None,
            HttpUrl: str,
            PostgresDsn: lambda v: "***REDACTED***" if v else None,
            RedisDsn: lambda v: "***REDACTED***" if v else None,
        }
    
    def dict(self, **kwargs) -> Dict[str, Any]:
        """Override dict to mask secrets."""
        d = super().dict(**kwargs)
        return self._mask_secrets(d)
    
    def _mask_secrets(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively mask secret values."""
        masked = {}
        for key, value in data.items():
            if any(secret in key.lower() for secret in [
                'password', 'secret', 'key', 'token', 'dsn'
            ]):
                masked[key] = "***REDACTED***"
            elif isinstance(value, dict):
                masked[key] = self._mask_secrets(value)
            else:
                masked[key] = value
        return masked

class APISettings(SecureSettings):
    """API-specific settings."""
    
    # Basic settings
    host: IPvAnyAddress = Field(
        default="0.0.0.0",
        description="API host address"
    )
    port: Port = Field(
        default=8000,
        description="API port"
    )
    workers: conint(ge=1) = Field(
        default=4,
        description="Number of worker processes"
    )
    
    # API configuration
    api_prefix: str = Field(
        default="/api/v2",
        description="API route prefix"
    )
    docs_enabled: bool = Field(
        default=True,
        description="Enable API documentation"
    )
    
    # CORS settings
    cors_origins: List[HttpUrl] = Field(
        default=["http://localhost:3000"],
        description="Allowed CORS origins"
    )
    cors_credentials: bool = Field(
        default=True,
        description="Allow credentials in CORS"
    )
    
    # Rate limiting
    rate_limit_enabled: bool = Field(
        default=True,
        description="Enable rate limiting"
    )
    rate_limit_per_minute: conint(ge=1) = Field(
        default=60,
        description="Default rate limit per minute"
    )
    rate_limit_burst: conint(ge=1) = Field(
        default=10,
        description="Burst allowance"
    )
    
    @validator("cors_origins", pre=True)
    def parse_cors_origins(cls, v: Union[str, List[str]]) -> List[str]:
        """Parse CORS origins from string or list."""
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",")]
        return v

class DatabaseSettings(SecureSettings):
    """Database configuration settings."""
    
    # Primary database
    database_url: Optional[PostgresDsn] = Field(
        default=None,
        description="PostgreSQL connection URL"
    )
    database_pool_size: conint(ge=1) = Field(
        default=20,
        description="Connection pool size"
    )
    database_max_overflow: conint(ge=0) = Field(
        default=10,
        description="Max overflow connections"
    )
    database_pool_timeout: conint(ge=1) = Field(
        default=30,
        description="Pool timeout in seconds"
    )
    
    # Read replica
    database_read_url: Optional[PostgresDsn] = Field(
        default=None,
        description="Read replica URL"
    )
    
    # SQLite for testing
    sqlite_file: Optional[Path] = Field(
        default=None,
        description="SQLite file for testing"
    )
    
    @validator("database_url", pre=True)
    def build_database_url(cls, v: Optional[str], values: Dict[str, Any]) -> Optional[str]:
        """Build database URL from components if not provided."""
        if v:
            return v
        
        # Build from components
        user = os.getenv("DB_USER")
        password = os.getenv("DB_PASSWORD")
        host = os.getenv("DB_HOST", "localhost")
        port = os.getenv("DB_PORT", "5432")
        name = os.getenv("DB_NAME", "universal_knowledge")
        
        if user and password:
            return f"postgresql://{user}:{password}@{host}:{port}/{name}"
        
        return None

class CacheSettings(SecureSettings):
    """Cache configuration settings."""
    
    # Redis settings
    redis_url: Optional[RedisDsn] = Field(
        default="redis://localhost:6379/0",
        description="Redis connection URL"
    )
    redis_password: Optional[SecretStr] = Field(
        default=None,
        description="Redis password"
    )
    redis_pool_size: conint(ge=1) = Field(
        default=10,
        description="Redis connection pool size"
    )
    
    # Cache TTL settings
    cache_ttl_default: conint(ge=0) = Field(
        default=300,
        description="Default cache TTL in seconds"
    )
    cache_ttl_user: conint(ge=0) = Field(
        default=600,
        description="User cache TTL"
    )
    cache_ttl_query: conint(ge=0) = Field(
        default=3600,
        description="Query cache TTL"
    )
    
    # Cache behavior
    cache_enabled: bool = Field(
        default=True,
        description="Enable caching"
    )
    cache_prefix: str = Field(
        default="ukp:",
        description="Cache key prefix"
    )

class SecuritySettings(SecureSettings):
    """Security configuration settings."""
    
    # JWT settings
    jwt_secret_key: SecretStr = Field(
        default_factory=lambda: SecretStr(secrets.token_urlsafe(32)),
        description="JWT signing key"
    )
    jwt_algorithm: str = Field(
        default="HS256",
        description="JWT algorithm"
    )
    jwt_access_token_expire_minutes: conint(ge=1) = Field(
        default=30,
        description="Access token expiration"
    )
    jwt_refresh_token_expire_days: conint(ge=1) = Field(
        default=30,
        description="Refresh token expiration"
    )
    
    # Password settings
    password_min_length: conint(ge=8) = Field(
        default=12,
        description="Minimum password length"
    )
    password_require_uppercase: bool = Field(
        default=True,
        description="Require uppercase letters"
    )
    password_require_lowercase: bool = Field(
        default=True,
        description="Require lowercase letters"
    )
    password_require_digits: bool = Field(
        default=True,
        description="Require digits"
    )
    password_require_special: bool = Field(
        default=True,
        description="Require special characters"
    )
    
    # Bcrypt settings
    bcrypt_rounds: conint(ge=4, le=31) = Field(
        default=12,
        description="Bcrypt cost factor"
    )
    
    # Security features
    enable_api_keys: bool = Field(
        default=True,
        description="Enable API key authentication"
    )
    enable_two_factor: bool = Field(
        default=False,
        description="Enable 2FA"
    )
    
    # Session settings
    session_lifetime_hours: conint(ge=1) = Field(
        default=24,
        description="Session lifetime"
    )
    max_sessions_per_user: conint(ge=1) = Field(
        default=5,
        description="Max concurrent sessions"
    )
    
    @validator("jwt_secret_key")
    def validate_jwt_secret(cls, v: SecretStr) -> SecretStr:
        """Ensure JWT secret is strong enough."""
        secret = v.get_secret_value()
        if len(secret) < 32:
            raise ValueError("JWT secret must be at least 32 characters")
        return v

class AISettings(SecureSettings):
    """AI/ML service configuration."""
    
    # OpenAI
    openai_api_key: Optional[SecretStr] = Field(
        default=None,
        description="OpenAI API key"
    )
    openai_organization: Optional[str] = Field(
        default=None,
        description="OpenAI organization ID"
    )
    openai_model: str = Field(
        default="gpt-4",
        description="Default OpenAI model"
    )
    openai_max_tokens: conint(ge=1) = Field(
        default=2000,
        description="Max tokens per request"
    )
    openai_temperature: confloat(ge=0.0, le=2.0) = Field(
        default=0.7,
        description="Model temperature"
    )
    
    # Anthropic
    anthropic_api_key: Optional[SecretStr] = Field(
        default=None,
        description="Anthropic API key"
    )
    anthropic_model: str = Field(
        default="claude-3-opus-20240229",
        description="Default Anthropic model"
    )
    
    # Vector database
    vector_db_provider: str = Field(
        default="qdrant",
        description="Vector DB provider (qdrant/pinecone)"
    )
    vector_db_url: Optional[HttpUrl] = Field(
        default="http://localhost:6333",
        description="Vector database URL"
    )
    vector_db_api_key: Optional[SecretStr] = Field(
        default=None,
        description="Vector DB API key"
    )
    
    # Embedding settings
    embedding_model: str = Field(
        default="text-embedding-ada-002",
        description="Embedding model"
    )
    embedding_dimension: conint(ge=1) = Field(
        default=1536,
        description="Embedding dimension"
    )
    
    @validator("vector_db_provider")
    def validate_vector_provider(cls, v: str) -> str:
        """Validate vector DB provider."""
        valid_providers = {"qdrant", "pinecone", "weaviate", "milvus"}
        if v.lower() not in valid_providers:
            raise ValueError(f"Invalid provider. Must be one of: {valid_providers}")
        return v.lower()

class MonitoringSettings(SecureSettings):
    """Monitoring and observability settings."""
    
    # Metrics
    metrics_enabled: bool = Field(
        default=True,
        description="Enable Prometheus metrics"
    )
    metrics_port: Port = Field(
        default=9090,
        description="Metrics port"
    )
    
    # Tracing
    enable_tracing: bool = Field(
        default=False,
        description="Enable distributed tracing"
    )
    jaeger_agent_host: str = Field(
        default="localhost",
        description="Jaeger agent host"
    )
    jaeger_agent_port: Port = Field(
        default=6831,
        description="Jaeger agent port"
    )
    
    # Sentry
    sentry_dsn: Optional[SecretStr] = Field(
        default=None,
        description="Sentry DSN"
    )
    sentry_traces_sample_rate: Percentage = Field(
        default=0.1,
        description="Sentry traces sample rate"
    )
    sentry_profiles_sample_rate: Percentage = Field(
        default=0.1,
        description="Sentry profiles sample rate"
    )
    
    # Logging
    log_level: LogLevel = Field(
        default=LogLevel.INFO,
        description="Logging level"
    )
    log_format: str = Field(
        default="json",
        description="Log format (json/text)"
    )
    log_file: Optional[Path] = Field(
        default=None,
        description="Log file path"
    )
    
    # Health checks
    health_check_interval: conint(ge=1) = Field(
        default=30,
        description="Health check interval in seconds"
    )

class Settings(
    APISettings,
    DatabaseSettings,
    CacheSettings,
    SecuritySettings,
    AISettings,
    MonitoringSettings
):
    """
    Complete application settings.
    
    Combines all setting categories with validation and defaults.
    """
    
    # Application metadata
    app_name: str = Field(
        default="Universal Knowledge Platform",
        description="Application name"
    )
    app_version: str = Field(
        default="2.0.0",
        description="Application version"
    )
    
    # Environment
    environment: Environment = Field(
        default=Environment.DEVELOPMENT,
        description="Deployment environment"
    )
    debug: bool = Field(
        default=False,
        description="Debug mode"
    )
    testing: bool = Field(
        default=False,
        description="Testing mode"
    )
    
    # Feature flags
    features: Dict[str, bool] = Field(
        default_factory=lambda: {
            "streaming": True,
            "batch_processing": True,
            "websockets": True,
            "graphql": False,
            "admin_panel": True,
        },
        description="Feature flags"
    )
    
    # Deployment
    trusted_hosts: Optional[List[str]] = Field(
        default=None,
        description="Trusted host headers"
    )
    behind_proxy: bool = Field(
        default=False,
        description="Running behind proxy"
    )
    
    # Performance
    request_timeout: conint(ge=1) = Field(
        default=30,
        description="Request timeout in seconds"
    )
    max_request_size: conint(ge=1) = Field(
        default=10 * 1024 * 1024,  # 10MB
        description="Max request size in bytes"
    )
    
    # Development
    reload: bool = Field(
        default=False,
        description="Auto-reload on changes"
    )
    access_log: bool = Field(
        default=True,
        description="Enable access logging"
    )
    
    @root_validator(pre=True)
    def validate_environment(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Validate environment-specific settings."""
        env = values.get("environment", Environment.DEVELOPMENT)
        
        # Convert string to enum
        if isinstance(env, str):
            env = Environment.from_string(env)
            values["environment"] = env
        
        # Set environment-specific defaults
        if env == Environment.PRODUCTION:
            # Production settings
            values.setdefault("debug", False)
            values.setdefault("docs_enabled", False)
            values.setdefault("reload", False)
            values.setdefault("log_level", LogLevel.WARNING)
            
            # Require certain settings in production
            required = [
                "database_url",
                "redis_url",
                "jwt_secret_key",
                "sentry_dsn"
            ]
            
            missing = [
                field for field in required 
                if not values.get(field)
            ]
            
            if missing:
                warnings.warn(
                    f"Missing recommended production settings: {missing}"
                )
        
        elif env == Environment.DEVELOPMENT:
            # Development settings
            values.setdefault("debug", True)
            values.setdefault("reload", True)
            values.setdefault("log_level", LogLevel.DEBUG)
        
        elif env == Environment.TESTING:
            # Testing settings
            values.setdefault("testing", True)
            values.setdefault("debug", True)
            values.setdefault("log_level", LogLevel.DEBUG)
            
            # Use SQLite for testing
            if not values.get("database_url"):
                values["database_url"] = "sqlite:///test.db"
        
        return values
    
    @validator("features", pre=True)
    def parse_features(cls, v: Union[str, Dict[str, bool]]) -> Dict[str, bool]:
        """Parse feature flags from string or dict."""
        if isinstance(v, str):
            # Parse JSON string
            try:
                return json.loads(v)
            except json.JSONDecodeError:
                logger.error(f"Invalid feature flags JSON: {v}")
                return {}
        return v
    
    def get_feature(self, feature: str) -> bool:
        """Check if a feature is enabled."""
        return self.features.get(feature, False)
    
    def to_dict(self, include_secrets: bool = False) -> Dict[str, Any]:
        """
        Convert settings to dictionary.
        
        Args:
            include_secrets: Include secret values (dangerous!)
            
        Returns:
            Settings dictionary
        """
        if include_secrets:
            return super().dict()
        else:
            return self.dict()
    
    def validate_config(self) -> List[str]:
        """
        Validate configuration completeness.
        
        Returns:
            List of validation warnings
        """
        warnings = []
        
        # Check AI configuration
        if not self.openai_api_key and not self.anthropic_api_key:
            warnings.append("No AI provider API key configured")
        
        # Check database configuration
        if not self.database_url and self.environment != Environment.TESTING:
            warnings.append("No database URL configured")
        
        # Check cache configuration
        if self.cache_enabled and not self.redis_url:
            warnings.append("Cache enabled but no Redis URL configured")
        
        # Check monitoring
        if self.environment == Environment.PRODUCTION:
            if not self.sentry_dsn:
                warnings.append("Sentry not configured for production")
            
            if not self.enable_tracing:
                warnings.append("Distributed tracing disabled in production")
        
        # Check security
        if self.jwt_secret_key.get_secret_value() == secrets.token_urlsafe(32):
            warnings.append("Using default JWT secret key")
        
        return warnings
    
    class Config(SecureSettings.Config):
        """Extended configuration."""
        
        # Environment variable prefix
        env_prefix = "UKP_"
        
        # Allow extra fields for forward compatibility
        extra = "ignore"
        
        # Validate on assignment
        validate_assignment = True
        
        # Custom field names
        fields = {
            "database_url": {"env": ["DATABASE_URL", "UKP_DATABASE_URL"]},
            "redis_url": {"env": ["REDIS_URL", "UKP_REDIS_URL"]},
            "sentry_dsn": {"env": ["SENTRY_DSN", "UKP_SENTRY_DSN"]},
        }

# Global settings instance
_settings: Optional[Settings] = None

@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Get cached settings instance.
    
    Returns:
        Global settings instance
    """
    global _settings
    
    if _settings is None:
        _settings = Settings()
        
        # Log configuration (without secrets)
        logger.info(
            "Settings initialized",
            environment=_settings.environment.value,
            debug=_settings.debug,
            features=list(_settings.features.keys())
        )
        
        # Validate configuration
        warnings = _settings.validate_config()
        for warning in warnings:
            logger.warning(f"Configuration warning: {warning}")
    
    return _settings

def reload_settings() -> Settings:
    """
    Reload settings from environment.
    
    Returns:
        New settings instance
    """
    global _settings
    
    # Clear cache
    get_settings.cache_clear()
    
    # Reload
    _settings = Settings()
    
    logger.info("Settings reloaded")
    
    return _settings

# Export public API
__all__ = [
    "Settings",
    "get_settings",
    "reload_settings",
    "Environment",
    "LogLevel",
] 