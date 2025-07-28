"""
API Versioning and Migration System - MAANG Standards.

This module implements comprehensive API versioning and migration
following MAANG best practices for API evolution and backward compatibility.

Features:
    - Semantic versioning support
    - Backward compatibility management
    - Migration strategies
    - Version deprecation handling
    - Feature flags per version
    - Migration guides
    - Breaking change detection
    - Version-specific documentation

Versioning Strategy:
    - URL-based versioning (/api/v1/, /api/v2/)
    - Header-based versioning (Accept: application/vnd.api+json;version=2)
    - Query parameter versioning (?version=2)
    - Automatic version detection
    - Graceful degradation

Migration Support:
    - Automatic data migration
    - Schema versioning
    - Breaking change notifications
    - Migration rollback support
    - Version compatibility matrix

Authors:
    - Universal Knowledge Platform Engineering Team
    
Version:
    2.0.0 (2024-12-28)
"""

import re
import json
import hashlib
from typing import (
    Optional, Dict, Any, List, Union, Callable,
    TypeVar, Protocol, Tuple, Set
)
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass, field
from enum import Enum
import structlog
from fastapi import Request, Response, HTTPException, Depends
from fastapi.routing import APIRoute
from pydantic import BaseModel, Field, validator

from api.exceptions import ValidationError, VersionError
from api.config import get_settings

logger = structlog.get_logger(__name__)

# Type definitions
T = TypeVar('T')

# Version types
class VersionType(str, Enum):
    """API version types."""
    MAJOR = "major"
    MINOR = "minor"
    PATCH = "patch"

class VersionStatus(str, Enum):
    """Version status."""
    ACTIVE = "active"
    DEPRECATED = "deprecated"
    SUNSET = "sunset"
    UNSUPPORTED = "unsupported"

# Version information
@dataclass
class APIVersion:
    """API version information."""
    
    version: str
    status: VersionStatus
    release_date: datetime
    sunset_date: Optional[datetime] = None
    breaking_changes: List[str] = field(default_factory=list)
    new_features: List[str] = field(default_factory=list)
    bug_fixes: List[str] = field(default_factory=list)
    migration_guide: Optional[str] = None
    documentation_url: Optional[str] = None
    
    def is_deprecated(self) -> bool:
        """Check if version is deprecated."""
        return self.status in [VersionStatus.DEPRECATED, VersionStatus.SUNSET]
    
    def is_supported(self) -> bool:
        """Check if version is still supported."""
        return self.status != VersionStatus.UNSUPPORTED
    
    def days_until_sunset(self) -> Optional[int]:
        """Get days until sunset date."""
        if not self.sunset_date:
            return None
        
        delta = self.sunset_date - datetime.now(timezone.utc)
        return max(0, delta.days)

# Version compatibility
@dataclass
class VersionCompatibility:
    """Version compatibility information."""
    
    from_version: str
    to_version: str
    compatible: bool
    migration_required: bool
    breaking_changes: List[str] = field(default_factory=list)
    migration_steps: List[str] = field(default_factory=list)
    estimated_migration_time: Optional[str] = None

# Feature flags per version
@dataclass
class FeatureFlag:
    """Feature flag configuration."""
    
    name: str
    description: str
    versions: Dict[str, bool] = field(default_factory=dict)
    default_value: bool = False
    deprecated_since: Optional[str] = None
    removed_since: Optional[str] = None

# Version manager
class VersionManager:
    """
    Comprehensive API version management system.
    
    Features:
    - Version registration and management
    - Compatibility checking
    - Migration handling
    - Feature flag management
    - Breaking change detection
    """
    
    def __init__(self):
        """Initialize version manager."""
        self.versions: Dict[str, APIVersion] = {}
        self.compatibility_matrix: Dict[str, Dict[str, VersionCompatibility]] = {}
        self.feature_flags: Dict[str, FeatureFlag] = {}
        self.migration_handlers: Dict[str, Callable] = {}
        
        # Register default versions
        self._register_default_versions()
    
    def _register_default_versions(self) -> None:
        """Register default API versions."""
        now = datetime.now(timezone.utc)
        
        # Version 1.0 (Legacy)
        self.register_version(
            APIVersion(
                version="v1",
                status=VersionStatus.DEPRECATED,
                release_date=now - timedelta(days=365),
                sunset_date=now + timedelta(days=180),
                breaking_changes=[
                    "Removed deprecated query format",
                    "Updated response schema",
                    "Changed authentication method"
                ],
                new_features=[
                    "Enhanced query processing",
                    "Improved error handling",
                    "Better performance"
                ],
                migration_guide="https://docs.example.com/migrate-v1-to-v2",
                documentation_url="https://docs.example.com/api/v1"
            )
        )
        
        # Version 2.0 (Current)
        self.register_version(
            APIVersion(
                version="v2",
                status=VersionStatus.ACTIVE,
                release_date=now - timedelta(days=30),
                new_features=[
                    "Streaming responses",
                    "Batch processing",
                    "Advanced analytics",
                    "Real-time collaboration"
                ],
                documentation_url="https://docs.example.com/api/v2"
            )
        )
        
        # Version 3.0 (Beta)
        self.register_version(
            APIVersion(
                version="v3",
                status=VersionStatus.ACTIVE,
                release_date=now,
                new_features=[
                    "GraphQL support",
                    "WebSocket subscriptions",
                    "Advanced caching",
                    "Machine learning features"
                ],
                documentation_url="https://docs.example.com/api/v3"
            )
        )
    
    def register_version(self, version: APIVersion) -> None:
        """Register a new API version."""
        self.versions[version.version] = version
        logger.info("API version registered", version=version.version, status=version.status)
    
    def get_version(self, version: str) -> Optional[APIVersion]:
        """Get version information."""
        return self.versions.get(version)
    
    def get_active_versions(self) -> List[APIVersion]:
        """Get all active versions."""
        return [v for v in self.versions.values() if v.status == VersionStatus.ACTIVE]
    
    def get_deprecated_versions(self) -> List[APIVersion]:
        """Get all deprecated versions."""
        return [v for v in self.versions.values() if v.is_deprecated()]
    
    def register_feature_flag(self, flag: FeatureFlag) -> None:
        """Register a feature flag."""
        self.feature_flags[flag.name] = flag
    
    def is_feature_enabled(self, feature: str, version: str) -> bool:
        """Check if a feature is enabled for a version."""
        flag = self.feature_flags.get(feature)
        if not flag:
            return False
        
        return flag.versions.get(version, flag.default_value)
    
    def register_migration_handler(
        self,
        from_version: str,
        to_version: str,
        handler: Callable
    ) -> None:
        """Register a migration handler."""
        key = f"{from_version}:{to_version}"
        self.migration_handlers[key] = handler
    
    def migrate_data(
        self,
        data: Dict[str, Any],
        from_version: str,
        to_version: str
    ) -> Dict[str, Any]:
        """Migrate data between versions."""
        key = f"{from_version}:{to_version}"
        handler = self.migration_handlers.get(key)
        
        if not handler:
            raise VersionError(f"No migration handler for {from_version} to {to_version}")
        
        try:
            return handler(data)
        except Exception as e:
            logger.error(
                "Migration failed",
                from_version=from_version,
                to_version=to_version,
                error=str(e)
            )
            raise VersionError(f"Migration failed: {str(e)}")
    
    def check_compatibility(
        self,
        from_version: str,
        to_version: str
    ) -> VersionCompatibility:
        """Check compatibility between versions."""
        key = f"{from_version}:{to_version}"
        
        if key in self.compatibility_matrix:
            return self.compatibility_matrix[key]
        
        # Default compatibility check
        from_ver = self.get_version(from_version)
        to_ver = self.get_version(to_version)
        
        if not from_ver or not to_ver:
            return VersionCompatibility(
                from_version=from_version,
                to_version=to_version,
                compatible=False,
                migration_required=True,
                breaking_changes=["Version not found"]
            )
        
        # Simple compatibility check
        compatible = from_ver.status != VersionStatus.UNSUPPORTED
        migration_required = from_version != to_version
        
        compatibility = VersionCompatibility(
            from_version=from_version,
            to_version=to_version,
            compatible=compatible,
            migration_required=migration_required,
            breaking_changes=to_ver.breaking_changes if migration_required else []
        )
        
        self.compatibility_matrix[key] = compatibility
        return compatibility

# Version detection
class VersionDetector:
    """Detect API version from request."""
    
    def __init__(self, default_version: str = "v2"):
        """Initialize version detector."""
        self.default_version = default_version
        self.version_pattern = re.compile(r'^v\d+$')
    
    def detect_version(self, request: Request) -> str:
        """Detect API version from request."""
        # Check URL path
        path_version = self._extract_version_from_path(request.url.path)
        if path_version:
            return path_version
        
        # Check Accept header
        accept_version = self._extract_version_from_header(request.headers.get("accept", ""))
        if accept_version:
            return accept_version
        
        # Check query parameter
        query_version = request.query_params.get("version")
        if query_version and self.version_pattern.match(query_version):
            return query_version
        
        # Check custom header
        custom_version = request.headers.get("X-API-Version")
        if custom_version and self.version_pattern.match(custom_version):
            return custom_version
        
        return self.default_version
    
    def _extract_version_from_path(self, path: str) -> Optional[str]:
        """Extract version from URL path."""
        parts = path.split("/")
        for part in parts:
            if self.version_pattern.match(part):
                return part
        return None
    
    def _extract_version_from_header(self, accept: str) -> Optional[str]:
        """Extract version from Accept header."""
        # Parse Accept header for version
        # Example: application/vnd.api+json;version=2
        version_match = re.search(r'version=(\d+)', accept)
        if version_match:
            return f"v{version_match.group(1)}"
        return None

# Version middleware
class VersionMiddleware:
    """Middleware for version handling."""
    
    def __init__(self, version_manager: VersionManager, version_detector: VersionDetector):
        """Initialize version middleware."""
        self.version_manager = version_manager
        self.version_detector = version_detector
    
    async def __call__(self, request: Request, call_next: Callable) -> Response:
        """Process request with version handling."""
        # Detect version
        version = self.version_detector.detect_version(request)
        
        # Check if version is supported
        version_info = self.version_manager.get_version(version)
        if not version_info or not version_info.is_supported():
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported API version: {version}"
            )
        
        # Add version info to request state
        request.state.api_version = version
        request.state.version_info = version_info
        
        # Add version headers to response
        response = await call_next(request)
        
        # Add version headers
        response.headers["X-API-Version"] = version
        response.headers["X-API-Status"] = version_info.status.value
        
        if version_info.is_deprecated():
            response.headers["X-API-Deprecated"] = "true"
            if version_info.sunset_date:
                response.headers["X-API-Sunset-Date"] = version_info.sunset_date.isoformat()
        
        return response

# Version-specific route handling
class VersionedRoute(APIRoute):
    """Route that handles version-specific logic."""
    
    def __init__(
        self,
        path: str,
        endpoint: Callable,
        version: str,
        **kwargs: Any
    ):
        """Initialize versioned route."""
        super().__init__(path, endpoint, **kwargs)
        self.version = version
    
    def get_route_handler(self) -> Callable:
        """Get route handler with version checking."""
        original_handler = super().get_route_handler()
        
        def versioned_handler(request: Request, *args: Any, **kwargs: Any) -> Any:
            # Check if request version matches route version
            detected_version = request.state.api_version
            if detected_version != self.version:
                raise HTTPException(
                    status_code=400,
                    detail=f"Route requires version {self.version}, got {detected_version}"
                )
            
            return original_handler(request, *args, **kwargs)
        
        return versioned_handler

# Version-specific models
class VersionedModel(BaseModel):
    """Base model with version support."""
    
    version: str = Field(..., description="API version")
    
    @validator('version')
    def validate_version(cls, v: str) -> str:
        """Validate version format."""
        if not re.match(r'^v\d+$', v):
            raise ValueError('Version must be in format v<number>')
        return v

# Migration utilities
class MigrationManager:
    """Manage data migrations between versions."""
    
    def __init__(self, version_manager: VersionManager):
        """Initialize migration manager."""
        self.version_manager = version_manager
        self.migration_history: List[Dict[str, Any]] = []
    
    def register_migration(
        self,
        from_version: str,
        to_version: str,
        migration_func: Callable,
        description: str
    ) -> None:
        """Register a migration function."""
        self.version_manager.register_migration_handler(from_version, to_version, migration_func)
        
        self.migration_history.append({
            'from_version': from_version,
            'to_version': to_version,
            'description': description,
            'registered_at': datetime.now(timezone.utc)
        })
    
    def migrate_request_data(
        self,
        data: Dict[str, Any],
        from_version: str,
        to_version: str
    ) -> Dict[str, Any]:
        """Migrate request data between versions."""
        if from_version == to_version:
            return data
        
        try:
            migrated_data = self.version_manager.migrate_data(data, from_version, to_version)
            logger.info(
                "Request data migrated",
                from_version=from_version,
                to_version=to_version
            )
            return migrated_data
        except Exception as e:
            logger.error(
                "Request migration failed",
                from_version=from_version,
                to_version=to_version,
                error=str(e)
            )
            raise
    
    def migrate_response_data(
        self,
        data: Dict[str, Any],
        from_version: str,
        to_version: str
    ) -> Dict[str, Any]:
        """Migrate response data between versions."""
        if from_version == to_version:
            return data
        
        try:
            migrated_data = self.version_manager.migrate_data(data, to_version, from_version)
            logger.info(
                "Response data migrated",
                from_version=from_version,
                to_version=to_version
            )
            return migrated_data
        except Exception as e:
            logger.error(
                "Response migration failed",
                from_version=from_version,
                to_version=to_version,
                error=str(e)
            )
            raise

# Version-specific dependencies
def get_api_version(request: Request) -> str:
    """Get API version from request."""
    return getattr(request.state, 'api_version', 'v2')

def get_version_info(request: Request) -> APIVersion:
    """Get version information from request."""
    return getattr(request.state, 'version_info')

def require_version(required_version: str):
    """Dependency to require specific API version."""
    def version_dependency(request: Request) -> str:
        current_version = get_api_version(request)
        if current_version != required_version:
            raise HTTPException(
                status_code=400,
                detail=f"Endpoint requires version {required_version}, got {current_version}"
            )
        return current_version
    
    return version_dependency

def require_min_version(min_version: str):
    """Dependency to require minimum API version."""
    def min_version_dependency(request: Request) -> str:
        current_version = get_api_version(request)
        current_num = int(current_version[1:])
        min_num = int(min_version[1:])
        
        if current_num < min_num:
            raise HTTPException(
                status_code=400,
                detail=f"Endpoint requires minimum version {min_version}, got {current_version}"
            )
        return current_version
    
    return min_version_dependency

# Global instances
_version_manager: Optional[VersionManager] = None
_version_detector: Optional[VersionDetector] = None
_migration_manager: Optional[MigrationManager] = None

def get_version_manager() -> VersionManager:
    """Get global version manager instance."""
    global _version_manager
    
    if _version_manager is None:
        _version_manager = VersionManager()
    
    return _version_manager

def get_version_detector() -> VersionDetector:
    """Get global version detector instance."""
    global _version_detector
    
    if _version_detector is None:
        _version_detector = VersionDetector()
    
    return _version_detector

def get_migration_manager() -> MigrationManager:
    """Get global migration manager instance."""
    global _migration_manager
    
    if _migration_manager is None:
        _migration_manager = MigrationManager(get_version_manager())
    
    return _migration_manager

# Migration functions
def migrate_v1_to_v2(data: Dict[str, Any]) -> Dict[str, Any]:
    """Migrate data from v1 to v2."""
    migrated = data.copy()
    
    # Handle breaking changes
    if 'query' in migrated:
        # v1 used 'text', v2 uses 'query'
        if 'text' in migrated:
            migrated['query'] = migrated.pop('text')
    
    # Add new fields
    if 'query' in migrated and 'max_tokens' not in migrated:
        migrated['max_tokens'] = 1000
    
    if 'confidence_threshold' not in migrated:
        migrated['confidence_threshold'] = 0.8
    
    return migrated

def migrate_v2_to_v1(data: Dict[str, Any]) -> Dict[str, Any]:
    """Migrate data from v2 to v1."""
    migrated = data.copy()
    
    # Handle breaking changes
    if 'query' in migrated:
        # v2 uses 'query', v1 used 'text'
        migrated['text'] = migrated.pop('query')
    
    # Remove new fields
    migrated.pop('max_tokens', None)
    migrated.pop('confidence_threshold', None)
    
    return migrated

# Initialize migrations
def initialize_migrations() -> None:
    """Initialize migration handlers."""
    migration_manager = get_migration_manager()
    
    # Register v1 to v2 migration
    migration_manager.register_migration(
        "v1", "v2",
        migrate_v1_to_v2,
        "Migrate from v1 to v2 API format"
    )
    
    # Register v2 to v1 migration
    migration_manager.register_migration(
        "v2", "v1",
        migrate_v2_to_v1,
        "Migrate from v2 to v1 API format"
    )

# Export public API
__all__ = [
    # Classes
    'VersionManager',
    'VersionDetector',
    'VersionMiddleware',
    'VersionedRoute',
    'VersionedModel',
    'MigrationManager',
    'APIVersion',
    'VersionCompatibility',
    'FeatureFlag',
    
    # Enums
    'VersionType',
    'VersionStatus',
    
    # Functions
    'get_version_manager',
    'get_version_detector',
    'get_migration_manager',
    'get_api_version',
    'get_version_info',
    'require_version',
    'require_min_version',
    'initialize_migrations',
    'migrate_v1_to_v2',
    'migrate_v2_to_v1',
] 