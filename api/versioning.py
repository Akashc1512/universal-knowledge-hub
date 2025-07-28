"""
API Versioning Module for Universal Knowledge Platform
Implements URL-based versioning with backward compatibility.
"""

import logging
from typing import Optional, Dict, Any, Callable
from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import JSONResponse
from functools import wraps
from datetime import datetime

logger = logging.getLogger(__name__)

# Version configuration
API_VERSIONS = {
    "v1": {
        "status": "stable",
        "deprecated": False,
        "sunset_date": None,
        "description": "Initial stable API version"
    },
    "v2": {
        "status": "beta",
        "deprecated": False,
        "sunset_date": None,
        "description": "Enhanced API with additional features"
    }
}

CURRENT_VERSION = "v1"
DEFAULT_VERSION = "v1"


class APIVersion:
    """API version handler."""
    
    def __init__(self, version: str):
        self.version = version
        self.router = APIRouter(prefix=f"/api/{version}")
        
    def is_deprecated(self) -> bool:
        """Check if this version is deprecated."""
        return API_VERSIONS.get(self.version, {}).get("deprecated", False)
    
    def get_sunset_date(self) -> Optional[datetime]:
        """Get sunset date for deprecated version."""
        sunset = API_VERSIONS.get(self.version, {}).get("sunset_date")
        return datetime.fromisoformat(sunset) if sunset else None


def version_deprecated(sunset_date: Optional[str] = None):
    """
    Decorator to mark an endpoint as deprecated.
    
    Args:
        sunset_date: ISO format date when the endpoint will be removed
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            request = kwargs.get("request") or args[0] if args else None
            
            if request and isinstance(request, Request):
                # Add deprecation headers
                headers = {
                    "X-API-Deprecated": "true",
                    "X-API-Deprecation-Date": datetime.now().isoformat()
                }
                
                if sunset_date:
                    headers["X-API-Sunset-Date"] = sunset_date
                
                # Log deprecation warning
                logger.warning(
                    f"Deprecated endpoint called: {request.url.path} "
                    f"by {request.client.host if request.client else 'unknown'}"
                )
            
            result = await func(*args, **kwargs)
            
            # Add headers to response if it's a Response object
            if hasattr(result, "headers"):
                for key, value in headers.items():
                    result.headers[key] = value
            
            return result
            
        return wrapper
    return decorator


def require_version(min_version: str, max_version: Optional[str] = None):
    """
    Decorator to require specific API version range.
    
    Args:
        min_version: Minimum required version
        max_version: Maximum allowed version (optional)
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            request = kwargs.get("request") or args[0] if args else None
            
            if request and isinstance(request, Request):
                # Extract version from path
                path_parts = request.url.path.split("/")
                version = None
                
                if "api" in path_parts:
                    api_index = path_parts.index("api")
                    if api_index + 1 < len(path_parts):
                        version = path_parts[api_index + 1]
                
                if not version:
                    version = DEFAULT_VERSION
                
                # Check version requirements
                if version < min_version:
                    raise HTTPException(
                        status_code=400,
                        detail=f"This endpoint requires API version {min_version} or higher"
                    )
                
                if max_version and version > max_version:
                    raise HTTPException(
                        status_code=400,
                        detail=f"This endpoint is not available in API version {version}"
                    )
            
            return await func(*args, **kwargs)
            
        return wrapper
    return decorator


class VersionedResponse:
    """Helper class for version-specific response formatting."""
    
    @staticmethod
    def format_response(data: Dict[str, Any], version: str) -> Dict[str, Any]:
        """
        Format response based on API version.
        
        Args:
            data: Response data
            version: API version
            
        Returns:
            Formatted response
        """
        # Base response format
        response = {
            "api_version": version,
            "timestamp": datetime.utcnow().isoformat(),
            "data": data
        }
        
        # Version-specific formatting
        if version == "v1":
            # V1 uses flat structure for backward compatibility
            if "error" in data:
                response = {
                    "error": data["error"],
                    "message": data.get("message", ""),
                    "request_id": data.get("request_id", "")
                }
            else:
                response = data
                
        elif version == "v2":
            # V2 uses consistent envelope structure
            response["status"] = "success" if "error" not in data else "error"
            
            # Add metadata
            response["metadata"] = {
                "version": version,
                "deprecated": API_VERSIONS.get(version, {}).get("deprecated", False)
            }
        
        return response


def create_versioned_app(app):
    """
    Create versioned routes for the application.
    
    Args:
        app: FastAPI application instance
    """
    # Import endpoint modules
    from api.endpoints_v1 import router as v1_router
    from api.endpoints_v2 import router as v2_router
    
    # Add version routers
    app.include_router(v1_router, prefix="/api/v1", tags=["v1"])
    app.include_router(v2_router, prefix="/api/v2", tags=["v2"])
    
    # Add version discovery endpoint
    @app.get("/api/versions")
    async def get_api_versions():
        """Get available API versions and their status."""
        return {
            "current_version": CURRENT_VERSION,
            "default_version": DEFAULT_VERSION,
            "versions": API_VERSIONS
        }
    
    # Add root API endpoint with version redirect
    @app.get("/api")
    async def api_root():
        """API root endpoint with version information."""
        return {
            "message": "Universal Knowledge Platform API",
            "current_version": CURRENT_VERSION,
            "available_versions": list(API_VERSIONS.keys()),
            "documentation": {
                "v1": "/api/v1/docs",
                "v2": "/api/v2/docs"
            }
        }


def version_middleware(app):
    """
    Middleware to handle version-specific logic.
    
    Args:
        app: FastAPI application instance
    """
    @app.middleware("http")
    async def version_handler(request: Request, call_next):
        # Extract version from path
        path_parts = request.url.path.split("/")
        version = DEFAULT_VERSION
        
        if "api" in path_parts:
            api_index = path_parts.index("api")
            if api_index + 1 < len(path_parts) and path_parts[api_index + 1].startswith("v"):
                version = path_parts[api_index + 1]
        
        # Check if version exists
        if version not in API_VERSIONS:
            return JSONResponse(
                status_code=404,
                content={
                    "error": "API version not found",
                    "available_versions": list(API_VERSIONS.keys())
                }
            )
        
        # Check if version is deprecated
        if API_VERSIONS[version]["deprecated"]:
            sunset_date = API_VERSIONS[version].get("sunset_date")
            
            # Add deprecation warning to response
            response = await call_next(request)
            response.headers["X-API-Deprecated"] = "true"
            
            if sunset_date:
                response.headers["X-API-Sunset"] = sunset_date
                
                # Check if past sunset date
                if datetime.now() > datetime.fromisoformat(sunset_date):
                    return JSONResponse(
                        status_code=410,
                        content={
                            "error": "API version has been sunset",
                            "message": f"Version {version} is no longer available",
                            "current_version": CURRENT_VERSION
                        }
                    )
            
            return response
        
        # Process request normally
        response = await call_next(request)
        
        # Add version header
        response.headers["X-API-Version"] = version
        
        return response


# Version-specific feature flags
FEATURE_FLAGS = {
    "v1": {
        "advanced_analytics": False,
        "multi_language": False,
        "streaming_responses": False,
        "batch_processing": False
    },
    "v2": {
        "advanced_analytics": True,
        "multi_language": True,
        "streaming_responses": True,
        "batch_processing": True
    }
}


def get_feature_flag(version: str, feature: str) -> bool:
    """
    Check if a feature is enabled for a specific version.
    
    Args:
        version: API version
        feature: Feature name
        
    Returns:
        True if feature is enabled
    """
    return FEATURE_FLAGS.get(version, {}).get(feature, False) 