"""
API Documentation Generator - MAANG Standards.

This module generates comprehensive API documentation following MAANG
best practices for developer experience and API discoverability.

Features:
    - OpenAPI 3.0 specification
    - Interactive API documentation
    - Code examples in multiple languages
    - Response schemas and examples
    - Error documentation
    - Authentication guides
    - Rate limiting documentation
    - Versioning information

Documentation Standards:
    - Clear endpoint descriptions
    - Request/response examples
    - Error code documentation
    - Authentication requirements
    - Rate limiting details
    - SDK examples

Authors:
    - Universal Knowledge Platform Engineering Team
    
Version:
    2.0.0 (2024-12-28)
"""

from typing import Dict, Any, List, Optional
from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi
from pydantic import BaseModel, Field
from enum import Enum
import json
import yaml
import logging

from api.config import get_settings
from api.models import (
    QueryRequest, QueryResponse, FeedbackRequest, FeedbackResponse,
    UserCreate, UserResponse, Token
)

# API documentation configuration
class APIVersion(str, Enum):
    """API versions."""
    V1 = "v1"
    V2 = "v2"

class DocumentationConfig:
    """Configuration for API documentation."""
    
    def __init__(self):
        self.settings = get_settings()
        
        # API metadata
        self.title = "Universal Knowledge Platform API"
        self.description = """
        # Universal Knowledge Platform API

        A comprehensive API for intelligent knowledge retrieval and processing.

        ## Features

        - **Intelligent Query Processing**: Advanced AI-powered query understanding
        - **Multi-Source Knowledge**: Integration with multiple knowledge bases
        - **Real-time Responses**: Fast, accurate responses with source citations
        - **User Management**: Secure authentication and user management
        - **Analytics**: Comprehensive usage analytics and insights
        - **Rate Limiting**: Fair usage policies with rate limiting
        - **Security**: Enterprise-grade security with threat detection

        ## Authentication

        The API uses JWT (JSON Web Token) authentication. Include the token in the Authorization header:

        ```
        Authorization: Bearer <your-token>
        ```

        ## Rate Limiting

        - **Standard Users**: 60 requests per minute
        - **Premium Users**: 200 requests per minute
        - **Enterprise Users**: 1000 requests per minute

        Rate limit headers are included in all responses:
        - `X-RateLimit-Limit`: Maximum requests per window
        - `X-RateLimit-Remaining`: Remaining requests in current window
        - `X-RateLimit-Reset`: Time when the rate limit resets

        ## Error Handling

        The API returns standard HTTP status codes and detailed error messages:

        - `400 Bad Request`: Invalid request data
        - `401 Unauthorized`: Invalid or missing authentication
        - `403 Forbidden`: Insufficient permissions
        - `404 Not Found`: Resource not found
        - `429 Too Many Requests`: Rate limit exceeded
        - `500 Internal Server Error`: Server error

        ## SDKs and Libraries

        Official SDKs are available for:
        - [Python](https://github.com/universal-knowledge-platform/python-sdk)
        - [JavaScript](https://github.com/universal-knowledge-platform/js-sdk)
        - [Java](https://github.com/universal-knowledge-platform/java-sdk)
        - [Go](https://github.com/universal-knowledge-platform/go-sdk)

        ## Support

        - **Documentation**: [https://docs.universal-knowledge-platform.com](https://docs.universal-knowledge-platform.com)
        - **API Status**: [https://status.universal-knowledge-platform.com](https://status.universal-knowledge-platform.com)
        - **Support**: [support@universal-knowledge-platform.com](mailto:support@universal-knowledge-platform.com)
        """
        
        self.version = "2.0.0"
        self.contact = {
            "name": "Universal Knowledge Platform Support",
            "email": "support@universal-knowledge-platform.com",
            "url": "https://universal-knowledge-platform.com/support"
        }
        
        self.license_info = {
            "name": "MIT License",
            "url": "https://opensource.org/licenses/MIT"
        }
        
        # Server configurations
        self.servers = [
            {
                "url": "https://api.universal-knowledge-platform.com",
                "description": "Production server"
            },
            {
                "url": "https://api-staging.universal-knowledge-platform.com",
                "description": "Staging server"
            },
            {
                "url": "http://localhost:8000",
                "description": "Local development server"
            }
        ]
        
        # Security schemes
        self.security_schemes = {
            "bearerAuth": {
                "type": "http",
                "scheme": "bearer",
                "bearerFormat": "JWT",
                "description": "JWT token for API authentication"
            }
        }
        
        # Tags for API organization
        self.tags = [
            {
                "name": "Authentication",
                "description": "User authentication and authorization endpoints"
            },
            {
                "name": "Queries",
                "description": "Knowledge query processing and retrieval"
            },
            {
                "name": "Feedback",
                "description": "User feedback and rating endpoints"
            },
            {
                "name": "Analytics",
                "description": "Usage analytics and insights"
            },
            {
                "name": "Users",
                "description": "User management endpoints"
            },
            {
                "name": "Health",
                "description": "System health and monitoring endpoints"
            }
        ]

# Enhanced request/response models for documentation
class QueryRequestDoc(QueryRequest):
    """Enhanced query request model with documentation."""
    
    class Config:
        schema_extra = {
            "example": {
                "query": "What is machine learning and how does it work?",
                "max_tokens": 1000,
                "confidence_threshold": 0.8,
                "search_type": "comprehensive",
                "language": "en",
                "include_sources": True,
                "metadata": {
                    "user_id": "user123",
                    "session_id": "session456",
                    "client_version": "2.0.0"
                }
            }
        }

class QueryResponseDoc(QueryResponse):
    """Enhanced query response model with documentation."""
    
    class Config:
        schema_extra = {
            "example": {
                "result": "Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed. It works by training algorithms on large datasets to identify patterns and make predictions or decisions.",
                "sources": [
                    {
                        "title": "Introduction to Machine Learning",
                        "url": "https://example.com/ml-intro",
                        "author": "Dr. John Smith",
                        "publication_date": "2023-01-15",
                        "relevance_score": 0.95
                    }
                ],
                "confidence": 0.92,
                "processing_time": 1.23,
                "tokens_used": 150,
                "query_id": "q_123456789"
            }
        }

class ErrorResponse(BaseModel):
    """Standard error response model."""
    
    error: Dict[str, Any] = Field(
        ...,
        description="Error details",
        example={
            "code": "E1001",
            "message": "Invalid input provided",
            "details": {
                "field": "query",
                "reason": "Query cannot be empty"
            },
            "correlation_id": "req_123456789"
        }
    )

class RateLimitResponse(BaseModel):
    """Rate limit error response."""
    
    error: Dict[str, Any] = Field(
        ...,
        description="Rate limit error details",
        example={
            "code": "E5001",
            "message": "Rate limit exceeded",
            "retry_after": 60,
            "limit": 60,
            "window": "minute"
        }
    )

# Documentation generator
class APIDocumentationGenerator:
    """
    Generates comprehensive API documentation.
    
    Features:
    - OpenAPI 3.0 specification
    - Interactive documentation
    - Code examples
    - Error documentation
    """
    
    def __init__(self, app: FastAPI):
        """Initialize documentation generator."""
        self.app = app
        self.config = DocumentationConfig()
    
    def generate_openapi_schema(self) -> Dict[str, Any]:
        """Generate enhanced OpenAPI schema."""
        if self.app.openapi_schema:
            return self.app.openapi_schema
        
        openapi_schema = get_openapi(
            title=self.config.title,
            version=self.config.version,
            description=self.config.description,
            routes=self.app.routes,
            servers=self.config.servers,
            tags=self.config.tags,
            contact=self.config.contact,
            license_info=self.config.license_info
        )
        
        # Add security schemes
        openapi_schema["components"]["securitySchemes"] = self.config.security_schemes
        
        # Add enhanced examples
        self._add_examples(openapi_schema)
        
        # Add error responses
        self._add_error_responses(openapi_schema)
        
        # Add rate limiting documentation
        self._add_rate_limiting_docs(openapi_schema)
        
        self.app.openapi_schema = openapi_schema
        return openapi_schema
    
    def _add_examples(self, schema: Dict[str, Any]) -> None:
        """Add comprehensive examples to schema."""
        # Query endpoint examples
        schema["paths"]["/api/v2/query"]["post"]["requestBody"]["content"]["application/json"]["examples"] = {
            "simple_query": {
                "summary": "Simple knowledge query",
                "value": {
                    "query": "What is artificial intelligence?",
                    "max_tokens": 500,
                    "confidence_threshold": 0.8,
                    "include_sources": True
                }
            },
            "complex_query": {
                "summary": "Complex query with metadata",
                "value": {
                    "query": "Explain the differences between supervised and unsupervised learning with examples",
                    "max_tokens": 1000,
                    "confidence_threshold": 0.9,
                    "search_type": "comprehensive",
                    "language": "en",
                    "include_sources": True,
                    "metadata": {
                        "user_id": "user123",
                        "session_id": "session456",
                        "client_version": "2.0.0"
                    }
                }
            }
        }
        
        # Authentication examples
        schema["paths"]["/auth/login"]["post"]["requestBody"]["content"]["application/x-www-form-urlencoded"]["examples"] = {
            "user_login": {
                "summary": "User login",
                "value": {
                    "username": "user@example.com",
                    "password": "securepassword123"
                }
            }
        }
    
    def _add_error_responses(self, schema: Dict[str, Any]) -> None:
        """Add comprehensive error response documentation."""
        error_responses = {
            "400": {
                "description": "Bad Request",
                "content": {
                    "application/json": {
                        "schema": {"$ref": "#/components/schemas/ErrorResponse"},
                        "examples": {
                            "validation_error": {
                                "summary": "Validation error",
                                "value": {
                                    "error": {
                                        "code": "E1001",
                                        "message": "Invalid input provided",
                                        "details": {
                                            "field": "query",
                                            "reason": "Query cannot be empty"
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            },
            "401": {
                "description": "Unauthorized",
                "content": {
                    "application/json": {
                        "schema": {"$ref": "#/components/schemas/ErrorResponse"},
                        "examples": {
                            "invalid_token": {
                                "summary": "Invalid authentication token",
                                "value": {
                                    "error": {
                                        "code": "E2001",
                                        "message": "Invalid or expired token",
                                        "details": {
                                            "reason": "Token validation failed"
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            },
            "403": {
                "description": "Forbidden",
                "content": {
                    "application/json": {
                        "schema": {"$ref": "#/components/schemas/ErrorResponse"},
                        "examples": {
                            "insufficient_permissions": {
                                "summary": "Insufficient permissions",
                                "value": {
                                    "error": {
                                        "code": "E3001",
                                        "message": "Insufficient permissions",
                                        "details": {
                                            "required_permission": "admin",
                                            "user_permissions": ["user"]
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            },
            "429": {
                "description": "Too Many Requests",
                "content": {
                    "application/json": {
                        "schema": {"$ref": "#/components/schemas/RateLimitResponse"},
                        "examples": {
                            "rate_limit_exceeded": {
                                "summary": "Rate limit exceeded",
                                "value": {
                                    "error": {
                                        "code": "E5001",
                                        "message": "Rate limit exceeded",
                                        "retry_after": 60,
                                        "limit": 60,
                                        "window": "minute"
                                    }
                                }
                            }
                        }
                    }
                }
            },
            "500": {
                "description": "Internal Server Error",
                "content": {
                    "application/json": {
                        "schema": {"$ref": "#/components/schemas/ErrorResponse"},
                        "examples": {
                            "internal_error": {
                                "summary": "Internal server error",
                                "value": {
                                    "error": {
                                        "code": "E9001",
                                        "message": "Internal server error",
                                        "correlation_id": "req_123456789"
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        
        # Add error responses to all endpoints
        for path in schema["paths"].values():
            for method in path.values():
                if isinstance(method, dict) and "responses" in method:
                    method["responses"].update(error_responses)
    
    def _add_rate_limiting_docs(self, schema: Dict[str, Any]) -> None:
        """Add rate limiting documentation."""
        rate_limit_headers = {
            "X-RateLimit-Limit": {
                "description": "Maximum requests per window",
                "schema": {"type": "integer", "example": 60}
            },
            "X-RateLimit-Remaining": {
                "description": "Remaining requests in current window",
                "schema": {"type": "integer", "example": 45}
            },
            "X-RateLimit-Reset": {
                "description": "Time when rate limit resets (Unix timestamp)",
                "schema": {"type": "integer", "example": 1640995200}
            }
        }
        
        # Add headers to all endpoints
        for path in schema["paths"].values():
            for method in path.values():
                if isinstance(method, dict) and "responses" in method:
                    for response in method["responses"].values():
                        if "headers" not in response:
                            response["headers"] = {}
                        response["headers"].update(rate_limit_headers)
    
    def generate_markdown_docs(self) -> str:
        """Generate markdown documentation."""
        schema = self.generate_openapi_schema()
        
        markdown = f"""# {self.config.title}

{self.config.description}

## API Reference

### Authentication

All API requests require authentication using JWT tokens.

```bash
curl -H "Authorization: Bearer <your-token>" \\
     https://api.universal-knowledge-platform.com/api/v2/query
```

### Endpoints

"""
        
        # Generate endpoint documentation
        for path, path_item in schema["paths"].items():
            for method, operation in path_item.items():
                if method.upper() in ["GET", "POST", "PUT", "DELETE", "PATCH"]:
                    markdown += self._generate_endpoint_doc(path, method, operation)
        
        return markdown
    
    def _generate_endpoint_doc(self, path: str, method: str, operation: Dict[str, Any]) -> str:
        """Generate documentation for a single endpoint."""
        doc = f"""
### {operation.get('summary', f'{method.upper()} {path}')}

**Endpoint:** `{method.upper()} {path}`

{operation.get('description', '')}

"""
        
        # Add parameters
        if "parameters" in operation:
            doc += "**Parameters:**\n\n"
            for param in operation["parameters"]:
                doc += f"- `{param['name']}` ({param.get('in', 'query')}) - {param.get('description', '')}\n"
            doc += "\n"
        
        # Add request body
        if "requestBody" in operation:
            doc += "**Request Body:**\n\n"
            doc += "```json\n"
            examples = operation["requestBody"]["content"]["application/json"].get("examples", {})
            if examples:
                first_example = list(examples.values())[0]
                doc += json.dumps(first_example["value"], indent=2)
            doc += "\n```\n\n"
        
        # Add responses
        doc += "**Responses:**\n\n"
        for status_code, response in operation["responses"].items():
            doc += f"- `{status_code}` - {response.get('description', '')}\n"
        
        doc += "\n"
        return doc
    
    def generate_sdk_examples(self) -> Dict[str, str]:
        """Generate SDK code examples."""
        examples = {
            "python": """
import requests

# Initialize client
api_key = "your-api-key"
base_url = "https://api.universal-knowledge-platform.com"

headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}

# Make a query
response = requests.post(
    f"{base_url}/api/v2/query",
    headers=headers,
    json={
        "query": "What is machine learning?",
        "max_tokens": 1000,
        "include_sources": True
    }
)

if response.status_code == 200:
    result = response.json()
    logger.info(f"Query successful - Answer: {result['result'][:100]}...")
    logger.debug(f"Query confidence: {result['confidence']}")
else:
    error_details = response.json() if response.content else {"error": "Unknown error"}
    logger.error(f"Query failed with status {response.status_code}: {error_details}")
""",
            "javascript": """
const axios = require('axios');

// Initialize client
const apiKey = 'your-api-key';
const baseUrl = 'https://api.universal-knowledge-platform.com';

const headers = {
    'Authorization': `Bearer ${apiKey}`,
    'Content-Type': 'application/json'
};

// Make a query
async function queryKnowledge() {
    try {
        const response = await axios.post(
            `${baseUrl}/api/v2/query`,
            {
                query: 'What is machine learning?',
                max_tokens: 1000,
                include_sources: true
            },
            { headers }
        );
        
        console.log('Answer:', response.data.result);
        console.log('Confidence:', response.data.confidence);
    } catch (error) {
        console.error('Error:', error.response.data);
    }
}

queryKnowledge();
""",
            "curl": """
# Make a query
curl -X POST "https://api.universal-knowledge-platform.com/api/v2/query" \\
     -H "Authorization: Bearer your-api-key" \\
     -H "Content-Type: application/json" \\
     -d '{
       "query": "What is machine learning?",
       "max_tokens": 1000,
       "include_sources": true
     }'
""",
            "java": """
import com.fasterxml.jackson.databind.ObjectMapper;
import okhttp3.*;

public class KnowledgeClient {
    private static final String BASE_URL = "https://api.universal-knowledge-platform.com";
    private static final String API_KEY = "your-api-key";
    
    public static void main(String[] args) throws Exception {
        OkHttpClient client = new OkHttpClient();
        ObjectMapper mapper = new ObjectMapper();
        
        // Create request
        String jsonBody = mapper.writeValueAsString(Map.of(
            "query", "What is machine learning?",
            "max_tokens", 1000,
            "include_sources", true
        ));
        
        Request request = new Request.Builder()
            .url(BASE_URL + "/api/v2/query")
            .addHeader("Authorization", "Bearer " + API_KEY)
            .addHeader("Content-Type", "application/json")
            .post(RequestBody.create(jsonBody, MediaType.get("application/json")))
            .build();
        
        try (Response response = client.newCall(request).execute()) {
            if (response.isSuccessful()) {
                String result = response.body().string();
                System.out.println("Response: " + result);
            } else {
                System.out.println("Error: " + response.code());
            }
        }
    }
}
"""
        }
        
        return examples

# Documentation setup function
def setup_documentation(app: FastAPI) -> None:
    """Set up comprehensive API documentation."""
    generator = APIDocumentationGenerator(app)
    
    # Generate OpenAPI schema
    app.openapi = generator.generate_openapi_schema
    
    # Add documentation routes
    @app.get("/docs/openapi.json", include_in_schema=False)
    async def get_openapi_json():
        """Get OpenAPI schema as JSON."""
        return generator.generate_openapi_schema()
    
    @app.get("/docs/openapi.yaml", include_in_schema=False)
    async def get_openapi_yaml():
        """Get OpenAPI schema as YAML."""
        schema = generator.generate_openapi_schema()
        return yaml.dump(schema, default_flow_style=False)
    
    @app.get("/docs/markdown", include_in_schema=False)
    async def get_markdown_docs():
        """Get API documentation as markdown."""
        return generator.generate_markdown_docs()
    
    @app.get("/docs/examples", include_in_schema=False)
    async def get_sdk_examples():
        """Get SDK code examples."""
        return generator.generate_sdk_examples()

# Export public API
__all__ = [
    'APIDocumentationGenerator',
    'DocumentationConfig',
    'setup_documentation',
    'QueryRequestDoc',
    'QueryResponseDoc',
    'ErrorResponse',
    'RateLimitResponse',
] 