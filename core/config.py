"""
Configuration management for the Universal Knowledge Platform.
Handles API keys, settings, and environment variables.
"""

import os
from typing import Optional, Dict, Any
from dataclasses import dataclass
from dotenv import load_dotenv
import logging

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


@dataclass
class APIConfig:
    """API configuration settings."""
    
    # OpenAI Configuration
    openai_api_key: Optional[str] = None
    openai_model: str = "gpt-4o-mini"
    openai_max_tokens: int = 4000
    openai_temperature: float = 0.1
    
    # Anthropic Configuration
    anthropic_api_key: Optional[str] = None
    anthropic_model: str = "claude-3-5-sonnet-20241022"
    anthropic_max_tokens: int = 4000
    
    # Pinecone Configuration
    pinecone_api_key: Optional[str] = None
    pinecone_environment: Optional[str] = None
    pinecone_index_name: str = "universal-knowledge-hub"
    
    # Vector Database Configuration
    vector_db_type: str = "pinecone"
    vector_dimension: int = 1536
    
    # Elasticsearch Configuration
    elasticsearch_url: str = "http://localhost:9200"
    elasticsearch_index: str = "knowledge_base"
    elasticsearch_api_key: Optional[str] = None
    elasticsearch_username: Optional[str] = None
    elasticsearch_password: Optional[str] = None
    
    # Redis Configuration
    redis_url: str = "redis://localhost:6379"
    redis_db: int = 0
    
    # PostgreSQL Configuration
    postgres_url: str = "postgresql://ukp_user:ukp_password@localhost:5432/ukp_db"
    
    # Token Budget Configuration
    daily_token_budget: int = 1_000_000
    max_tokens_per_query: int = 10_000
    
    # Server Configuration
    host: str = "0.0.0.0"
    port: int = 8002
    reload: bool = False
    
    # Logging Configuration
    log_level: str = "INFO"
    log_format: str = "detailed"
    
    # Security Configuration
    secret_key: str = "your_secret_key_here"
    cors_origins: list = None
    
    # Monitoring Configuration
    enable_metrics: bool = True
    enable_health_checks: bool = True
    prometheus_port: int = 9090
    grafana_port: int = 3000
    
    # Development Configuration
    debug: bool = True
    enable_hot_reload: bool = True
    enable_api_docs: bool = True
    
    def __post_init__(self):
        """Initialize configuration from environment variables."""
        # OpenAI
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.openai_model = os.getenv("OPENAI_MODEL", self.openai_model)
        self.openai_max_tokens = int(os.getenv("OPENAI_MAX_TOKENS", self.openai_max_tokens))
        self.openai_temperature = float(os.getenv("OPENAI_TEMPERATURE", self.openai_temperature))
        
        # Anthropic
        self.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
        self.anthropic_model = os.getenv("ANTHROPIC_MODEL", self.anthropic_model)
        self.anthropic_max_tokens = int(os.getenv("ANTHROPIC_MAX_TOKENS", self.anthropic_max_tokens))
        
        # Pinecone
        self.pinecone_api_key = os.getenv("PINECONE_API_KEY")
        self.pinecone_environment = os.getenv("PINECONE_ENVIRONMENT")
        self.pinecone_index_name = os.getenv("PINECONE_INDEX_NAME", self.pinecone_index_name)
        
        # Vector Database
        self.vector_db_type = os.getenv("VECTOR_DB_TYPE", self.vector_db_type)
        self.vector_dimension = int(os.getenv("VECTOR_DIMENSION", self.vector_dimension))
        
        # Elasticsearch
        self.elasticsearch_url = os.getenv("ELASTICSEARCH_URL", self.elasticsearch_url)
        self.elasticsearch_index = os.getenv("ELASTICSEARCH_INDEX", self.elasticsearch_index)
        self.elasticsearch_api_key = os.getenv("ELASTICSEARCH_API_KEY")
        self.elasticsearch_username = os.getenv("ELASTICSEARCH_USERNAME")
        self.elasticsearch_password = os.getenv("ELASTICSEARCH_PASSWORD")
        
        # Redis
        self.redis_url = os.getenv("REDIS_URL", self.redis_url)
        self.redis_db = int(os.getenv("REDIS_DB", self.redis_db))
        
        # PostgreSQL
        self.postgres_url = os.getenv("POSTGRES_URL", self.postgres_url)
        
        # Token Budget
        self.daily_token_budget = int(os.getenv("DAILY_TOKEN_BUDGET", self.daily_token_budget))
        self.max_tokens_per_query = int(os.getenv("MAX_TOKENS_PER_QUERY", self.max_tokens_per_query))
        
        # Server
        self.host = os.getenv("UKP_HOST", self.host)
        self.port = int(os.getenv("UKP_PORT", self.port))
        self.reload = os.getenv("UKP_RELOAD", "false").lower() == "true"
        
        # Logging
        self.log_level = os.getenv("LOG_LEVEL", self.log_level)
        self.log_format = os.getenv("LOG_FORMAT", self.log_format)
        
        # Security
        self.secret_key = os.getenv("SECRET_KEY", self.secret_key)
        cors_origins_str = os.getenv("CORS_ORIGINS", '["http://localhost:3000", "http://localhost:8002"]')
        self.cors_origins = eval(cors_origins_str) if cors_origins_str else []
        
        # Monitoring
        self.enable_metrics = os.getenv("ENABLE_METRICS", "true").lower() == "true"
        self.enable_health_checks = os.getenv("ENABLE_HEALTH_CHECKS", "true").lower() == "true"
        self.prometheus_port = int(os.getenv("PROMETHEUS_PORT", self.prometheus_port))
        self.grafana_port = int(os.getenv("GRAFANA_PORT", self.grafana_port))
        
        # Development
        self.debug = os.getenv("DEBUG", "true").lower() == "true"
        self.enable_hot_reload = os.getenv("ENABLE_HOT_RELOAD", "true").lower() == "true"
        self.enable_api_docs = os.getenv("ENABLE_API_DOCS", "true").lower() == "true"
        
        self._validate_config()
    
    def _validate_config(self):
        """Validate configuration and log warnings for missing API keys."""
        missing_keys = []
        
        if not self.openai_api_key:
            missing_keys.append("OPENAI_API_KEY")
        if not self.anthropic_api_key:
            missing_keys.append("ANTHROPIC_API_KEY")
        if not self.pinecone_api_key:
            missing_keys.append("PINECONE_API_KEY")
        
        if missing_keys:
            logger.warning(f"Missing API keys: {', '.join(missing_keys)}")
            logger.info("Some features may be limited without these API keys")
        else:
            logger.info("✅ All API keys are configured")
    
    def get_openai_config(self) -> Dict[str, Any]:
        """Get OpenAI configuration."""
        return {
            "api_key": self.openai_api_key,
            "model": self.openai_model,
            "max_tokens": self.openai_max_tokens,
            "temperature": self.openai_temperature
        }
    
    def get_anthropic_config(self) -> Dict[str, Any]:
        """Get Anthropic configuration."""
        return {
            "api_key": self.anthropic_api_key,
            "model": self.anthropic_model,
            "max_tokens": self.anthropic_max_tokens
        }
    
    def get_pinecone_config(self) -> Dict[str, Any]:
        """Get Pinecone configuration."""
        return {
            "api_key": self.pinecone_api_key,
            "environment": self.pinecone_environment,
            "index_name": self.pinecone_index_name
        }
    
    def get_elasticsearch_config(self) -> Dict[str, Any]:
        """Get Elasticsearch configuration."""
        return {
            "url": self.elasticsearch_url,
            "index_name": self.elasticsearch_index,
            "api_key": self.elasticsearch_api_key,
            "username": self.elasticsearch_username,
            "password": self.elasticsearch_password
        }
    
    def is_fully_configured(self) -> bool:
        """Check if all required API keys are configured."""
        return all([
            self.openai_api_key,
            self.anthropic_api_key,
            self.pinecone_api_key
        ])


# Global configuration instance
config = APIConfig() 