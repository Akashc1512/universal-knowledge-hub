#!/usr/bin/env python3
"""
Verify Environment Configuration Script.

This script checks that all required environment variables are properly
configured and provides helpful feedback for missing or invalid values.
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dotenv import load_dotenv
import re

# Load .env file
env_path = Path('.env')
if env_path.exists():
    load_dotenv(env_path)
    print(f"✅ Loaded .env file from: {env_path.absolute()}")
else:
    print(f"❌ No .env file found at: {env_path.absolute()}")
    print("   Please create one from env.template")
    sys.exit(1)

# Define required and optional environment variables
REQUIRED_VARS = {
    # Core security
    'SECRET_KEY': 'Application secret key for JWT tokens',
    'API_KEY_SECRET': 'Secret for API key generation',
    
    # External APIs (at least one AI provider needed)
    'OPENAI_API_KEY': 'OpenAI API key (or use Anthropic)',
    
    # Vector databases (at least one needed)
    'PINECONE_API_KEY': 'Pinecone API key (or use Qdrant/Elasticsearch)',
}

OPTIONAL_BUT_RECOMMENDED = {
    # AI Providers
    'ANTHROPIC_API_KEY': 'Anthropic Claude API key',
    
    # Vector Databases
    'PINECONE_ENVIRONMENT': 'Pinecone environment (e.g., us-west1-gcp)',
    'PINECONE_INDEX_NAME': 'Pinecone index name',
    'QDRANT_URL': 'Qdrant URL if using Qdrant',
    'ELASTICSEARCH_URL': 'Elasticsearch URL if using Elasticsearch',
    
    # Caching
    'REDIS_URL': 'Redis URL for caching',
    
    # Security
    'ENCRYPTION_KEY': 'Key for encrypting sensitive data',
    'ADMIN_API_KEY': 'Admin API key',
    
    # Performance
    'MAX_CONCURRENT_WORKERS': 'Maximum concurrent workers',
    'RATE_LIMIT_REQUESTS': 'Rate limit requests per window',
}

SERVICE_GROUPS = {
    'AI Providers': {
        'vars': ['OPENAI_API_KEY', 'ANTHROPIC_API_KEY'],
        'required': 1,  # At least one required
        'description': 'At least one AI provider must be configured'
    },
    'Vector Databases': {
        'vars': ['PINECONE_API_KEY', 'QDRANT_URL', 'ELASTICSEARCH_URL'],
        'required': 1,  # At least one required
        'description': 'At least one vector database must be configured'
    },
    'Security Keys': {
        'vars': ['SECRET_KEY', 'API_KEY_SECRET'],
        'required': 2,  # All required
        'description': 'Security keys must be set and not use default values'
    }
}

def check_env_var(var_name: str, description: str, required: bool = True) -> Tuple[bool, str]:
    """Check if an environment variable is set and valid."""
    value = os.getenv(var_name)
    
    if not value:
        return False, f"Not set - {description}"
    
    # Check for placeholder values
    placeholder_patterns = [
        r'your-.*-here',
        r'change-.*-production',
        r'your-.*-key',
        r'your-.*-password',
        r'xxx',
        r'todo',
        r'placeholder'
    ]
    
    for pattern in placeholder_patterns:
        if re.search(pattern, value.lower()):
            return False, f"Still using placeholder value - {description}"
    
    # Check minimum length for keys
    if 'KEY' in var_name and len(value) < 16:
        return False, f"Key too short (min 16 chars) - {description}"
    
    # Mask sensitive values in output
    if any(sensitive in var_name for sensitive in ['KEY', 'PASSWORD', 'SECRET']):
        masked_value = value[:4] + '*' * (len(value) - 8) + value[-4:] if len(value) > 8 else '*' * len(value)
        return True, f"Set ({masked_value}) - {description}"
    
    return True, f"Set ({value[:20]}...) - {description}"

def check_service_groups() -> Dict[str, Tuple[bool, str]]:
    """Check service groups to ensure at least one option is configured."""
    results = {}
    
    for group_name, config in SERVICE_GROUPS.items():
        configured_count = 0
        configured_vars = []
        
        for var in config['vars']:
            value = os.getenv(var)
            if value and not any(p in value.lower() for p in ['your-', 'change-', 'placeholder']):
                configured_count += 1
                configured_vars.append(var)
        
        if configured_count >= config['required']:
            results[group_name] = (True, f"{configured_count}/{len(config['vars'])} configured: {', '.join(configured_vars)}")
        else:
            results[group_name] = (False, f"Need at least {config['required']}, but only {configured_count} configured. {config['description']}")
    
    return results

def check_service_connectivity():
    """Check if configured services are accessible."""
    print("\n" + "="*60)
    print("SERVICE CONNECTIVITY CHECKS:")
    print("="*60)
    
    # Redis connectivity
    redis_url = os.getenv('REDIS_URL')
    if redis_url and not any(p in redis_url for p in ['your-', 'change-']):
        try:
            import redis
            r = redis.from_url(redis_url)
            r.ping()
            print("✅ Redis: Connected successfully")
        except ImportError:
            print("⚠️  Redis: redis package not installed")
        except Exception as e:
            print(f"❌ Redis: Connection failed - {str(e)}")
    
    # Elasticsearch connectivity
    es_url = os.getenv('ELASTICSEARCH_URL')
    if es_url and not any(p in es_url for p in ['your-', 'change-']):
        try:
            import requests
            response = requests.get(es_url, timeout=5)
            if response.status_code == 200:
                print("✅ Elasticsearch: Connected successfully")
            else:
                print(f"❌ Elasticsearch: Connection failed - Status {response.status_code}")
        except ImportError:
            print("⚠️  Elasticsearch: requests package not installed")
        except Exception as e:
            print(f"❌ Elasticsearch: Connection failed - {str(e)}")

def main():
    """Main verification function."""
    print("ENVIRONMENT CONFIGURATION VERIFICATION")
    print("="*60)
    
    # Check required variables
    print("\nREQUIRED VARIABLES:")
    print("-"*40)
    all_good = True
    for var, desc in REQUIRED_VARS.items():
        success, message = check_env_var(var, desc, required=True)
        print(f"{'✅' if success else '❌'} {var}: {message}")
        if not success:
            all_good = False
    
    # Check optional variables
    print("\nOPTIONAL BUT RECOMMENDED:")
    print("-"*40)
    for var, desc in OPTIONAL_BUT_RECOMMENDED.items():
        success, message = check_env_var(var, desc, required=False)
        print(f"{'✅' if success else '⚠️ '} {var}: {message}")
    
    # Check service groups
    print("\nSERVICE GROUPS:")
    print("-"*40)
    group_results = check_service_groups()
    for group, (success, message) in group_results.items():
        print(f"{'✅' if success else '❌'} {group}: {message}")
        if not success:
            all_good = False
    
    # Check service connectivity
    check_service_connectivity()
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY:")
    if all_good:
        print("✅ All required environment variables are properly configured!")
        print("   The backend should be able to start successfully.")
    else:
        print("❌ Some required environment variables are missing or invalid.")
        print("   Please update your .env file with the missing values.")
        print("\nTIPS:")
        print("- Copy env.template to .env if you haven't already")
        print("- Replace all placeholder values with real API keys")
        print("- For development, you can use dummy values for services you're not using")
        print("- At minimum, you need one AI provider and one vector database configured")
    
    return 0 if all_good else 1

if __name__ == '__main__':
    sys.exit(main()) 