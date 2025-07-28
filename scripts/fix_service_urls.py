#!/usr/bin/env python3
"""Fix service URL formatting in .env file."""

import re
from pathlib import Path

def fix_env_file():
    """Fix URL formatting issues in .env file."""
    env_path = Path('.env')
    
    if not env_path.exists():
        print("❌ No .env file found!")
        return
    
    # Read the file
    with open(env_path, 'r') as f:
        content = f.read()
    
    # Fix Elasticsearch URL (remove quotes)
    content = re.sub(
        r'ELASTICSEARCH_URL="([^"]+)"',
        r'ELASTICSEARCH_URL=\1',
        content
    )
    
    # Fix Pinecone API key (remove quotes if present)
    content = re.sub(
        r'PINECONE_API_KEY="([^"]+)"',
        r'PINECONE_API_KEY=\1',
        content
    )
    
    # Fix Anthropic API key (remove quotes if present)
    content = re.sub(
        r'ANTHROPIC_API_KEY="([^"]+)"',
        r'ANTHROPIC_API_KEY=\1',
        content
    )
    
    # Fix OpenAI API key (remove quotes if present)
    content = re.sub(
        r'OPENAI_API_KEY="([^"]+)"',
        r'OPENAI_API_KEY=\1',
        content
    )
    
    # Write back
    with open(env_path, 'w') as f:
        f.write(content)
    
    print("✅ Fixed URL formatting in .env file")
    print("   - Removed quotes from API keys and URLs")
    print("\nNote: Redis connection issues are expected if:")
    print("   - Redis server is not running locally")
    print("   - AWS ElastiCache is not accessible from your network")
    print("   - You can disable Redis caching if not needed")

if __name__ == '__main__':
    fix_env_file() 