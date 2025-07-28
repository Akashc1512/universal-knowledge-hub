#!/usr/bin/env python3
"""
Generate secure keys for environment configuration.

This script generates cryptographically secure keys for the
security-related environment variables.
"""

import secrets
import string
import base64
from pathlib import Path

def generate_secure_key(length: int = 32) -> str:
    """Generate a cryptographically secure key."""
    alphabet = string.ascii_letters + string.digits + "!@#$%^&*"
    return ''.join(secrets.choice(alphabet) for _ in range(length))

def generate_base64_key(length: int = 32) -> str:
    """Generate a base64-encoded secure key."""
    random_bytes = secrets.token_bytes(length)
    return base64.b64encode(random_bytes).decode('utf-8')

def main():
    """Generate and display secure keys."""
    print("SECURE KEY GENERATOR")
    print("="*60)
    print("\nGenerated secure values for your .env file:")
    print("(Copy and paste these into your .env file)\n")
    
    # Generate keys
    secret_key = generate_secure_key(64)
    api_key_secret = generate_secure_key(64)
    encryption_key = generate_base64_key(32)
    admin_api_key = f"admin-{generate_secure_key(48)}"
    user_api_key = f"user-{generate_secure_key(48)}"
    readonly_api_key = f"readonly-{generate_secure_key(48)}"
    
    print("# Security Configuration")
    print(f"SECRET_KEY={secret_key}")
    print(f"API_KEY_SECRET={api_key_secret}")
    print(f"ENCRYPTION_KEY={encryption_key}")
    print()
    print("# API Keys for Authentication")
    print(f"ADMIN_API_KEY={admin_api_key}")
    print(f"USER_API_KEY={user_api_key}")
    print(f"READONLY_API_KEY={readonly_api_key}")
    
    print("\n" + "="*60)
    print("IMPORTANT SECURITY NOTES:")
    print("- These keys are cryptographically secure")
    print("- Save them in a secure location")
    print("- Never commit them to version control")
    print("- Use different keys for production")
    print("- Rotate keys periodically")
    
    # Check if .env.backup exists
    env_path = Path('.env')
    backup_path = Path('.env.backup')
    
    if env_path.exists() and not backup_path.exists():
        print(f"\nTIP: Consider backing up your current .env file:")
        print(f"     cp .env .env.backup")

if __name__ == '__main__':
    main() 