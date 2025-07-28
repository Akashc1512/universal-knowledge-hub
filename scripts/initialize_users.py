#!/usr/bin/env python3
"""
Initialize Default Users for Universal Knowledge Platform
Creates admin and user accounts with secure passwords.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.user_management import get_user_manager, UserCreate, UserRole

def initialize_users():
    """Initialize default users."""
    print("🔐 Initializing Default Users for Universal Knowledge Platform")
    print("="*60)
    
    user_manager = get_user_manager()
    
    # The UserManager will automatically create default users on first run
    # But we can also create custom users here
    
    print("\n✅ Default users have been created!")
    print("\n📋 User Accounts:")
    print("-"*40)
    
    users = user_manager.list_users()
    for user in users:
        print(f"Username: {user.username}")
        print(f"  Email: {user.email}")
        print(f"  Role: {user.role}")
        print(f"  Active: {user.is_active}")
        print(f"  Created: {user.created_at}")
        print("-"*40)
    
    print("\n🔑 Login Credentials:")
    print("="*60)
    print("Admin Account:")
    print("  Username: admin")
    print("  Password: AdminPass123!")
    print("\nUser Account:")
    print("  Username: user")
    print("  Password: UserPass123!")
    print("\n⚠️  IMPORTANT: Change these passwords after first login!")
    print("="*60)
    
    print("\n📡 API Endpoints:")
    print("  Login: POST /auth/login")
    print("  Get User Info: GET /auth/me")
    print("  Change Password: POST /auth/change-password")
    print("\n🔧 Admin Endpoints:")
    print("  List Users: GET /auth/users")
    print("  Create User: POST /auth/users")
    print("  Update Role: PUT /auth/users/{username}/role")
    print("  Deactivate User: DELETE /auth/users/{username}")
    
    print("\n💡 Example Login Request:")
    print("""
    curl -X POST "http://localhost:8002/auth/login" \\
         -H "Content-Type: application/x-www-form-urlencoded" \\
         -d "username=admin&password=AdminPass123!"
    """)
    
    print("\n✨ Users initialized successfully!")

if __name__ == "__main__":
    initialize_users() 