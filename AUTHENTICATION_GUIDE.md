# 🔐 Authentication Guide - Universal Knowledge Platform

## Overview

The Universal Knowledge Platform now includes a comprehensive authentication system with user management, role-based access control, and JWT token authentication.

## 🎯 Default User Accounts

When you first start the application, two default accounts are automatically created:

### Admin Account
- **Username**: `admin`
- **Password**: `AdminPass123!`
- **Role**: admin
- **Permissions**: Full access to all features

### User Account
- **Username**: `user`
- **Password**: `UserPass123!`
- **Role**: user
- **Permissions**: Standard user access

⚠️ **IMPORTANT**: Change these passwords immediately after first login!

## 🚀 Quick Start

### 1. Initialize Users
```bash
python scripts/initialize_users.py
```

### 2. Login via API
```bash
# Login as admin
curl -X POST "http://localhost:8002/auth/login" \
     -H "Content-Type: application/x-www-form-urlencoded" \
     -d "username=admin&password=AdminPass123!"

# Response:
{
  "access_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
  "token_type": "bearer",
  "expires_in": 1800
}
```

### 3. Use the Token
```bash
# Make authenticated request
curl -X POST "http://localhost:8002/query" \
     -H "Authorization: Bearer YOUR_ACCESS_TOKEN" \
     -H "Content-Type: application/json" \
     -d '{"query": "What is AI?"}'
```

## 📡 Authentication Endpoints

### Public Endpoints

#### Login
```http
POST /auth/login
Content-Type: application/x-www-form-urlencoded

username=admin&password=AdminPass123!
```

#### Register New User
```http
POST /auth/register
Content-Type: application/json

{
  "username": "newuser",
  "email": "newuser@example.com",
  "password": "SecurePass123!",
  "full_name": "New User"
}
```

### Authenticated Endpoints

#### Get Current User Info
```http
GET /auth/me
Authorization: Bearer YOUR_ACCESS_TOKEN
```

#### Change Password
```http
POST /auth/change-password
Authorization: Bearer YOUR_ACCESS_TOKEN
Content-Type: application/json

{
  "current_password": "OldPassword123!",
  "new_password": "NewPassword123!"
}
```

### Admin-Only Endpoints

#### List All Users
```http
GET /auth/users
Authorization: Bearer ADMIN_ACCESS_TOKEN
```

#### Create New User (with role)
```http
POST /auth/users
Authorization: Bearer ADMIN_ACCESS_TOKEN
Content-Type: application/json

{
  "username": "poweruser",
  "email": "power@example.com",
  "password": "PowerPass123!",
  "role": "admin",
  "full_name": "Power User"
}
```

#### Update User Role
```http
PUT /auth/users/{username}/role
Authorization: Bearer ADMIN_ACCESS_TOKEN
Content-Type: application/json

{
  "new_role": "admin"
}
```

#### Deactivate User
```http
DELETE /auth/users/{username}
Authorization: Bearer ADMIN_ACCESS_TOKEN
```

## 🔑 User Roles

### admin
- Full access to all features
- Can manage other users
- Can access admin endpoints
- Higher rate limits

### user
- Standard access to query endpoints
- Can change own password
- Cannot access admin features
- Standard rate limits

### readonly
- Read-only access
- Cannot make queries
- Limited to viewing data
- Lower rate limits

## 🛡️ Security Features

1. **Password Hashing**: Bcrypt with salt
2. **JWT Tokens**: Expire after 30 minutes (configurable)
3. **Rate Limiting**: Per-user limits based on role
4. **Session Management**: Automatic logout on inactivity
5. **Secure Storage**: User data stored in encrypted JSON file

## 💻 Frontend Integration

### JavaScript Example
```javascript
// Login
async function login(username, password) {
  const formData = new URLSearchParams();
  formData.append('username', username);
  formData.append('password', password);
  
  const response = await fetch('http://localhost:8002/auth/login', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/x-www-form-urlencoded',
    },
    body: formData
  });
  
  const data = await response.json();
  
  // Store token
  localStorage.setItem('access_token', data.access_token);
  
  return data;
}

// Make authenticated request
async function makeQuery(query) {
  const token = localStorage.getItem('access_token');
  
  const response = await fetch('http://localhost:8002/query', {
    method: 'POST',
    headers: {
      'Authorization': `Bearer ${token}`,
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({ query })
  });
  
  return await response.json();
}
```

### React Hook Example
```jsx
import { useState, useEffect } from 'react';

function useAuth() {
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);
  
  useEffect(() => {
    const token = localStorage.getItem('access_token');
    if (token) {
      fetchUserInfo(token);
    } else {
      setLoading(false);
    }
  }, []);
  
  const login = async (username, password) => {
    // ... login logic
  };
  
  const logout = () => {
    localStorage.removeItem('access_token');
    setUser(null);
  };
  
  return { user, loading, login, logout };
}
```

## 🔧 Configuration

### Environment Variables
```bash
# JWT Configuration
SECRET_KEY=your-secret-key-here
ACCESS_TOKEN_EXPIRE_MINUTES=30

# Password Requirements
MIN_PASSWORD_LENGTH=8
REQUIRE_SPECIAL_CHARS=true
```

### User Storage
Users are stored in `data/users.json`. To backup users:
```bash
cp data/users.json data/users.backup.json
```

## 🚨 Troubleshooting

### Forgot Password
Currently, password reset requires admin intervention:
1. Admin logs in
2. Admin creates new temporary password
3. User logs in with temporary password
4. User changes password

### Token Expired
Tokens expire after 30 minutes. Simply login again to get a new token.

### Account Locked
If an account is deactivated, an admin must reactivate it.

## 📊 API Testing with Postman

1. **Import Collection**: Use the provided Postman collection
2. **Set Environment**: Configure `base_url` as `http://localhost:8002`
3. **Login First**: Run the login request to get token
4. **Use Token**: Token is automatically set in environment

## 🎯 Best Practices

1. **Always use HTTPS in production**
2. **Change default passwords immediately**
3. **Implement password complexity requirements**
4. **Regular token rotation**
5. **Monitor failed login attempts**
6. **Backup user data regularly**

---

**Need Help?** Check the logs in `logs/` directory or run with debug logging enabled. 