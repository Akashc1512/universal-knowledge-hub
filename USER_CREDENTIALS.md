# 🔐 User Credentials - Universal Knowledge Platform

## Your Login Credentials

### Admin Account
- **Username**: `admin`
- **Password**: `AdminPass123!`
- **Access Level**: Full administrative access

### User Account  
- **Username**: `user`
- **Password**: `UserPass123!`
- **Access Level**: Standard user access

## How to Login

### 1. Using cURL (Command Line)
```bash
# Admin login
curl -X POST "http://localhost:8002/auth/login" \
     -H "Content-Type: application/x-www-form-urlencoded" \
     -d "username=admin&password=AdminPass123!"

# User login
curl -X POST "http://localhost:8002/auth/login" \
     -H "Content-Type: application/x-www-form-urlencoded" \
     -d "username=user&password=UserPass123!"
```

### 2. Using Postman or API Client
- **URL**: `http://localhost:8002/auth/login`
- **Method**: POST
- **Content-Type**: `application/x-www-form-urlencoded`
- **Body**: 
  - username: `admin` (or `user`)
  - password: `AdminPass123!` (or `UserPass123!`)

### 3. Response Format
```json
{
  "access_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
  "token_type": "bearer",
  "expires_in": 1800
}
```

## Using the Access Token

After login, use the access token in the Authorization header:

```bash
curl -X POST "http://localhost:8002/query" \
     -H "Authorization: Bearer YOUR_ACCESS_TOKEN" \
     -H "Content-Type: application/json" \
     -d '{"query": "What is artificial intelligence?"}'
```

## Important Notes

⚠️ **Security Warning**: 
- Change these default passwords immediately after first login!
- Use the `/auth/change-password` endpoint to update your password

## Quick Start

1. Start the API server:
   ```bash
   python start_api.py
   ```

2. Login with your credentials

3. Use the access token for all API requests

---

**Need help?** See the full [Authentication Guide](AUTHENTICATION_GUIDE.md) for more details. 