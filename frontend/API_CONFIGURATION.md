# API Configuration Guide

## Environment Variables

The frontend application uses the following environment variables for API configuration:

### Required Variables

- `NEXT_PUBLIC_API_BASE_URL`: The base URL of the backend API
  - Default: `http://localhost:8002`
  - Example: `http://localhost:8002` or `https://api.example.com`

### Optional Variables

- `NEXT_PUBLIC_API_KEY`: API key for authentication
  - Default: `user-key-456` (if not set)
  - Options:
    - Leave empty to use default user key
    - Set to `admin-key-123` for admin access (analytics, etc.)
    - Set to any other key that matches your backend configuration

## Configuration Files

### .env.local
Create this file in the frontend directory with your configuration:

```bash
# API Configuration
NEXT_PUBLIC_API_BASE_URL=http://localhost:8002
NEXT_PUBLIC_API_KEY=

# App Configuration
NEXT_PUBLIC_APP_NAME="Universal Knowledge Platform"
NEXT_PUBLIC_VERSION=1.0.0

# Feature Flags
NEXT_PUBLIC_ENABLE_FEEDBACK=true
NEXT_PUBLIC_ENABLE_ANALYTICS=true
NEXT_PUBLIC_ENABLE_EXPERT_MODE=false
```

## API Key Management

### User vs Admin Access

- **User Key** (`user-key-456`): Standard access for querying and feedback
- **Admin Key** (`admin-key-123`): Enhanced access including analytics and system monitoring

### Switching Keys

1. **Development**: Edit `.env.local` and restart the development server
2. **Production**: Update environment variables in your deployment platform

### Security Notes

- API keys are included in all requests as `X-API-Key` header
- Keys are visible in browser network tab (as they're client-side)
- For production, consider implementing server-side proxy for sensitive operations

## Backend Integration

The frontend automatically includes the API key in all requests to the backend. If you change the backend API keys, update the frontend environment variables to match.

### Request Headers

All API requests include:
- `Content-Type: application/json`
- `X-API-Key: [your-api-key]`

### Error Handling

The application handles various HTTP status codes:
- `401`: Authentication required (check API key)
- `403`: Query blocked by content guidelines
- `503`: Service temporarily unavailable
- `429`: Too many requests
- `500`: Server error

## Development Setup

1. Copy `env.example` to `.env.local`
2. Update the configuration as needed
3. Restart the development server: `npm run dev`

## Production Deployment

1. Set environment variables in your deployment platform
2. Ensure `NEXT_PUBLIC_API_BASE_URL` points to your production backend
3. Set appropriate API keys for production access 