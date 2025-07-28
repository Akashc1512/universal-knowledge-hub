# Docker Frontend Integration Guide

This guide explains how to run the Universal Knowledge Platform with both backend and frontend using Docker.

## 🚀 Quick Start

### Development Setup (Recommended)

The easiest way to develop is to run the backend in Docker and the frontend with Next.js dev server:

```bash
# 1. Start backend services
docker-compose -f docker-compose.frontend-dev.yml up -d

# 2. Run frontend with hot reload
cd frontend
npm install
npm run dev
```

- Backend API: http://localhost:8002
- Frontend: http://localhost:3001 (or 3000 if available)

### Full Stack Docker Setup

To run everything in Docker containers:

```bash
# Build and start all services
docker-compose -f docker-compose.fullstack.yml up --build
```

- Frontend: http://localhost:3001
- Backend API: http://localhost:8002
- Nginx Proxy: http://localhost (optional)

## 📦 Docker Configurations

### 1. Development Configuration (`docker-compose.frontend-dev.yml`)

Minimal setup for development:
- Backend API with hot reload
- Redis for caching
- PostgreSQL for data
- Frontend runs separately with `npm run dev`

### 2. Full Stack Configuration (`docker-compose.fullstack.yml`)

Complete containerized setup:
- Backend API container
- Frontend Next.js container
- All supporting services (Redis, PostgreSQL, Elasticsearch)
- Optional Nginx reverse proxy

### 3. Frontend Dockerfile (`Dockerfile.frontend`)

Multi-stage build for Next.js:
- Optimized for production
- Standalone output for smaller images
- Non-root user for security

## 🔧 Configuration

### Environment Variables

#### Backend (.env)
```env
CORS_ORIGINS=http://localhost:3000,http://localhost:3001
ADMIN_API_KEY=admin-key-123
USER_API_KEY=user-key-456
READONLY_API_KEY=readonly-key-789
```

#### Frontend (frontend/.env.local)
```env
NEXT_PUBLIC_API_BASE_URL=http://localhost:8002
NEXT_PUBLIC_API_KEY=user-key-456
```

### Port Mappings

| Service | Internal Port | External Port |
|---------|--------------|---------------|
| Backend API | 8000 | 8002 |
| Frontend | 3000 | 3001 |
| Redis | 6379 | 6379 |
| PostgreSQL | 5432 | 5432 |
| Elasticsearch | 9200 | 9200 |

## 🧪 Testing Integration

### 1. Manual Testing

1. Open http://localhost:3001
2. Submit a query in the UI
3. Verify the answer and citations render correctly

### 2. Check Services

```bash
# Backend health
curl http://localhost:8002/health

# Test query with API key
curl -X POST http://localhost:8002/query \
  -H "Content-Type: application/json" \
  -H "X-API-Key: user-key-456" \
  -d '{"query": "What is Docker?"}'
```

### 3. View Logs

```bash
# All services
docker-compose -f docker-compose.fullstack.yml logs -f

# Specific service
docker-compose -f docker-compose.fullstack.yml logs -f backend
docker-compose -f docker-compose.fullstack.yml logs -f frontend
```

## 🚢 Production Deployment

### Option 1: Docker Compose

```bash
# Use production configuration
docker-compose -f docker-compose.fullstack.yml up -d
```

### Option 2: Separate Deployments

1. **Backend**: Deploy to any container service (ECS, Cloud Run, etc.)
2. **Frontend**: 
   - Deploy to Vercel (recommended for Next.js)
   - Or build static files: `npm run build && npm run export`
   - Serve with Nginx or CDN

### Option 3: Kubernetes

Use the Docker images with Kubernetes manifests (not included).

## 🔍 Troubleshooting

### Frontend can't connect to backend

1. Check CORS origins in backend
2. Verify API base URL in frontend
3. Ensure API key is correct

### Port conflicts

- Grafana uses 3000 by default (conflicts with Next.js)
- Frontend is configured to use 3001 to avoid conflicts
- Adjust ports in docker-compose files if needed

### Build failures

```bash
# Clean rebuild
docker-compose -f docker-compose.fullstack.yml down
docker-compose -f docker-compose.fullstack.yml build --no-cache
docker-compose -f docker-compose.fullstack.yml up
```

## 🏗️ Architecture

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Browser   │────▶│  Frontend   │────▶│   Backend   │
│             │     │  (Next.js)  │     │   (FastAPI) │
└─────────────┘     └─────────────┘     └─────────────┘
                           │                     │
                           │                     ▼
                           │              ┌─────────────┐
                           │              │    Redis    │
                           │              └─────────────┘
                           │                     │
                           │                     ▼
                           │              ┌─────────────┐
                           └─────────────▶│ PostgreSQL  │
                                         └─────────────┘
```

## 📝 Notes

- The frontend is configured to work with API key authentication
- CORS is configured to allow both localhost:3000 and localhost:3001
- For production, update environment variables and API keys
- Consider using secrets management for sensitive data 