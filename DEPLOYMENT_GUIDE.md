# Deployment Guide - MAANG Standards

This guide provides comprehensive deployment instructions for the Universal Knowledge Platform following MAANG (Meta, Amazon, Apple, Netflix, Google) engineering standards.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Environment Setup](#environment-setup)
3. [Local Development](#local-development)
4. [Docker Deployment](#docker-deployment)
5. [Kubernetes Deployment](#kubernetes-deployment)
6. [CI/CD Pipeline](#cicd-pipeline)
7. [Monitoring & Observability](#monitoring--observability)
8. [Security Configuration](#security-configuration)
9. [Performance Optimization](#performance-optimization)
10. [Troubleshooting](#troubleshooting)

## Prerequisites

### System Requirements

- **Python**: 3.11+
- **Docker**: 20.10+
- **Kubernetes**: 1.24+
- **Redis**: 7.0+
- **PostgreSQL**: 14+
- **Elasticsearch**: 8.11+
- **Qdrant**: 1.7+

### Required Tools

```bash
# Install required tools
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Install additional tools
pip install docker-compose kubectl helm
```

## Environment Setup

### 1. Environment Variables

Create `.env` file with the following configuration:

```bash
# Application Settings
ENVIRONMENT=production
DEBUG=false
LOG_LEVEL=INFO
SECRET_KEY=your-super-secret-key-change-in-production
DATABASE_URL=postgresql://user:pass@localhost:5432/ukp
REDIS_URL=redis://localhost:6379

# External Services
ELASTICSEARCH_URL=http://localhost:9200
VECTOR_DB_URL=http://localhost:6333
SPARQL_ENDPOINT=http://localhost:7200/repositories/knowledge

# AI Services
OPENAI_API_KEY=sk-your-openai-key
ANTHROPIC_API_KEY=sk-your-anthropic-key

# Security
CORS_ORIGINS=https://yourdomain.com,https://www.yourdomain.com
ALLOWED_HOSTS=api.yourdomain.com,yourdomain.com

# Monitoring
PROMETHEUS_ENABLED=true
GRAFANA_ENABLED=true
JAEGER_ENABLED=true

# Cache Settings
CACHE_ENABLED=true
CACHE_TTL=300
CACHE_MAX_SIZE=10000

# Rate Limiting
RATE_LIMIT_ENABLED=true
RATE_LIMIT_REQUESTS_PER_MINUTE=60
RATE_LIMIT_BURST_SIZE=10

# Security
SECURITY_ENABLED=true
THREAT_DETECTION_ENABLED=true
ENCRYPTION_ENABLED=true
```

### 2. Database Setup

```sql
-- Create database
CREATE DATABASE ukp;

-- Create user
CREATE USER ukp_user WITH PASSWORD 'secure_password';

-- Grant permissions
GRANT ALL PRIVILEGES ON DATABASE ukp TO ukp_user;
```

### 3. Redis Setup

```bash
# Install Redis
sudo apt-get install redis-server

# Configure Redis
sudo nano /etc/redis/redis.conf

# Key configurations:
# maxmemory 1gb
# maxmemory-policy allkeys-lru
# save 900 1
# save 300 10
# save 60 10000

# Start Redis
sudo systemctl start redis
sudo systemctl enable redis
```

## Local Development

### 1. Quick Start

```bash
# Clone repository
git clone https://github.com/your-org/universal-knowledge-platform.git
cd universal-knowledge-platform

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Set up environment
cp .env.example .env
# Edit .env with your configuration

# Initialize database
python scripts/initialize_database.py

# Create default users
python scripts/initialize_users.py

# Run development server
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### 2. Development with Docker Compose

```bash
# Start all services
docker-compose -f docker-compose.dev.yml up -d

# View logs
docker-compose -f docker-compose.dev.yml logs -f

# Stop services
docker-compose -f docker-compose.dev.yml down
```

### 3. Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test categories
pytest tests/ -v -m unit
pytest tests/ -v -m integration
pytest tests/ -v -m performance
pytest tests/ -v -m security

# Run with coverage
pytest tests/ -v --cov=api --cov-report=html

# Run linting
flake8 api/ tests/
black --check api/ tests/
mypy api/
```

## Docker Deployment

### 1. Build Docker Image

```bash
# Build production image
docker build -t universal-knowledge-platform:latest .

# Build with specific target
docker build --target production -t universal-knowledge-platform:prod .

# Build for different architectures
docker buildx build --platform linux/amd64,linux/arm64 -t universal-knowledge-platform:latest .
```

### 2. Docker Compose Production

```yaml
# docker-compose.prod.yml
version: '3.8'

services:
  api:
    image: universal-knowledge-platform:latest
    ports:
      - "8000:8000"
    environment:
      - ENVIRONMENT=production
      - DATABASE_URL=postgresql://user:pass@db:5432/ukp
      - REDIS_URL=redis://redis:6379
    depends_on:
      - db
      - redis
      - elasticsearch
      - qdrant
    volumes:
      - app-data:/app/data
      - app-logs:/app/logs
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  db:
    image: postgres:15
    environment:
      POSTGRES_DB: ukp
      POSTGRES_USER: user
      POSTGRES_PASSWORD: pass
    volumes:
      - postgres-data:/var/lib/postgresql/data
    restart: unless-stopped

  redis:
    image: redis:7-alpine
    command: redis-server --appendonly yes
    volumes:
      - redis-data:/data
    restart: unless-stopped

  elasticsearch:
    image: docker.elastic.co/elasticsearch/elasticsearch:8.11.0
    environment:
      - discovery.type=single-node
      - xpack.security.enabled=false
      - "ES_JAVA_OPTS=-Xms512m -Xmx512m"
    volumes:
      - elasticsearch-data:/usr/share/elasticsearch/data
    restart: unless-stopped

  qdrant:
    image: qdrant/qdrant:latest
    ports:
      - "6333:6333"
    volumes:
      - qdrant-data:/qdrant/storage
    restart: unless-stopped

volumes:
  app-data:
  app-logs:
  postgres-data:
  redis-data:
  elasticsearch-data:
  qdrant-data:
```

### 3. Deploy with Docker Compose

```bash
# Deploy production stack
docker-compose -f docker-compose.prod.yml up -d

# Scale API service
docker-compose -f docker-compose.prod.yml up -d --scale api=3

# View logs
docker-compose -f docker-compose.prod.yml logs -f api

# Update deployment
docker-compose -f docker-compose.prod.yml pull
docker-compose -f docker-compose.prod.yml up -d
```

## Kubernetes Deployment

### 1. Prerequisites

```bash
# Install kubectl
curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
sudo install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl

# Install Helm
curl https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash

# Configure kubectl for your cluster
kubectl config use-context your-cluster-context
```

### 2. Deploy to Kubernetes

```bash
# Create namespace
kubectl create namespace universal-knowledge-platform

# Apply secrets
kubectl apply -f k8s/secrets.yaml

# Apply configuration
kubectl apply -f k8s/configmap.yaml

# Deploy application
kubectl apply -f k8s/deployment.yaml

# Verify deployment
kubectl get pods -n universal-knowledge-platform
kubectl get services -n universal-knowledge-platform
kubectl get ingress -n universal-knowledge-platform
```

### 3. Helm Chart Deployment

```bash
# Add Helm repository
helm repo add universal-knowledge-platform https://your-helm-repo.com

# Install with Helm
helm install ukp universal-knowledge-platform/universal-knowledge-platform \
  --namespace universal-knowledge-platform \
  --set environment=production \
  --set replicaCount=3 \
  --set resources.requests.memory=512Mi \
  --set resources.requests.cpu=250m

# Upgrade deployment
helm upgrade ukp universal-knowledge-platform/universal-knowledge-platform \
  --namespace universal-knowledge-platform \
  --set image.tag=v2.0.1
```

## CI/CD Pipeline

### 1. GitHub Actions Setup

The CI/CD pipeline is configured in `.github/workflows/ci-cd.yml` and includes:

- **Code Quality**: Linting, formatting, type checking
- **Unit Tests**: Fast, isolated tests
- **Integration Tests**: External service integration
- **Performance Tests**: Load testing and benchmarks
- **Security Tests**: Vulnerability scanning
- **Docker Build**: Multi-architecture images
- **Deployment**: Staging and production environments

### 2. Pipeline Stages

```yaml
# Pipeline stages
code-quality -> unit-tests -> integration-tests -> docker-build -> deploy-staging -> deploy-production
```

### 3. Environment Configuration

Configure GitHub environments:

```bash
# Staging environment
kubectl config set-context staging-cluster
kubectl create namespace staging

# Production environment
kubectl config set-context production-cluster
kubectl create namespace production
```

## Monitoring & Observability

### 1. Prometheus Configuration

```yaml
# prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'universal-knowledge-platform'
    static_configs:
      - targets: ['api:8000']
    metrics_path: '/metrics'
    scrape_interval: 10s
```

### 2. Grafana Dashboards

Import the following dashboards:

- **API Performance**: Request rate, response time, error rate
- **System Resources**: CPU, memory, disk usage
- **Security Metrics**: Threat detection, rate limiting
- **Business Metrics**: User activity, query performance

### 3. Alerting Rules

```yaml
# alerting-rules.yml
groups:
  - name: api_alerts
    rules:
      - alert: HighErrorRate
        expr: rate(http_requests_total{status=~"5.."}[5m]) > 0.1
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "High error rate detected"
          description: "Error rate is {{ $value }} errors per second"

      - alert: HighResponseTime
        expr: histogram_quantile(0.95, http_request_duration_seconds) > 2
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High response time detected"
          description: "95th percentile response time is {{ $value }} seconds"
```

## Security Configuration

### 1. Network Security

```bash
# Configure firewall
sudo ufw allow 22/tcp
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
sudo ufw enable

# Configure SSL/TLS
certbot --nginx -d api.yourdomain.com
```

### 2. Application Security

```python
# Security headers configuration
SECURITY_HEADERS = {
    "Content-Security-Policy": "default-src 'self'",
    "X-Frame-Options": "DENY",
    "X-Content-Type-Options": "nosniff",
    "X-XSS-Protection": "1; mode=block",
    "Referrer-Policy": "strict-origin-when-cross-origin",
    "Permissions-Policy": "geolocation=(), microphone=(), camera=()",
    "Strict-Transport-Security": "max-age=31536000; includeSubDomains"
}
```

### 3. Secret Management

```bash
# Use Kubernetes secrets
kubectl create secret generic app-secrets \
  --from-literal=database-url="postgresql://user:pass@db:5432/ukp" \
  --from-literal=secret-key="your-secret-key" \
  --from-literal=openai-api-key="sk-your-key"

# Use external secret management
helm install external-secrets external-secrets/external-secrets \
  --namespace external-secrets \
  --create-namespace
```

## Performance Optimization

### 1. Application Optimization

```python
# Database connection pooling
DATABASE_CONFIG = {
    "pool_size": 20,
    "max_overflow": 30,
    "pool_pre_ping": True,
    "pool_recycle": 3600
}

# Cache configuration
CACHE_CONFIG = {
    "default_ttl": 300,
    "max_size": 10000,
    "compression_threshold": 1024
}

# Rate limiting
RATE_LIMIT_CONFIG = {
    "requests_per_minute": 60,
    "burst_size": 10,
    "algorithm": "token_bucket"
}
```

### 2. Infrastructure Optimization

```yaml
# Resource limits
resources:
  requests:
    memory: "512Mi"
    cpu: "250m"
  limits:
    memory: "1Gi"
    cpu: "500m"

# Horizontal Pod Autoscaler
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: api-hpa
spec:
  minReplicas: 3
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
```

### 3. Monitoring Performance

```bash
# Monitor application performance
kubectl top pods -n universal-knowledge-platform

# Check resource usage
kubectl describe hpa api-hpa -n universal-knowledge-platform

# View metrics
curl http://api.yourdomain.com/metrics
```

## Troubleshooting

### 1. Common Issues

#### Application Won't Start

```bash
# Check logs
kubectl logs -f deployment/universal-knowledge-platform

# Check environment variables
kubectl describe pod <pod-name>

# Check resource limits
kubectl describe pod <pod-name> | grep -A 10 "Events:"
```

#### Database Connection Issues

```bash
# Test database connectivity
kubectl exec -it <pod-name> -- python -c "
import psycopg2
try:
    conn = psycopg2.connect('$DATABASE_URL')
    print('Database connection successful')
except Exception as e:
    print(f'Database connection failed: {e}')
"
```

#### High Memory Usage

```bash
# Check memory usage
kubectl top pods -n universal-knowledge-platform

# Analyze memory profile
kubectl exec -it <pod-name> -- python -c "
import psutil
print(f'Memory usage: {psutil.virtual_memory().percent}%')
"
```

### 2. Debug Commands

```bash
# Get pod information
kubectl get pods -n universal-knowledge-platform

# Describe pod
kubectl describe pod <pod-name> -n universal-knowledge-platform

# Execute command in pod
kubectl exec -it <pod-name> -n universal-knowledge-platform -- /bin/bash

# Port forward for debugging
kubectl port-forward <pod-name> 8000:8000 -n universal-knowledge-platform

# View logs
kubectl logs -f <pod-name> -n universal-knowledge-platform
```

### 3. Performance Debugging

```bash
# Check slow queries
kubectl exec -it <pod-name> -- python -c "
from api.database import get_db
db = get_db()
result = db.execute('SELECT * FROM pg_stat_statements ORDER BY mean_time DESC LIMIT 10')
for row in result:
    print(f'{row.query}: {row.mean_time}ms')
"

# Check cache hit rate
curl http://api.yourdomain.com/metrics | grep cache

# Check rate limiting
curl http://api.yourdomain.com/metrics | grep rate_limit
```

## Support

For deployment issues:

1. **Check Documentation**: Review this guide and API documentation
2. **Monitor Logs**: Use the logging commands above
3. **Check Metrics**: Monitor application metrics
4. **Contact Support**: Reach out to the engineering team

## Contributing

To contribute to deployment improvements:

1. **Fork Repository**: Create your own fork
2. **Create Branch**: Make changes in a feature branch
3. **Test Changes**: Ensure all tests pass
4. **Submit PR**: Create a pull request with detailed description
5. **Code Review**: Address feedback from reviewers

---

**Note**: This deployment guide follows MAANG engineering standards for reliability, scalability, and maintainability. Always test changes in staging before deploying to production. 