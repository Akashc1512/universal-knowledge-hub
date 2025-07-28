# 🔑 Complete Guide to Getting API Keys for Universal Knowledge Platform

## 📋 Quick Links to Get Started

### 🤖 AI Services (Already Have)
- ✅ **OpenAI** - You already have this
- ✅ **Anthropic** - You already have this

### 📊 Vector Databases
- ✅ **Pinecone** - You already have this
- 🔗 **Qdrant** - [Get Started](#qdrant-optional)

### 🔍 Search & Databases
- ✅ **Elasticsearch** - You already have this
- ✅ **Neo4j** - You already have this
- 🔗 **Redis** - [Setup Guide](#redis-cache)

### ☁️ Cloud Services
- ✅ **AWS** - You already have this
- ✅ **SSL Certificate** - You already have this

---

## 🚀 Services You Still Need to Set Up

### 1. Redis Cache (Required for Performance)

#### Option A: Local Redis with Docker (Recommended for Development)
```bash
# Run Redis locally
docker run -d --name redis-stack -p 6379:6379 redis/redis-stack-server:latest

# Your .env configuration:
REDIS_URL=redis://localhost:6379/0
REDIS_PASSWORD=  # Leave empty for local development
```

#### Option B: Redis Cloud (Free Tier Available)
1. Visit [Redis Cloud](https://redis.com/try-free/)
2. Sign up for free account
3. Create a new database (30MB free)
4. Copy the connection details:
   ```
   REDIS_URL=redis://default:YOUR_PASSWORD@YOUR_ENDPOINT:PORT
   ```

### 2. PostgreSQL Database (For Persistent Storage)

#### Option A: Local PostgreSQL with Docker
```bash
# Run PostgreSQL locally
docker run -d \
  --name postgres \
  -e POSTGRES_PASSWORD=your_secure_password \
  -e POSTGRES_DB=ukp_db \
  -p 5432:5432 \
  postgres:15

# Your .env configuration:
DATABASE_HOST=localhost
DATABASE_PORT=5432
DATABASE_NAME=ukp_db
DATABASE_USERNAME=postgres
DATABASE_PASSWORD=your_secure_password
```

#### Option B: AWS RDS (Already have AWS)
Since you have AWS configured:
1. Go to AWS RDS Console
2. Create a PostgreSQL instance (free tier eligible)
3. Note the endpoint and credentials

### 3. Qdrant Vector Database (Optional - Alternative to Pinecone)

#### Option A: Qdrant Cloud (Free Tier)
1. Visit [Qdrant Cloud](https://cloud.qdrant.io/)
2. Sign up for free account
3. Create a cluster (1GB free storage)
4. Get your API key and URL:
   ```
   QDRANT_URL=https://YOUR-CLUSTER-ID.qdrant.io
   QDRANT_API_KEY=YOUR_API_KEY
   ```

#### Option B: Local Qdrant with Docker
```bash
docker run -d -p 6333:6333 qdrant/qdrant

# Your .env configuration:
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=  # Not needed for local
```

### 4. Application Secrets (Generate These)

Generate secure secrets for your application:

```bash
# Generate SECRET_KEY
python -c "import secrets; print('SECRET_KEY=' + secrets.token_urlsafe(32))"

# Generate API_KEY_SECRET
python -c "import secrets; print('API_KEY_SECRET=' + secrets.token_urlsafe(32))"

# Generate ENCRYPTION_KEY
python -c "import secrets; print('ENCRYPTION_KEY=' + secrets.token_urlsafe(32))"

# Generate Admin API Key
python -c "import secrets; print('ADMIN_API_KEY=admin-' + secrets.token_urlsafe(32))"

# Generate User API Key
python -c "import secrets; print('USER_API_KEY=user-' + secrets.token_urlsafe(32))"

# Generate Readonly API Key
python -c "import secrets; print('READONLY_API_KEY=readonly-' + secrets.token_urlsafe(32))"
```

### 5. Monitoring Services (Optional but Recommended)

#### Prometheus (Metrics)
```bash
# Run Prometheus locally
docker run -d -p 9090:9090 prom/prometheus

# Your .env configuration:
PROMETHEUS_URL=http://localhost:9090
```

#### Grafana (Dashboards)
```bash
# Run Grafana locally
docker run -d -p 3000:3000 grafana/grafana

# Your .env configuration:
GRAFANA_URL=http://localhost:3000
GRAFANA_USERNAME=admin
GRAFANA_PASSWORD=admin  # Change on first login
```

---

## 📝 Complete .env Configuration

Here's your complete .env file with all services:

```bash
# =============================================================================
# AI SERVICES (You already have these)
# =============================================================================
OPENAI_API_KEY=your-actual-openai-key
ANTHROPIC_API_KEY=your-actual-anthropic-key

# =============================================================================
# VECTOR DATABASES
# =============================================================================
# Pinecone (You already have this)
PINECONE_API_KEY=your-actual-pinecone-key
PINECONE_ENVIRONMENT=us-west1-gcp
PINECONE_INDEX_NAME=ukp-knowledge-base

# Qdrant (Optional - choose one option above)
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=

# =============================================================================
# DATABASES & CACHE
# =============================================================================
# PostgreSQL (Choose local or AWS RDS)
DATABASE_HOST=localhost
DATABASE_PORT=5432
DATABASE_NAME=ukp_db
DATABASE_USERNAME=postgres
DATABASE_PASSWORD=your_secure_password

# Elasticsearch (You already have this)
ELASTICSEARCH_URL=your-elasticsearch-url
ELASTICSEARCH_USERNAME=elastic
ELASTICSEARCH_PASSWORD=your-elasticsearch-password
ELASTICSEARCH_INDEX=universal-knowledge-hub

# Neo4j (You already have this)
NEO4J_URI=your-neo4j-uri
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your-neo4j-password

# Redis (Choose local or cloud)
REDIS_URL=redis://localhost:6379/0
REDIS_PASSWORD=

# =============================================================================
# APPLICATION SECRETS (Generate these!)
# =============================================================================
SECRET_KEY=GENERATE_WITH_COMMAND_ABOVE
API_KEY_SECRET=GENERATE_WITH_COMMAND_ABOVE
ENCRYPTION_KEY=GENERATE_WITH_COMMAND_ABOVE

# API Keys for Authentication
ADMIN_API_KEY=GENERATE_WITH_COMMAND_ABOVE
USER_API_KEY=GENERATE_WITH_COMMAND_ABOVE
READONLY_API_KEY=GENERATE_WITH_COMMAND_ABOVE

# =============================================================================
# AWS (You already have this)
# =============================================================================
AWS_ACCESS_KEY_ID=your-aws-access-key
AWS_SECRET_ACCESS_KEY=your-aws-secret-key
AWS_REGION=us-east-1

# =============================================================================
# MONITORING (Optional)
# =============================================================================
PROMETHEUS_URL=http://localhost:9090
GRAFANA_URL=http://localhost:3000
GRAFANA_USERNAME=admin
GRAFANA_PASSWORD=admin
```

---

## 🚀 Quick Start Commands

### 1. Start All Local Services with Docker Compose

Create a `docker-compose.yml` file:

```yaml
version: '3.8'

services:
  redis:
    image: redis/redis-stack-server:latest
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data

  postgres:
    image: postgres:15
    environment:
      POSTGRES_PASSWORD: your_secure_password
      POSTGRES_DB: ukp_db
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data

  elasticsearch:
    image: elasticsearch:8.11.0
    environment:
      - discovery.type=single-node
      - xpack.security.enabled=false
    ports:
      - "9200:9200"
    volumes:
      - es_data:/usr/share/elasticsearch/data

  neo4j:
    image: neo4j:latest
    environment:
      NEO4J_AUTH: neo4j/your_neo4j_password
    ports:
      - "7474:7474"
      - "7687:7687"
    volumes:
      - neo4j_data:/data

  qdrant:
    image: qdrant/qdrant
    ports:
      - "6333:6333"
    volumes:
      - qdrant_data:/qdrant/storage

volumes:
  redis_data:
  postgres_data:
  es_data:
  neo4j_data:
  qdrant_data:
```

Then run:
```bash
# Start all services
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f
```

### 2. Generate All Secrets at Once

```bash
# Run this script to generate all secrets
cat << 'EOF' > generate_secrets.py
import secrets

print("# Generated Secrets - Add these to your .env file")
print("# " + "="*50)
print(f"SECRET_KEY={secrets.token_urlsafe(32)}")
print(f"API_KEY_SECRET={secrets.token_urlsafe(32)}")
print(f"ENCRYPTION_KEY={secrets.token_urlsafe(32)}")
print(f"ADMIN_API_KEY=admin-{secrets.token_urlsafe(32)}")
print(f"USER_API_KEY=user-{secrets.token_urlsafe(32)}")
print(f"READONLY_API_KEY=readonly-{secrets.token_urlsafe(32)}")
print("# " + "="*50)
EOF

python generate_secrets.py
```

### 3. Update Your .env File

```bash
# Edit your .env file
nano .env

# Add the generated secrets and service URLs
# Save and exit (Ctrl+X, Y, Enter)
```

### 4. Verify Everything is Working

```bash
# Test your configuration
python test_basic_setup.py

# If all good, start the server
python start_server.py
```

---

## 🔍 Service Connection Strings Summary

| Service | Local Development | Cloud Option |
|---------|------------------|--------------|
| Redis | `redis://localhost:6379/0` | `redis://user:pass@host:port` |
| PostgreSQL | `postgresql://postgres:password@localhost:5432/ukp_db` | AWS RDS endpoint |
| Elasticsearch | `http://localhost:9200` | Elastic Cloud endpoint |
| Neo4j | `bolt://localhost:7687` | Neo4j Aura endpoint |
| Qdrant | `http://localhost:6333` | `https://cluster.qdrant.io` |

---

## ❓ Troubleshooting

### Docker Not Installed?
```bash
# Install Docker (Ubuntu/Debian)
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Add user to docker group
sudo usermod -aG docker $USER
# Log out and back in
```

### Port Already in Use?
```bash
# Check what's using a port (e.g., 6379)
sudo lsof -i :6379

# Kill the process if needed
sudo kill -9 <PID>
```

### Can't Connect to Service?
1. Check if container is running: `docker ps`
2. Check logs: `docker logs <container_name>`
3. Test connection: `telnet localhost <port>`

---

## 🎯 Next Steps

1. **Choose your setup approach**:
   - **Quick Start**: Use Docker Compose for all services locally
   - **Production**: Use cloud services for reliability

2. **Generate all secrets** using the script above

3. **Update your .env file** with all values

4. **Start your services** and test the application

Need help with any specific service? Let me know which one you'd like to set up first! 