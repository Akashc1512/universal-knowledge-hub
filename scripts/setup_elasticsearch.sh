#!/bin/bash
# Setup Elasticsearch locally with Docker
# Created: December 19, 2024

echo "🔍 Setting up Elasticsearch..."

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    echo "   Visit: https://docs.docker.com/get-docker/"
    exit 1
fi

# Pull and run Elasticsearch
echo "📦 Pulling Elasticsearch Docker image..."
docker pull elasticsearch:8.11.0

echo "🚀 Starting Elasticsearch..."
docker run -d \
  --name elasticsearch \
  -p 9200:9200 \
  -p 9300:9300 \
  -e "discovery.type=single-node" \
  -e "xpack.security.enabled=false" \
  -e "ES_JAVA_OPTS=-Xms512m -Xmx512m" \
  elasticsearch:8.11.0

echo "⏳ Waiting for Elasticsearch to start..."
sleep 10

# Test connection
if curl -s http://localhost:9200 > /dev/null; then
    echo "✅ Elasticsearch is running at http://localhost:9200"
    echo ""
    echo "📝 Update your .env file:"
    echo "   ELASTICSEARCH_URL=http://localhost:9200"
else
    echo "❌ Failed to connect to Elasticsearch"
fi 