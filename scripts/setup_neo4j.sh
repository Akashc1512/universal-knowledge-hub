#!/bin/bash
# Setup Neo4j locally with Docker
# Created: December 19, 2024

echo "🔗 Setting up Neo4j Knowledge Graph..."

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    echo "   Visit: https://docs.docker.com/get-docker/"
    exit 1
fi

# Pull and run Neo4j
echo "📦 Pulling Neo4j Docker image..."
docker pull neo4j:5.15.0

echo "🚀 Starting Neo4j..."
docker run -d \
  --name neo4j \
  -p 7474:7474 \
  -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/password123 \
  -e NEO4J_PLUGINS='["apoc"]' \
  -v $HOME/neo4j/data:/data \
  -v $HOME/neo4j/logs:/logs \
  neo4j:5.15.0

echo "⏳ Waiting for Neo4j to start..."
sleep 15

# Test connection
if curl -s http://localhost:7474 > /dev/null; then
    echo "✅ Neo4j is running!"
    echo "   Web Interface: http://localhost:7474"
    echo "   Bolt URL: bolt://localhost:7687"
    echo ""
    echo "📝 Update your .env file:"
    echo "   NEO4J_URI=bolt://localhost:7687"
    echo "   NEO4J_USERNAME=neo4j"
    echo "   NEO4J_PASSWORD=password123"
else
    echo "❌ Failed to connect to Neo4j"
fi 