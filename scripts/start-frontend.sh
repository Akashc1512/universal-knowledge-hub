#!/bin/bash

# Start Next.js Frontend Server
# This script starts the Next.js frontend with hot reload

cd "$(dirname "$0")/../frontend"

echo "🚀 Starting Next.js frontend server..."
echo "🌐 Server will be available at: http://localhost:3000"
echo ""

# Start the Next.js development server
npm run dev 