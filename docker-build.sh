#!/bin/bash
# Docker build helper script

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Default values
BUILD_TYPE="production"
TAG="latest"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --dev|--development)
            BUILD_TYPE="development"
            shift
            ;;
        --test|--testing)
            BUILD_TYPE="testing"
            shift
            ;;
        --tag)
            TAG="$2"
            shift 2
            ;;
        --help)
            echo "Usage: ./docker-build.sh [options]"
            echo "Options:"
            echo "  --dev, --development  Build development image"
            echo "  --test, --testing     Build testing image"
            echo "  --tag TAG            Tag for the image (default: latest)"
            echo "  --help               Show this help message"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

echo -e "${GREEN}Building Docker image...${NC}"
echo "Build type: $BUILD_TYPE"
echo "Tag: $TAG"

# Check if .env.example exists
if [ ! -f ".env.example" ]; then
    echo -e "${YELLOW}Warning: .env.example not found. Creating from env.template...${NC}"
    cp env.template .env.example
fi

# Build the image
IMAGE_NAME="ukp-backend"
FULL_TAG="${IMAGE_NAME}:${TAG}"

echo -e "${GREEN}Building ${FULL_TAG} (${BUILD_TYPE})...${NC}"

docker build \
    --target "${BUILD_TYPE}" \
    --tag "${FULL_TAG}" \
    --tag "${IMAGE_NAME}:${BUILD_TYPE}" \
    .

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Build successful!${NC}"
    echo -e "${GREEN}Image tagged as:${NC}"
    echo "  - ${FULL_TAG}"
    echo "  - ${IMAGE_NAME}:${BUILD_TYPE}"
    
    echo -e "\n${GREEN}To run the container:${NC}"
    case $BUILD_TYPE in
        production)
            echo "docker run -p 8002:8002 --env-file .env ${FULL_TAG}"
            ;;
        development)
            echo "docker run -p 8002:8002 -v \$(pwd):/app --env-file .env ${FULL_TAG}"
            ;;
        testing)
            echo "docker run --env-file .env ${FULL_TAG}"
            ;;
    esac
else
    echo -e "${RED}❌ Build failed!${NC}"
    exit 1
fi 