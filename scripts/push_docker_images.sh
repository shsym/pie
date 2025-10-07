#!/bin/bash
# Push PIE Docker images to Docker Hub
# Usage: ./scripts/push_docker_images.sh [username]
# Default username: sslee0cs

set -e

# Check if running as root
if [ "$EUID" -eq 0 ]; then
    SUDO=""
else
    SUDO="sudo"
fi

# Configuration
USERNAME="${1:-sslee0cs}"
IMAGE_NAME="pie"

echo "================================"
echo "PIE Docker Image Push Script"
echo "================================"
echo "Username: $USERNAME"
echo ""

# Check if Docker is running
if ! $SUDO docker info >/dev/null 2>&1; then
    echo "Error: Docker is not running"
    exit 1
fi

# Check if logged in to Docker Hub
if ! $SUDO docker info 2>/dev/null | grep -q "Username"; then
    echo "Warning: You may not be logged in to Docker Hub"
    echo "Run: $SUDO docker login"
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

echo "Pushing images to Docker Hub..."
echo ""

# Tag and push latest
echo "1. Tagging and pushing ${USERNAME}/${IMAGE_NAME}:latest"
$SUDO docker tag ${IMAGE_NAME}:latest ${USERNAME}/${IMAGE_NAME}:latest
$SUDO docker push ${USERNAME}/${IMAGE_NAME}:latest
echo "✓ ${USERNAME}/${IMAGE_NAME}:latest pushed successfully"
echo ""

# Tag and push dev
echo "2. Tagging and pushing ${USERNAME}/${IMAGE_NAME}:dev"
$SUDO docker tag ${IMAGE_NAME}:dev ${USERNAME}/${IMAGE_NAME}:dev
$SUDO docker push ${USERNAME}/${IMAGE_NAME}:dev
echo "✓ ${USERNAME}/${IMAGE_NAME}:dev pushed successfully"
echo ""

echo "================================"
echo "All images pushed successfully!"
echo "================================"
echo ""
echo "Available images:"
echo "  - ${USERNAME}/${IMAGE_NAME}:latest (production)"
echo "  - ${USERNAME}/${IMAGE_NAME}:dev (development/RunPod)"
echo ""
echo "Pull commands:"
echo "  docker pull ${USERNAME}/${IMAGE_NAME}:latest"
echo "  docker pull ${USERNAME}/${IMAGE_NAME}:dev"
