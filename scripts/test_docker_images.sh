#!/bin/bash
# Test script for PIE Docker images
# Validates build, server startup, authentication, and inferlet execution
#
# This script performs a complete end-to-end test of the PIE Docker setup:
# 1. Checks prerequisites (docker, pie-cli, SSH keys)
# 2. Builds Docker images using build_docker_images.sh
# 3. Starts PIE server in Docker with authentication
# 4. Configures pie-cli to connect to the server
# 5. Tests connection with ping
# 6. Submits echo inferlet (simple test)
# 7. Submits text completion inferlet (full LLM test)
# 8. Cleans up containers and temporary files
#
# Prerequisites:
#   - Docker installed and running
#   - pie-cli installed (cargo install --path client/cli)
#   - SSH key pair at ~/.ssh/id_ed25519
#   - NVIDIA GPU with CUDA support (for Docker)
#
# Usage:
#   ./scripts/test_docker_images.sh
#
# Exit codes:
#   0 - All tests passed
#   1 - Test failed (check output for details)

set -e  # Exit on any error

echo "=== PIE Docker Image Test Suite ==="
echo ""

# Check prerequisites
check_prerequisites() {
    local missing=0

    echo "Checking prerequisites..."

    if ! command -v docker &> /dev/null; then
        echo "❌ docker not found"
        missing=1
    else
        echo "✅ docker found"
    fi

    if ! command -v pie-cli &> /dev/null; then
        echo "❌ pie-cli not found"
        echo "   Install with: cargo install --path client/cli"
        missing=1
    else
        echo "✅ pie-cli found"
    fi

    if [ ! -f ~/.ssh/id_ed25519 ] && [ ! -f ~/.ssh/id_ed25519.pub ]; then
        echo "❌ SSH key pair not found at ~/.ssh/id_ed25519"
        echo "   Generate with: ssh-keygen -t ed25519 -C \"your_email@example.com\""
        missing=1
    else
        echo "✅ SSH key pair found"
    fi

    if [ $missing -eq 1 ]; then
        echo ""
        echo "Please install missing prerequisites and try again."
        exit 1
    fi

    echo ""
}

check_prerequisites