#!/bin/bash
# Test script for PIE Docker images
# Validates build, server startup, authentication, and inferlet execution
#
# This script performs a complete end-to-end test of the PIE Docker setup:
# 1. Checks prerequisites (docker, pie-cli, SSH keys)
# 2. Verifies Docker images exist (does NOT build them)
# 3. Starts PIE server in Docker with authentication
# 4. Configures pie-cli to connect to the server
# 5. Tests connection with ping
# 6. Submits echo inferlet (simple test)
# 7. Submits text completion inferlet (full LLM test)
# 8. Cleans up containers and temporary files
#
# Note: This script assumes Docker images are already built.
# To build images first, run: ./scripts/build_docker_images.sh
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

# Cleanup function
cleanup() {
    echo ""
    echo "=== Cleanup ==="
    sudo docker stop pie-test-server 2>/dev/null || true
    sudo docker rm pie-test-server 2>/dev/null || true
    rm -f ./echo.wasm ./text_completion.wasm 2>/dev/null || true
    echo "✅ Cleanup complete"
}

# Set trap to cleanup on exit
trap cleanup EXIT

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

# Verify Docker images exist
verify_images() {
    echo "=== Verifying Docker Images Exist ==="
    echo ""

    # Check if pie:latest image exists
    if ! sudo docker images | grep -q "^pie.*latest"; then
        echo "❌ Docker image 'pie:latest' not found"
        echo ""
        echo "Please build Docker images first:"
        echo "  ./scripts/build_docker_images.sh"
        echo ""
        exit 1
    fi

    local images=$(sudo docker images | grep -E "^pie" | wc -l)

    echo "Found $images pie image(s):"
    sudo docker images | grep -E "^(REPOSITORY|pie)"
    echo ""
    echo "✅ Images verified"
    echo ""
}

# Start PIE server with authentication
start_server() {
    echo "=== Starting PIE Server ==="
    echo ""

    local username=$(whoami)
    local ssh_key=$(cat ~/.ssh/id_ed25519.pub)

    echo "Username: $username"
    echo "Container: pie-test-server"
    echo "Port: 8080"
    echo ""

    sudo docker run -d --gpus all -p 8080:8080 \
        --name pie-test-server \
        -e PIE_AUTH_USER="$username" \
        -e "PIE_AUTH_KEY=$ssh_key" \
        -v ~/.cache:/root/.cache \
        pie:latest

    echo "Waiting for server startup..."
    sleep 10

    echo ""
    echo "✅ Server started"
    echo ""
}

# Verify server is running and authentication configured
verify_server() {
    echo "=== Verifying Server Status ==="
    echo ""

    # Check container is running
    if ! sudo docker ps | grep -q pie-test-server; then
        echo "❌ Server container not running"
        sudo docker logs pie-test-server 2>&1 | tail -20
        exit 1
    fi

    # Check logs for auth setup
    local logs=$(sudo docker logs pie-test-server 2>&1)

    if echo "$logs" | grep -q "🔐 Setting up authentication"; then
        echo "✅ Authentication setup initiated"
    else
        echo "⚠️  No authentication setup message found"
    fi

    if echo "$logs" | grep -q "✅ Authentication configured successfully"; then
        echo "✅ Authentication configured"
    else
        echo "❌ Authentication configuration failed"
        echo "$logs" | grep -E "(Authentication|Error)" | tail -10
        exit 1
    fi

    if echo "$logs" | grep -q "PIE runtime started successfully"; then
        echo "✅ PIE runtime started"
    else
        echo "❌ PIE runtime failed to start"
        echo "$logs" | tail -20
        exit 1
    fi

    if echo "$logs" | grep -q "Handler initialized with flashinfer backend"; then
        echo "✅ FlashInfer backend initialized"
    else
        echo "⚠️  Backend initialization not confirmed"
    fi

    echo ""
    echo "✅ Server verification complete"
    echo ""
}

# Configure pie-cli to connect to Docker server
configure_client() {
    echo "=== Configuring pie-cli ==="
    echo ""

    local username=$(whoami)

    # Initialize config (will prompt if exists, so check first)
    if [ ! -f ~/.pie_cli/config.toml ]; then
        echo "Creating new pie-cli config..."
        pie-cli config init --enable-auth true
    else
        echo "Using existing pie-cli config"
    fi

    # Update config with test server settings
    pie-cli config update \
        --username "$username" \
        --host localhost \
        --port 8080 \
        --private-key-path ~/.ssh/id_ed25519

    echo ""
    echo "Current config:"
    pie-cli config show

    echo ""
    echo "✅ pie-cli configured"
    echo ""
}

# Test connection to server
test_connection() {
    echo "=== Testing Connection ==="
    echo ""

    if pie-cli ping; then
        echo ""
        echo "✅ Connection successful"
    else
        echo ""
        echo "❌ Connection failed"
        echo "Server logs:"
        sudo docker logs pie-test-server 2>&1 | tail -20
        exit 1
    fi

    echo ""
}

# Copy inferlets from Docker to host
copy_inferlets() {
    echo "=== Copying Inferlets from Container ==="
    echo ""

    sudo docker cp pie-test-server:/workspace/example-apps/echo.wasm ./echo.wasm
    sudo docker cp pie-test-server:/workspace/example-apps/text_completion.wasm ./text_completion.wasm

    if [ -f ./echo.wasm ] && [ -f ./text_completion.wasm ]; then
        echo "✅ Inferlets copied:"
        ls -lh ./echo.wasm ./text_completion.wasm
    else
        echo "❌ Failed to copy inferlets"
        exit 1
    fi

    echo ""
}

# Test echo inferlet (simple, no model needed)
test_echo() {
    echo "=== Testing Echo Inferlet ==="
    echo ""

    local output=$(pie-cli submit ./echo.wasm -- --text "Docker test successful!" 2>&1)

    echo "$output"

    if echo "$output" | grep -q "Docker test successful!"; then
        echo ""
        echo "✅ Echo inferlet test passed"
    else
        echo ""
        echo "❌ Echo inferlet test failed"
        exit 1
    fi

    echo ""
}

# Test text completion inferlet (uses LLM)
test_text_completion() {
    echo "=== Testing Text Completion Inferlet ==="
    echo ""

    local output=$(pie-cli submit ./text_completion.wasm -- --prompt "hello" 2>&1)

    echo "$output"

    # Verify output contains expected elements
    local checks_passed=0

    if echo "$output" | grep -q "Inferlet hash:"; then
        echo "✅ Inferlet uploaded"
        ((checks_passed++))
    fi

    if echo "$output" | grep -q "Inferlet launched with ID:"; then
        echo "✅ Inferlet launched"
        ((checks_passed++))
    fi

    if echo "$output" | grep -q "Output:"; then
        echo "✅ Got output from LLM"
        ((checks_passed++))
    fi

    if echo "$output" | grep -q "Per token latency:"; then
        echo "✅ Performance metrics reported"
        ((checks_passed++))
    fi

    if echo "$output" | grep -q "Completed:"; then
        echo "✅ Inferlet completed"
        ((checks_passed++))
    fi

    if [ $checks_passed -eq 5 ]; then
        echo ""
        echo "✅ Text completion inferlet test passed (5/5 checks)"
    else
        echo ""
        echo "⚠️  Text completion test incomplete ($checks_passed/5 checks passed)"
        exit 1
    fi

    echo ""
}

# Main execution
check_prerequisites
verify_images
start_server
verify_server
configure_client
test_connection
copy_inferlets
test_echo
test_text_completion

# Print final success message
echo "=========================================="
echo "✅ All Docker tests passed!"
echo "=========================================="
echo ""
echo "Summary:"
echo "  ✅ Docker images verified"
echo "  ✅ Server started with authentication"
echo "  ✅ pie-cli connected successfully"
echo "  ✅ Echo inferlet executed"
echo "  ✅ Text completion inferlet executed"
echo ""