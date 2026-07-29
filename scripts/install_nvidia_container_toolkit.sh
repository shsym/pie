#!/bin/bash
# Script to install NVIDIA Container Toolkit for Docker GPU support
# This enables Docker containers to access NVIDIA GPUs

set -e

echo "=================================================="
echo "NVIDIA Container Toolkit Installation Script"
echo "=================================================="
echo ""

# Check if running as root
if [ "$EUID" -eq 0 ]; then
    SUDO=""
else
    SUDO="sudo"
fi

# Check if nvidia-smi is available
if ! command -v nvidia-smi &> /dev/null; then
    echo "ERROR: nvidia-smi not found. Please install NVIDIA drivers first."
    exit 1
fi

echo "✓ NVIDIA driver detected:"
nvidia-smi --query-gpu=driver_version,name --format=csv,noheader
echo ""

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "ERROR: Docker not found. Please install Docker first."
    exit 1
fi

echo "✓ Docker detected:"
docker --version
echo ""

# Detect OS
if [ -f /etc/os-release ]; then
    . /etc/os-release
    OS=$ID
    VER=$VERSION_ID
else
    echo "ERROR: Cannot detect OS. /etc/os-release not found."
    exit 1
fi

echo "Detected OS: $OS $VER"
echo ""

# Install based on OS
case "$OS" in
    ubuntu|debian)
        echo "Installing NVIDIA Container Toolkit for Debian/Ubuntu..."

        # Install required dependencies
        echo "→ Installing required dependencies..."
        $SUDO apt-get update
        $SUDO apt-get install -y curl gnupg

        # Configure the repository
        echo "→ Configuring NVIDIA Container Toolkit repository..."
        curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
            $SUDO gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

        curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
            sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
            $SUDO tee /etc/apt/sources.list.d/nvidia-container-toolkit.list > /dev/null

        # Install the toolkit
        echo "→ Installing nvidia-container-toolkit package..."
        $SUDO apt-get update
        $SUDO apt-get install -y nvidia-container-toolkit
        ;;

    centos|rhel|fedora)
        echo "Installing NVIDIA Container Toolkit for RHEL/CentOS/Fedora..."

        # Configure the repository
        echo "→ Configuring NVIDIA Container Toolkit repository..."
        curl -s -L https://nvidia.github.io/libnvidia-container/stable/rpm/nvidia-container-toolkit.repo | \
            $SUDO tee /etc/yum.repos.d/nvidia-container-toolkit.repo > /dev/null

        # Install the toolkit
        echo "→ Installing nvidia-container-toolkit package..."
        $SUDO yum install -y nvidia-container-toolkit
        ;;

    *)
        echo "ERROR: Unsupported OS: $OS"
        echo "Please install manually from: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html"
        exit 1
        ;;
esac

# Configure Docker to use NVIDIA runtime
echo ""
echo "→ Configuring Docker to use NVIDIA runtime..."
$SUDO nvidia-ctk runtime configure --runtime=docker

# Restart Docker (if systemd is available)
if command -v systemctl &> /dev/null && systemctl is-system-running &> /dev/null; then
    echo "→ Restarting Docker daemon..."
    $SUDO systemctl restart docker
else
    echo "⚠️  Systemd is not available."
    echo "Pie doest not support nested containers yet."
fi

# Verify installation
echo ""
echo "=================================================="
echo "Verifying Installation"
echo "=================================================="
echo ""

echo "Testing GPU access in Docker..."
if $SUDO docker run --rm --gpus all nvidia/cuda:12.6.1-base-ubuntu24.04 nvidia-smi > /dev/null 2>&1; then
    echo "✓ SUCCESS! NVIDIA Container Toolkit is working correctly."
    echo ""
    echo "You can now run PIE with GPU support:"
    echo "  docker run --gpus all -it pie:latest"
    echo ""
    echo "GPU Info:"
    $SUDO docker run --rm --gpus all nvidia/cuda:12.6.1-base-ubuntu24.04 nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
else
    echo "✗ ERROR: GPU test failed."
    echo ""
    echo "Detailed error output:"
    $SUDO docker run --rm --gpus all nvidia/cuda:12.6.1-base-ubuntu24.04 nvidia-smi 2>&1 || true
    exit 1
fi

echo ""
echo "Installation complete!"
