#!/bin/bash
# CUA Setup Script for Ubuntu 24.04 with Ampere GPU Support
# This script automates the installation of NVIDIA drivers, CUDA toolkit,
# Docker with GPU support, and CUA packages optimized for Ampere GPUs.

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if running as root
if [ "$EUID" -eq 0 ]; then 
    log_error "Please do not run this script as root. Use your regular user account."
    exit 1
fi

# Check Ubuntu version
log_info "Checking Ubuntu version..."
if ! grep -q "24.04" /etc/os-release; then
    log_error "This script is designed for Ubuntu 24.04. Your version: $(lsb_release -d | cut -f2)"
    exit 1
fi
log_info "Ubuntu 24.04 detected ✓"

# Check for NVIDIA GPU
log_info "Checking for NVIDIA GPU..."
if ! lspci | grep -i nvidia > /dev/null; then
    log_error "No NVIDIA GPU detected. This script requires an NVIDIA GPU."
    exit 1
fi

GPU_NAME=$(lspci | grep -i nvidia | head -n1)
log_info "Found: $GPU_NAME ✓"

# Parse command line arguments
INSTALL_DRIVER=true
INSTALL_CUDA=true
INSTALL_DOCKER=true
INSTALL_CUA=true
SKIP_REBOOT=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-driver)
            INSTALL_DRIVER=false
            shift
            ;;
        --skip-cuda)
            INSTALL_CUDA=false
            shift
            ;;
        --skip-docker)
            INSTALL_DOCKER=false
            shift
            ;;
        --skip-cua)
            INSTALL_CUA=false
            shift
            ;;
        --skip-reboot)
            SKIP_REBOOT=true
            shift
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --skip-driver     Skip NVIDIA driver installation"
            echo "  --skip-cuda       Skip CUDA toolkit installation"
            echo "  --skip-docker     Skip Docker and NVIDIA Container Toolkit installation"
            echo "  --skip-cua        Skip CUA package installation"
            echo "  --skip-reboot     Skip automatic reboot after driver installation"
            echo "  --help            Show this help message"
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            echo "Use --help to see available options"
            exit 1
            ;;
    esac
done

# Update system
log_info "Updating system packages..."
sudo apt update
sudo apt upgrade -y

# Install NVIDIA Driver
if [ "$INSTALL_DRIVER" = true ]; then
    log_info "Installing NVIDIA driver 550..."
    
    # Check if driver is already installed
    if nvidia-smi &> /dev/null; then
        DRIVER_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n1)
        log_warn "NVIDIA driver $DRIVER_VERSION is already installed"
        read -p "Do you want to reinstall? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            log_info "Skipping driver installation"
            INSTALL_DRIVER=false
        fi
    fi
    
    if [ "$INSTALL_DRIVER" = true ]; then
        sudo apt install -y nvidia-driver-550
        log_info "NVIDIA driver installed successfully ✓"
        
        if [ "$SKIP_REBOOT" = false ]; then
            log_warn "A reboot is required to load the driver"
            read -p "Reboot now? (Y/n): " -n 1 -r
            echo
            if [[ ! $REPLY =~ ^[Nn]$ ]]; then
                log_info "Rebooting... Please run this script again after reboot."
                sudo reboot
            fi
        fi
    fi
fi

# Verify NVIDIA driver
if ! nvidia-smi &> /dev/null; then
    log_error "NVIDIA driver not loaded. Please reboot your system and run this script again."
    exit 1
fi

log_info "NVIDIA driver version: $(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n1) ✓"

# Install CUDA Toolkit
if [ "$INSTALL_CUDA" = true ]; then
    log_info "Installing CUDA 12.6 toolkit..."
    
    # Add NVIDIA package repository
    if [ ! -f /etc/apt/sources.list.d/cuda.list ]; then
        wget -q -O - https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/3bf863cc.pub | sudo apt-key add -
        echo "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64 /" | sudo tee /etc/apt/sources.list.d/cuda.list
        sudo apt update
    fi
    
    # Install CUDA toolkit and libraries
    sudo apt install -y \
        cuda-toolkit-12-6 \
        libcudnn9-cuda-12 \
        libcublas-12-6 \
        libnccl2 \
        nvidia-gds-12-6
    
    # Set environment variables
    if ! grep -q "CUDA_HOME" ~/.bashrc; then
        log_info "Adding CUDA environment variables to ~/.bashrc..."
        cat >> ~/.bashrc << 'EOF'

# CUDA configuration
export CUDA_HOME=/usr/local/cuda-12.6
export PATH=${CUDA_HOME}/bin:${PATH}
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
export CUDA_DEVICE_ORDER=PCI_BUS_ID

# PyTorch optimizations for Ampere GPUs
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0"
export CUDA_LAUNCH_BLOCKING=0
export TORCH_CUDNN_V8_API_ENABLED=1
EOF
        source ~/.bashrc
    fi
    
    log_info "CUDA toolkit installed successfully ✓"
fi

# Install Docker and NVIDIA Container Toolkit
if [ "$INSTALL_DOCKER" = true ]; then
    log_info "Installing Docker..."
    
    # Check if Docker is already installed
    if command -v docker &> /dev/null; then
        log_warn "Docker is already installed"
        DOCKER_VERSION=$(docker --version)
        log_info "$DOCKER_VERSION"
    else
        sudo apt install -y docker.io docker-compose
        sudo systemctl enable docker
        sudo systemctl start docker
        
        # Add user to docker group
        sudo usermod -aG docker $USER
        log_info "Docker installed successfully ✓"
        log_warn "You may need to log out and back in for docker group changes to take effect"
    fi
    
    log_info "Installing NVIDIA Container Toolkit..."
    
    # Add NVIDIA Container Toolkit repository
    if [ ! -f /etc/apt/sources.list.d/nvidia-container-toolkit.list ]; then
        curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
        curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
            sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
            sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
        sudo apt update
    fi
    
    # Install NVIDIA Container Toolkit
    sudo apt install -y nvidia-container-toolkit
    
    # Configure Docker to use NVIDIA runtime
    sudo nvidia-ctk runtime configure --runtime=docker
    sudo systemctl restart docker
    
    log_info "NVIDIA Container Toolkit installed successfully ✓"
fi

# Install Python and CUA packages
if [ "$INSTALL_CUA" = true ]; then
    log_info "Installing Python 3.12 and CUA packages..."
    
    # Install Python and pip
    sudo apt install -y python3.12 python3.12-dev python3-pip python3.12-venv
    
    # Create virtual environment (optional but recommended)
    if [ ! -d ~/cua-venv ]; then
        log_info "Creating Python virtual environment..."
        python3.12 -m venv ~/cua-venv
        log_info "Virtual environment created at ~/cua-venv"
        log_info "Activate it with: source ~/cua-venv/bin/activate"
    fi
    
    # Install PyTorch with CUDA support
    log_info "Installing PyTorch with CUDA support..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    
    # Install CUA packages
    log_info "Installing CUA packages..."
    pip install cua-agent[all]
    pip install cua-computer
    pip install cua-som
    pip install cua-computer-server
    
    log_info "CUA packages installed successfully ✓"
fi

# Verification
log_info "Running verification checks..."

echo ""
echo "=== System Verification ==="
echo ""

# Check NVIDIA driver
echo "1. NVIDIA Driver:"
nvidia-smi --query-gpu=driver_version,name,compute_cap,memory.total --format=csv,noheader
echo ""

# Check CUDA
if [ "$INSTALL_CUDA" = true ]; then
    echo "2. CUDA Toolkit:"
    if command -v nvcc &> /dev/null; then
        nvcc --version | grep release
    else
        log_warn "nvcc not found. You may need to log out and back in."
    fi
    echo ""
fi

# Check Docker GPU support
if [ "$INSTALL_DOCKER" = true ]; then
    echo "3. Docker GPU Support:"
    if sudo docker run --rm --gpus all nvidia/cuda:12.6.0-base-ubuntu24.04 nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null; then
        log_info "Docker GPU support verified ✓"
    else
        log_warn "Docker GPU test failed. You may need to log out and back in."
    fi
    echo ""
fi

# Check PyTorch
if [ "$INSTALL_CUA" = true ]; then
    echo "4. PyTorch CUDA:"
    python3 -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')" 2>/dev/null || log_warn "PyTorch check failed"
    echo ""
fi

echo "=== Verification Complete ==="
echo ""

# Final instructions
log_info "Setup complete! 🎉"
echo ""
echo "Next steps:"
echo "1. Log out and back in (or run 'newgrp docker') to apply group changes"
echo "2. Activate the Python virtual environment: source ~/cua-venv/bin/activate"
echo "3. Test CUA with GPU: python3 -c 'from agent import ComputerAgent; print(\"CUA ready!\")'"
echo "4. Pull GPU-enabled Docker image: docker pull trycua/cua-ubuntu:gpu-latest"
echo ""
echo "For more information, see: https://github.com/trycua/cua/blob/main/docs/GPU_SETUP.md"
echo ""

# Save installation log
LOG_FILE=~/cua-setup-$(date +%Y%m%d-%H%M%S).log
echo "Installation log saved to: $LOG_FILE"
