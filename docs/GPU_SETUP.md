# GPU Setup Guide for Ubuntu 24.04 with Ampere GPUs

This guide provides instructions for setting up CUA on Ubuntu 24.04 with NVIDIA Ampere GPU support for optimal performance.

## Table of Contents

1. [System Requirements](#system-requirements)
2. [NVIDIA Driver Installation](#nvidia-driver-installation)
3. [CUDA Toolkit Installation](#cuda-toolkit-installation)
4. [Docker GPU Support](#docker-gpu-support)
5. [PyTorch GPU Optimization](#pytorch-gpu-optimization)
6. [Verification](#verification)
7. [Troubleshooting](#troubleshooting)

## System Requirements

### Supported GPUs (Ampere Architecture and Newer)

- **Data Center GPUs**: A100, A40, A30, A10
- **Workstation GPUs**: A6000, A5500, A5000, A4500, A4000
- **Consumer GPUs**: 
  - RTX 3000 series: 3060, 3060 Ti, 3070, 3070 Ti, 3080, 3080 Ti, 3090, 3090 Ti
  - RTX 4000 series: 4060, 4060 Ti, 4070, 4070 Ti, 4080, 4090
  - RTX 5000 series: 5060, 5070, 5080, 5090
- **High-Performance Computing**: H100, H200

### Compute Capabilities

- **Ampere (SM 8.0)**: A100
- **Ampere (SM 8.6)**: RTX 3000/4000 series, A6000, A5000, A4000
- **Ada Lovelace (SM 8.9)**: RTX 4090, L4, L40
- **Hopper (SM 9.0)**: H100, H200

### OS Requirements

- Ubuntu 24.04 LTS (Noble Numbat)
- Kernel 6.8 or newer
- x86_64 architecture

## NVIDIA Driver Installation

### Option 1: Using Ubuntu's Package Manager (Recommended)

```bash
# Update package list
sudo apt update

# Install NVIDIA driver 550 or newer (recommended for Ampere)
sudo apt install -y nvidia-driver-550

# Reboot to load the driver
sudo reboot
```

### Option 2: Using NVIDIA's Official Repository

```bash
# Add NVIDIA package repositories
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update

# Install latest NVIDIA driver
sudo apt install -y nvidia-driver-560

# Reboot to load the driver
sudo reboot
```

### Verify Driver Installation

```bash
# Check NVIDIA driver version
nvidia-smi

# Expected output should show:
# - Driver Version: 550.x or newer
# - CUDA Version: 12.4 or newer
# - Your GPU model and memory
```

## CUDA Toolkit Installation

### Install CUDA 12.6 (Optimized for Ampere)

```bash
# Install CUDA toolkit
sudo apt install -y cuda-toolkit-12-6

# Install cuDNN (Deep Neural Network library)
sudo apt install -y libcudnn9-cuda-12

# Install additional CUDA libraries
sudo apt install -y \
    libcublas-12-6 \
    libnccl2 \
    nvidia-gds-12-6

# Set environment variables
cat >> ~/.bashrc << 'EOF'
# CUDA configuration
export CUDA_HOME=/usr/local/cuda-12.6
export PATH=${CUDA_HOME}/bin:${PATH}
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
export CUDA_DEVICE_ORDER=PCI_BUS_ID
EOF

source ~/.bashrc
```

### Verify CUDA Installation

```bash
# Check CUDA version
nvcc --version

# Run CUDA samples (optional)
cuda-install-samples-12.6.sh ~
cd ~/NVIDIA_CUDA-12.6_Samples/1_Utilities/deviceQuery
make
./deviceQuery

# Expected output should show your GPU with Compute Capability 8.0 or higher
```

## Docker GPU Support

### Install Docker (if not already installed)

```bash
# Install Docker
sudo apt update
sudo apt install -y docker.io docker-compose

# Add your user to docker group
sudo usermod -aG docker $USER
newgrp docker
```

### Install NVIDIA Container Toolkit

```bash
# Add NVIDIA Container Toolkit repository
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# Install NVIDIA Container Toolkit
sudo apt update
sudo apt install -y nvidia-container-toolkit

# Configure Docker to use NVIDIA runtime
sudo nvidia-ctk runtime configure --runtime=docker

# Restart Docker
sudo systemctl restart docker
```

### Verify Docker GPU Support

```bash
# Test GPU access in Docker
docker run --rm --gpus all nvidia/cuda:12.6.0-base-ubuntu24.04 nvidia-smi

# Expected output should show your GPU information
```

## PyTorch GPU Optimization

### Install Python and PyTorch with CUDA Support

```bash
# Install Python 3.12 and pip
sudo apt install -y python3.12 python3.12-dev python3-pip

# Install PyTorch with CUDA 12.x support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install CUA packages with GPU support
pip install cua-agent[all]
pip install cua-som  # Includes GPU-accelerated vision models
```

### Optimize PyTorch for Ampere GPUs

Add these environment variables to your `~/.bashrc`:

```bash
cat >> ~/.bashrc << 'EOF'
# PyTorch optimizations for Ampere GPUs
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0"
export CUDA_LAUNCH_BLOCKING=0  # Enable async kernel launches
export TORCH_CUDNN_V8_API_ENABLED=1  # Enable cuDNN v8 API
EOF

source ~/.bashrc
```

### Configure PyTorch in Python

Create a configuration file for PyTorch optimizations:

```python
# ~/cua_gpu_config.py
import torch
import os

def configure_ampere_gpu():
    """Configure PyTorch for optimal Ampere GPU performance."""
    
    if not torch.cuda.is_available():
        print("CUDA is not available. Running on CPU.")
        return False
    
    # Display GPU information
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA version: {torch.version.cuda}")
    print(f"cuDNN version: {torch.backends.cudnn.version()}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"\nGPU {i}: {props.name}")
        print(f"  Compute Capability: {props.major}.{props.minor}")
        print(f"  Total Memory: {props.total_memory / 1024**3:.2f} GB")
    
    # Enable TF32 for Ampere GPUs (faster matrix operations)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    # Enable cuDNN autotuner for optimal algorithm selection
    torch.backends.cudnn.benchmark = True
    
    # Enable mixed precision training
    torch.set_float32_matmul_precision('high')
    
    print("\nAmpere GPU optimizations enabled!")
    return True

if __name__ == "__main__":
    configure_ampere_gpu()
```

## Verification

### Complete System Check

Run this comprehensive verification script:

```bash
# Create verification script
cat > ~/verify_gpu_setup.sh << 'EOF'
#!/bin/bash

echo "=== GPU Setup Verification ==="
echo ""

echo "1. NVIDIA Driver:"
nvidia-smi --query-gpu=driver_version,name,compute_cap --format=csv,noheader
echo ""

echo "2. CUDA Toolkit:"
nvcc --version | grep release
echo ""

echo "3. Docker GPU Support:"
docker run --rm --gpus all nvidia/cuda:12.6.0-base-ubuntu24.04 nvidia-smi --query-gpu=name --format=csv,noheader
echo ""

echo "4. PyTorch CUDA:"
python3 -c "import torch; print(f'PyTorch CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
echo ""

echo "=== Verification Complete ==="
EOF

chmod +x ~/verify_gpu_setup.sh
~/verify_gpu_setup.sh
```

### Run CUA with GPU

Test CUA with GPU acceleration:

```python
from agent import ComputerAgent
from computer import Computer
import torch

# Verify GPU is available
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Compute Capability: {torch.cuda.get_device_capability(0)}")

# Create GPU-enabled Docker container
computer = Computer(
    os_type="linux",
    provider_type="docker",
    image="trycua/cua-ubuntu:gpu-latest",
    name="gpu-automation-container"
)

# Use GPU-accelerated model
agent = ComputerAgent(
    model="huggingface-local/ByteDance-Seed/UI-TARS-1.5-7B",
    tools=[computer],
    device="cuda"  # Use GPU
)

# Run agent
async for result in agent.run("Take a screenshot"):
    print(result)
```

## Troubleshooting

### Issue: "CUDA out of memory" errors

**Solutions:**
1. Reduce batch size in your models
2. Enable memory-efficient attention:
   ```python
   export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:512
   ```
3. Clear GPU cache periodically:
   ```python
   import torch
   torch.cuda.empty_cache()
   ```

### Issue: GPU not detected by PyTorch

**Solutions:**
1. Verify driver installation: `nvidia-smi`
2. Reinstall PyTorch with CUDA support:
   ```bash
   pip uninstall torch torchvision torchaudio
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   ```
3. Check CUDA version compatibility: `python -c "import torch; print(torch.version.cuda)"`

### Issue: Docker container cannot access GPU

**Solutions:**
1. Verify NVIDIA Container Toolkit: `docker run --rm --gpus all nvidia/cuda:12.6.0-base-ubuntu24.04 nvidia-smi`
2. Restart Docker: `sudo systemctl restart docker`
3. Check Docker GPU runtime: `docker info | grep -i runtime`

### Issue: Poor GPU performance

**Solutions:**
1. Enable TF32 operations (Ampere GPUs):
   ```python
   torch.backends.cuda.matmul.allow_tf32 = True
   torch.backends.cudnn.allow_tf32 = True
   ```
2. Use mixed precision training with `torch.cuda.amp`
3. Enable cuDNN autotuner: `torch.backends.cudnn.benchmark = True`
4. Check GPU utilization: `nvidia-smi dmon`

### Issue: Driver version mismatch

**Solution:**
```bash
# Remove old drivers
sudo apt purge nvidia-* -y
sudo apt autoremove -y

# Reinstall latest driver
sudo apt install -y nvidia-driver-550

# Reboot
sudo reboot
```

## Performance Tips

### 1. Tensor Cores Optimization (Ampere)

Ampere GPUs have dedicated Tensor Cores for fast matrix operations. Enable TF32:

```python
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
```

### 2. Multi-GPU Setup

For systems with multiple GPUs:

```python
# Use specific GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Use multiple GPUs with DataParallel
model = torch.nn.DataParallel(model)

# Use multiple GPUs with DistributedDataParallel (recommended)
torch.distributed.init_process_group(backend='nccl')
```

### 3. Memory Management

Monitor GPU memory usage:

```python
import torch

# Get memory stats
print(f"Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
print(f"Reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")

# Clear cache when needed
torch.cuda.empty_cache()
```

### 4. Async Operations

Leverage async GPU operations for better performance:

```python
# Enable async kernel launches (default)
os.environ["CUDA_LAUNCH_BLOCKING"] = "0"

# Use CUDA streams for concurrent operations
stream1 = torch.cuda.Stream()
stream2 = torch.cuda.Stream()

with torch.cuda.stream(stream1):
    # GPU operation 1
    pass

with torch.cuda.stream(stream2):
    # GPU operation 2 (runs concurrently)
    pass
```

## Additional Resources

- [NVIDIA CUDA Toolkit Documentation](https://docs.nvidia.com/cuda/)
- [PyTorch CUDA Documentation](https://pytorch.org/docs/stable/cuda.html)
- [NVIDIA Ampere Architecture White Paper](https://www.nvidia.com/en-us/data-center/ampere-architecture/)
- [CUA Documentation](https://cua.ai/docs)

## Support

For issues specific to CUA GPU support, please:
1. Check the [CUA GitHub Issues](https://github.com/trycua/cua/issues)
2. Join the [CUA Discord](https://discord.com/invite/mVnXXpdE85)
3. Consult the [CUA Documentation](https://cua.ai/docs)
