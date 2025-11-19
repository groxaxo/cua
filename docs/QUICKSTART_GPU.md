# Quick Start: CUA with GPU Support on Ubuntu 24.04

Get started with CUA using NVIDIA Ampere GPUs in under 10 minutes!

## Prerequisites

- Ubuntu 24.04 LTS
- NVIDIA Ampere GPU (RTX 3000/4000/5000, A100, H100, etc.)
- At least 16GB RAM recommended
- 50GB free disk space

## Fast Track Installation

### Option 1: Automated Setup (Recommended)

Run our automated setup script that handles everything:

```bash
# Download and run the setup script
curl -fsSL https://raw.githubusercontent.com/trycua/cua/main/scripts/setup-ubuntu-24.04-gpu.sh -o setup-gpu.sh
chmod +x setup-gpu.sh
./setup-gpu.sh
```

The script will:
- ✅ Install NVIDIA drivers (550+)
- ✅ Install CUDA 12.6 toolkit
- ✅ Configure Docker with GPU support
- ✅ Install CUA packages with PyTorch GPU support
- ✅ Verify the installation

After the script completes, log out and back in, then proceed to [Verify Installation](#verify-installation).

### Option 2: Manual Installation

#### Step 1: Install NVIDIA Driver

```bash
sudo apt update
sudo apt install -y nvidia-driver-550
sudo reboot
```

After reboot, verify:
```bash
nvidia-smi
```

#### Step 2: Install CUDA Toolkit

```bash
# Add NVIDIA CUDA repository
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update

# Install CUDA 12.6
sudo apt install -y cuda-toolkit-12-6 libcudnn9-cuda-12
```

#### Step 3: Install Docker with GPU Support

```bash
# Install Docker
sudo apt install -y docker.io docker-compose
sudo usermod -aG docker $USER

# Install NVIDIA Container Toolkit
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt update
sudo apt install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

Log out and back in for group changes to take effect.

#### Step 4: Install CUA with GPU Support

```bash
# Clone the repository
git clone https://github.com/trycua/cua.git
cd cua

# Install Python packages with GPU support
pip install -r requirements-gpu.txt
```

## Verify Installation

Run this verification script:

```python
# test_gpu.py
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"cuDNN version: {torch.backends.cudnn.version()}")

if torch.cuda.is_available():
    print(f"\nGPU Device: {torch.cuda.get_device_name(0)}")
    print(f"Compute Capability: {torch.cuda.get_device_capability(0)}")
    print(f"Total Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    # Test GPU computation
    x = torch.randn(1000, 1000).cuda()
    y = torch.randn(1000, 1000).cuda()
    z = torch.matmul(x, y)
    print("\n✅ GPU computation test passed!")
else:
    print("\n❌ CUDA not available")
```

Run it:
```bash
python test_gpu.py
```

Expected output:
```
PyTorch version: 2.9.0
CUDA available: True
CUDA version: 12.1
cuDNN version: 90100

GPU Device: NVIDIA GeForce RTX 4090
Compute Capability: (8, 9)
Total Memory: 24.00 GB

✅ GPU computation test passed!
```

## Your First GPU-Accelerated Agent

### Example 1: Run a GPU-Accelerated Model Locally

```python
from agent import ComputerAgent
from computer import Computer
import torch

# Verify GPU
print(f"Using device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

# Create a computer instance
computer = Computer(
    os_type="linux",
    provider_type="docker",
    image="trycua/cua-ubuntu:gpu-latest",
    name="my-gpu-agent"
)

# Use a GPU-accelerated model
agent = ComputerAgent(
    model="huggingface-local/ByteDance-Seed/UI-TARS-1.5-7B",
    tools=[computer],
    device="cuda"  # Use GPU
)

# Run the agent
async def main():
    await computer.run()
    
    async for result in agent.run("Take a screenshot and describe what you see"):
        if result.get("output"):
            for item in result["output"]:
                if item["type"] == "message":
                    print(item["content"][0]["text"])
    
    await computer.close()

import asyncio
asyncio.run(main())
```

### Example 2: GPU-Enabled Docker Container

```bash
# Pull GPU-enabled image
docker pull trycua/cua-ubuntu:gpu-latest

# Run with GPU support
docker run -it --gpus all \
  --shm-size=2g \
  -p 6901:6901 \
  -p 8000:8000 \
  -e VNCOPTIONS=-disableBasicAuth \
  trycua/cua-ubuntu:gpu-latest

# Access desktop at http://localhost:6901
```

### Example 3: Using Docker Compose for Multi-GPU Setup

```bash
# Navigate to Kasm directory
cd libs/kasm

# Start GPU-enabled services
docker-compose -f docker-compose.gpu.yml up -d

# Check status
docker-compose -f docker-compose.gpu.yml ps

# View logs
docker-compose -f docker-compose.gpu.yml logs -f

# Stop services
docker-compose -f docker-compose.gpu.yml down
```

## Performance Tips

### 1. Enable TF32 for Faster Computation (Ampere GPUs)

```python
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
```

### 2. Use Mixed Precision Training

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

# In your training loop
with autocast():
    output = model(input)
    loss = criterion(output, target)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

### 3. Monitor GPU Utilization

```bash
# Watch GPU usage in real-time
watch -n 1 nvidia-smi

# Or use gpustat for better formatting
pip install gpustat
gpustat -i 1
```

### 4. Optimize Memory Usage

```python
import torch

# Clear cache periodically
torch.cuda.empty_cache()

# Use gradient checkpointing for large models
model.gradient_checkpointing_enable()

# Enable memory efficient attention
from torch.nn.attention import SDPBackend, sdpa_kernel

with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
    output = model(input)
```

## Troubleshooting

### GPU Not Detected

```bash
# Check driver
nvidia-smi

# Check CUDA installation
nvcc --version

# Check PyTorch CUDA
python -c "import torch; print(torch.cuda.is_available())"
```

### Out of Memory Errors

```bash
# Reduce batch size
# Enable gradient checkpointing
# Use mixed precision training

# Or set memory fraction
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
```

### Docker GPU Issues

```bash
# Restart Docker
sudo systemctl restart docker

# Test GPU in Docker
docker run --rm --gpus all nvidia/cuda:12.6.0-base-ubuntu24.04 nvidia-smi
```

## Next Steps

- 📚 Read the [Complete GPU Setup Guide](GPU_SETUP.md)
- 🔧 Learn about [PyTorch Optimization](https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html)
- 🚀 Explore [CUA Examples](../examples/)
- 💬 Join our [Discord Community](https://discord.com/invite/mVnXXpdE85)

## Benchmarking Your Setup

Run this benchmark to test your GPU performance:

```python
import torch
import time

def benchmark_gpu():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Benchmark matrix multiplication
    sizes = [1000, 2000, 4000, 8000]
    for size in sizes:
        x = torch.randn(size, size, device=device)
        y = torch.randn(size, size, device=device)
        
        # Warmup
        for _ in range(10):
            _ = torch.matmul(x, y)
        
        # Benchmark
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(100):
            z = torch.matmul(x, y)
        torch.cuda.synchronize()
        end = time.time()
        
        elapsed = end - start
        tflops = (2 * size**3 * 100) / elapsed / 1e12
        print(f"Size {size}x{size}: {elapsed:.3f}s, {tflops:.2f} TFLOPS")

if __name__ == "__main__":
    benchmark_gpu()
```

Expected performance on RTX 4090:
- 1000x1000: ~0.01s, ~200 TFLOPS
- 2000x2000: ~0.02s, ~800 TFLOPS
- 4000x4000: ~0.10s, ~1280 TFLOPS
- 8000x8000: ~0.80s, ~1600 TFLOPS

## Support

Need help? 
- 📖 [Full Documentation](https://cua.ai/docs)
- 💬 [Discord Community](https://discord.com/invite/mVnXXpdE85)
- 🐛 [GitHub Issues](https://github.com/trycua/cua/issues)
