# Ubuntu 24.04 Migration Guide

This document describes the changes made to support Ubuntu 24.04 (Noble Numbat) and NVIDIA Ampere GPU optimization in the CUA framework.

## Overview

CUA now fully supports Ubuntu 24.04 LTS with optimizations for NVIDIA Ampere GPUs (compute capability 8.0 and higher). This includes RTX 3000/4000/5000 series, A-series data center GPUs, and H-series high-performance computing GPUs.

## What Changed

### Base System

| Component | Before (22.04) | After (24.04) |
|-----------|---------------|---------------|
| Base OS | Ubuntu 22.04 (Jammy) | Ubuntu 24.04 (Noble) |
| Python | 3.11 | 3.12 |
| Kernel | 5.15 | 6.8 |
| Docker Base | kasmweb/core-ubuntu-jammy:1.17.0 | kasmweb/core-ubuntu-noble:1.17.0 |

### GPU Support

| Feature | Status | Details |
|---------|--------|---------|
| CUDA Toolkit | ✅ | Version 12.6 |
| cuDNN | ✅ | Version 9 |
| TensorRT | Optional | Can be added |
| Compute Capability | ✅ | 8.0, 8.6, 8.9, 9.0 |
| Multi-GPU | ✅ | NCCL support |

## Breaking Changes

### None

This update is fully backward compatible. CPU-only setups will continue to work without any changes.

## New Features

### 1. GPU-Enabled Docker Images

```bash
# New GPU image
docker pull trycua/cua-ubuntu:gpu-latest

# Existing CPU image (still supported)
docker pull trycua/cua-ubuntu:latest
```

### 2. Automated Setup

```bash
# One-command setup for Ubuntu 24.04 with GPU
curl -fsSL https://raw.githubusercontent.com/trycua/cua/main/scripts/setup-ubuntu-24.04-gpu.sh | bash
```

### 3. GPU Requirements File

```bash
# Install CUA with GPU support
pip install -r requirements-gpu.txt
```

### 4. Environment Templates

```bash
# Copy GPU configuration template
cp .env.gpu.example .env.local
# Edit with your settings
```

### 5. Build Tools

```bash
# Build GPU-enabled images
cd libs/kasm
./build-gpu.sh
```

## Migration Steps

### For Existing Users (CPU-only)

**No action required!** Your existing setup will continue to work.

To upgrade to Ubuntu 24.04:

1. Pull the new base image:
   ```bash
   docker pull trycua/cua-ubuntu:latest
   ```

2. Rebuild any custom containers:
   ```bash
   cd libs/kasm
   docker build -t my-custom-cua .
   ```

### For New GPU Users

1. **Check GPU Compatibility**
   ```bash
   lspci | grep -i nvidia
   ```
   Ensure you have an Ampere GPU or newer.

2. **Run Setup Script**
   ```bash
   curl -fsSL https://raw.githubusercontent.com/trycua/cua/main/scripts/setup-ubuntu-24.04-gpu.sh -o setup.sh
   chmod +x setup.sh
   ./setup.sh
   ```

3. **Verify Installation**
   ```bash
   nvidia-smi
   docker run --rm --gpus all nvidia/cuda:12.6.0-base-ubuntu24.04 nvidia-smi
   python -c "import torch; print(torch.cuda.is_available())"
   ```

### For Existing GPU Users (Upgrading from 22.04)

1. **Backup Current Setup**
   ```bash
   docker ps -a > containers_backup.txt
   docker images > images_backup.txt
   ```

2. **Update NVIDIA Driver** (if needed)
   ```bash
   sudo apt update
   sudo apt install -y nvidia-driver-550
   sudo reboot
   ```

3. **Pull New Images**
   ```bash
   docker pull trycua/cua-ubuntu:gpu-latest
   ```

4. **Update Environment Variables**
   - Copy `.env.gpu.example` to `.env.local`
   - Add Ampere-specific optimizations:
     ```bash
     export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
     export TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0"
     ```

5. **Test GPU Access**
   ```bash
   docker run --rm --gpus all trycua/cua-ubuntu:gpu-latest nvidia-smi
   ```

## Performance Improvements

### Ampere-Specific Optimizations

1. **TF32 Operations** (5-10x speedup for certain operations)
   - Automatically enabled for Ampere GPUs
   - No code changes required

2. **Memory Management**
   - Expandable memory segments
   - Reduced fragmentation
   - Better multi-GPU memory coordination

3. **Faster Matrix Operations**
   - Tensor Core optimization
   - cuDNN v8 API
   - Optimized BLAS libraries

### Benchmarks

Expected performance improvements on Ampere vs. previous generation:

| Operation | Improvement |
|-----------|-------------|
| Matrix Multiplication (FP32) | 2-3x faster |
| Matrix Multiplication (TF32) | 5-10x faster |
| Mixed Precision Training | 3-5x faster |
| Inference Throughput | 2-4x faster |

## Configuration Examples

### Single GPU Setup

```python
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from agent import ComputerAgent
agent = ComputerAgent(
    model="huggingface-local/ByteDance-Seed/UI-TARS-1.5-7B",
    device="cuda"
)
```

### Multi-GPU Setup

```python
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

import torch
# Use DataParallel or DistributedDataParallel
model = torch.nn.DataParallel(model)
```

### Docker Compose (Multi-GPU)

```yaml
services:
  cua-gpu:
    image: trycua/cua-ubuntu:gpu-latest
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
```

## Troubleshooting

### GPU Not Detected

**Symptoms:** `torch.cuda.is_available()` returns `False`

**Solutions:**
1. Check driver: `nvidia-smi`
2. Reinstall CUDA toolkit: `sudo apt install -y cuda-toolkit-12-6`
3. Reinstall PyTorch: `pip install torch --force-reinstall --index-url https://download.pytorch.org/whl/cu121`

### CUDA Out of Memory

**Symptoms:** RuntimeError: CUDA out of memory

**Solutions:**
1. Reduce batch size
2. Enable gradient checkpointing
3. Use mixed precision: `torch.cuda.amp`
4. Set memory fraction:
   ```python
   torch.cuda.set_per_process_memory_fraction(0.8, device=0)
   ```

### Docker GPU Not Working

**Symptoms:** `docker: Error response from daemon: could not select device driver`

**Solutions:**
1. Install NVIDIA Container Toolkit:
   ```bash
   curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
   sudo apt update
   sudo apt install -y nvidia-container-toolkit
   sudo systemctl restart docker
   ```

2. Test GPU access:
   ```bash
   docker run --rm --gpus all nvidia/cuda:12.6.0-base-ubuntu24.04 nvidia-smi
   ```

### Poor Performance

**Symptoms:** GPU utilization < 50%

**Solutions:**
1. Enable TF32:
   ```python
   torch.backends.cuda.matmul.allow_tf32 = True
   torch.backends.cudnn.allow_tf32 = True
   ```

2. Enable cuDNN autotuner:
   ```python
   torch.backends.cudnn.benchmark = True
   ```

3. Check for CPU bottlenecks:
   ```bash
   nvidia-smi dmon
   ```

## Compatibility Matrix

### Supported GPUs

| GPU Series | Compute Capability | Supported | Recommended |
|------------|-------------------|-----------|-------------|
| RTX 3000 | 8.6 | ✅ | ✅ |
| RTX 4000 | 8.9 | ✅ | ✅ |
| RTX 5000 | 9.0 | ✅ | ✅ |
| A100 | 8.0 | ✅ | ✅ |
| A40 | 8.6 | ✅ | ✅ |
| A30 | 8.0 | ✅ | ✅ |
| A10 | 8.6 | ✅ | ✅ |
| H100 | 9.0 | ✅ | ✅ |
| RTX 2000 | 7.5 | ⚠️ | ❌ |
| GTX 1000 | 6.1 | ⚠️ | ❌ |

✅ Fully supported  
⚠️ Works but not optimized  
❌ Not recommended

### Software Versions

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| Ubuntu | 24.04 | 24.04 LTS |
| NVIDIA Driver | 550 | 560+ |
| CUDA | 12.0 | 12.6 |
| cuDNN | 8.0 | 9.0 |
| PyTorch | 2.0 | 2.9+ |
| Python | 3.12 | 3.12 |

## Resources

- [Full GPU Setup Guide](GPU_SETUP.md)
- [Quick Start Guide](QUICKSTART_GPU.md)
- [NVIDIA Ampere Architecture](https://www.nvidia.com/en-us/data-center/ampere-architecture/)
- [PyTorch CUDA Documentation](https://pytorch.org/docs/stable/cuda.html)
- [Docker GPU Support](https://docs.docker.com/config/containers/resource_constraints/#gpu)

## Support

- 📖 [CUA Documentation](https://cua.ai/docs)
- 💬 [Discord Community](https://discord.com/invite/mVnXXpdE85)
- 🐛 [GitHub Issues](https://github.com/trycua/cua/issues)

## Changelog

### Version 0.5.0 (2025-11-19)

#### Added
- Ubuntu 24.04 (Noble) support
- NVIDIA Ampere GPU optimization
- CUDA 12.6 support
- GPU-enabled Docker images
- Automated setup script (`setup-ubuntu-24.04-gpu.sh`)
- Build automation (`build-gpu.sh`)
- Docker Compose GPU configuration
- Comprehensive GPU documentation
- Quick start guide for GPU users
- Environment configuration templates
- GPU requirements file

#### Changed
- Base OS from Ubuntu 22.04 to 24.04
- Python from 3.11 to 3.12
- Docker base image to Noble
- Optimized for Ampere compute capabilities

#### Fixed
- GPU memory management with expandable segments
- CUDA device ordering consistency
- Multi-GPU support improvements

## License

Same as CUA project (MIT License)
