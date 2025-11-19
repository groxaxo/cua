# CUA Ubuntu Container

Containerized virtual desktop for Computer-Using Agents (CUA). Utilizes Kasm's MIT-licensed Ubuntu XFCE container as a base with computer-server pre-installed.

## Features

- Ubuntu 24.04 (Noble Numbat) with XFCE desktop environment
- Pre-installed computer-server for remote computer control
- VNC access for visual desktop interaction
- Python 3.12 with necessary libraries
- Screen capture tools (gnome-screenshot, wmctrl, ffmpeg)
- Clipboard utilities (xclip, socat)
- Optional NVIDIA GPU support for Ampere architecture (A100, RTX 3000/4000/5000 series, H100)

## Usage

### Build Options

#### Standard Build (CPU-only)

Use Docker Buildx to build and push a multi-architecture image for both `linux/amd64` and `linux/arm64` in a single command. Replace `trycua` with your Docker Hub username or your registry namespace as needed.

```bash
# Login to your registry first (Docker Hub shown here)
docker login

# Build and push for amd64 and arm64 in one step (CPU-only)
docker buildx build \
  --platform linux/amd64,linux/arm64 \
  -t trycua/cua-ubuntu:latest \
  --push \
  .
```

#### GPU-Enabled Build (Ampere GPUs)

For NVIDIA Ampere GPU support (A100, RTX 3000/4000/5000 series, H100), use the GPU-specific Dockerfile:

##### Using the build script (recommended)

```bash
# Build GPU image locally
./build-gpu.sh

# Build and push to registry
./build-gpu.sh --registry trycua --push

# Build CPU-only multi-arch image
./build-gpu.sh --cpu-only --tag latest --push
```

##### Manual build commands

```bash
# Build with GPU support (x86_64 only)
docker buildx build \
  --platform linux/amd64 \
  -f Dockerfile.gpu \
  -t trycua/cua-ubuntu:gpu-latest \
  --push \
  .

# Or build with GPU support using build arg in standard Dockerfile
docker buildx build \
  --platform linux/amd64 \
  --build-arg INSTALL_GPU_SUPPORT=true \
  -t trycua/cua-ubuntu:gpu-latest \
  --push \
  .
```

### Running the Container Manually

#### CPU-only Container

```bash
docker run --rm -it --shm-size=512m -p 6901:6901 -p 8000:8000 -e VNCOPTIONS=-disableBasicAuth cua-ubuntu:latest
```

#### GPU-enabled Container

For GPU support, you need to pass the GPU device to the container:

```bash
# Using NVIDIA Container Runtime
docker run --rm -it \
  --gpus all \
  --shm-size=512m \
  -p 6901:6901 \
  -p 8000:8000 \
  -e VNCOPTIONS=-disableBasicAuth \
  cua-ubuntu:gpu-latest

# Or specify specific GPU
docker run --rm -it \
  --gpus '"device=0"' \
  --shm-size=512m \
  -p 6901:6901 \
  -p 8000:8000 \
  -e VNCOPTIONS=-disableBasicAuth \
  cua-ubuntu:gpu-latest
```

- **VNC Access**: Available at `http://localhost:6901`
- **Computer Server API**: Available at `http://localhost:8000`

### GPU Requirements

To use GPU-enabled containers, you need:

1. **NVIDIA GPU** with Ampere architecture or newer (Compute Capability 8.0+):
   - A100, A40, A30, A10, A6000
   - RTX 3000 series (3060, 3070, 3080, 3090)
   - RTX 4000 series (4060, 4070, 4080, 4090)
   - RTX 5000 series
   - H100, H200

2. **NVIDIA Driver** version 550 or newer on the host system

3. **NVIDIA Container Toolkit** installed on the host:
   ```bash
   # Install NVIDIA Container Toolkit on Ubuntu 24.04
   curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
   curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
     sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
     sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
   sudo apt-get update
   sudo apt-get install -y nvidia-container-toolkit
   sudo systemctl restart docker
   ```

### Using with CUA Docker Provider

This container is designed to work with the CUA Docker provider for automated container management:

```python
from computer.providers.factory import VMProviderFactory

# Create docker provider
provider = VMProviderFactory.create_provider(
    provider_type="docker",
    image="cua-ubuntu:latest",
    port=8000,  # computer-server API port
    noVNC_port=6901  # VNC port
)

# Run a container
async with provider:
    vm_info = await provider.run_vm(
        image="cua-ubuntu:latest",
        name="my-cua-container",
        run_opts={
            "memory": "4GB",
            "cpu": 2,
            "vnc_port": 6901,
            "api_port": 8000
        }
    )
```

## Container Configuration

### Ports

- **6901**: VNC web interface (noVNC)
- **8080**: Computer-server API endpoint

### Environment Variables

- `VNC_PW`: VNC password (default: "password")
- `DISPLAY`: X11 display (set to ":0")

### Volumes

- `/home/kasm-user/storage`: Persistent storage mount point
- `/home/kasm-user/shared`: Shared folder mount point

## Creating Filesystem Snapshots

You can create a filesystem snapshot of the container at any time:

```bash
docker commit <container_id> cua-ubuntu-snapshot:latest
```

Then run the snapshot:

```bash
docker run --rm -it --shm-size=512m -p 6901:6901 -p 8080:8080 -e VNCOPTIONS=-disableBasicAuth cua-ubuntu-snapshot:latest
```

Memory snapshots are available using the experimental `docker checkpoint` command. [Docker Checkpoint Documentation](https://docs.docker.com/reference/cli/docker/checkpoint/)

## Integration with CUA System

This container integrates seamlessly with the CUA computer provider system:

- **Automatic Management**: Use the Docker provider for lifecycle management
- **Resource Control**: Configure memory, CPU, and storage limits
- **Network Access**: Automatic port mapping and IP detection
- **Storage Persistence**: Mount host directories for persistent data
- **Monitoring**: Real-time container status and health checking
