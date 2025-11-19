#!/bin/bash
# Build script for GPU-enabled CUA Ubuntu container
# Optimized for NVIDIA Ampere GPUs on Ubuntu 24.04

set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Default values
IMAGE_NAME="cua-ubuntu"
IMAGE_TAG="gpu-latest"
REGISTRY=""
PUSH=false
PLATFORM="linux/amd64"
DOCKERFILE="Dockerfile.gpu"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --name)
            IMAGE_NAME="$2"
            shift 2
            ;;
        --tag)
            IMAGE_TAG="$2"
            shift 2
            ;;
        --registry)
            REGISTRY="$2/"
            shift 2
            ;;
        --push)
            PUSH=true
            shift
            ;;
        --cpu-only)
            DOCKERFILE="Dockerfile"
            IMAGE_TAG="latest"
            PLATFORM="linux/amd64,linux/arm64"
            shift
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --name NAME        Image name (default: cua-ubuntu)"
            echo "  --tag TAG          Image tag (default: gpu-latest)"
            echo "  --registry REG     Docker registry (e.g., docker.io/username)"
            echo "  --push             Push image to registry after build"
            echo "  --cpu-only         Build CPU-only version (multi-arch)"
            echo "  --help             Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0                                    # Build GPU image locally"
            echo "  $0 --registry trycua --push          # Build and push to trycua/cua-ubuntu:gpu-latest"
            echo "  $0 --cpu-only --tag latest --push    # Build multi-arch CPU image"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            echo "Use --help to see available options"
            exit 1
            ;;
    esac
done

FULL_IMAGE_NAME="${REGISTRY}${IMAGE_NAME}:${IMAGE_TAG}"

echo -e "${GREEN}Building CUA Ubuntu Container${NC}"
echo "================================"
echo "Dockerfile: $DOCKERFILE"
echo "Image name: $FULL_IMAGE_NAME"
echo "Platform:   $PLATFORM"
echo "Push:       $PUSH"
echo ""

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo -e "${RED}Error: Docker is not installed${NC}"
    exit 1
fi

# Check if running GPU build and NVIDIA runtime is available
if [ "$DOCKERFILE" = "Dockerfile.gpu" ]; then
    echo -e "${YELLOW}Checking NVIDIA Docker support...${NC}"
    if docker run --rm --gpus all nvidia/cuda:12.6.0-base-ubuntu24.04 nvidia-smi &> /dev/null; then
        echo -e "${GREEN}✓ NVIDIA Docker support detected${NC}"
    else
        echo -e "${YELLOW}⚠ Warning: NVIDIA Docker support not available${NC}"
        echo "  The image will build, but GPU features won't work at runtime"
        echo "  Install nvidia-container-toolkit to enable GPU support"
        echo ""
        read -p "Continue anyway? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi
fi

# Check if buildx is available for multi-platform builds
if [[ "$PLATFORM" == *","* ]]; then
    if ! docker buildx version &> /dev/null; then
        echo -e "${RED}Error: Docker buildx is required for multi-platform builds${NC}"
        echo "Install it with: docker buildx create --use"
        exit 1
    fi
    BUILD_CMD="docker buildx build --platform $PLATFORM"
    if [ "$PUSH" = true ]; then
        BUILD_CMD="$BUILD_CMD --push"
    else
        BUILD_CMD="$BUILD_CMD --load"
    fi
else
    BUILD_CMD="docker build"
fi

# Build the image
echo -e "${GREEN}Starting build...${NC}"
echo ""

$BUILD_CMD \
    -f "$DOCKERFILE" \
    -t "$FULL_IMAGE_NAME" \
    --build-arg BUILDKIT_INLINE_CACHE=1 \
    .

BUILD_EXIT_CODE=$?

if [ $BUILD_EXIT_CODE -eq 0 ]; then
    echo ""
    echo -e "${GREEN}✓ Build successful!${NC}"
    echo ""
    
    # Get image size
    if [ "$PUSH" = false ] && [[ "$PLATFORM" != *","* ]]; then
        IMAGE_SIZE=$(docker images "$FULL_IMAGE_NAME" --format "{{.Size}}")
        echo "Image size: $IMAGE_SIZE"
        echo ""
    fi
    
    # Push to registry if requested
    if [ "$PUSH" = true ] && [[ "$PLATFORM" != *","* ]]; then
        echo -e "${GREEN}Pushing to registry...${NC}"
        docker push "$FULL_IMAGE_NAME"
        if [ $? -eq 0 ]; then
            echo -e "${GREEN}✓ Push successful!${NC}"
        else
            echo -e "${RED}✗ Push failed${NC}"
            exit 1
        fi
    fi
    
    # Print usage instructions
    echo ""
    echo -e "${GREEN}Usage:${NC}"
    echo ""
    
    if [ "$DOCKERFILE" = "Dockerfile.gpu" ]; then
        echo "Run with GPU support:"
        echo "  docker run -it --gpus all --shm-size=2g -p 6901:6901 -p 8000:8000 \\"
        echo "    -e VNCOPTIONS=-disableBasicAuth $FULL_IMAGE_NAME"
        echo ""
        echo "Or use docker-compose:"
        echo "  docker-compose -f docker-compose.gpu.yml up -d"
    else
        echo "Run the container:"
        echo "  docker run -it --shm-size=512m -p 6901:6901 -p 8000:8000 \\"
        echo "    -e VNCOPTIONS=-disableBasicAuth $FULL_IMAGE_NAME"
    fi
    
    echo ""
    echo "Access the desktop at: http://localhost:6901"
    echo "Computer Server API: http://localhost:8000"
    echo ""
    
else
    echo ""
    echo -e "${RED}✗ Build failed${NC}"
    exit 1
fi

# Run basic tests if image was built locally
if [ "$PUSH" = false ] && [[ "$PLATFORM" != *","* ]]; then
    echo -e "${YELLOW}Running basic tests...${NC}"
    
    # Test 1: Check if image exists
    if docker images "$FULL_IMAGE_NAME" | grep -q "$IMAGE_TAG"; then
        echo -e "${GREEN}✓ Image exists${NC}"
    else
        echo -e "${RED}✗ Image not found${NC}"
        exit 1
    fi
    
    # Test 2: Start container and check if it runs
    echo "Testing container startup..."
    CONTAINER_ID=$(docker run -d --rm "$FULL_IMAGE_NAME" sleep 30)
    sleep 2
    
    if docker ps | grep -q "$CONTAINER_ID"; then
        echo -e "${GREEN}✓ Container starts successfully${NC}"
        docker stop "$CONTAINER_ID" > /dev/null
    else
        echo -e "${RED}✗ Container failed to start${NC}"
        exit 1
    fi
    
    # Test 3: Check Python version
    echo "Checking Python version..."
    PYTHON_VERSION=$(docker run --rm "$FULL_IMAGE_NAME" python3 --version)
    if [[ "$PYTHON_VERSION" == *"3.12"* ]]; then
        echo -e "${GREEN}✓ Python 3.12 installed${NC}"
    else
        echo -e "${YELLOW}⚠ Python version: $PYTHON_VERSION${NC}"
    fi
    
    # Test 4: Check if GPU image has CUDA
    if [ "$DOCKERFILE" = "Dockerfile.gpu" ]; then
        echo "Checking CUDA installation..."
        if docker run --rm "$FULL_IMAGE_NAME" nvcc --version &> /dev/null; then
            CUDA_VERSION=$(docker run --rm "$FULL_IMAGE_NAME" nvcc --version | grep "release" | awk '{print $5}' | sed 's/,//')
            echo -e "${GREEN}✓ CUDA $CUDA_VERSION installed${NC}"
        else
            echo -e "${YELLOW}⚠ CUDA not found (might be expected for non-GPU hosts)${NC}"
        fi
    fi
    
    echo ""
    echo -e "${GREEN}All tests passed!${NC}"
fi

echo ""
echo -e "${GREEN}Done!${NC}"
