# docker/images/ubuntu22.04-cuda12.4.1/env.mk

# Base OS & CUDA
UBUNTU_VERSION = 22.04
CUDA_VERSION = 12.4.131
IMAGE_TYPE = devel

# --- Stateful Feature Flags ---
CUDNN_VERSION = -1

# Build Orchestration Defaults
IMAGE_REPO = dxr
AUTO_INSTALL_REPO_URL = https://github.com/DXR-keonghun6612/Utility-Belt.git
