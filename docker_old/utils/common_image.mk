# docker/utils/common_image.mk
# ------------------------------------------------------------------
# Orchestrator for Layered Build System
# ------------------------------------------------------------------

IMAGE_REPO ?= dxr

# Config Handling
# Calculate Absolute Path for SPEC file to pass to sub-makes
# SPEC variable comes from the calling Makefile (e.g. docker/images/Makefile)
SPEC_FILE_PATH := $(abspath $(SPEC).mk)

# Load the spec in the orchestrator too (to get version info for Init Base)
-include $(SPEC_FILE_PATH)

# CUDA_VERSION Parsing (12.4.131 -> 12.4.1)
CUDA_IMAGE_VERSION = $(shell echo $(CUDA_VERSION) | sed -E 's/([0-9]+\.[0-9]+\.)([0-9]).*/\1\2/')

TAG_OS = u$(UBUNTU_VERSION)
TAG_CUDA = cu$(CUDA_VERSION)

# Initial Base Image (Step 0)
CUDNN_SUFFIX = -$(IMAGE_TYPE)
ifneq ($(CUDNN_VERSION),)
    CUDNN_SUFFIX = -cudnn-$(IMAGE_TYPE)
endif

INIT_BASE_IMAGE = nvidia/cuda:$(CUDA_IMAGE_VERSION)$(CUDNN_SUFFIX)-ubuntu$(UBUNTU_VERSION)
INIT_TAG_PREFIX = $(TAG_OS)-$(TAG_CUDA)

# Paths
ROOT_DIR = $(CURDIR)/../../
UTILS_DIR = ../utils

# Export Common Variables for sub-makes
# Only system-wide settings are exported. 
# Component settings are loaded by sub-makes via SPEC_FILE.
AUTO_INSTALL_REPO_URL ?= https://github.com/DXR-keonghun6612/Utility-Belt.git
export IMAGE_REPO
export ROOT_DIR
export AUTO_INSTALL_REPO_URL
export SSH_PORT
export APT_PACKAGES

.PHONY: build force-build

build:
	@echo "=========================================================="
	@echo " Starting Pipeline Build"
	@echo " Spec File:    $(SPEC_FILE_PATH)"
	@echo " Initial Base: $(INIT_BASE_IMAGE)"
	@echo "=========================================================="
	@set -e; \
	CURRENT_IMG="$(INIT_BASE_IMAGE)"; \
	CURRENT_TAG="$(INIT_TAG_PREFIX)"; \
	\
	echo ">>> [Orchestrator] Requesting CuDNN Update Layer..."; \
	STEP0_OUT=$$($(MAKE) --no-print-directory -s -f $(UTILS_DIR)/step_cudnn.mk run-step \
		BASE_IMAGE="$$CURRENT_IMG" \
		TAG_PREFIX="$$CURRENT_TAG" \
		SPEC_FILE="$(SPEC_FILE_PATH)"); \
	CURRENT_IMG="$$STEP0_OUT"; \
	CURRENT_TAG=$$(echo "$$CURRENT_IMG" | cut -d: -f2); \
	echo ">>> [Orchestrator] Step 0 Result: $$CURRENT_IMG"; \
	\
	echo ">>> [Orchestrator] Requesting ROS2 Layer..."; \
	STEP1_OUT=$$($(MAKE) --no-print-directory -s -f $(UTILS_DIR)/step_ros2.mk run-step \
		BASE_IMAGE="$$CURRENT_IMG" \
		TAG_PREFIX="$$CURRENT_TAG" \
		SPEC_FILE="$(SPEC_FILE_PATH)"); \
	CURRENT_IMG="$$STEP1_OUT"; \
	CURRENT_TAG=$$(echo "$$CURRENT_IMG" | cut -d: -f2); \
	echo ">>> [Orchestrator] Step 1 Result: $$CURRENT_IMG"; \
	\
	echo ">>> [Orchestrator] Requesting OpenCV Layer..."; \
	STEP2_OUT=$$($(MAKE) --no-print-directory -s -f $(UTILS_DIR)/step_opencv.mk run-step \
		BASE_IMAGE="$$CURRENT_IMG" \
		TAG_PREFIX="$$CURRENT_TAG" \
		SPEC_FILE="$(SPEC_FILE_PATH)"); \
	CURRENT_IMG="$$STEP2_OUT"; \
	CURRENT_TAG=$$(echo "$$CURRENT_IMG" | cut -d: -f2); \
	echo ">>> [Orchestrator] Step 2 Result: $$CURRENT_IMG"; \
	\
	echo ">>> [Orchestrator] Requesting Final System Layer..."; \
	STEP4_OUT=$$($(MAKE) --no-print-directory -s -f $(UTILS_DIR)/step_system.mk run-step \
		BASE_IMAGE="$$CURRENT_IMG" \
		TAG_PREFIX="$$CURRENT_TAG" \
		SPEC_FILE="$(SPEC_FILE_PATH)"); \
	CURRENT_IMG="$$STEP4_OUT"; \
	echo ">>> [Orchestrator] Step 4 Result: $$CURRENT_IMG"; \
	\
	echo "=========================================================="; \
	echo " Pipeline Completed Successfully."; \
	echo " Final Image: $$CURRENT_IMG"; \
	echo "==========================================================";

force-build: build
