# docker/utils/step_opencv.mk
# ------------------------------------------------------------------
# Step 2: OpenCV Build Logic
# ------------------------------------------------------------------

# Load Configuration
-include $(SPEC_FILE)

.PHONY: run-step

# --- Variables ---
OPENCV_SUFFIX = -cv$(OPENCV_VERSION)
NEW_TAG = $(TAG_PREFIX)$(OPENCV_SUFFIX)
NEW_IMAGE = $(IMAGE_REPO):$(NEW_TAG)

CONFIG_FILENAME = auto_install_config_cv_$(OPENCV_VERSION).conf
CONFIG_FILE_ABS = $(ROOT_DIR)/docker/images/$(CONFIG_FILENAME)
CONFIG_BUILD_ARG = docker/images/$(CONFIG_FILENAME)

# Artifacts Logic
STAGING_DIR = $(ROOT_DIR)/build_for_container
SOURCE_POOL = $(ROOT_DIR)/build_pool
OPENCV_GPU_ARCH_SAFE = $(subst ;,_,$(OPENCV_GPU_ARCH))
OPENCV_BUILD_PATH = $(OPENCV_ARTIFACT_NAME)/build-cuda-v$(CUDA_VERSION)-dn$(CUDNN_VERSION)-$(OPENCV_GPU_ARCH_SAFE)-cpp$(OPENCV_CPP_STD)

run-step:
ifeq ($(ENABLE_OPENCV),yes)
	@set -e; \
	if [ -z "$$(docker images -q $(NEW_IMAGE))" ]; then \
		echo ">>> [Step: OpenCV] Building $(NEW_IMAGE) from $(BASE_IMAGE)..." >&2; \
		echo "Preparing artifacts in build_for_container/..." >&2; \
		find $(STAGING_DIR) -mindepth 1 ! -name '.gitkeep' -delete; \
		echo "Copying OpenCV sources:" >&2; \
		for d in $(SOURCE_POOL)/opencv*; do \
			if [ -d "$$d" ]; then \
				rel_d=$${d#$(ROOT_DIR)/}; \
				target_name=$$(basename "$$d"); \
				echo "  source: $$rel_d" >&2; \
				echo "  in_container: /opt/build/$$target_name" >&2; \
				cp -r "$$d" "$(STAGING_DIR)/$$target_name"; \
			fi; \
		done; \
		if [ "$(OPENCV_USE_PREBUILT)" = "yes" ]; then \
			if [ -e "$(SOURCE_POOL)/$(OPENCV_BUILD_PATH)" ]; then \
				echo "Copying OpenCV artifact:" >&2; \
				echo "  source: build_pool/$(OPENCV_BUILD_PATH)" >&2; \
				echo "  in_container: /opt/build/$(OPENCV_BUILD_PATH)" >&2; \
				mkdir -p "$(STAGING_DIR)/$$(dirname $(OPENCV_BUILD_PATH))"; \
				cp -r "$(SOURCE_POOL)/$(OPENCV_BUILD_PATH)" "$(STAGING_DIR)/$(OPENCV_BUILD_PATH)"; \
			else \
				echo "Warning: Artifact $(OPENCV_BUILD_PATH) not found!" >&2; \
			fi; \
		fi; \
		echo "# OpenCV Config" > $(CONFIG_FILE_ABS); \
		echo "[OPENCV_PROFILE]" >> $(CONFIG_FILE_ABS); \
		echo "version=$(OPENCV_VERSION)" >> $(CONFIG_FILE_ABS); \
		echo "with_cuda=$(OPENCV_WITH_CUDA)" >> $(CONFIG_FILE_ABS); \
		echo "gpu_arch=$(OPENCV_GPU_ARCH)" >> $(CONFIG_FILE_ABS); \
		echo "jobs=$(OPENCV_JOBS)" >> $(CONFIG_FILE_ABS); \
		echo "install_prefix=$(OPENCV_INSTALL_PREFIX)" >> $(CONFIG_FILE_ABS); \
		echo "work_dir=$(OPENCV_WORK_DIR)" >> $(CONFIG_FILE_ABS); \
		echo "python_path=$(OPENCV_PYTHON_PATH)" >> $(CONFIG_FILE_ABS); \
		echo "cpp_std=$(OPENCV_CPP_STD)" >> $(CONFIG_FILE_ABS); \
		\
		docker build \
			--progress=plain \
			--build-arg BASE_IMAGE=$(BASE_IMAGE) \
			--build-arg INSTALL_CONFIG=$(CONFIG_BUILD_ARG) \
			--build-arg REPO_URL=$(AUTO_INSTALL_REPO_URL) \
			--build-arg SSH_PORT=$(SSH_PORT) \
			-t $(NEW_IMAGE) -f $(ROOT_DIR)/docker/images/Dockerfile $(ROOT_DIR) >&2; \
		rm -f $(CONFIG_FILE_ABS); \
	else \
		echo ">>> [Step: OpenCV] Image $(NEW_IMAGE) already exists. Skipping." >&2; \
	fi
	@echo $(NEW_IMAGE)
else
	@echo ">>> [Step: OpenCV] Skipped." >&2
	@echo $(BASE_IMAGE)
endif
