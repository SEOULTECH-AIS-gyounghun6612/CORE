# docker/utils/step_cudnn.mk
# ------------------------------------------------------------------
# Step 0: CuDNN Modification / Update Logic
# ------------------------------------------------------------------

# Load Configuration
-include $(SPEC_FILE)

.PHONY: run-step

# --- Variables ---
# If we update CuDNN, we append a suffix to indicate this change
CUDNN_UPDATE_SUFFIX = -cudnn$(CUDNN_VERSION)
NEW_TAG = $(TAG_PREFIX)$(CUDNN_UPDATE_SUFFIX)
NEW_IMAGE = $(IMAGE_REPO):$(NEW_TAG)

CONFIG_FILENAME = auto_install_config_cudnn.conf
CONFIG_FILE_ABS = $(ROOT_DIR)/docker/images/$(CONFIG_FILENAME)
CONFIG_BUILD_ARG = docker/images/$(CONFIG_FILENAME)

run-step:
	@set -e; \
	if [ -z "$$(docker images -q $(NEW_IMAGE))" ]; then \
		echo ">>> [Step: CuDNN] Updating CuDNN to $(CUDNN_VERSION) on $(BASE_IMAGE)..." >&2; \
		echo "# CuDNN Config" > $(CONFIG_FILE_ABS); \
		echo "[CUDNN_LIBRARY_PROFILE]" >> $(CONFIG_FILE_ABS); \
		echo "version=$(CUDNN_VERSION)" >> $(CONFIG_FILE_ABS); \
		\
		docker build \
			--build-arg BASE_IMAGE=$(BASE_IMAGE) \
			--build-arg INSTALL_CONFIG=$(CONFIG_BUILD_ARG) \
			--build-arg REPO_URL=$(AUTO_INSTALL_REPO_URL) \
			--build-arg SSH_PORT=$(SSH_PORT) \
			-t $(NEW_IMAGE) -f $(ROOT_DIR)/docker/images/Dockerfile $(ROOT_DIR) >&2; \
		rm -f $(CONFIG_FILE_ABS); \
	else \
		echo ">>> [Step: CuDNN] Image $(NEW_IMAGE) already exists. Skipping." >&2; \
	fi
	@echo $(NEW_IMAGE)
