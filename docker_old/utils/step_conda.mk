# docker/utils/step_conda.mk
# ------------------------------------------------------------------
# Step 3: Conda Build Logic
# ------------------------------------------------------------------

# Load Configuration
-include $(SPEC_FILE)

.PHONY: run-step

# --- Variables ---
CONDA_SUFFIX = -conda
NEW_TAG = $(TAG_PREFIX)$(CONDA_SUFFIX)
NEW_IMAGE = $(IMAGE_REPO):$(NEW_TAG)

CONFIG_FILENAME = auto_install_config_conda.conf
CONFIG_FILE_ABS = $(ROOT_DIR)/docker/images/$(CONFIG_FILENAME)
CONFIG_BUILD_ARG = docker/images/$(CONFIG_FILENAME)

run-step:
ifeq ($(ENABLE_CONDA),yes)
	@if [ -z "$$(docker images -q $(NEW_IMAGE))" ]; then \
		echo ">>> [Step: Conda] Building $(NEW_IMAGE) from $(BASE_IMAGE)..." >&2; \
		echo "# Conda Config" > $(CONFIG_FILE_ABS); \
		echo "[CONDA_PROFILE]" >> $(CONFIG_FILE_ABS); \
		echo "mode=$(CONDA_MODE)" >> $(CONFIG_FILE_ABS); \
		echo "type=$(CONDA_TYPE)" >> $(CONFIG_FILE_ABS); \
		\
		docker build \
			--build-arg BASE_IMAGE=$(BASE_IMAGE) \
			--build-arg INSTALL_CONFIG=$(CONFIG_BUILD_ARG) \
			--build-arg REPO_URL=$(AUTO_INSTALL_REPO_URL) \
			--build-arg SSH_PORT=$(SSH_PORT) \
			-t $(NEW_IMAGE) -f $(ROOT_DIR)/docker/images/Dockerfile $(ROOT_DIR) >&2; \
		rm -f $(CONFIG_FILE_ABS); \
	else \
		echo ">>> [Step: Conda] Image $(NEW_IMAGE) already exists. Skipping." >&2; \
	fi
	@echo $(NEW_IMAGE)
else
	@echo ">>> [Step: Conda] Skipped." >&2
	@echo $(BASE_IMAGE)
endif
