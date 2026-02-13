# docker/utils/step_system.mk
# ------------------------------------------------------------------
# Step 4: Final System Packages Logic
# ------------------------------------------------------------------

# Load Configuration
-include $(SPEC_FILE)

.PHONY: run-step

# --- Variables ---
SYSTEM_SUFFIX = -custom
NEW_TAG = $(TAG_PREFIX)$(SYSTEM_SUFFIX)
NEW_IMAGE = $(IMAGE_REPO):$(NEW_TAG)

CONFIG_FILENAME = auto_install_config_system.conf
CONFIG_FILE_ABS = $(ROOT_DIR)/docker/images/$(CONFIG_FILENAME)
CONFIG_BUILD_ARG = docker/images/$(CONFIG_FILENAME)

run-step:
ifeq ($(ENABLE_SYSTEM_PACKAGES),yes)
	@set -e; \
	if [ -z "$$(docker images -q $(NEW_IMAGE))" ]; then \
		echo ">>> [Step: System] Installing additional packages on $(BASE_IMAGE)..." >&2; \
		echo "# System Config" > $(CONFIG_FILE_ABS); \
		echo "[PACKAGES_LIST]" >> $(CONFIG_FILE_ABS); \
		for pkg in $(ADDITIONAL_PACKAGES); do echo "$$pkg=" >> $(CONFIG_FILE_ABS); done; \
		\
		docker build \
			--build-arg BASE_IMAGE=$(BASE_IMAGE) \
			--build-arg INSTALL_CONFIG=$(CONFIG_BUILD_ARG) \
			--build-arg REPO_URL=$(AUTO_INSTALL_REPO_URL) \
			--build-arg SSH_PORT=$(SSH_PORT) \
			-t $(NEW_IMAGE) -f $(ROOT_DIR)/docker/images/Dockerfile $(ROOT_DIR) >&2; \
		rm -f $(CONFIG_FILE_ABS); \
	else \
		echo ">>> [Step: System] Image $(NEW_IMAGE) already exists. Skipping." >&2; \
	fi
	@echo $(NEW_IMAGE)
else
	@echo ">>> [Step: System] Skipped." >&2
	@echo $(BASE_IMAGE)
endif
