# docker/utils/step_ros2.mk
# ------------------------------------------------------------------
# Step 1: ROS2 Build Logic
# ------------------------------------------------------------------

# Load Configuration
-include $(SPEC_FILE)

.PHONY: run-step

# --- Variables ---
# Determine output image name
ROS2_SUFFIX = -ros-$(ROS2_DISTRO)
NEW_TAG = $(TAG_PREFIX)$(ROS2_SUFFIX)
NEW_IMAGE = $(IMAGE_REPO):$(NEW_TAG)

# Config file path
CONFIG_FILENAME = auto_install_config_ros2_$(ROS2_DISTRO).conf
CONFIG_FILE_ABS = $(ROOT_DIR)/docker/images/$(CONFIG_FILENAME)
CONFIG_BUILD_ARG = docker/images/$(CONFIG_FILENAME)

run-step:
ifeq ($(ENABLE_ROS2),yes)
	@set -e; \
	if [ -z "$$(docker images -q $(NEW_IMAGE))" ]; then \
		echo ">>> [Step: ROS2] Building $(NEW_IMAGE) from $(BASE_IMAGE)..." >&2; \
		echo "# ROS2 Config" > $(CONFIG_FILE_ABS); \
		echo "[ROS2_PROFILE]" >> $(CONFIG_FILE_ABS); \
		echo "distro=$(ROS2_DISTRO)" >> $(CONFIG_FILE_ABS); \
		echo "type=$(ROS2_TYPE)" >> $(CONFIG_FILE_ABS); \
		\
		docker build \
			--build-arg BASE_IMAGE=$(BASE_IMAGE) \
			--build-arg INSTALL_CONFIG=$(CONFIG_BUILD_ARG) \
			--build-arg REPO_URL=$(AUTO_INSTALL_REPO_URL) \
			--build-arg SSH_PORT=$(SSH_PORT) \
			-t $(NEW_IMAGE) -f $(ROOT_DIR)/docker/images/Dockerfile $(ROOT_DIR) >&2; \
		rm -f $(CONFIG_FILE_ABS); \
	else \
		echo ">>> [Step: ROS2] Image $(NEW_IMAGE) already exists. Skipping." >&2; \
	fi
	@echo $(NEW_IMAGE)
else
	@echo ">>> [Step: ROS2] Skipped." >&2
	@echo $(BASE_IMAGE)
endif
