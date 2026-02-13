# docker/utils/image_builder.mk
# ------------------------------------------------------------------
# Unified Modular Build Engine
# ------------------------------------------------------------------

# This file acts as both the Orchestrator and the Builder.
SELF := $(lastword $(MAKEFILE_LIST))
DOCKER_DIR := $(ROOT_DIR)/docker

# Debug Mode: Controls general visibility but NOT internal data piping
ifeq ($(DEBUG),1)
    PRINT_DIR :=
else
    PRINT_DIR := --no-print-directory
endif

# --- [Phase 1: Common Initialization] ---
ifdef FLAVOR
    FLAVOR_DIR := $(DOCKER_DIR)/images/$(FLAVOR)
    -include $(FLAVOR_DIR)/env.mk
    
    # Derived Global Variables
    CUDA_SHORT := $(shell echo $(CUDA_VERSION) | sed -E 's/([0-9]+\.[0-9]+\.)([0-9]).*/\1\2/')
    TAG_PREFIX_BASE := u$(UBUNTU_VERSION)-cu$(CUDA_VERSION)
    INIT_BASE_IMAGE := nvidia/cuda:$(CUDA_SHORT)-$(IMAGE_TYPE)-ubuntu$(UBUNTU_VERSION)
    
    ifdef MANIFEST
        MANIFEST_PATH := $(FLAVOR_DIR)/$(MANIFEST).mk
        -include $(MANIFEST_PATH)
    endif
endif

# --- [Phase 2: Orchestrator Mode] ---
.PHONY: build
build:
	@echo "=========================================================="
	@echo " Starting Modular Pipeline Build (Unified Engine)"
	@echo " Flavor:   $(FLAVOR)"
	@echo " Manifest: $(MANIFEST)"
	@echo "=========================================================="
	@set -e; \
	CURRENT_IMG="$(INIT_BASE_IMAGE)"; \
	CURRENT_TAG="$(TAG_PREFIX_BASE)"; \
	STATE_VARS="CUDA_VERSION=$(CUDA_VERSION) \
	            CUDNN_VERSION=$(CUDNN_VERSION) \
	            UBUNTU_VERSION=$(UBUNTU_VERSION) \
	            IMAGE_TYPE=$(IMAGE_TYPE) \
	            IMAGE_REPO=$(IMAGE_REPO) \
	            AUTO_INSTALL_REPO_URL=$(AUTO_INSTALL_REPO_URL) \
	            TAG_PREFIX=$$CURRENT_TAG"; \
	step_count=1; \
	for entry in $(STEPS); do \
		step_folder=$$(echo $$entry | cut -d: -f1); \
		profile_name=$$(echo $$entry | cut -d: -f2); \
		echo ">>> [Step $$step_count] Building: $$step_folder (Profile: $$profile_name)..." >&2; \
		# 데이터 파싱을 위해 하위 make 실행 시 --no-print-directory 를 반드시 강제함 \
		OUT=$$($(MAKE) -f $(SELF) run-step --no-print-directory -s \
			ROOT_DIR="$(ROOT_DIR)" \
			STEP_DIR="$(FLAVOR_DIR)/$$step_folder" \
			PROFILE_FILE="$$profile_name" \
			BASE_IMAGE="$$CURRENT_IMG" \
			$$STATE_VARS) || exit 1; \
		if [ -z "$$OUT" ]; then \
			echo "Error: Step $$step_folder failed to return output." >&2; \
			exit 1; \
		fi; \
		NEW_IMG=$$(echo "$$OUT" | cut -d'|' -f1); \
		UPDATES=$$(echo "$$OUT" | cut -d'|' -f2); \
		if [ -n "$$UPDATES" ]; then \
			echo "    State Update Captured: $$UPDATES" >&2; \
			STATE_VARS="$$STATE_VARS $$UPDATES"; \
		fi; \
		CURRENT_IMG="$$NEW_IMG"; \
		CURRENT_TAG=$$(echo "$$CURRENT_IMG" | cut -d: -f2); \
		STATE_VARS="$$(echo $$STATE_VARS | sed "s/TAG_PREFIX=[^ ]*/TAG_PREFIX=$$CURRENT_TAG/")"; \
		step_count=$$(($$step_count + 1)); \
	done; \
	FINAL_IMAGE_NAME="$$CURRENT_IMG"; \
	echo ">>> [Stamping] Recording image name in: $(MANIFEST_PATH)"; \
	sed -i '/^BUILT_IMAGE_NAME =/d' $(MANIFEST_PATH); \
	echo "BUILT_IMAGE_NAME = $$FINAL_IMAGE_NAME" >> $(MANIFEST_PATH); \
	echo "=========================================================="; \
	echo " Pipeline Completed Successfully."; \
	echo " Final Image: $$FINAL_IMAGE_NAME"; \
	echo "==========================================================";


# --- [Phase 3: Worker Mode] ---
.PHONY: run-step build-conf clean-conf copy_artifacts

ifdef STEP_DIR
    -include $(STEP_DIR)/logic.mk
    -include $(STEP_DIR)/$(PROFILE_FILE).mk
    
    TAG_SUFFIX ?= -$(notdir $(STEP_DIR))
    NEW_TAG = $(TAG_PREFIX)$(TAG_SUFFIX)
    NEW_IMAGE = $(IMAGE_REPO):$(NEW_TAG)
    STAGING_DIR := $(ROOT_DIR)/build_for_container
    CONFIG_FILE_ABS := $(STEP_DIR)/auto_generated.conf
    CONFIG_BUILD_ARG := $(subst $(ROOT_DIR)/,,$(CONFIG_FILE_ABS))
endif

run-step:
	@set -e; \
	if [ -z "$$(docker images -q $(NEW_IMAGE))" ]; then \
		$(MAKE) -f $(SELF) build-conf --no-print-directory \
			STEP_DIR=$(STEP_DIR) \
			PROFILE_FILE=$(PROFILE_FILE) \
			TAG_PREFIX="$(TAG_PREFIX)" \
			IMAGE_REPO="$(IMAGE_REPO)" >&2; \
		if [ -n "$(PRE_BUILD_HOOK)" ]; then \
			$(MAKE) -f $(SELF) $(PRE_BUILD_HOOK) --no-print-directory \
				ROOT_DIR=$(ROOT_DIR) \
				STEP_DIR=$(STEP_DIR) \
				PROFILE_FILE=$(PROFILE_FILE) \
				TAG_PREFIX="$(TAG_PREFIX)" \
				CUDA_VERSION="$(CUDA_VERSION)" \
				CUDNN_VERSION="$(CUDNN_VERSION)" >&2; \
		fi; \
		docker build \
			--no-cache \
			--progress=plain \
			--build-arg BASE_IMAGE=$(BASE_IMAGE) \
			--build-arg INSTALL_CONFIG=$(CONFIG_BUILD_ARG) \
			--build-arg REPO_URL=$(AUTO_INSTALL_REPO_URL) \
			--build-arg SSH_PORT=$(SSH_PORT) \
			-t $(NEW_IMAGE) -f $(ROOT_DIR)/docker/Dockerfile $(ROOT_DIR) >&2; \
	fi; \
	echo "$(NEW_IMAGE)|$(STEP_EXPORT_VARS)"

build-conf:
	@echo "# Auto-generated" > $(CONFIG_FILE_ABS)
	@echo "[$(CONF_SECTION)]" >> $(CONFIG_FILE_ABS)
	@$(foreach v,$(filter CONF_%,$(.VARIABLES)), \
		if [ "$(v)" != "CONF_SECTION" ]; then \
			echo "$(subst CONF_,,$(v))=$($(v))" >> $(CONFIG_FILE_ABS); \
		fi; )

clean-conf:
	@rm -f $(CONFIG_FILE_ABS)

copy_artifacts:
	@echo "Preparing artifacts for $(notdir $(STEP_DIR))..." >&2
	@find $(STAGING_DIR) -mindepth 1 ! -name '.gitkeep' -delete
	@if [ -n "$(ARTIFACT_NAME)" ]; then \
		BUNDLE_DIR=$(ROOT_DIR)/build_pool/$(ARTIFACT_NAME); \
		if [ -d "$$BUNDLE_DIR" ]; then \
			mkdir -p "$(STAGING_DIR)/$(ARTIFACT_NAME)"; \
			cp -r $$BUNDLE_DIR/* "$(STAGING_DIR)/$(ARTIFACT_NAME)/"; \
			find "$(STAGING_DIR)/$(ARTIFACT_NAME)" -type d -name "build*" -exec rm -rf {} +; \
		fi; \
	fi
	@if [ "$(USE_PREBUILT)" = "yes" ]; then \
		if [ -n "$(ARTIFACT_PATH)" ]; then \
			mkdir -p "$(STAGING_DIR)/$$(dirname $(ARTIFACT_PATH))"; \
			cp -r $(ROOT_DIR)/build_pool/$(ARTIFACT_PATH)* "$(STAGING_DIR)/$$(dirname $(ARTIFACT_PATH))/"; \
		fi; \
	fi
