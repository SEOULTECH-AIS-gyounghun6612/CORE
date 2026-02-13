# docker/utils/common_container.mk
# ------------------------------------------------------------------
# Unified Container Management Logic
# ------------------------------------------------------------------

# Required variables passed from root Makefile:
# ROOT_DIR, FLAVOR, MANIFEST, CON_SPEC

DOCKER_DIR := $(ROOT_DIR)/docker
FLAVOR_DIR := $(DOCKER_DIR)/images/$(FLAVOR)

# 1. 환경 및 매니페스트 로드
-include $(FLAVOR_DIR)/env.mk
-include $(FLAVOR_DIR)/$(MANIFEST).mk
-include $(DOCKER_DIR)/containers/$(CON_SPEC).mk

# 2. 이미지 이름 결정
# 매니페스트에 기록된 BUILT_IMAGE_NAME 이 있으면 사용, 없으면 동적 계산
ifdef BUILT_IMAGE_NAME
    IMAGE_NAME := $(BUILT_IMAGE_NAME)
else
    # Derived Variables (Fallback logic)
    CUDA_SHORT := $(shell echo $(CUDA_VERSION) | sed -E 's/([0-9]+\.[0-9]+\.)([0-9]).*/\1\2/')
    TAG_PREFIX_BASE := u$(UBUNTU_VERSION)-cu$(CUDA_VERSION)
    IMAGE_NAME := $(IMAGE_REPO):$(TAG_PREFIX_BASE)-$(IMAGE_TAG)
endif

# 3. GPU 플래그 자동 설정
ifneq ($(CUDA_VERSION),none)
    ifneq ($(CUDA_VERSION),)
        DOCKER_GPU_FLAGS ?= --gpus all
    endif
endif

.PHONY: run stop shell clean

run:
	@echo ">>> Starting container: $(CONTAINER_NAME) using $(IMAGE_NAME)"
	@mkdir -p $(DATA_DIR)
	docker run -it -d \
		--name $(CONTAINER_NAME) \
		--network $(DOCKER_NETWORK) \
		$(DOCKER_GPU_FLAGS) \
		$(DOCKER_DEVICES) \
		$(DOCKER_EXTRA_FLAGS) \
		-v $(DATA_DIR):$(CONTAINER_DATA_DIR) \
		$(IMAGE_NAME)

stop:
	@echo ">>> Stopping and removing container: $(CONTAINER_NAME)"
	-docker stop $(CONTAINER_NAME)
	-docker rm $(CONTAINER_NAME)

clean: stop

shell:
	docker exec -it $(CONTAINER_NAME) /bin/bash
