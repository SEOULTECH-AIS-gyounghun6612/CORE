# 컨테이너 실행을 위한 공통 로직

# CUDA 사용 여부에 따라 GPU 플래그 자동 설정
ifneq ($(CUDA_VERSION),none)
    ifneq ($(CUDA_VERSION),)
        DOCKER_GPU_FLAGS ?= --gpus all
    endif
endif

.PHONY: run stop shell clean

run:
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
	-docker stop $(CONTAINER_NAME)
	-docker rm $(CONTAINER_NAME)

clean: stop

shell:
	docker exec -it $(CONTAINER_NAME) /bin/bash
