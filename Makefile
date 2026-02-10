# 프로젝트 및 이미지 이름 설정
PROJECT ?= computer_vision
AUTO_INSTALL_REPO_URL = https://github.com/DXR-keonghun6612/Utility-Belt.git

# 프로젝트별 설정 파일 로드
BUILD_CONFIG = docker/$(PROJECT)/build_config.mk
include $(BUILD_CONFIG)

# cuDNN 접미사 설정
ifeq ($(USE_CUDNN),yes)
    CUDNN_SUFFIX = -cudnn-$(IMAGE_TYPE)
else
    CUDNN_SUFFIX = -$(IMAGE_TYPE)
endif

# 이미지 주소 동적 생성
ifeq ($(CUDA_VERSION),none)
    $(error Error: Non-CUDA builds are not supported. Please specify a valid CUDA_VERSION)
endif

BASE_IMAGE = nvidia/cuda:$(CUDA_VERSION)$(CUDNN_SUFFIX)-ubuntu$(UBUNTU_VERSION)

# TODO: 추가적인 옵션으로 OpenCV와 같은 빌드 설치 처리하는 로직 추가 필요
#       이미 빌드 설치하는 스크립트는 존재 -> 빌드 디렉토리를 연결하는 방향으로 구현 필요

.PHONY: build run clean rebuild shell check-config

check-config:
	@if [ ! -f $(INSTALL_CONFIG) ]; then echo "Error: Config file $(INSTALL_CONFIG) not found."; exit 1; fi

build: check-config
	docker build \
		--build-arg BASE_IMAGE=$(BASE_IMAGE) \
		--build-arg INSTALL_CONFIG=$(INSTALL_CONFIG) \
		--build-arg REPO_URL=$(AUTO_INSTALL_REPO_URL) \
		--build-arg SSH_PORT=$(SSH_PORT) \
		-t $(IMAGE_NAME) -f docker/$(PROJECT)/Dockerfile .

run:
	@mkdir -p $(DATA_DIR)
	# 도커 컨테이너 실행: 설정된 네트워크, 장치, 볼륨 마운트 적용
	docker run -it -d \
		--name $(CONTAINER_NAME) \
		--network $(DOCKER_NETWORK) \
		$(DOCKER_DEVICES) \
		$(DOCKER_EXTRA_FLAGS) \
		-v $(DATA_DIR):$(CONTAINER_DATA_DIR) \
		$(IMAGE_NAME)

clean:
	-docker rm -f $(CONTAINER_NAME)

rebuild: clean build run

shell:
	docker exec -it $(CONTAINER_NAME) /bin/bash
