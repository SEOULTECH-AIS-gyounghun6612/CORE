# docker/containers/for_SL/with_cv.mk
# 참조할 이미지 설정
FLAVOR = ubuntu22.04-cuda12.4.1
MANIFEST = cv4.11.0-ros2.humble

# IMAGE_NAME은 시스템(common_container.mk)에 의해 자동 생성됨

# 컨테이너 이름
CONTAINER_NAME = with_cv

# 데이터 저장 경로 (호스트 기준)
DATA_DIR = $(HOME)/data
CONTAINER_DATA_DIR = /root/data

# 컨테이너 실행 설정
DOCKER_NETWORK = host
# DOCKER_DEVICES = --device=/dev/ttyUSB0:/dev/ttyUSB0
DOCKER_EXTRA_FLAGS = 
