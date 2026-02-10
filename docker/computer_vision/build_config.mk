# 컴퓨터 비전용 이미지 빌드 설정
IMAGE_NAME = asap_robot_image
CONTAINER_NAME = managed_module

CUDA_VERSION = 12.4.1
USE_CUDNN = yes
IMAGE_TYPE = devel
UBUNTU_VERSION = 22.04
INSTALL_CONFIG = package_config/with_ros2.conf

# 데이터 저장 경로 (호스트 기준)
DATA_DIR = $(HOME)/data
# OPENCV_DIR = $(HOME)/libs/opencv

# 컨테이너 실행 설정
DOCKER_NETWORK = host
SSH_PORT = 2222

# 하드웨어 장치 연결 설정 (UART, 카메라 등 특정 장치만 연결할 때 사용)
# 예: --device=/dev/ttyUSB0:/dev/ttyUSB0 --device=/dev/video0:/dev/video0
DOCKER_DEVICES = --device=/dev/ttyUSB0:/dev/ttyUSB0

# 기타 도커 실행 플래그 (필요 시 --privileged 등을 추가할 수 있으나 권장되지 않음)
DOCKER_EXTRA_FLAGS = 

# 컨테이너 내부 마운트 경로 설정
CONTAINER_DATA_DIR = /root/data
# CONTAINER_OPENCV_DIR = /opt/opencv
