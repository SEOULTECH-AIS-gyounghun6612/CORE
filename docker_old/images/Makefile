# 지정된 SPEC 경로의 설정을 로드하여 이미지를 빌드
# SPEC 예: ubuntu22.04-cuda12.4.1/cv4.10.0-ros2.humble
SPEC ?= ubuntu22.04-cuda12.4.1/cv4.10.0-ros2.humble

# 설정 파일 로드 (.mk 확장자 추가)
-include $(SPEC).mk

# 공통 빌드 로직 로드
include ../utils/common_image.mk
