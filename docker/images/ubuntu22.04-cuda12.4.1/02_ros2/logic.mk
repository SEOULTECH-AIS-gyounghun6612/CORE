# docker/images/ubuntu22.04-cuda12.4.1/02_ros2/logic.mk

# 1. 인자(Data) 로드
-include $(STEP_DIR)/$(PROFILE_FILE).mk

# 2. 로직 본체
TAG_SUFFIX = -ros2_$(VERSION)_$(VARIANT)

CONF_SECTION = ROS2_PROFILE
CONF_distro = $(VERSION)
CONF_variant = $(VARIANT)

# ROS2는 별도의 파이프라인 전파 변수가 없으므로 빈 값 유지 (필요 시 추가 가능)
STEP_EXPORT_VARS = 
