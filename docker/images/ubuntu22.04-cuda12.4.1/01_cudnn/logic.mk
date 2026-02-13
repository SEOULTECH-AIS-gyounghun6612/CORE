# docker/images/ubuntu22.04-cuda12.4.1/01_cudnn/logic.mk

# 1. 인자(Data) 로드
-include $(STEP_DIR)/$(PROFILE_FILE).mk

# 2. 로직 본체
TAG_SUFFIX = -cudnn_$(VERSION)

CONF_SECTION = CUDNN_LIBRARY_PROFILE
CONF_version = $(VERSION)

# 다음 스텝으로 CuDNN 버전을 전파
STEP_EXPORT_VARS = CUDNN_VERSION=$(VERSION)
