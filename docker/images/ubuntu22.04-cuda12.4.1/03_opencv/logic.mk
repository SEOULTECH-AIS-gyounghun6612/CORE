# docker/images/ubuntu22.04-cuda12.4.1/03_opencv/logic.mk

# 1. 인자(Data) 로드
-include $(STEP_DIR)/$(PROFILE_FILE).mk

# 2. 로직 본체
PRE_BUILD_HOOK = copy_artifacts
USE_PREBUILT = yes
ARTIFACT_NAME = opencv-$(VERSION)

# 상태(CUDNN_VERSION)에 따른 지능형 분기 및 태그/경로 생성
ifeq ($(CUDNN_VERSION),-1)
    # [CPU Build Mode]
    TAG_SUFFIX = -cv$(VERSION)-cpu-cpp$(CPP_STD)
    CONF_with_cuda = OFF
    CONF_gpu_arch = 
    ARTIFACT_PATH = $(ARTIFACT_NAME)/build-cpu-cpp$(CPP_STD)
else
    # [CUDA Build Mode]
    # 태그에 CUDA/CuDNN 정보를 명시하여 가독성 증대
    TAG_SUFFIX = -cv$(VERSION)-cuda-dn$(CUDNN_VERSION)-cpp$(CPP_STD)
    CONF_with_cuda = ON
    CONF_gpu_arch = $(GPU_ARCH)
    GPU_ARCH_SAFE = $(subst ;,_,$(GPU_ARCH))
    ARTIFACT_PATH = $(ARTIFACT_NAME)/build-cuda-v$(CUDA_VERSION)-dn$(CUDNN_VERSION)-$(GPU_ARCH_SAFE)-cpp$(CPP_STD)
endif

# 컨테이너 주입용 공통 설정
CONF_SECTION = OPENCV_PROFILE
CONF_version = $(VERSION)
CONF_jobs = $(JOBS)
CONF_install_prefix = /usr/local
CONF_work_dir = /opt/build
CONF_python_path = /usr/bin/python3
CONF_cpp_std = $(CPP_STD)

STEP_EXPORT_VARS = 
