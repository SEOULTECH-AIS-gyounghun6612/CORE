# 지정된 CON_SPEC 경로의 설정을 로드하여 컨테이너를 실행
CON_SPEC ?= for_SL/with_cv

# 1. 컨테이너 설정 로드
-include $(CON_SPEC).mk

# 2. 이미지 태그 연동을 위해 이미지 설정 파일을 명시적으로 로드
# 컨테이너 설정(.mk)에 정의된 SPEC 변수를 사용합니다.
ifdef SPEC
    # docker/containers/ 디렉토리 기준이므로 ../images/ 로 접근
    -include ../images/$(SPEC).mk
endif

# 3. 이미지 이름 및 태그 생성을 위한 빌드 로직 로드
include ../utils/common_image.mk

# 4. 공통 실행 로직 로드
include ../utils/common_container.mk
