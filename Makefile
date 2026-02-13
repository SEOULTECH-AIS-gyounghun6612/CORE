# Root Makefile
# Usage:
#   make build SPEC=docker/images/ubuntu22.04-cuda12.4.1/dn9.19.0.56-ros2_humble-cv_4.11.0.mk
#   make run SPEC=... CON_SPEC=for_SL/with_cv

ROOT_DIR := $(CURDIR)
export ROOT_DIR

# 1. SPEC 파일 경로 수신 (기본값 설정)
SPEC ?= docker/images/ubuntu22.04-cuda12.4.1/dn9.19.0.56-ros2_humble-cv_4.11.0.mk
CON_SPEC ?= for_SL/with_cv

# 2. 경로 분석 (Path Parsing)
MANIFEST := $(basename $(notdir $(SPEC)))
FLAVOR := $(notdir $(patsubst %/,%,$(dir $(SPEC))))

.PHONY: build build-debug run stop shell clean

build:
	@echo ">>> Target Manifest: $(MANIFEST) (Flavor: $(FLAVOR))"
	$(MAKE) -f $(ROOT_DIR)/docker/utils/image_builder.mk build FLAVOR=$(FLAVOR) MANIFEST=$(MANIFEST)

build-debug:
	@echo ">>> [DEBUG MODE] Target Manifest: $(MANIFEST) (Flavor: $(FLAVOR))"
	$(MAKE) -f $(ROOT_DIR)/docker/utils/image_builder.mk build FLAVOR=$(FLAVOR) MANIFEST=$(MANIFEST) DEBUG=1

run:
	$(MAKE) -f $(ROOT_DIR)/docker/utils/common_container.mk run CON_SPEC=$(CON_SPEC) FLAVOR=$(FLAVOR) MANIFEST=$(MANIFEST)

stop:
	$(MAKE) -f $(ROOT_DIR)/docker/utils/common_container.mk stop CON_SPEC=$(CON_SPEC) FLAVOR=$(FLAVOR) MANIFEST=$(MANIFEST)

shell:
	$(MAKE) -f $(ROOT_DIR)/docker/utils/common_container.mk shell CON_SPEC=$(CON_SPEC) FLAVOR=$(FLAVOR) MANIFEST=$(MANIFEST)

clean:
	$(MAKE) -f $(ROOT_DIR)/docker/utils/common_container.mk clean CON_SPEC=$(CON_SPEC) FLAVOR=$(FLAVOR) MANIFEST=$(MANIFEST)
