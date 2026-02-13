# 0. 빌드 인자 정의
ARG BASE_IMAGE=nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04
FROM ${BASE_IMAGE}

# 스테이지 내부에서 사용할 변수 재선언
ARG REPO_URL="https://github.com/DXR-keonghun6612/Utility-Belt.git"
ARG DEBIAN_FRONTEND=noninteractive

# 1. 필수 기본 패키지 및 OpenGL 의존성 설치
RUN apt-get update && apt-get install -y \
    git \
    tzdata \
    libglu1-mesa-dev \
    freeglut3-dev \
    mesa-common-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    && rm -rf /var/lib/apt/lists/*

# 2. GitHub에서 프로젝트 클론 (Submodule 포함)
WORKDIR /opt
RUN rm -rf asap && git clone -b ASAP_linux --recursive ${REPO_URL} asap

WORKDIR /opt/asap

# 3. 빌드 결과물 주입
RUN mkdir -p /opt/build
COPY build_for_container /opt/build
RUN find /opt/build -maxdepth 2 && du -sh /opt/build/*

ARG INSTALL_CONFIG=docker/images/auto_install_config.conf
COPY ${INSTALL_CONFIG} ./conf/config_auto.conf

# 4. ASAP 자동 설치 실행
RUN chmod +x ./ASAP_auto.sh \
    && bash -e ./ASAP_auto.sh ./conf/config_auto.conf \
    && rm -rf /opt/build
