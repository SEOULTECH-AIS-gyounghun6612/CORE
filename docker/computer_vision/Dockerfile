# 0. 빌드 인자 정의 (Makefile에서 완성된 이미지명을 전달받음)
ARG BASE_IMAGE=nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04
FROM ${BASE_IMAGE}

# 스테이지 내부에서 사용할 변수 재선언
ARG REPO_URL="https://github.com/DXR-keonghun6612/Utility-Belt.git"
ARG DEBIAN_FRONTEND=noninteractive

# 1. 필수 기본 패키지 및 git 설치
RUN apt-get update && apt-get install -y git tzdata \
    && rm -rf /var/lib/apt/lists/*

# 2. GitHub에서 프로젝트 클론 (Submodule 포함)
WORKDIR /opt
RUN git clone -b ASAP_linux --recursive ${REPO_URL} asap

WORKDIR /opt/asap

ARG INSTALL_CONFIG=docker/config.conf

# 3. 설정 파일 복사 (현재 작업 디렉토리 기준)
# 빌드 인자로 받은 호스트의 설정 파일을 컨테이너의 현재 위치로 가져옵니다.
COPY ${INSTALL_CONFIG} ./config_auto.conf

# 4. ASAP 자동 설치 실행
RUN chmod +x ./ASAP_auto.sh \
    && ./ASAP_auto.sh ./config_auto.conf

# 4. SSH 서비스 설정 (Host Network 모드 대응)
# 포트를 변경하여 호스트의 SSH(22)와 충돌 방지
ARG SSH_PORT=2222
RUN mkdir -p /run/sshd \
    && echo "root:root" | chpasswd \
    && sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config \
    && sed -i "s/#Port 22/Port ${SSH_PORT}/" /etc/ssh/sshd_config \
    # 이미 Port 22가 활성화되어 있을 경우를 대비해 치환 (주석 없는 경우)
    && sed -i "s/^Port 22/Port ${SSH_PORT}/" /etc/ssh/sshd_config || true

# 5. 진입점 설정
EXPOSE ${SSH_PORT}
CMD ["/usr/sbin/sshd", "-D"]
