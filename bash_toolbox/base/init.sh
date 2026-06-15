#!/bin/bash
# ==============================================================================
# 파일명: init.sh
# 설명: 베이스 라이브러리 로드 및 전역 환경 초기화.
# ==============================================================================

# ==============================================================================
# 1. 시스템 컨텍스트 초기화 (권한, 사용자, 그룹)
# ==============================================================================
_initialize_system_context() {
    export G_IS_ROOT
    if [[ $EUID -eq 0 ]]; then
        G_IS_ROOT=true
    else
        G_IS_ROOT=false
    fi

    export G_IS_SUDO
    if [[ -n "${SUDO_USER}" ]]; then
        G_IS_SUDO=true
    else
        G_IS_SUDO=false
    fi

    export G_ACTUAL_USER
    if [[ "${G_IS_SUDO}" == "true" ]]; then
        G_ACTUAL_USER="${SUDO_USER}"
    elif [[ -n "${USER}" ]]; then
        G_ACTUAL_USER="${USER}"
    else
        G_ACTUAL_USER=$(id -un)
    fi

    export G_ACTUAL_GROUP
    G_ACTUAL_GROUP=$(id -gn "${G_ACTUAL_USER}")

    export G_SUDO_PREFIX
    if [[ "${G_IS_ROOT}" == "false" ]]; then
        G_SUDO_PREFIX="sudo"
    else
        G_SUDO_PREFIX=""
    fi
}

# ==============================================================================
# 2. 프로젝트 디렉토리 구조 초기화
# ==============================================================================
# @description 필요한 디렉토리(log, conf, state)를 정의하고 생성합니다.
# @param $1 project_root
_initialize_directories() {
    local root_dir="$1"
    
    export G_LOG_DIR="${root_dir}/log"
    export G_CONF_DIR="${root_dir}/conf"
    export G_STATE_DIR="${root_dir}/state"

    local dirs=("${G_LOG_DIR}" "${G_CONF_DIR}" "${G_STATE_DIR}")

    for dir in "${dirs[@]}"; do
        [[ ! -d "${dir}" ]] && mkdir -p "${dir}" 2>/dev/null
    done
}

# ==============================================================================
# 3. 기본 파일 초기화 및 준비
# ==============================================================================
# @description 로그, 상태, 설정 파일의 경로를 정의하고 기본 파일을 생성/복사합니다.
# @param $1 custom_config (선택)
# @param $2 custom_template (선택)
_initialize_files() {
    local custom_config="$1"
    local custom_template="$2"

    # --- 경로 정의 ---
    export G_LOG_FILE
    G_LOG_FILE="${G_LOG_DIR}/asap_$(date +%Y%m%d).log"
    
    export G_STATE_FILE="${G_STATE_DIR}/system.state"
    export CONFIG_FILE="${custom_config:-${G_CONF_DIR}/default.conf}"

    export TEMPLATE_FILE
    if [[ -n "${custom_template}" ]]; then
        TEMPLATE_FILE="${custom_template}"
    else
        local default_template="${G_PROJECT_ROOT}/template/config.conf"
        [[ ! -f "${default_template}" ]] && default_template="${G_PROJECT_ROOT}/template/ubuntu_config.conf"
        TEMPLATE_FILE="${default_template}"
    fi

    # --- 파일 준비 (Bootstrap) ---
    [[ ! -f "${G_LOG_FILE}" ]] && touch "${G_LOG_FILE}" 2>/dev/null
    [[ ! -f "${G_STATE_FILE}" ]] && touch "${G_STATE_FILE}" 2>/dev/null

    if [[ ! -f "${CONFIG_FILE}" ]]; then
        if [[ -f "${TEMPLATE_FILE}" ]]; then
            cp "${TEMPLATE_FILE}" "${CONFIG_FILE}" 2>/dev/null
        else
            touch "${CONFIG_FILE}" 2>/dev/null
        fi
    fi
}

# ==============================================================================
# 공개 초기화 함수 (Bootstrap)
# ==============================================================================
# @description 베이스 모듈들을 로드하고 시스템 환경을 일괄 초기화합니다.
# @param $1 project_root
# @param $2 custom_config (선택)
# @param $3 custom_template (선택)
initialize_core_environment() {
    local root_dir="${1:-$(pwd)}"
    local config_path="$2"
    local template_path="$3"

    # 1. 경로 확정
    export G_PROJECT_ROOT="${root_dir}"
    local base_dir
    base_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

    # 2. 의존성 모듈 로드 (함수 내부 로딩)
    if [[ -f "${base_dir}/common.sh" ]]; then
        source "${base_dir}/common.sh"
    else
        echo "[FATAL] common.sh not found." >&2; return 1
    fi

    if [[ -f "${base_dir}/profile_engine.sh" ]]; then
        source "${base_dir}/profile_engine.sh"
    else
        echo "[FATAL] profile_engine.sh not found." >&2; return 1
    fi

    # 3. 환경 초기화 수행
    _initialize_system_context
    _initialize_directories "${G_PROJECT_ROOT}"
    _initialize_files "${config_path}" "${template_path}"

    log_success "Core environment initialized successfully (Root: ${G_PROJECT_ROOT})"
}
