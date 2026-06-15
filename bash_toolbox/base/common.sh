#!/bin/bash
# ==============================================================================
# 파일명: common.sh
# 설명: 프로젝트 전반에서 사용되는 공통 보조 유틸리티 함수.
# ==============================================================================

# ------------------------------------------------------------------------------
# @description 메시지를 터미널에 출력하고 로그 파일에 저장합니다. (내부 엔진)
# @param $1 level (INFO, WARN, ERROR, SUCCESS)
# @param $2 message
# ------------------------------------------------------------------------------
log_message() {
    local level="$1"
    local message="$2"
    local timestamp; timestamp=$(date "+%Y-%m-%d %H:%M:%S")
    local log_content="[${timestamp}] [${level}] ${message}"

    # 1. 터미널 출력 (색상 적용, 표준 에러로 출력하여 명령어 치환 시 캡처 방지)
    case "${level}" in
        "INFO")    echo -e "\e[34m[INFO]\e[0m ${message}" >&2 ;;
        "WARN")    echo -e "\e[33m[WARN]\e[0m ${message}" >&2 ;;
        "ERROR")   echo -e "\e[31m[ERROR]\e[0m ${message}" >&2 ;;
        "SUCCESS") echo -e "\e[32m[SUCCESS]\e[0m ${message}" >&2 ;;
        *)         echo "[${level}] ${message}" >&2 ;;
    esac

    # 2. 로그 파일 저장
    if [[ -n "${G_LOG_FILE}" ]]; then
        echo "${log_content}" >> "${G_LOG_FILE}" 2>/dev/null
    fi
}

# --- 직관적인 로깅을 위한 하위 함수들 ---
log_info()    { log_message "INFO" "$1"; }
log_warn()    { log_message "WARN" "$1"; }
log_error()   { log_message "ERROR" "$1"; }
log_success() { log_message "SUCCESS" "$1"; }

# ------------------------------------------------------------------------------
# @description 임시 파일을 최종 목적지로 이동하고, 필요시 소유권 복원.
# @param $1 source_path (임시 파일 경로)
# @param $2 destination_path (최종 파일 경로)
# ------------------------------------------------------------------------------
commit_file_change() {
    local source_path="$1"
    local destination_path="$2"

    # 기본 소유자와 권한 설정 (파일이 새로 생성될 경우 사용).
    local owner="root"
    local group="root"
    local mode="0644"

    # 대상 파일이 이미 존재하면, 기존 소유자와 권한을 그대로 사용.
    if [[ -f "${destination_path}" ]]; then
        owner=$(stat -c '%U' "${destination_path}")
        group=$(stat -c '%G' "${destination_path}")
        mode=$(stat -c '%a' "${destination_path}")
    fi

    log_info "Committing changes to '${destination_path}'..."

    # install 명령어로 파일 복사, 소유자/그룹/권한 설정을 한 번에 처리.
    if ${G_SUDO_PREFIX} install -o "${owner}" -g "${group}" -m "${mode}" "${source_path}" "${destination_path}"; then
        return 0
    else
        log_error "Failed to commit file changes to '${destination_path}'."
        return 1
    fi
}
