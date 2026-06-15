#!/bin/bash
# ==============================================================================
# 파일명: load_core.sh
# 설명: CORE 라이브러리의 진입점. 모든 핵심 모듈을 로드하고 초기화합니다.
# ==============================================================================

# @description 핵심 라이브러리를 로드하고 설정을 초기화합니다.
# @param $1 project_root (프로젝트 루트 경로)
# @param $2 custom_conf (선택: 사용자 설정 파일 경로)
# @param $3 custom_template (선택: 템플릿 파일 경로)
# @param $4 is_interactive (선택: 대화형 UI 사용 여부, 기본값: true)
load_core_libraries() {
    local project_root="$1"
    local custom_conf="$2"
    local custom_template="$3"
    local is_interactive="${4:-true}"
    
    # 현재 스크립트의 디렉토리 경로
    local core_dir
    core_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

    # --- 1. 베이스 초기화 모듈 로드 및 실행 ---
    if [[ -f "${core_dir}/base/init.sh" ]]; then
        # shellcheck source=script/core/base/init.sh
        source "${core_dir}/base/init.sh"
        if ! initialize_core_environment "${project_root}" "${custom_conf}" "${custom_template}"; then
            return 1
        fi
    else
        echo "[FATAL] Failed to load base/init.sh. Aborting." >&2
        return 1
    fi

    # --- 2. 패키지 관리 모듈 로드 및 실행 ---
    if [[ -f "${core_dir}/package/init.sh" ]]; then
        # shellcheck source=script/core/package/init.sh
        source "${core_dir}/package/init.sh"
        if ! initialize_package_environment; then
            return 1
        fi
    else
        log_error "Failed to load package/init.sh. Aborting."
        return 1
    fi

    # --- 3. UI 모듈 로드 및 실행 (상호작용 모드일 때만) ---
    if [[ "${is_interactive}" == "true" ]]; then
        if [[ -f "${core_dir}/ui/init.sh" ]]; then
            # shellcheck source=script/core/ui/init.sh
            source "${core_dir}/ui/init.sh"
            if ! initialize_ui_environment; then
                return 1
            fi
        else
            log_error "Failed to load ui/init.sh. Aborting."
            return 1
        fi
    fi

    # --- 4. 기타 기능 모듈 로드 (02_*.sh 패턴 유지) ---
    for script in "${core_dir}"/02_*.sh; do
        if [[ -f "${script}" ]]; then
            # shellcheck source=/dev/null
            source "${script}"
        fi
    done

    return 0
}
