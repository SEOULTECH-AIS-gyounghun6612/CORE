#!/bin/bash
# ==============================================================================
# 파일명: init.sh
# 설명: TUI 모듈 초기화 및 진입점.
# ==============================================================================

# @description UI에 필요한 패키지(dialog)를 확인하고 초기화합니다.
initialize_ui_environment() {
    local ui_dir
    ui_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

    # 1. 하위 모듈 로드
    local modules=("info.sh" "select.sh" "input.sh" "form.sh")
    for module in "${modules[@]}"; do
        if [[ -f "${ui_dir}/${module}" ]]; then
        # shellcheck source=dev/null
            source "${ui_dir}/${module}"
        else
            log_error "FATAL: Failed to load UI module: ${module}"
            return 1
        fi
    done

    # 2. 필수 패키지 확인 및 설치 보장
    if command -v ensure_packages_installed &>/dev/null; then
        ensure_packages_installed "PACKAGES_LIST" "Dialog utility" "dialog" || return $?
    else
        log_warn "'ensure_packages_installed' function not found. Skipping 'dialog' check."
    fi

    log_success "UI environment initialized successfully."
    return 0
}
