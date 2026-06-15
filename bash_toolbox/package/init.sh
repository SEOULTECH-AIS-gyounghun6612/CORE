#!/bin/bash
# ==============================================================================
# 파일명: init.sh
# 설명: 패키지 관리 모듈 초기화 및 진입점.
# ==============================================================================

# ------------------------------------------------------------------------------
# @description 현재 시스템의 패키지 관리자 타입을 반환합니다.
# ------------------------------------------------------------------------------
_get_package_manager_type() {
    if command -v dpkg &>/dev/null; then
        echo "dpkg"
    elif command -v rpm &>/dev/null; then
        echo "rpm"
    elif command -v pacman &>/dev/null; then
        echo "pacman"
    else
        return 1
    fi
}

# ------------------------------------------------------------------------------
# @description 패키지 관리 유틸리티를 초기화하고 명령어를 매핑합니다.
# ------------------------------------------------------------------------------
initialize_package_environment() {
    local pkg_dir
    pkg_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

    # 1. 시스템 패키지 관리자 감지 (모듈 로드 전에 먼저 수행)
    local pkg_type
    pkg_type=$(_get_package_manager_type) || {
        log_error "No supported package manager found."
        return 3
    }

    # 2. 공통 하위 모듈 로드 (manager, state)
    local modules=("manager.sh" "state.sh")
    for module in "${modules[@]}"; do
        if [[ -f "${pkg_dir}/${module}" ]]; then
            source "${pkg_dir}/${module}"
        else
            log_error "FATAL: Failed to load package module: ${module}"
            return 1
        fi
    done

    # 3. OS 특정 모듈 로드 (Debian/Ubuntu 계열일 경우에만 apt.sh 로드)
    if [[ "${pkg_type}" == "dpkg" ]]; then
        if [[ -f "${pkg_dir}/apt.sh" ]]; then
            source "${pkg_dir}/apt.sh"
        else
            log_warn "dpkg system detected, but apt.sh module is missing."
        fi
    fi

    # 4. 명령어 매핑
    export _PKG_INSTALL_CMD=""
    export _PKG_UPDATE_CMD=""
    export _PKG_REMOVE_CMD=""
    export _PKG_AUTOREMOVE_CMD=""
    export _PKG_CHECK_FN=""

    case "${pkg_type}" in
        "dpkg")
            export _PKG_CHECK_FN="_check_pkg_dpkg"
            export _PKG_UPDATE_CMD="${G_SUDO_PREFIX} apt-get update"
            export _PKG_INSTALL_CMD="${G_SUDO_PREFIX} apt-get install -y"
            export _PKG_REMOVE_CMD="${G_SUDO_PREFIX} apt-get purge -y"
            export _PKG_AUTOREMOVE_CMD="${G_SUDO_PREFIX} apt-get autoremove -y"
            ;;
        "rpm")
            export _PKG_CHECK_FN="_check_pkg_rpm"
            export _PKG_INSTALL_CMD="${G_SUDO_PREFIX} yum install -y"
            export _PKG_REMOVE_CMD="${G_SUDO_PREFIX} yum remove -y"
            export _PKG_AUTOREMOVE_CMD="${G_SUDO_PREFIX} yum autoremove -y"
            ;;
        "pacman")
            export _PKG_CHECK_FN="_check_pkg_pacman"
            export _PKG_INSTALL_CMD="${G_SUDO_PREFIX} pacman -S --noconfirm"
            export _PKG_REMOVE_CMD="${G_SUDO_PREFIX} pacman -Rns --noconfirm"
            export _PKG_AUTOREMOVE_CMD="${G_SUDO_PREFIX} pacman -Rns \$(pacman -Qdtq)"
            ;;
    esac

    # 5. 필수 공통 패키지 설치 확인 (wget, curl)
    local essential_pkgs=("wget" "curl")
    if ! ensure_packages_installed "PACKAGES_LIST" "Core Utilities" "${essential_pkgs[@]}"; then
        log_error "Failed to ensure essential packages: ${essential_pkgs[*]}"
        return 5
    fi

    log_success "Package management environment initialized successfully (${pkg_type})."
    return 0
}
