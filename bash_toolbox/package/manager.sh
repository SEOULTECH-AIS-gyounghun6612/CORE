#!/bin/bash
# ==============================================================================
# 파일명: manager.sh
# 설명: OS 비종속 패키지 관리자 추상화 계층.
# ==============================================================================

# ==============================================================================
# 1. 패키지 상태 확인 헬퍼
# ==============================================================================
_check_pkg_dpkg()   { dpkg-query -W -f='${Status}' "$1" 2>/dev/null | grep -q 'install ok installed'; }
_check_pkg_rpm()    { rpm -q "$1" &>/dev/null; }
_check_pkg_pacman() { pacman -Q "$1" &>/dev/null; }

# ==============================================================================
# 2. 공개 패키지 관리 함수
# ==============================================================================

# @description 특정 패키지의 설치 여부를 확인합니다.
# @param $1 package_name
is_package_installed() {
    local pkg_name="$1"
    if [[ -z "${pkg_name}" ]]; then
        log_warn "Package name cannot be empty."
        return 1
    fi

    "${_PKG_CHECK_FN}" "${pkg_name}"
    return $?
}

# @description 여러 패키지의 설치 여부를 일괄 확인합니다.
# @param $1... pkg_names
# @stdout 설치되지 않은 패키지 목록
check_package() {
    if [[ "$#" -lt 1 ]]; then
        log_warn "No packages provided to check."
        return 2
    fi
    
    local packages_to_install=()
    local all_installed=true

    log_info "Checking package installation status..."

    for pkg_name in "$@"; do
        if is_package_installed "${pkg_name}"; then
            log_success "Package '${pkg_name}' is already installed."
        else
            log_warn "Package '${pkg_name}' is not installed."
            packages_to_install+=("${pkg_name}")
            all_installed=false
        fi
    done

    if [[ "${all_installed}" == "true" ]]; then
        return 0
    else
        echo "${packages_to_install[@]}"
        return 1
    fi
}

# @description 하나 이상의 패키지를 설치합니다.
# @param $1... pkg_names
install_package() {
    # 업데이트 명령어가 설정된 경우 (apt 계열 등) 세션당 최초 1회만 실행
    if [[ -n "${_PKG_UPDATE_CMD}" ]] && [[ "${G_PKG_LIST_UPDATED}" != "true" ]]; then
        log_info "Updating package list (first time in this session)..."
        if ${_PKG_UPDATE_CMD}; then
            export G_PKG_LIST_UPDATED="true"
        else
            log_error "Package list update failed."
            return 1
        fi
    fi

    log_info "Installing packages: $*"
    if ${_PKG_INSTALL_CMD} "$@"; then
        log_success "Packages installed successfully."
        return 0
    else
        log_error "Package installation failed: $*"
        return 2
    fi
}

# @description 하나 이상의 패키지를 제거합니다.
# @param $1... pkg_names
uninstall_package_from_system() {
    local requested_packages=("$@")
    local successfully_removed=false

    log_info "Starting sequential package removal process..."

    for pkg in "${requested_packages[@]}"; do
        if is_package_installed "${pkg}"; then
            if ${_PKG_REMOVE_CMD} "${pkg}"; then
                log_success "Successfully removed '${pkg}'."
                successfully_removed=true
            else
                log_error "Failed to remove '${pkg}'."
            fi
        else
            log_warn "Package '${pkg}' is already not installed."
        fi
    done

    # 불필요 의존성 제거
    if [[ "${successfully_removed}" == "true" ]] && [[ -n "${_PKG_AUTOREMOVE_CMD}" ]]; then
        log_info "Running autoremove to clean up dependencies..."
        ${_PKG_AUTOREMOVE_CMD} || log_warn "Autoremove command failed."
    fi
}

# @description 대화형 설치 스크립트를 실행합니다.
# @param $1 installer_path
_run_interactive_installer() {
    local installer_path="$1"
    if [[ -z "${installer_path}" ]] || [[ ! -f "${installer_path}" ]]; then
        log_error "Installer script not found at: ${installer_path}"
        return 1
    fi

    clear
    echo "=============================================================================="
    echo " Starting Interactive Installation"
    echo "=============================================================================="
    echo "An external installer is about to run."
    echo "Please follow the on-screen instructions."
    echo "Press Enter to begin the installation..."
    read -r
    
    bash "${installer_path}"
    return $?
}
