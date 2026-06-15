#!/bin/bash
# ==============================================================================
# 파일명: state.sh
# 설명: 설정 파일과 연동한 패키지 설치 상태 관리 및 동기화 로직.
# ==============================================================================

# ------------------------------------------------------------------------------
# @description 패키지 설치 상태를 설정 파일에 기록합니다. (내부 전용)
# ------------------------------------------------------------------------------
_record_state() {
    local config_file="$1"
    local section_name="$2"
    local comment="$3"
    local write_value="$4"
    shift 4
    local keys_to_record=("$@")

    if [[ ${#keys_to_record[@]} -eq 0 ]]; then
        log_warn "No keys to record in section [${section_name}]."
        return 0
    fi

    # 주석 및 섹션 추가
    if command -v add_config_comment &>/dev/null; then
        add_config_comment "${config_file}" "${section_name}" "${comment}"
    fi
    if command -v add_config_section &>/dev/null; then
        add_config_section "${config_file}" "${section_name}"
    fi

    log_info "Recording value '${write_value}' for ${#keys_to_record[@]} keys in section [${section_name}]."

    local key_value_pairs=()
    for key in "${keys_to_record[@]}"; do
        key_value_pairs+=("${key}" "${write_value}")
    done

    if ! set_config_value "${config_file}" "${section_name}" "${key_value_pairs[@]}"; then
        log_error "Failed to write configuration file."
        return 4
    fi

    log_success "Successfully recorded timestamps in section [${section_name}]."
    return 0
}

# ------------------------------------------------------------------------------
# @description 제시된 패키지 목록을 기준으로 설치 상태를 동기화합니다.
# ------------------------------------------------------------------------------
sync_package() {
    if [[ "$#" -lt 3 ]]; then
        log_error "Usage: sync_package <section_name> <comment> <pkg1> [pkg2]..."
        return 3
    fi

    local section_name="$1"
    local comment="$2"
    shift 2
    local requested_pkgs=("$@")
    
    local timestamp
    timestamp=$(date "+%Y-%m-%dT%H:%M:%S")

    log_info "Syncing packages in section [${section_name}]: ${requested_pkgs[*]}"

    # 1. 실제 설치가 필요한 패키지 확인
    local not_installed
    not_installed=$(check_package "${requested_pkgs[@]}")
    local status=$?

    local not_installed_pkgs=()
    if [[ $status -ne 0 ]]; then
        read -r -a not_installed_pkgs <<< "${not_installed}"
        log_info "Packages to be installed: ${not_installed_pkgs[*]}"
    fi

    # 2. 이미 설치되어 있지만 설정 파일에 기록이 없는 패키지 확인
    local pkgs_to_record=()
    for pkg in "${requested_pkgs[@]}"; do
        local is_missing=false
        for nip in "${not_installed_pkgs[@]}"; do
            [[ "$pkg" == "$nip" ]] && is_missing=true && break
        done

        if [[ "$is_missing" == "false" ]]; then
            local current_val
            current_val=$(get_config_value "${G_STATE_FILE}" "${section_name}" "${pkg}")
            [[ -z "${current_val}" ]] && pkgs_to_record+=("${pkg}")
        fi
    done

    # 3. 이미 설치된 패키지 중 미기록분 기록
    if [[ ${#pkgs_to_record[@]} -gt 0 ]]; then
        log_info "Recording already installed packages: ${pkgs_to_record[*]}"
        _record_state "${G_STATE_FILE}" "${section_name}" "${comment}" "${timestamp}" "${pkgs_to_record[@]}" || return 4
    fi

    # 4. 미설치 패키지 설치 및 기록
    if [[ ${#not_installed_pkgs[@]} -gt 0 ]]; then
        install_package "${not_installed_pkgs[@]}" || return 2
        log_info "Recording newly installed packages: ${not_installed_pkgs[*]}"
        _record_state "${G_STATE_FILE}" "${section_name}" "${comment}" "${timestamp}" "${not_installed_pkgs[@]}" || return 4
    fi

    log_success "Package sync completed for [${section_name}]."
    return 0
}

# ------------------------------------------------------------------------------
# @description 패키지를 제거하고 설정 파일의 기록을 삭제합니다.
# ------------------------------------------------------------------------------
uninstall_package() {
    if [[ "$#" -lt 2 ]]; then
        log_error "Usage: uninstall_package <section_name> <pkg1> [pkg2]..."
        return 3
    fi
    local section_name="$1"
    shift
    local requested_packages=("$@")

    # 시스템에서 패키지 제거 수행 (manager.sh 함수 호출)
    uninstall_package_from_system "${requested_packages[@]}"

    # 설정 파일 기록 삭제
    log_info "Clearing installation records from configuration file..."
    local key_value_pairs=()
    for pkg in "${requested_packages[@]}"; do
        key_value_pairs+=("${pkg}" "")
    done
    
    _record_state "${G_STATE_FILE}" "${section_name}" "" "" "${requested_packages[@]}"
}

# ------------------------------------------------------------------------------
# @description 필수 패키지의 설치 상태를 보장합니다.
# ------------------------------------------------------------------------------
ensure_packages_installed() {
    local section_name="$1"
    local comment="${2:-Required Packages}"
    shift 2
    local packages=("$@")

    [[ ${#packages[@]} -eq 0 ]] && return 0

    if [[ "${G_IS_ROOT}" == "false" && -z "${G_SUDO_PREFIX}" ]]; then
        if check_package "${packages[@]}" > /dev/null; then
            return 0
        else
            log_error "Essential packages missing and no sudo available: ${packages[*]}"
            return 5
        fi
    fi

    sync_package "${section_name}" "${comment}" "${packages[@]}"
}
