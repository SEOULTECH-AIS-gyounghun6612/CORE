#!/bin/bash
# ==============================================================================
# 파일명: apt.sh
# 설명: Debian/Ubuntu 계열 시스템 전용 패키지 유틸리티.
# ==============================================================================

# ------------------------------------------------------------------------------
# @description APT 저장을 위한 GPG 키를 다운로드하고 설치합니다.
# @param $1 url (키 다운로드 URL)
# @param $2 dest_path (키 저장 경로)
# @param $3 dearmor (선택: "true"이면 gpg --dearmor 실행, 기본값 "false")
# ------------------------------------------------------------------------------
install_apt_gpg_key() {
    local url="$1"
    local dest_path="$2"
    local do_dearmor="${3:-false}"

    if [[ -z "$url" ]] || [[ -z "$dest_path" ]]; then
        log_error "Usage: install_apt_gpg_key <url> <dest_path> [dearmor]"
        return 1
    fi

    if ! command -v curl &>/dev/null; then
        log_warn "'curl' command not found. Key installation might fail."
    fi

    log_info "Installing APT GPG key from ${url}..."

    local tmp_file
    tmp_file=$(mktemp)

    local success=false
    if [[ "${do_dearmor}" == "true" ]]; then
        if curl -fsSL "${url}" | gpg --dearmor > "${tmp_file}"; then
            success=true
        fi
    else
        if curl -fsSL "${url}" -o "${tmp_file}"; then
            success=true
        fi
    fi

    if [[ "$success" != "true" ]]; then
        log_error "Failed to download GPG key from ${url}"
        rm -f "${tmp_file}"
        return 1
    fi

    local dest_dir
    dest_dir=$(dirname "${dest_path}")
    if [[ ! -d "${dest_dir}" ]]; then
        ${G_SUDO_PREFIX} install -m 0755 -d "${dest_dir}"
    fi

    if ${G_SUDO_PREFIX} install -m 0644 "${tmp_file}" "${dest_path}"; then
        log_success "GPG key installed to ${dest_path}"
        rm -f "${tmp_file}"
        return 0
    else
        log_error "Failed to install GPG key to ${dest_path}"
        rm -f "${tmp_file}"
        return 1
    fi
}

# ------------------------------------------------------------------------------
# @description APT 저장소를 추가하고 패키지 목록을 업데이트합니다.
# @param $1 repo_filename (저장소 파일 이름, 확장자 제외)
# @param $2 repo_content (저장소 정의 내용)
# ------------------------------------------------------------------------------
setup_apt_repository() {
    local repo_filename="$1"
    local repo_content="$2"
    local list_file="/etc/apt/sources.list.d/${repo_filename}.list"

    if [[ -z "$repo_filename" ]] || [[ -z "$repo_content" ]]; then
        log_error "Usage: setup_apt_repository <filename> <content>"
        return 1
    fi

    if [[ -f "${list_file}" ]]; then
        log_info "Repository '${repo_filename}' is already configured."
        return 0
    fi

    log_info "Adding APT repository: ${repo_filename}..."
    if [[ ! -d "/etc/apt/sources.list.d" ]]; then
         ${G_SUDO_PREFIX} install -m 0755 -d /etc/apt/sources.list.d
    fi

    if ! echo "${repo_content}" | ${G_SUDO_PREFIX} tee "${list_file}" > /dev/null; then
        log_error "Failed to write to ${list_file}."
        return 1
    fi

    log_info "Updating package list for ${repo_filename}..."
    if ! ${_PKG_UPDATE_CMD}; then
        log_error "'apt update' failed after adding repository."
        ${G_SUDO_PREFIX} rm -f "${list_file}"
        return 1
    fi

    return 0
}
