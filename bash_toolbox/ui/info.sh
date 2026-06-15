#!/bin/bash
# ==============================================================================
# 파일명: info.sh
# 설명: 정보 표시 및 확인을 위한 TUI 위젯 모듈.
# ==============================================================================

# @description Yes/No 확인창을 표시합니다.
# @param $1 prompt 표시할 메시지
# @param $2 [title] 박스의 제목 (기본값: "Confirmation")
# @param $3 [height] 높이 (기본값: 10)
# @param $4 [width] 너비 (기본값: 60)
# @return 'Yes' 선택 시 0, 그 외 1
ui_confirm() {
    local prompt="$1"
    local title="${2:-Confirmation}"
    local height="${3:-10}"
    local width="${4:-60}"
    
    dialog --title "${title}" --yesno "${prompt}" "${height}" "${width}"
    return $?
}

# @description 단순 정보/경고 메시지 박스를 표시합니다.
# @param $1 message 표시할 메시지
# @param $2 [title] 박스의 제목 (기본값: "Information")
# @param $3 [height] 높이 (기본값: 0)
# @param $4 [width] 너비 (기본값: 0)
ui_message_box() {
    local message="$1"
    local title="${2:-Information}"
    local height="${3:-0}"
    local width="${4:-0}"

    dialog --title "${title}" --msgbox "\n${message}" "${height}" "${width}" --stdout
}

# @description 텍스트 파일의 내용을 표시하는 박스를 띄웁니다.
# @param $1 file_path 표시할 파일 경로
# @param $2 [title] 제목
# @param $3 [height] 높이
# @param $4 [width] 너비
ui_show_textbox() {
    local file_path="$1"
    local title="${2:-Text View}"
    local height="${3:-20}"
    local width="${4:-70}"

    dialog --title "${title}" --textbox "${file_path}" "${height}" "${width}"
}
