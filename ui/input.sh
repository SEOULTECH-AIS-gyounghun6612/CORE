#!/bin/bash
# ==============================================================================
# 파일명: input.sh
# 설명: 사용자 입력을 받기 위한 TUI 위젯 모듈.
# ==============================================================================

# @description 사용자 입력을 받는 입력 박스를 표시합니다.
# @param $1 prompt 표시할 메시지
# @param $2 [title] 박스의 제목
# @param $3 [init_val] 초기값
# @param $4 [height] 높이
# @param $5 [width] 너비
ui_input_box() {
    local prompt="$1"
    local title="${2:-Input}"
    local init_val="${3:-}"
    local height="${4:-10}"
    local width="${5:-60}"

    local result
    result=$(dialog --title "${title}" --inputbox "${prompt}" "${height}" "${width}" "${init_val}" --stdout)
    
    if [[ $? -ne 0 ]]; then
        return 1
    else
        echo "${result}"
        return 0
    fi
}

# @description 비밀번호 입력 박스를 표시합니다.
# @param $1 prompt 표시할 메시지
# @param $2 [title] 박스의 제목
# @param $3 [height] 높이
# @param $4 [width] 너비
ui_password_box() {
    local prompt="$1"
    local title="${2:-Password}"
    local height="${3:-10}"
    local width="${4:-60}"

    local result
    result=$(dialog --title "${title}" --insecure --passwordbox "${prompt}" "${height}" "${width}" --stdout)

    if [[ $? -ne 0 ]]; then
        return 1
    else
        echo "${result}"
        return 0
    fi
}

# @description 파일 선택 대화상자를 표시합니다.
# @param $1 init_path 초기 경로
# @param $2 [title] 제목
# @param $3 [height] 높이
# @param $4 [width] 너비
ui_file_select() {
    local init_path="$1"
    local title="${2:-Select File}"
    local height="${3:-14}"
    local width="${4:-70}"

    local result
    result=$(dialog --title "${title}" --fselect "${init_path}" "${height}" "${width}" --stdout)

    if [[ $? -ne 0 ]]; then
        return 1
    else
        echo "${result}"
        return 0
    fi
}
