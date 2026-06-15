#!/bin/bash
# ==============================================================================
# 파일명: form.sh
# 설명: 복합 양식 입력을 위한 TUI 위젯 모듈.
# ==============================================================================

# @description 폼(Form)을 생성하여 여러 입력을 한 번에 받습니다.
# @param $1 backtitle
# @param $2 title
# @param $3 prompt
# @param $4 height
# @param $5 width
# @param $6 form_height
# @param $7... 필드 정의 (Label y x Item y x flen ilen) ...
ui_create_form() {
    local backtitle="$1" title="$2" prompt="$3"
    local height="$4" width="$5" form_height="$6"
    shift 6
    local fields=("$@")

    local result
    result=$(dialog --clear \
        --backtitle "${backtitle}" \
        --title "${title}" \
        --form "${prompt}" "${height}" "${width}" "${form_height}" \
        "${fields[@]}" \
        --stdout)

    if [[ $? -ne 0 ]]; then
        return 1
    else
        echo "${result}"
        return 0
    fi
}

# @description 연속적인 단계가 있는 폼(Form)을 생성합니다. (OK 버튼 대신 Next 버튼 사용)
ui_create_continuous_form() {
    local backtitle="$1" title="$2" prompt="$3"
    local height="$4" width="$5" form_height="$6"
    shift 6
    local fields=("$@")

    local result
    result=$(dialog --clear \
        --ok-label "Next" \
        --backtitle "${backtitle}" \
        --title "${title}" \
        --form "${prompt}" "${height}" "${width}" "${form_height}" \
        "${fields[@]}" \
        --stdout)

    if [[ $? -ne 0 ]]; then
        return 1
    else
        echo "${result}"
        return 0
    fi
}
