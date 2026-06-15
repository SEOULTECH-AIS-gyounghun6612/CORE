#!/bin/bash
# ==============================================================================
# 파일명: select.sh
# 설명: 메뉴 및 목록 선택을 위한 TUI 위젯 모듈.
# ==============================================================================

# @description 동적으로 dialog를 생성하고 선택된 결과를 반환합니다. (범용)
# @param $1 widget_type: "menu", "checklist", "radiolist"
# @param $2..$7 backtitle, title, prompt, height, width, list_height
# @param $8... 항목들
ui_create_selection() {
    local widget_type="$1"
    local backtitle="$2" title="$3" prompt="$4"
    local height="$5" width="$6" list_height="$7"
    shift 7
    local options=("$@")

    local ui_option
    case "${widget_type}" in
        "menu")      ui_option="--menu" ;;
        "checklist") ui_option="--checklist" ;;
        "radiolist") ui_option="--radiolist" ;;
        *) log_error "Invalid widget_type: ${widget_type}"; return 1 ;;
    esac

    local result
    result=$(dialog --clear \
        --backtitle "${backtitle}" \
        --title "${title}" \
        "${ui_option}" "${prompt}" "${height}" "${width}" "${list_height}" \
        "${options[@]}" \
        --stdout)

    if [[ $? -ne 0 ]]; then
        echo "CANCEL"
    else
        echo "${result}"
    fi
}

# @description 동적으로 메뉴를 생성하고 사용자의 선택(태그)을 반환합니다.
ui_create_menu() {
    local backtitle="$1" title="$2" prompt="$3"
    local height="$4" width="$5" menu_height="$6"
    shift 6
    local menu_items=("$@")

    ui_create_selection "menu" "${backtitle}" "${title}" "${prompt}" "${height}" "${width}" "${menu_height}" "${menu_items[@]}"
}

# @description 동적으로 체크리스트를 생성하고 선택된 태그들을 반환합니다.
ui_create_checklist() {
    local backtitle="$1" title="$2" prompt="$3"
    local height="$4" width="$5" list_height="$6"
    shift 6
    local checklist_options=("$@")

    ui_create_selection "checklist" "${backtitle}" "${title}" "${prompt}" "${height}" "${width}" "${list_height}" "${checklist_options[@]}"
}
