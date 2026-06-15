#!/bin/bash
# ==============================================================================
# 파일명: profile_engine.sh
# 설명: awk 기반의 INI 형식 프로필 및 설정 분석 엔진.
# ==============================================================================

# Bash 확장 패턴 매칭 활성화 (parse_config_to_array 등에서 사용)
shopt -s extglob

parse_config_to_array() {
    # ★★★ 핵심: 첫 번째 인자로 받은 '이름'으로 연관 배열을 참조
    local -n target_array="$1"
    # 함수를 여러 번 호출할 경우를 대비해 배열을 깨끗하게 초기화
    target_array=()

    while IFS='=' read -r key value; do
        # 1. 키/값 앞뒤 공백 제거 (trim)
        key="${key##*( )}"
        key="${key%%*( )}"
        value="${value##*( )}"
        value="${value%%*( )}"
        
        # 2. 값의 따옴표 제거 (예: "value" -> value)
        value="${value#\"}"; value="${value%\"}"
        
        # 키가 비어있지 않은 경우에만 배열에 추가 (안정성 강화)
        if [[ -n "$key" ]]; then
            target_array["${key}"]="${value}"
        fi
    done

    # 분석된 결과 로깅 (SC2034 대응)
    if [[ ${#target_array[@]} -gt 0 ]]; then
        log_info "Parsed keys into '$1': ${!target_array[*]}"
    fi
}

# ------------------------------------------------------------------------------
# @description 설정 파일에서 특정 섹션의 모든 키 목록을 반환. (awk 버전)
# @param $1 section (섹션명)
# @param $2 config_file (설정 파일 경로)
# ------------------------------------------------------------------------------
get_config_keys() {
    local section="$1"
    local config_file="$2"

    awk -v section="[${section}]" '
        # 현재 줄이 대상 섹션 헤더이면, in_section 플래그를 1로 설정하고 다음 줄로 넘어감
        $0 == section { in_section=1; next }
        # 다른 섹션 헤더를 만나면 플래그를 0으로 초기화
        /^\s*\[/ { in_section=0 }
        # 대상 섹션 내부이고, 주석이 아니며, "="를 포함한 라인이면
        in_section && !/^\s*#/ && /=/ {
            # "="를 기준으로 첫 번째 필드를 키로 간주
            key=$0
            sub(/=.*/, "", key)
            # 키의 앞뒤 공백 제거
            gsub(/^[ \t]+|[ \t]+$/, "", key)
            print key
        }
    ' "${config_file}"
}

# ------------------------------------------------------------------------------
# @description 설정 파일의 특정 섹션에서 모든 키-값 쌍을 반환. (awk 버전)
# @param $1 config_file (설정 파일 경로)
# @param $2 section (섹션명)
# ------------------------------------------------------------------------------
get_config_section() {
    local config_file="$1"
    local section="$2"

    awk -v section="[${section}]" '
        BEGIN { FS="=" }
        $0 == section { in_section=1; next }
        /^\s*\[/ { in_section=0; next }
        in_section && !/^\s*#/ && NF > 1 {
            key = $1
            gsub(/^[ \t]+|[ \t]+$/, "", key)
            
            value = $0
            sub(/^[^=]*=/, "", value)
            sub(/\s*#.*/, "", value)
            gsub(/^[ \t]+|[ \t]+$/, "", value)
            
            print key "=" value
        }
    ' "${config_file}"
}


##
# @description 설정 파일에 지정된 섹션이 존재하지 않으면 파일 끝에 추가합니다.
# @param $1 config_file (설정 파일 경로)
# @param $2 section (추가할 섹션명)
# @return 0: 이미 존재하거나 성공적으로 추가됨, 1: 실패
#
add_config_section() {
    local config_file="$1"
    local section_name="$2"

    # grep -q: 조용한 모드, -x: 라인 전체 일치, -F: 고정 문자열로 검색
    if ! grep -q -Fx "[${section_name}]" "${config_file}"; then
        echo -e "\n[${section_name}]" >> "${config_file}"
    fi
}

##
# @description 설정 파일에서 특정 접두사를 가진 모든 섹션 이름을 찾아 목록으로 반환합니다.
# @param $1 config_file (설정 파일 경로)
# @param $2 profile_prefix (찾을 프로필 섹션의 접두사, 예: "STORAGE_PROFILE_")
# @return stdout 섹션 이름 목록
#
get_profile_list() {
    local config_file="$1"
    local profile_prefix="$2"

    if [[ ! -f "$config_file" ]]; then return 1; fi

    awk -v prefix="${profile_prefix}" '
        # 1단계: 정리된 라인이 섹션 헤더 형식인지 확인합니다. (예: "[...]")
        /^\s*\[.*\]\s*$/ {
            # 2단계: 대괄호 안의 섹션 이름만 추출하고 다시 한번 정리합니다.
            section_name = $0;
            gsub(/^\[|\]$/, "", section_name); # 앞/뒤 대괄호만 제거
            gsub(/^[ \t]+|[ \t]+$/, "", section_name); # 대괄호 안쪽 공백도 제거

            # 3단계: 정리된 섹션 이름이 원하는 접두사로 시작하는지 확인합니다.
            # index(string, substring) 함수는 substring이 시작되는 위치를 반환합니다.
            # 시작 위치가 1이면, 해당 문자열로 시작한다는 의미입니다.
            if (index(section_name, prefix) == 1) {
                print section_name;
            }
        }
    ' "${config_file}"
}


# ------------------------------------------------------------------------------
# @description 설정 파일의 특정 섹션에서 키에 해당하는 값을 추출. (awk 버전)
# @param $1 (설정 파일 경로)
# @param $2 section (섹션명)
# @param $3 key (키 이름)
# ------------------------------------------------------------------------------
get_config_value() {
    local config_file="$1"
    local section="$2"
    shift 2
    local keys_to_find=("$@")

    # 실제 설정 파일에서 값 조회
    awk -v section="[${section}]" -v keys="${keys_to_find[*]}" '
        BEGIN { FS="="; split(keys, search_keys, " ") }
        $0 == section { in_section=1; next }
        /^\s*\[/ { in_section=0 }
        in_section && !/^\s*#/ {
            current_key = $1
            gsub(/^[ \t]+|[ \t]+$/, "", current_key)
            for (i in search_keys) {
                if (current_key == search_keys[i]) {
                    value = $0
                    sub(/^[^=]*=/, "", value)
                    sub(/\s*#.*/, "", value)
                    gsub(/^[ \t]+|[ \t]+$/, "", value)
                    print value
                    break
                }
            }
        }
    ' "${config_file}"
}

# ------------------------------------------------------------------------------
# @description 설정 파일의 특정 섹션에서 값에 해당하는 키를 추출. (awk 버전)
# @param $1 config_file (설정 파일 경로)
# @param $2 section (섹션명)
# @param $3 검색 값
# ------------------------------------------------------------------------------
get_config_key() {
    local config_file="$1"
    local section="$2"
    local search_value="$3" # 단일 검색 값

    awk -v section="[${section}]" -v value="${search_value}" '
        # 필드 구분자를 "="로 설정합니다.
        BEGIN { FS="=" }

        # 현재 줄이 찾고 있는 섹션과 일치하면, in_section 플래그를 1로 설정하고 다음 줄로 넘어갑니다.
        $0 == section { in_section=1; next }

        # 다른 섹션이 시작되면, in_section 플래그를 0으로 설정합니다.
        /^\s*\[/ { in_section=0 }

        # 현재 섹션 내에 있고 주석(#)이 아닌 줄을 처리합니다.
        in_section && !/^\s*#/ {
            # 키(key) 처리: 첫 번째 필드($1)에서 앞뒤 공백을 제거합니다.
            current_key = $1
            gsub(/^[ \t]+|[ \t]+$/, "", current_key)

            # 값(value) 처리: "=" 이후의 모든 문자열을 가져옵니다.
            current_value = $0
            sub(/^[^=]*=/, "", current_value) # 키와 "=" 부분을 제거합니다.
            sub(/\s*#.*/, "", current_value)   # 주석을 제거합니다.
            gsub(/^[ \t]+|[ \t]+$/, "", current_value) # 값의 앞뒤 공백을 제거합니다.
            sub(/^"/, "", current_value); sub(/"$/, "", current_value) # 따옴표를 제거합니다.

            # 처리된 값이 찾고자 하는 값(value)과 일치하는지 확인하고, 일치하면 키를 출력합니다.
            if (current_value == value) {
                print current_key
            }
        }
    ' "${config_file}"
}

##
# @description 설정 파일의 특정 섹션 끝에 주석을 추가합니다.
# @param $1 config_file
# @param $2 section
# @param $3 comment
add_config_comment() {
    local config_file="$1"
    local section="$2"
    local comment="$3"
    
    if [[ -z "$comment" || "$comment" == "none" ]]; then return 0; fi
    
    # 섹션이 없으면 생성
    add_config_section "$config_file" "$section"
    
    local tmp_file; tmp_file=$(mktemp)
    
    # awk를 사용하여 섹션의 끝(다음 섹션 시작 전 또는 파일 끝)에 주석을 추가
    # 빈 줄(blank lines)을 버퍼링하여 주석이 빈 줄보다 먼저 오도록 처리
    awk -v section="[${section}]" -v comment="# ${comment}" '
        $0 == section { in_section=1; print; next }
        /^\s*\[/ { 
            if (in_section) { 
                print comment; 
                printf "%s", blank_buffer;
                blank_buffer = "";
                in_section = 0 
            } 
            print; 
            next 
        }
        in_section && /^\s*$/ {
            blank_buffer = blank_buffer $0 "\n";
            next
        }
        in_section {
            printf "%s", blank_buffer;
            blank_buffer = "";
            print;
            next
        }
        { print }
        END { 
            if (in_section) {
                print comment;
                printf "%s", blank_buffer;
            }
        }
    ' "$config_file" > "$tmp_file"
    
    commit_file_change "$tmp_file" "$config_file"
    rm -f "$tmp_file"
}

# ------------------------------------------------------------------------------
# @description 설정 파일의 특정 섹션에 키-값 쌍을 설정하거나 업데이트. (awk 버전)
# @param $1 config_file (설정 파일 경로)
# @param $2 section (섹션명)
# @param $3... (키-값 쌍, 예: key1 value1 key2 value2 ...)
# ------------------------------------------------------------------------------
set_config_value() {
    local config_file="$1"
    local section="$2"
    shift 2
    local pairs=("$@")

    # 1. 입력 유효성 검사: 인자 개수가 짝수인지 확인합니다 (키와 값의 쌍이므로).
    if (( ${#pairs[@]} % 2 != 0 )); then
        echo "[ERROR] The number of key-value pair arguments must be even." >&2
        return 1
    fi

    if [[ ! -f "${config_file}" || ! -r "${config_file}" ]]; then
        echo "[ERROR] Config file '${config_file}' does not exist or is not readable." >&2
        return 1
    fi

    local tmp_file; tmp_file=$(mktemp)
    trap 'rm -f "${tmp_file}"' EXIT ERR  # Cleanup on exit or error

    # 2. 키와 값을 구분자(|)로 연결한 문자열로 변환 (공백 포함 안전).
    local keys=""
    local values=""
    local sep="|"  # 구분자: 키/값에 |가 포함될 수 있으면 $'\x1F' 등으로 변경.
    for (( i=0; i<${#pairs[@]}; i+=2 )); do
        if [[ $i -gt 0 ]]; then
            keys+="${sep}"
            values+="${sep}"
        fi
        keys+="${pairs[i]}"
        values+="${pairs[i+1]}"
    done

    # 3. awk를 사용하여 설정 파일을 처리합니다.
    if ! awk -v section="[${section}]" -v keys="${keys}" -v values="${values}" -v sep="${sep}" '
    BEGIN {
        FS = "=";  
        OFS = "="; 
        in_section = 0;
        blank_buffer = "";

        split(keys, key_arr, sep);
        split(values, val_arr, sep);

        for (i in key_arr) {
            k = key_arr[i];
            v = val_arr[i];
            if (k in new_pairs) {
                print "[WARN] Duplicate key: " k > "/dev/stderr";
            }
            new_pairs[k] = v;
            found[k] = 0;  
        }
    }

    $0 == section {
        in_section = 1;
        print;
        next;
    }

    in_section == 1 {
        if (/^\s*\[/) {
            for (k in found) {
                if (found[k] == 0) {
                    print k, new_pairs[k];
                }
            }
            printf "%s", blank_buffer;
            blank_buffer = "";
            in_section = 0;
            print;
            next;
        }

        if (/^\s*$/) {
            blank_buffer = blank_buffer $0 "\n";
            next;
        }

        # 주석 라인 처리
        if (/^\s*[#;]/) {
            printf "%s", blank_buffer;
            blank_buffer = "";
            print;
            next;
        }

        current_key = $1;
        gsub(/^[ \t]+|[ \t]+$/, "", current_key);  
        
        current_value = substr($0, index($0, "=") + 1);
        gsub(/^[ \t]+|[ \t]+$/, "", current_value);  
        
        printf "%s", blank_buffer;
        blank_buffer = "";

        if (current_key in new_pairs) {
            print current_key, new_pairs[current_key];
            found[current_key] = 1;
            next;
        } else {
            print current_key, current_value;
            next;
        }
    }

    { print; }

    END {
        if (in_section == 1) {
            for (k in found) {
                if (found[k] == 0) {
                    print k, new_pairs[k];
                }
            }
            printf "%s", blank_buffer;
        }
    }' "${config_file}" > "${tmp_file}"; then
        echo "[ERROR] awk processing failed." >&2
        return 1
    fi
    
    # 4. 임시 파일을 원본으로 커밋.
    commit_file_change "${tmp_file}" "${config_file}"
}

# ------------------------------------------------------------------------------
# @description 설정 파일에서 특정 섹션 전체를 삭제. (awk 버전)
# @param $1 config_file (설정 파일 경로)
# @param $2 section (삭제할 섹션명)
# ------------------------------------------------------------------------------
delete_config_section() {
    # 인자를 지역 변수에 저장.
    local config_file="$1"
    local section_name="$2"
    
    # 안전한 파일 수정을 위한 임시 파일 생성.
    local tmp_file; tmp_file=$(mktemp)

    # awk로 대상 섹션을 제외한 내용만 임시 파일에 저장.
    awk -v section="[${section_name}]" '
        # 상태 변수 초기화.
        BEGIN { in_section_to_delete = 0 }

        # 대상 섹션 시작 감지 (대소문자 무시, 주석/공백 허용).
        tolower($0) ~ tolower("^\\s*#*\\s*" section) {
            in_section_to_delete = 1
            # 대상 섹션 시작 줄은 건너뛰기.
            next
        }

        # 삭제 섹션 내부에서 새로운 섹션 헤더를 만나면 삭제 섹션 종료.
        in_section_to_delete && /^\s*#*\s*\[.*\]/ {
            in_section_to_delete = 0
        }

        # 삭제할 섹션이 아닐 경우에만 줄 출력.
        !in_section_to_delete { print }

    ' "${config_file}" > "${tmp_file}"

    commit_file_change "${tmp_file}" "${config_file}"
}

# ------------------------------------------------------------------------------
# @description 설정 파일의 특정 섹션 전체를 주석 처리/해제. (awk 버전)
# @param $1 config_file (설정 파일 경로)
# @param $2 section (대상 섹션명)
# @param $3 state ("enable" 또는 "disable")
# ------------------------------------------------------------------------------
toggle_config_section() {
    # 인자를 지역 변수에 저장.
    local config_file="$1"
    local section_name="$2"
    local state="$3"

    # 안전한 파일 수정을 위한 임시 파일 생성.
    local tmp_file; tmp_file=$(mktemp)

    awk -v section="[${section_name}]" -v state="${state}" '
        # 상태 변수 초기화.
        BEGIN { in_section = 0 }

        # 대상 섹션 시작 감지.
        tolower($0) ~ tolower("^\\s*#*\\s*" section) { in_section = 1 }
        
        # 다른 섹션 시작 감지 시, 대상 섹션 종료.
        in_section && /^\s*#*\s*\[.*\]/ && tolower($0) !~ tolower(section) {
            in_section = 0
        }

        # 대상 섹션 내부일 경우, 상태에 따라 라인 수정.
        if (in_section) {
            if (state == "enable") {
                # 활성화: 라인 시작의 주석과 공백 제거.
                sub(/^\s*#\s*/, "")
            } else { # state == "disable"
                # 비활성화: 주석이 없는 경우에만 추가.
                if ($0 !~ /^\s*#/) {
                    $0 = "# " $0
                }
            }
        }
        
        # 수정되거나 원본 라인 출력.
        { print }

    ' "${config_file}" > "${tmp_file}"

    commit_file_change "${tmp_file}" "${config_file}"
}
