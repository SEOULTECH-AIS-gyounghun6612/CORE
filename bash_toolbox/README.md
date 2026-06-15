# CORE: Universal Shell Script Toolbox

## 개요 (Overview)

* Purpose: 리눅스 서버 관리 및 자동화를 위한 범용 쉘 스크립트 라이브러리.
* Design: OS 비종속(Agnostic), 계층형 모듈 설계, 환경 부트스트랩 지원.

---

## 모듈 구성 (Modules)

### 1. `base/` (시스템 기초 계층)
* `init.sh`: 전역 시스템 컨텍스트(권한, 사용자), 디렉토리 구조 초기화 및 베이스 모듈 로더.
* `common.sh`: 프로젝트 전반의 공통 유틸리티 및 표준 로깅 함수 (`log_info`, `log_success` 등).
* `profile_engine.sh`: awk 기반의 INI 형식 프로필 및 설정 분석 엔진.

### 2. `package/` (패키지 관리 계층)
* `init.sh`: 시스템 패키지 관리자(`apt`, `yum`, `pacman`) 감지 및 명령어 매핑.
* `manager.sh`: OS 통합 패키지 설치/제거/확인 인터페이스.
* `state.sh`: `G_STATE_FILE`을 활용한 패키지 설치 상태 동기화 및 이력 관리.
* `apt.sh`: Debian/Ubuntu 계열 전용 유틸리티 (GPG 키, 저장소 관리).

### 3. `ui/` (사용자 인터페이스 계층)
* `init.sh`: `dialog` 기반 TUI 환경 초기화 및 하위 위젯 로드.
* `info.sh` / `select.sh` / `input.sh` / `form.sh`: 목적별 TUI 위젯 (메시지박스, 메뉴, 입력폼 등).

### 4. `core.sh` (통합 로더)
* Role: 라이브러리 전체의 진입점. 각 계층의 `init.sh`를 순차적으로 실행하여 런타임 환경을 구축.

---

## 사용법 (Usage)

### 1. 라이브러리 로드 및 초기화 (Bootstrap)

```bash
#!/bin/bash

# 1. 로더 소스
source "./script/core/core.sh"

# 2. 통합 초기화 호출 (루트 경로, 설정 파일, 템플릿 지정)
# 내부적으로 시스템 권한, 로그/상태 디렉토리, 패키지 및 UI 환경을 모두 준비합니다.
load_core_libraries "$(pwd)" "./conf/config.conf" "./template/config.conf"
```

### 2. 주요 기능 활용 예시

표준 로깅
```bash
log_info "작업을 시작합니다."
log_success "설정이 완료되었습니다."
log_error "오류가 발생했습니다."
```

패키지 상태 동기화
```bash
# 시스템에 패키지를 설치하고 설치 이력을 state/system.state에 기록
sync_package "TOOLS" "System Utilities" "vim" "htop" "git"
```

TUI 대화창
```bash
if ui_confirm "진행하시겠습니까?" "확인"; then
    log_info "사용자가 확인을 선택했습니다."
fi
```
