# Provisioning Runtime Image & Native conTainer

Docker 컨테이너 환경 최적화 및 구축 자동화 프로젝트

## 목표

- 시스템(제어, 비전 검출, 사물 인식, UI 등)의 안정적/효율적 운영 환경 구축
- 프로젝트별 컨테이너 환경 모듈화 및 신속한 배포
- 순차적 레이어드 빌드(Sequential Layered Build) 시스템을 통한 빌드 시간 단축 및 재사용성 극대화

## 시스템 구조 (System Architecture)

### 1. 이미지 관리 (`docker/images/`)

- 계층 구조: `OS-CUDA 버전 / 기술 스택.mk` 파일 형태로 관리
- 파이프라인 빌드: 단일 통 이미지가 아닌, 기술 스택별로 이미지를 순차적으로 빌드하여 적층
    - Step 0: CuDNN Update (Optional)
    - Step 1: ROS2 (Middleware Layer)
    - Step 2: OpenCV (Vision Library Layer)
    - Step 3: Conda (Python Environment Layer - Optional)
    - Step 4: System Packages (Final Customization Layer)
- 효율성: 상위 레이어(예: ROS2)가 이미 존재하면 재사용하여 OpenCV만 추가 빌드하는 방식

### 2. 컨테이너 관리 (`docker/containers/`)

- 프로젝트 단위: 프로젝트별, 모듈별 실행 환경 정의
- 이미지 참조: 빌드된 최종 단계의 이미지를 참조하여 독립적인 컨테이너 인스턴스 실행

### 3. 유틸리티 (`docker/utils/`)

- `common_image.mk`: 전체 빌드 파이프라인을 조율(Orchestrate)하는 메인 로직
- `step_*.mk`: 각 단계별(ROS2, OpenCV, Conda 등) 독립적인 빌드 로직
- `common_container.mk`: 컨테이너 실행, 정지, 쉘 접속 등 공통 룰 정의

## 사용 방법 (Usage)

루트 디렉토리의 `Makefile`을 통해 모든 작업을 수행할 수 있습니다.

### 1. 이미지 빌드 (Image Build)

설정된 기술 스택을 순서대로 확인하며 필요한 레이어만 빌드합니다.

```bash
# 기본 스펙 빌드 (파이프라인 자동 실행)
make build

# 특정 스펙 빌드
make build SPEC=ubuntu22.04-cuda12.4.1/cv4.10.0-ros2.humble

# 이미지 강제 재빌드 (전체 파이프라인 재실행)
make force-build SPEC=ubuntu22.04-cuda12.4.1/cv4.10.0-ros2.humble
```

빌드 프로세스 예시:
1. `nvidia/cuda` 베이스 이미지 확인
2. `...-ros-humble` 이미지 확인 -> 없으면 빌드, 있으면 Skip
3. `...-ros-humble-cv4.10.0` 이미지 확인 -> (위 이미지를 FROM으로) 없으면 빌드
4. 최종 이미지 태그 생성 완료

### 2. 컨테이너 실행 (Container Run)

```bash
# 기본 컨테이너 실행
make run

# 특정 프로젝트 컨테이너 실행
make run CON_SPEC=for_SL/with_cv
```

### 3. 관리 커맨드

```bash
# 컨테이너 쉘 접속
make shell CON_SPEC=for_SL/with_cv

# 컨테이너 정지 및 제거
make stop CON_SPEC=for_SL/with_cv
make clean CON_SPEC=for_SL/with_cv
```

## 주요 특징

- Layered Build System: 무거운 컴포넌트(ROS2, OpenCV)를 분리하여 빌드 시간을 획기적으로 단축하고 캐싱 효율을 높임
- Dependency Management: ASAP 스크립트와 연동하여 컨테이너 내부 의존성 자동 설치
- Artifact Injection: `build_pool`의 아티팩트를 스테이지별로 적절히 주입
- Auto-Recovery: 중간 단계 이미지가 없으면 자동으로 감지하여 하위 레이어부터 다시 빌드

## TODO

- [x] 이미지 및 컨테이너 설정 분리 및 모듈화
- [x] 기술 스택별 이미지 명명 규칙 자동화
- [x] 순차적 레이어드 빌드 시스템 구현 (Make Pipeline)
- [ ] 다양한 도메인별(제어, 비전, UI) 특화 스택 프로필 확충
- [ ] 다중 컨테이너 오케스트레이션 가이드 추가
