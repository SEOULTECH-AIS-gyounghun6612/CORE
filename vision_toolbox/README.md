# Vision Toolbox

비전 및 3D 공간 연산 기능들을 체계적으로 구성한 Python 기반 도구 모음입니다. 2D 픽셀 영역부터 3D 전역 좌표계까지의 연산을 계층화하여, 비전 알고리즘 개발 및 연구의 효율성을 극대화하는 것을 목표로 합니다.

## Core Architecture

본 라이브러리는 좌표계의 위계(Hierarchy)에 따라 패키지를 분리하여 설계되었습니다.

- **`images/` (Pixel Space)**: 이미지 파일 I/O 및 2D 픽셀 평면에서의 원자적 연산(Filter, Morphology, 2D Geometry).
- **`camera/` (Camera Space)**: 카메라 로컬 3D 공간 연산. 광학 모델(Intrinsics), 투영(Projection), 시점 변환(Extrinsics).
- **`world/` (World Space)**: 전역 3D 공간 연산. 좌표계 통합(Registration), 정렬(Alignment), 스케일 조정.
- **`pipeline/` (Framework)**: 각 계층의 연산들을 `Step` 단위로 추상화하여 순차적/반복적 흐름으로 제어하는 범용 프레임워크.

## Update Plan

### Infrastructure & Framework

- [x] `pipeline` 엔진 최상위 승격 및 범용화 (`BaseState` 도입)
- [x] 이미지 처리 로직 `images/functional` 및 `images/io.py`로 재배치
- [ ] 패키지 간 의존성 규칙 정립 (`world` -> `camera` -> `images`)

### Domain Modules

- [ ] **Camera Module**: `Asset` 의존성을 제거한 새로운 카메라 모델 및 광학 연산 구현
- [ ] **World Module**: 전역 좌표계 변환 및 다중 시점 데이터 통합 로직 구축
- [ ] **IO Integration**: `utils/file.py` 로직을 도메인별 `io` 모듈로 통합 및 최적화

## Installation

### Requirements

- [python_toolbox](https://github.com/SEOULTECH-AIS-gyounghun6612/CORE/tree/python_toolbox)
- numpy
- scipy
- opencv-python

### Install via pip

```bash
pip install git+https://github.com/SEOULTECH-AIS-gyounghun6612/CORE.git@vision_toolbox
```
