# simulation

`simulation/`은 `scene/`과 `render/`를 조합해 데이터셋 캡처 파이프라인을 구성하는 상위 자동화 계층임.

`scene/`이 장면 상태를 관리하고 `render/`가 단일 프레임 렌더를 담당한다면, `simulation/`은 객체/카메라 랜덤화, 다중 샘플 반복, 결과 저장을 담당함.

```text
scene/  ──►  render/
    └──────► simulation/
```

## 역할

- `core/config.py`는 백엔드 무관 캡처 설정과 랜덤화 범위를 정의함
- `core/engine.py`는 캡처 엔진 추상 계약을 정의함
- `core/exporter.py`는 렌더 결과를 이미지·배열·메타데이터 파일로 저장함
- `blender/engine.py`는 Blender 렌더러 기반의 실제 데이터셋 캡처 엔진을 제공함

## 구조

```text
simulation/
├── __init__.py             # public API re-export
├── README.md
├── COOKBOOK.md             # 공통 설정, 출력 규약, 사용 패턴
│
├── core/
│   ├── config.py           # Sim_Config, Randomize_Range, sampling helper
│   ├── engine.py           # Base_Capture_Engine
│   └── exporter.py         # Result_Exporter
│
└── blender/
    ├── __init__.py
    ├── engine.py           # Blender_Capture_Engine
    └── COOKBOOK.md
```

## 문서

- 공통 설정과 출력 형식: [COOKBOOK.md](./COOKBOOK.md)
- Blender 백엔드 캡처 엔진: [blender/COOKBOOK.md](./blender/COOKBOOK.md)
