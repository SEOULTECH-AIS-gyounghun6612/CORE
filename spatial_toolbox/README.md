# spatial_toolbox

`spatial_toolbox`는 3D scene graph 관리, 멀티패스 렌더링, 반복 캡처 자동화를 위한 Python 패키지임.

핵심 구성은 세 계층으로 나뉨.

- `scene/`: 장면 데이터 모델, 에셋 캐시, JSON/USD import-export
- `render/`: OpenGL / Blender 기반 이미지 렌더링
- `simulation/`: 랜덤화, 반복 샘플링, 결과 저장 자동화

```text
scene/  ──►  render/
    └──────► simulation/
```

## 설계 경계

- `scene/`은 순수 데이터 레이어이며 `render/`나 UI에 의존하지 않음
- `render/`는 `scene.Controller`를 입력으로 받아 채널별 이미지를 생성함
- `simulation/`은 `scene`과 `render`를 조합해 데이터셋 캡처를 수행함
- JSON / USD는 `scene/file`이 담당하고, 원시 geometry 로드는 `scene/asset/file`이 담당함

## 패키지 구조

```text
spatial_toolbox/
├── scene/
│   ├── README.md
│   ├── COOKBOOK.md
│   ├── asset/             # geometry asset, cache, similarity
│   ├── node/              # scene graph node types, traversal, transform
│   ├── file/              # scene JSON / USD import-export
│   └── stage.py           # Controller
│
├── render/
│   ├── README.md
│   ├── COOKBOOK.md
│   ├── core/              # renderer contract, channels, camera resolution
│   ├── openGL/            # OpenGL backend
│   └── blender/           # Blender backend
│
└── simulation/
    ├── README.md
    ├── COOKBOOK.md
    ├── core/              # config, exporter, engine contract
    └── blender/           # Blender capture engine
```

## 문서

| 문서 | 역할 |
|---|---|
| [README.md](./README.md) | 프로젝트 개요, 구조, 설치 |
| [COOKBOOK.md](./COOKBOOK.md) | 빠른 시작과 상위 사용 흐름 |
| [ROADMAP.md](./ROADMAP.md) | 리팩토링 진행 현황과 TODO |
| [scene/README.md](./spatial_toolbox/scene/README.md) | scene 아키텍처 개요 |
| [scene/COOKBOOK.md](./spatial_toolbox/scene/COOKBOOK.md) | scene 사용 예제 |
| [render/README.md](./spatial_toolbox/render/README.md) | render 아키텍처 개요 |
| [render/COOKBOOK.md](./spatial_toolbox/render/COOKBOOK.md) | render 사용 예제 |
| [simulation/README.md](./spatial_toolbox/simulation/README.md) | simulation 아키텍처 개요 |
| [simulation/COOKBOOK.md](./spatial_toolbox/simulation/COOKBOOK.md) | simulation 사용 예제 |

## 설치

Python `3.11` 환경을 기준으로 함.

```bash
pip install -e .
```

주요 런타임 의존성은 `numpy`, `trimesh`, `PyOpenGL`, `scipy`, `Pillow`, `usd-core`, `bpy`임. Blender 백엔드를 쓰지 않으면 `bpy`가 필요한 경로를 호출하지 않는 한 lazy import 상태로 유지됨.
