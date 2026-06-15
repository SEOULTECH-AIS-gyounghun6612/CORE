# render

`render/`는 `scene.Controller`가 구성한 씬을 실제 이미지로 변환하는 렌더 백엔드 계층임.

`scene/`이 데이터 모델과 씬 상태를 담당한다면, `render/`는 그 상태를 받아 채널별 이미지(`rgb`, `depth`, `normal`, `segmentation`)로 평가하는 역할을 맡음.

```text
scene/  ──►  render/
```

## 역할

- `render/core`는 공통 렌더 계약을 정의함
- `render/openGL`은 즉시형 OpenGL 기반 렌더러를 제공함
- `render/blender`는 USD 브리지를 통한 Blender headless 렌더러를 제공함

## 구조

```text
render/
├── __init__.py             # public API re-export
├── README.md
├── COOKBOOK.md             # 공통 사용 패턴, 채널 규약
│
├── core/
│   ├── camera.py           # Resolve_cameras
│   ├── channel.py          # RGB/DEPTH/NORMAL/SEGMENTATION 상수
│   └── renderer.py         # Renderer, Render_Request, Render_Result
│
├── openGL/
│   ├── renderer.py         # OpenGL_Renderer
│   ├── context.py          # EGL / embedded context
│   ├── passes/             # rgb/depth/normal/segmentation pass
│   ├── draw/               # mesh draw dispatcher + VBO cache
│   └── utils/              # projection matrix, id color, VBO helper
│
└── blender/
    ├── renderer.py         # Blender_Renderer
    ├── session.py          # lazy bpy import
    └── scene/              # USD import, camera setup, compositor output
```

## 문서

- 공통 렌더 요청/결과와 카메라 해석 규약: [COOKBOOK.md](./COOKBOOK.md)
- OpenGL 백엔드 사용법: [openGL/COOKBOOK.md](./openGL/COOKBOOK.md)
- Blender 백엔드 사용법: [blender/COOKBOOK.md](./blender/COOKBOOK.md)
