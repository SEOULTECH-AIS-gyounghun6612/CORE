# Render 모듈 레시피

[← 최상위 COOKBOOK](../../COOKBOOK.md)

`render/`는 `scene.Controller`가 구성한 씬을 실제 이미지로 변환하는 백엔드 계층임. 공통 렌더 계약은 `render/core`에 있고, 현재 구현 백엔드는 `openGL`과 `blender` 두 가지임.

---

## 하위 모듈

| 모듈 | 역할 | 문서 |
|---|---|---|
| **core** | 채널 이름, 렌더 요청/결과 컨테이너, 카메라 해석 유틸, Renderer 추상 계약 | 이 문서 |
| **openGL** | 즉시형 OpenGL 기반 멀티패스 렌더러 | [openGL/COOKBOOK](./openGL/COOKBOOK.md) |
| **blender** | USD 브리지 기반 Blender headless 렌더러 | [blender/COOKBOOK](./blender/COOKBOOK.md) |

---

## 공통 렌더 계약

```python
from spatial_toolbox.render import (
    RGB,
    DEPTH,
    SEGMENTATION,
    Render_Request,
)

request = Render_Request(
    channels=[RGB, DEPTH, SEGMENTATION],
)
```

- `Render_Request.channels`는 실행할 렌더 채널 목록임.
- `Renderer.Render(...)`의 반환값은 `dict[camera_label, Render_Result]` 구조임.
- `Render_Result.images[channel]`에 픽셀 배열이, `Render_Result.metadata[channel]`에 부가 정보가 들어감.

---

## 레시피 1: 카메라 해석 규약

```python
from spatial_toolbox.render import Resolve_cameras

cameras = Resolve_cameras(scene, ["main_camera"])
```

`camera_labels`는 두 방식으로 해석됨.

- label이 `Camera` 노드를 직접 가리키면 그 카메라를 사용함
- label이 일반 노드를 가리키면 그 하위 트리에서 발견되는 모든 `Camera`를 순서대로 사용함

존재하지 않는 label이 들어오면 `KeyError`, 빈 리스트면 `ValueError`가 발생함.

---

## 레시피 2: 공통 렌더 호출 패턴

```python
from spatial_toolbox.render import OpenGL_Renderer, RGB, DEPTH, Render_Request

renderer = OpenGL_Renderer(1024, 768)
request = Render_Request(channels=[RGB, DEPTH])

with renderer:
    results = renderer.Render(
        scene=scene,
        camera_labels=["main_camera"],
        request=request,
    )

rgb = results["main_camera"].Get_image(RGB)
depth = results["main_camera"].Get_image(DEPTH)
```

모든 백엔드는 동일한 `Render(scene, camera_labels, request)` 시그니처를 따름.

---

## 채널 규약

| 채널 | key | 기본 결과 타입 |
|---|---|---|
| RGB | `"rgb"` | `(H, W, 3) uint8` |
| Depth | `"depth"` | `(H, W) float32` |
| Normal | `"normal"` | `(H, W, 3) uint8` |
| Segmentation | `"segmentation"` | OpenGL: `(H, W, 3) uint8`, Blender: `(H, W) int32` |

세부 포맷과 metadata는 백엔드별 문서를 참조.
