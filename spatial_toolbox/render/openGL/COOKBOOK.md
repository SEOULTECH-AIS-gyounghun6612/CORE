# OpenGL 렌더러 레시피

[← Render COOKBOOK](../COOKBOOK.md)

`render/openGL`은 `scene.Controller`의 render queue를 즉시형 OpenGL로 그려 RGB, Depth, Normal, Segmentation 채널을 반환하는 백엔드임.

---

## 레시피 1: 기본 렌더링

```python
from spatial_toolbox.render import (
    DEPTH,
    RGB,
    OpenGL_Renderer,
    Render_Request,
)

renderer = OpenGL_Renderer(1024, 768)
request = Render_Request(channels=[RGB, DEPTH])

with renderer:
    results = renderer.Render(scene, ["main_camera"], request)

rgb = results["main_camera"].Get_image(RGB)
depth = results["main_camera"].Get_image(DEPTH)
```

- `OpenGL_Renderer(w, h)`는 고정 해상도 렌더러 인스턴스를 만듦.
- `Setup()`에서 GL context를 준비하고, `Teardown()`에서 VBO 캐시를 포함한 내부 리소스를 정리함.

---

## 레시피 2: 지원 채널과 결과 형식

```python
from spatial_toolbox.render import (
    DEPTH,
    NORMAL,
    RGB,
    SEGMENTATION,
    Render_Request,
)

request = Render_Request(channels=[RGB, DEPTH, NORMAL, SEGMENTATION])
```

각 채널은 다음처럼 반환됨.

- `rgb`: `(H, W, 3) uint8`
- `depth`: `(H, W) float32`, near/far clip을 사용해 선형화된 거리
- `normal`: `(H, W, 3) uint8`, `[-1, 1]` normal을 `[0, 255]`로 인코딩
- `segmentation`: `(H, W, 3) uint8`, object id를 RGB로 인코딩

---

## 레시피 3: Segmentation metadata 사용

```python
from spatial_toolbox.render import SEGMENTATION

seg = results["main_camera"].Get_image(SEGMENTATION)
meta = results["main_camera"].Get_metadata(SEGMENTATION)
id_map = meta["id_map"]
```

`id_map`은 `(r, g, b)` 색상 키를 실제 `scene` 노드 객체에 매핑한 dict임. segmentation 패스 실행 전마다 renderer가 내부 id 상태를 초기화하므로, 한 번의 렌더 결과 안에서는 일관된 매핑을 보장함.

---

## 레시피 4: 조명 설정 변경

```python
class Lighting_Config:
    light_position = [5.0, 8.0, 10.0, 1.0]
    light_diffuse = [1.0, 1.0, 1.0, 1.0]
    light_ambient = [0.2, 0.2, 0.2, 1.0]
    light_specular = [0.1, 0.1, 0.1, 1.0]
    material_specular = [0.15, 0.15, 0.15, 1.0]
    material_shininess = 8.0

renderer = OpenGL_Renderer(1024, 768)
renderer.Configure_lighting(Lighting_Config)
```

현재 조명 상태는 RGB 패스에서 사용되고, Normal / Depth / Segmentation 패스는 비조명 모드로 렌더됨.

---

## 내부 동작 요약

- 카메라 투영 행렬은 `Build_gl_projection(camera.intrinsic, scene.unit_length)`로 계산됨
- `scene.Get_render_queue()`에서 받은 `Mesh` 노드만 그려짐
- geometry는 `node.source_key`를 통해 `ASSET_CACHE`에서 조회됨
- 동일 geometry는 `source_key` 기준 VBO 캐시를 재사용함
- GL context는 가능하면 `EGL_Context`, 실패 시 `Embedded_Context`를 사용함
