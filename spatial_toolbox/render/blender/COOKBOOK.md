# Blender 렌더러 레시피

[← Render COOKBOOK](../COOKBOOK.md)

`render/blender`는 `scene.Controller`를 USD로 내보낸 뒤 Blender에 다시 적재해서 RGB, Depth, Normal, Segmentation을 뽑아내는 headless 백엔드임.

---

## 레시피 1: 기본 렌더링

```python
from spatial_toolbox.render import (
    Blender_Renderer,
    DEPTH,
    RGB,
    Render_Request,
)

renderer = Blender_Renderer()
request = Render_Request(channels=[RGB, DEPTH])

with renderer:
    results = renderer.Render(scene, ["main_camera"], request)

rgb = results["main_camera"].Get_image(RGB)
depth = results["main_camera"].Get_image(DEPTH)
```

이 백엔드는 런타임에 `bpy` import가 가능해야 하며, 내부적으로 `Blender_Session`이 첫 접근 시 `bpy`를 lazy import함.

---

## 레시피 2: 내부 파이프라인 이해

한 번의 `Render()` 호출은 대략 아래 순서로 동작함.

1. `scene.Controller`를 USD 파일로 export
2. Blender factory scene으로 초기화
3. USD import
4. 요청된 카메라별로 Blender camera 설정
5. 필요한 render pass와 compositor output 구성
6. 채널 이미지를 임시 디렉터리로 저장
7. 파일을 다시 numpy 배열로 로드

즉 Blender 백엔드는 직접 scene graph를 옮겨 그리는 방식이 아니라, `scene/file/usd.py`를 브리지로 사용하는 교환형 렌더러임.

---

## 레시피 3: 채널별 결과와 metadata

```python
from spatial_toolbox.render import NORMAL, SEGMENTATION

normal = results["main_camera"].Get_image(NORMAL)
seg = results["main_camera"].Get_image(SEGMENTATION)
meta = results["main_camera"].Get_metadata(SEGMENTATION)
```

채널별 기본 형식은 다음과 같음.

- `rgb`: `(H, W, 3) uint8`
- `depth`: `(H, W) float32`
- `normal`: `(H, W, 3) uint8`
- `segmentation`: `(H, W) int32`

Segmentation metadata에는 아래 값이 들어감.

- `id_map`: `pass_index -> scene node`
- `label_map`: `pass_index -> node.label`

---

## 레시피 4: 카메라 규약

Blender 카메라는 `scene.node.type.camera.Camera`의 `intrinsic`과 `world_matrix`에서 직접 구성됨.

- `fx`, `fy`, `cx`, `cy`는 Blender 카메라 lens / shift로 변환됨
- `near_clip`, `far_clip`은 `scene.unit_length`를 반영해 Blender clip range로 변환됨
- 출력 해상도는 camera intrinsic의 `width`, `height`를 그대로 사용함

따라서 여러 카메라를 렌더할 때는 렌더러 생성 시 해상도를 고정할 필요가 없고, 카메라별 intrinsic이 실제 출력 크기를 결정함.
