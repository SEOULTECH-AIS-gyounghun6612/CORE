# Spatial Toolbox Cookbook

`spatial_toolbox`를 처음 사용할 때 필요한 상위 흐름만 빠르게 정리한 문서임. 세부 API는 각 하위 모듈 문서를 참조.

## 모듈 문서

| 모듈 | 설명 | 문서 |
|---|---|---|
| **Scene** | 씬 그래프, 에셋, 장면 import-export | [scene/COOKBOOK](./spatial_toolbox/scene/COOKBOOK.md) |
| **Render** | 공통 렌더 계약과 백엔드 사용법 | [render/COOKBOOK](./spatial_toolbox/render/COOKBOOK.md) |
| **Simulation** | 랜덤화, 반복 캡처, 결과 저장 | [simulation/COOKBOOK](./spatial_toolbox/simulation/COOKBOOK.md) |
| **OpenGL** | 즉시형 OpenGL 멀티패스 렌더링 | [render/openGL/COOKBOOK](./spatial_toolbox/render/openGL/COOKBOOK.md) |
| **Blender Render** | USD 브리지 기반 headless 렌더링 | [render/blender/COOKBOOK](./spatial_toolbox/render/blender/COOKBOOK.md) |
| **Blender Capture** | Blender 기반 데이터셋 캡처 엔진 | [simulation/blender/COOKBOOK](./spatial_toolbox/simulation/blender/COOKBOOK.md) |

## 빠른 시작 1: 씬 로드와 노드 구성

```python
from spatial_toolbox.scene import Controller
from spatial_toolbox.scene.node.type.camera import Camera

scene = Controller(unit_length=1.0)

keys = scene.Register_from_file("model.obj", unit_length=0.001)
node = scene.Build_node_from_cache(keys[0], label="model")
scene.Add_node(node)

camera = Camera(label="main_camera")
scene.Add_node(camera)
```

이 단계의 목적은 `scene.Controller` 안에 geometry를 참조하는 노드와 카메라를 배치하는 것임.

## 빠른 시작 2: 단일 프레임 렌더링

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

렌더 호출의 공통 입력은 항상 아래 세 가지임.

- `scene`: `scene.Controller`
- `camera_labels`: 렌더 대상 카메라 label 목록
- `request`: 채널 목록을 담은 `Render_Request`

## 빠른 시작 3: 반복 캡처 데이터셋 생성

```python
from pathlib import Path

from spatial_toolbox.simulation import Blender_Capture_Engine, Randomize_Range, Sim_Config

config = Sim_Config(
    target_label="target",
    camera_labels=["main_camera"],
    num_samples=4,
    seed=1234,
    obj=Randomize_Range(ry=[-10.0, 10.0]),
)

engine = Blender_Capture_Engine()
engine.Capture(
    scene=scene,
    config=config,
    output_dir=Path("captures"),
)
```

이 경로는 `target` 그룹의 직속 자식들을 하나씩 단독 표시하면서 RGB / Depth / Normal / Segmentation 결과와 메타데이터를 저장함.

## 언제 어떤 모듈을 보는가

- scene graph를 만들거나 장면을 저장/복원하려면 `scene/COOKBOOK.md`
- 단일 프레임 렌더 결과를 얻으려면 `render/COOKBOOK.md`
- OpenGL 백엔드 세부 규약이 필요하면 `render/openGL/COOKBOOK.md`
- Blender 렌더 백엔드 세부 규약이 필요하면 `render/blender/COOKBOOK.md`
- 반복 캡처와 출력 파일 규약이 필요하면 `simulation/COOKBOOK.md`
