# Blender Capture 레시피

[← Simulation COOKBOOK](../COOKBOOK.md)

`simulation/blender`는 `render.Blender_Renderer`를 사용해 대상 객체를 순회하며 다중 샘플 캡처를 수행하는 데이터셋 생성 엔진임.

---

## 레시피 1: 기본 캡처 실행

```python
from pathlib import Path

from spatial_toolbox.scene import Controller
from spatial_toolbox.simulation import Blender_Capture_Engine, Sim_Config

scene = Controller()
scene.Import("scene.json")

engine = Blender_Capture_Engine()
config = Sim_Config(
    target_label="target",
    camera_labels=["main_camera"],
    num_samples=8,
    output_layout="per_object",
    seed=1234,
)

engine.Capture(
    scene=scene,
    config=config,
    output_dir=Path("captures"),
)
```

기본 채널은 `rgb`, `depth`, `normal`, `segmentation` 네 가지임.

---

## 레시피 2: 특정 채널만 저장하기

```python
from spatial_toolbox.render import RGB, DEPTH
from spatial_toolbox.simulation import Blender_Capture_Engine

engine = Blender_Capture_Engine(channels=[RGB, DEPTH])
```

엔진 생성 시 채널 목록을 넘기면 내부 `Render_Request`가 그 채널들로만 구성됨.

---

## 레시피 3: 카메라별 랜덤화 재정의

```python
from spatial_toolbox.simulation import Randomize_Range, Sim_Config

config = Sim_Config(
    camera_labels=["cam_front", "cam_side"],
    cam=Randomize_Range(tx=[-0.01, 0.01], rz=[-2.0, 2.0]),
    cam_overrides={
        "cam_side": Randomize_Range(tx=[-0.03, 0.03], ry=[-5.0, 5.0]),
    },
)
```

`cam_overrides[camera_label]`가 있으면 해당 카메라는 공통 `cam` 대신 override 범위를 사용함.

---

## 레시피 4: 내부 동작 순서

한 번의 `Capture()` 호출은 아래 순서로 동작함.

1. `target_label`에 해당하는 Xform 그룹 탐색
2. `camera_labels`를 실제 `Camera` 노드 목록으로 해석
3. 대상 그룹 전체를 숨김
4. 자식 객체를 하나만 보이게 설정
5. 객체와 카메라에 랜덤 delta 적용
6. `Blender_Renderer.Render(...)` 호출
7. 카메라별 결과를 `Result_Exporter`로 저장
8. 원래 transform 복구 후 다음 샘플로 진행

캡처 도중 객체는 한 번에 하나만 visible 상태가 되므로, segmentation과 metadata가 객체 단위 샘플링에 맞춰 정리됨.

---

## 레시피 5: 진행률 콜백 받기

```python
def on_progress(current: int, total: int, message: str) -> None:
    print(f"[{current}/{total}] {message}")

engine.Capture(
    scene=scene,
    config=config,
    output_dir=Path("captures"),
    progress_callback=on_progress,
)
```

콜백 시그니처는 `(current_frame, total_frames, message)`임.
