# Simulation 모듈 레시피

[← 최상위 COOKBOOK](../../COOKBOOK.md)

`simulation/`은 장면 단건 렌더가 아니라, 대상 객체를 순회하며 랜덤화된 다중 프레임을 캡처하고 결과를 파일로 저장하는 데이터셋 생성 계층임.

---

## 하위 모듈

| 모듈 | 역할 | 문서 |
|---|---|---|
| **core** | 설정, 랜덤화, 출력 저장 규약 | 이 문서 |
| **blender** | Blender 렌더러 기반 캡처 엔진 | [blender/COOKBOOK](./blender/COOKBOOK.md) |

---

## 장면 규약

캡처 엔진은 현재 다음 씬 규약을 전제로 함.

- 대상 객체들은 `label == config.target_label` 이고 `prim_type == "Xform"`인 그룹의 직속 자식이어야 함
- 카메라는 `config.camera_labels`로 씬 트리에서 해석됨
- 캡처 중에는 대상 자식 노드를 하나씩 단독 가시화해서 프레임을 생성함

즉 `target` 그룹 아래에 여러 객체를 두고, 각 객체를 독립 샘플 단위로 순회하는 구조가 기본 패턴임.

---

## 레시피 1: Sim_Config 만들기

```python
from spatial_toolbox.simulation import Randomize_Range, Sim_Config

config = Sim_Config(
    scene_path="scene.json",
    target_label="target",
    camera_labels=["main_camera"],
    num_samples=10,
    output_layout="per_object",
    seed=1234,
    cam=Randomize_Range(tx=[-0.02, 0.02], ty=[-0.02, 0.02], rz=[-3.0, 3.0]),
    obj=Randomize_Range(tx=[-0.01, 0.01], ry=[-10.0, 10.0]),
)
```

- `cam`은 모든 카메라에 공통 적용되는 extrinsic 델타 범위임
- `cam_overrides`를 주면 특정 카메라별 범위를 재정의할 수 있음
- `obj`는 대상 객체의 `local_rigid`에 곱해지는 랜덤 델타 범위임

---

## 레시피 2: 랜덤화 규약 이해

```python
from spatial_toolbox.simulation import (
    Randomize_Range,
    Sample_delta_matrix,
    Sample_translation,
)

delta = Sample_delta_matrix(
    Randomize_Range(tx=[-0.1, 0.1], rz=[-15.0, 15.0])
)

tx, ty, tz = Sample_translation(
    Randomize_Range(tx=[-0.1, 0.1], ty=0.0, tz=0.0)
)
```

- `float` 값은 고정값으로 사용됨
- `[min, max]` 리스트는 균등분포 샘플링 범위로 해석됨
- `Sample_delta_matrix()`는 `Build_transform(...)` 기반 4x4 rigid delta를 만듦

---

## 레시피 3: 출력 파일 구조

`Result_Exporter`는 채널 결과를 파일 시스템에 아래 형식으로 저장함.

```text
<output_dir>/
├── rgb_000000.png
├── depth_000000.npy
├── normal_000000.png
├── segmentation_000000.npy
└── metadata_000000.json
```

저장 규약은 다음과 같음.

- `float32` 배열은 `.npy`
- 2D 정수 배열도 `.npy`
- 그 외 이미지형 배열은 `.png`

메타데이터 JSON에는 최소한 아래 정보가 들어감.

- 카메라 label / intrinsic / world extrinsic
- render 설정(`Sim_Config.Serialize()`)
- 채널별 metadata
- 엔진이 추가한 부가 정보(`object`, `sample` 등)

---

## 레시피 4: 백엔드 호출 패턴

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
    num_samples=4,
)

engine.Capture(
    scene=scene,
    config=config,
    output_dir=Path("outputs"),
)
```

세부 동작은 Blender 엔진 문서를 참조.
