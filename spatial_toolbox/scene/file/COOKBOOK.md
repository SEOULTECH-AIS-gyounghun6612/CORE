# File 모듈 레시피

[← Scene 모듈 COOKBOOK으로 돌아가기](../COOKBOOK.md)

`scene/file` 모듈은 `Controller`가 보유한 씬 상태를 JSON 또는 USD로 저장·복원하는 장면 persistence 계층임. 원시 3D geometry 로딩은 이 모듈이 아니라 `scene/asset/file`이 담당함.

---

## 레시피 1: 확장자 기반 장면 저장

```python
from spatial_toolbox.scene import Controller

ctrl = Controller()
keys = ctrl.Register_from_file("robot.obj")
ctrl.Add_node(ctrl.Build_node_from_cache(keys[0], label="robot"))

ctrl.Export("scene.json")
ctrl.Export("scene.usda")
```

`Controller.Export()`는 내부적으로 `scene.file.Export_to()`를 호출하고, 확장자에 따라 아래 exporter를 선택함.

- `.json`
- `.usd`
- `.usda`
- `.usdc`

---

## 레시피 2: 장면 복원

```python
from spatial_toolbox.scene import Controller

ctrl = Controller()
ctrl.Import("scene.json")

render_queue = ctrl.Get_render_queue()
```

`Controller.Import()`는 파일에서 `Scene_State(root, unit_length)`를 복원한 뒤 현재 씬을 교체함.

---

## 레시피 3: JSON 포맷 직접 사용

```python
from spatial_toolbox.scene.file import Export_to_json, Import_from_json

Export_to_json("scene.json", ctrl.root, ctrl.unit_length)
scene_state = Import_from_json("scene.json")
```

JSON 경로는 `Data_Schema.Serialize()` 결과를 기반으로 트리를 직렬화하며, import 시 `NODE_REGISTRY`로 각 `prim_type`의 노드 클래스를 다시 구성함.

---

## 레시피 4: USD 포맷 특성

```python
from spatial_toolbox.scene.file import Export_to_usd, Import_from_usd

Export_to_usd("scene.usda", ctrl.root, ctrl.unit_length)
scene_state = Import_from_usd("scene.usda")
```

USD 경로에서는 다음 규약을 사용함.

- 메시 geometry는 `/_Prototypes` 아래에 저장됨
- 씬 노드는 prototype에 internal reference를 걸고 `instanceable=True`로 표시됨
- 원본 `source_key`는 custom data(`focusSourceKey`, `focusProtoRef`)로 보존됨
- import 시 prototype geometry를 `ASSET_CACHE`에 다시 등록한 뒤 씬 계층을 복원함
