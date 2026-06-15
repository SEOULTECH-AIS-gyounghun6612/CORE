# Asset 모듈 레시피

[← Scene 모듈 COOKBOOK으로 돌아가기](../COOKBOOK.md)

`scene/asset` 모듈은 파일에서 로드된 원시 geometry를 타입별 에셋으로 관리하고, `ASSET_CACHE`를 통해 디스크 I/O를 줄이며, 필요 시 형상 유사도 비교 유틸리티를 제공함.

---

## 레시피 1: 에셋 캐시 등록 및 조회

```python
from spatial_toolbox.scene import ASSET_CACHE

keys = ASSET_CACHE.Add_from_file("path/to/model.obj", unit_length=0.001)

if keys:
    asset = ASSET_CACHE.Get(keys[0], is_hold=True)
    print(asset.label, asset.unit_length)
```

- `Add_from_file()`은 현재 `.obj`를 읽어 `Mesh` 에셋으로 분해 등록함.
- `Get(..., is_hold=True)`는 캐시 원본 참조를 반환함.
- `Get(..., is_hold=False)` 또는 기본값은 `deepcopy`된 독립 인스턴스를 반환함.

---

## 레시피 2: 타입별 조회와 캐시 정리

```python
from spatial_toolbox.scene.asset.type.mesh import Mesh

all_assets = ASSET_CACHE.Get_all()
mesh_assets = ASSET_CACHE.Get_by_type(Mesh)
all_keys = ASSET_CACHE.Get_paths()

removed = ASSET_CACHE.Remove(keys[0])
ASSET_CACHE.Clear()
```

캐시는 타입별 버킷으로 분리되어 있고, 키는 절대 경로 기준으로 정규화됨. 하나의 scene 파일에서 여러 geometry가 분해될 경우 `"{abs_path}#{geo_name}"` 형식의 fragment 키가 사용됨.

---

## 레시피 3: Controller와 함께 사용하기

```python
from spatial_toolbox.scene import ASSET_CACHE, Controller

ctrl = Controller(unit_length=1.0)
keys = ctrl.Register_from_file("scan.obj", unit_length=0.001)

node = ctrl.Build_node_from_cache(keys[0])
ctrl.Add_node(node)

asset = ASSET_CACHE.Get(keys[0], is_hold=True)
```

이 패턴에서는 geometry는 `asset`이 들고 있고, 씬 배치 정보는 `node`가 들고 있음. 둘은 `source_key`로 연결됨.

---

## 레시피 4: 메시 유사도 평가

```python
from spatial_toolbox.scene.asset.utils import (
    Calculate_match_rate,
    Calculate_scan_match_rate,
)

asset_a = ASSET_CACHE.Get("A.obj", is_hold=True)
asset_b = ASSET_CACHE.Get("B.obj", is_hold=True)

rate = Calculate_match_rate(
    asset_a.geometry,
    asset_b.geometry,
    num_samples=5000,
    threshold=0.01,
)

scan_rate = Calculate_scan_match_rate(
    asset_a.geometry,
    asset_b.geometry,
)
```

- `Calculate_match_rate()`는 표면 샘플링 기반의 기하 일치율 계산에 적합함.
- `Calculate_scan_match_rate()`는 부분 중첩이나 스캔본 비교처럼 정합이 필요한 경우에 사용함.
