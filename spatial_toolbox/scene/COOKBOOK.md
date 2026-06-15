# Scene 모듈 레시피

[← 최상위 COOKBOOK](../../../COOKBOOK.md)

`scene/`은 순수 데이터 레이어임. 씬 그래프 노드 정의, 에셋 캐시, 장면 import/export를 담당하며 렌더러·UI에 의존하지 않음.

---

## 하위 모듈

| 모듈 | 역할 | 문서 |
|---|---|---|
| **node** | 씬 그래프 노드 타입, Controller 사용 패턴, 순회·변환 유틸 | [node/COOKBOOK](./node/COOKBOOK.md) |
| **asset** | 에셋 타입, ASSET_CACHE, 원시 geometry 로드, 유사도 평가 | [asset/COOKBOOK](./asset/COOKBOOK.md) |
| **file** | 씬 상태 직렬화 / 역직렬화 (JSON, USD) | [file/COOKBOOK](./file/COOKBOOK.md) |

---

## 의존 흐름

```text
asset/file  ──►  asset/cache
                      ▲
                      │
node/type  ◄──── stage.py ────► scene/file
```

- 원시 3D 파일(`.obj`) 로드는 `scene/asset/file`이 담당함.
- 씬 상태(`Controller.root`, `unit_length`)의 저장/복원은 `scene/file`이 담당함.
- `stage.py`의 `Controller`가 두 계층을 묶는 진입점임.

---

## 빠른 시작

```python
from spatial_toolbox.scene import Controller

ctrl = Controller(unit_length=1.0)

# 1. 파일을 읽어 ASSET_CACHE에 등록
keys = ctrl.Register_from_file("model.obj", unit_length=0.001)

# 2. 캐시 키로 Mesh 노드 생성
node = ctrl.Build_node_from_cache(keys[0])

# 3. 씬 루트에 추가
ctrl.Add_node(node)

# 4. 현재 씬을 JSON 또는 USD로 저장
ctrl.Export("scene.json")

# 5. 다른 컨트롤러에서 다시 복원
ctrl2 = Controller()
ctrl2.Import("scene.json")
```

---

## 언제 무엇을 쓰는가

- geometry를 메모리에 올리고 재사용하려면 `ASSET_CACHE` 또는 `Controller.Register_from_file`
- geometry를 씬에 배치하려면 `Controller.Build_node_from_cache` + `Controller.Add_node`
- 장면 전체를 저장/복원하려면 `Controller.Export` / `Controller.Import`

세부 예시는 각 하위 문서를 참조.
