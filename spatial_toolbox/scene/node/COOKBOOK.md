# Node 모듈 레시피

[← Scene COOKBOOK](../COOKBOOK.md)

`scene/node`는 씬 그래프의 노드 타입 정의와 순회·변환 유틸을 제공하고, 트리 조립은 루트 `stage.py`의 `Controller`가 담당함.

---

## 하위 문서

| 주제 | 문서 |
|---|---|
| Camera / Camera_Intrinsic | [type/COOKBOOK](./type/COOKBOOK.md) |

---

## 노드 타입 개요

| 클래스 | prim_type | 용도 |
|---|---|---|
| `Base_Node` | `"Xform"` | 모든 노드의 공통 베이스 |
| `Group` | `"Xform"` | 구조용 그룹 노드 |
| `Mesh` | `"Mesh"` | `source_key`로 `ASSET_CACHE`를 참조하는 메시 노드 |
| `Camera` | `"Camera"` | 카메라 extrinsic + intrinsic |

---

## 레시피 1: Controller로 씬 구성하기

```python
from spatial_toolbox.scene import Controller

ctrl = Controller(unit_length=1.0)

# 1. 파일 → ASSET_CACHE 등록
keys = ctrl.Register_from_file("robot.obj", unit_length=0.001)

# 2. 캐시 키 → Mesh 노드 생성
robot = ctrl.Build_node_from_cache(keys[0], label="robot")

# 3. 노드를 루트에 추가
ctrl.Add_node(robot)
```

`Build_node_from_cache()`는 자산의 `unit_length`와 씬의 `unit_length`를 비교해 `node.unit_scale`을 자동 계산함.

---

## 레시피 2: Group(Xform) 구조 노드 만들기

`Controller.Add_node()`는 일반 노드는 복제해서 붙이고, `prim_type == "Xform"`인 노드는 "컨테이너 자체"를 붙이지 않고 자식들을 평탄화해서 삽입함. 빈 그룹을 트리에 만들고 싶으면 `node=None`으로 호출해야 함.

```python
from spatial_toolbox.scene.node.type.group import Group

# 빈 Group 생성
ctrl.Add_node()
group = ctrl.root.children[-1]
group.label = "robots"

# Mesh 노드를 group 아래에 추가
ctrl.Add_node(robot, parent=group)

# 이미 자식을 가진 Xform을 넣으면 컨테이너는 생략되고 자식만 복제됨
template = Group(label="template")
template.children.append(robot.Clone(label_name="robot_copy"))
ctrl.Add_node(template, parent=group)
```

---

## 레시피 3: 노드 이동·분리·초기화

```python
# 노드 이동
ctrl.Move_node(robot, old=group, new=ctrl.root)

# 노드 분리
detached = ctrl.Pop_node(robot, parent=ctrl.root)

# 씬 전체 초기화
ctrl.Clear()
```

---

## 레시피 4: 변환 행렬 제어

모든 노드는 `local_rigid`(4x4), `scale`(3-vector), `unit_scale`(scalar)로 최종 `world_matrix`를 구성함.

```python
from spatial_toolbox.scene.node.utils.transform import (
    Build_transform,
    Decompose_transform,
)

robot.local_rigid = Build_transform(tx=1.0, ty=0.0, tz=0.5, rz=45.0)
world = robot.world_matrix

tx, ty, tz, rx, ry, rz = Decompose_transform(robot.local_rigid)
```

`local_rigid`, `scale`, `unit_scale`, `parent`가 바뀌면 Dirty Flag가 전파되어 자신과 자손의 `world_matrix` 캐시가 자동 무효화됨.

---

## 레시피 5: 가시성 제어

```python
robot.visible = False

group.visible = False   # 자식 전체에 False 전파
group.visible = True    # group만 True, 자식은 자동 복구되지 않음
```

`visible=False`만 자손에게 전파되고, `True`는 전파되지 않음. 자식별 가시성 상태를 보존하려는 규약임.

---

## 레시피 6: 트리 순회와 렌더 큐

```python
from spatial_toolbox.scene.node.utils.traversal import walk_nodes
from spatial_toolbox.scene.node.type.mesh import Mesh

mesh_nodes = list(walk_nodes(ctrl.root, lambda n: isinstance(n, Mesh)))
render_queue = ctrl.Get_render_queue()

target = next(
    walk_nodes(ctrl.root, lambda n: n.label == "robot"),
    None,
)
```

`Get_render_queue()`는 현재 구현상 `Mesh`이면서 `is_renderable`이 `True`인 노드만 반환함.

---

## 레시피 7: 복제와 경로

```python
clone = robot.Clone()
clone2 = robot.Clone(label_name="robot_copy")

print(robot.prim_path)   # "/World_Root/robots/robot"
data = robot.Serialize()
```
