# Camera 레시피

[← Node COOKBOOK](../COOKBOOK.md)

`Camera` 노드는 `Base_Node`를 상속하며, OpenCV 규약의 핀홀 카메라 내부 파라미터(`Camera_Intrinsic`)를 보유함.

---

## Camera_Intrinsic 파라미터

| 필드 | 기본값 | 설명 |
|---|---|---|
| `width`, `height` | `1920`, `1080` | 이미지 해상도 (픽셀) |
| `fx`, `fy` | `1000.0`, `1000.0` | 초점 거리 (픽셀) |
| `cx`, `cy` | `960.0`, `540.0` | 주점 (principal point) |
| `distortion` | `[0.0] * 8` | 왜곡 계수 (`k1~k6`, `p1`, `p2`) |
| `near_clip` | `0.1` | 근거리 클리핑 평면 |
| `far_clip` | `1000.0` | 원거리 클리핑 평면 |

---

## 레시피 1: Camera 노드 생성 및 씬 등록

```python
from spatial_toolbox.scene import Controller
from spatial_toolbox.scene.node.type.camera import Camera, Camera_Intrinsic
from spatial_toolbox.scene.node.utils.transform import Build_transform

ctrl = Controller()

intrinsic = Camera_Intrinsic(
    width=1920,
    height=1080,
    fx=1050.0,
    fy=1050.0,
    cx=960.0,
    cy=540.0,
    near_clip=0.1,
    far_clip=500.0,
)

cam = Camera(label="main_camera", intrinsic=intrinsic)
cam.local_rigid = Build_transform(tz=2.0)

ctrl.Add_node(cam)
```

---

## 레시피 2: FOV 조회

```python
print(f"fov_x: {cam.intrinsic.fov_x:.1f}°")
print(f"fov_y: {cam.intrinsic.fov_y:.1f}°")
```

`fov_x`, `fov_y`는 내부 파라미터로부터 계산되는 읽기 전용 프로퍼티임.

---

## 레시피 3: 직렬화 / 역직렬화

```python
data = cam.intrinsic.Serialize()
restored = Camera_Intrinsic(**data)

cam_data = cam.Serialize()
restored_cam = Camera(**cam_data)
```

`Camera` 직렬화 시 `intrinsic`은 `intrinsic_meta` 형태로 저장되며, 생성자에서 자동 복원됨.

---

## 레시피 4: Camera 복제

```python
cam2 = cam.Clone(label_name="sub_camera")
cam2.local_rigid = Build_transform(tx=0.5, tz=2.0)
```

`Camera.Clone()`은 `intrinsic`을 깊은 복사하므로, 두 카메라의 내부 파라미터는 독립적으로 수정 가능함.
