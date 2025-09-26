# COOKBOOK

python_toolbox 사용 예시 모음.

패키지 단위로 분리된 사용법은 각 폴더의 COOKBOOK 문서를 참고하세요. 단일 파일 모듈(`data_schema`, `registry`, `system`)은 본 문서에 확인 가능합니다.

| 패키지 | 문서 |
|--------|------|
| `file/` | [file 사용 예시](python_toolbox/file/COOKBOOK.md) — 파일 I/O 디스패치, 포맷 클래스, 그룹 분할 |
| `project/` | [project 사용 예시](python_toolbox/project/COOKBOOK.md) — Base_Config, Project_Template |

---

## Data_Schema — 직렬화/추출 코어

dataclass에 두 가지 dict 변환 기능을 부여합니다.

- **Serialize**: 재귀 dict 변환 + 키 리매핑. 저장/전송 목적의 dict 산출.
- **Extract**: 평탄화 + 원본 타입 유지. 함수 호출/주입 목적의 dict 산출.

```python
from dataclasses import dataclass, field
from python_toolbox import Data_Schema

@dataclass
class Inner(Data_Schema):
    a: int = 10
    b: int = 20

@dataclass
class Outer(Data_Schema):
    name: str = "demo"
    inner: Inner = field(default_factory=Inner)

obj = Outer()
obj.Serialize()
# {"name": "demo", "inner": {"a": 10, "b": 20}}   <- 중첩 보존

obj.Extract()
# {"name": "demo", "a": 10, "b": 20}              <- 평탄화
```

### 직렬화 제어 ClassVar

```python
from typing import ClassVar
from dataclasses import dataclass, field
from python_toolbox import Data_Schema

@dataclass
class Model_Schema(Data_Schema):
    name: str = "resnet50"
    weights: bytes = b""        # 직렬화 제외
    extra: dict = field(default_factory=lambda: {"k": "v"})

    __exclude_serialize__: ClassVar[set] = {"weights"}
    __custom_keys__: ClassVar[dict] = {"name": "model_name"}
    __custom_serializers__: ClassVar[dict] = {
        "extra": lambda v: list(v.items()),     # dict → tuple list로 직렬화
    }

Model_Schema().Serialize()
# {"model_name": "resnet50", "extra": [("k", "v")]}
```

### 추출 제어 ClassVar

```python
@dataclass
class Loader_Schema(Data_Schema):
    batch_size: int = 32
    extra_kwargs: dict = field(default_factory=lambda: {"momentum": 0.9})
    _internal: str = "skip"

    __exclude_extract__: ClassVar[set] = {"_internal"}
    __unpack_extract__: ClassVar[set] = {"extra_kwargs"}

Loader_Schema().Extract()
# {"batch_size": 32, "momentum": 0.9}     <- _internal 제외, extra_kwargs 평탄화
```

### MRO 누적 병합

부모 클래스의 ClassVar 규칙은 자식에 자동 누적됩니다.

```python
@dataclass
class A(Data_Schema):
    __exclude_serialize__: ClassVar[set] = {"a_secret"}

@dataclass
class B(A):
    __exclude_serialize__: ClassVar[set] = {"b_secret"}

B.__exclude_serialize__   # {"a_secret", "b_secret"}
```

병합은 누적 전용입니다. 부모 항목 제거가 필요하면 조부모 레벨에서 새 분기 클래스를 정의해야 합니다.

### 신규 ClassVar 등록

도메인 특화 ClassVar를 추가하려면 `__merge_specs__`에 한 줄만 등록하면 됩니다.

```python
@dataclass
class Custom_Schema(Data_Schema):
    __merge_specs__: ClassVar[dict] = {"__my_filter__": set}
    __my_filter__: ClassVar[set] = {"some_field"}
```

`__merge_specs__` 자체도 누적 병합되므로 부모 entry는 명시 복사할 필요가 없습니다. container_type은 `update` 메서드를 가진 타입(set, dict 등)이어야 합니다.

---

## Registry — 컴포넌트 등록 및 조회

타입/시그니처 안전성을 보장하는 모듈 레지스트리입니다.

### 클래스 레지스트리

```python
from python_toolbox import Registry

class Loss_Base:
    pass

LOSSES = Registry[type[Loss_Base]]("losses", Loss_Base)

@LOSSES.Register_module("ce_loss")
class CE_Loss(Loss_Base):
    pass

cls = LOSSES.Get("ce_loss")     # CE_Loss 클래스 반환
```

### Callable 레지스트리

```python
from typing import Callable, Any
from python_toolbox import Registry

COLLATE_FNS = Registry[Callable[[Any], Any]](
    "collate_fn", Callable[[Any], Any]
)

@COLLATE_FNS.Register_module("default")
def default_collate(batch):
    ...
```

등록 시점에 클래스 상속 계층 또는 Callable 파라미터 개수가 검증됩니다. 불일치 시 `TypeError`가 즉시 발생합니다 (XNOR 검증).

---

## String — 문자열 유틸리티

```python
from python_toolbox.system import String

# 카운터 정렬 (예: 배치 로그)
String.Count_auto_align(3, 100)                  # "003/100"
String.Count_auto_align(3, 100, is_right=False)  # "3  /100"

# 진행바 출력
for i in range(1, 101):
    String.Progress_bar(i, 100, prefix="Train", suffix="Complete")
```

## Time_Utils — 시간 유틸리티

```python
from python_toolbox.system import Time_Utils
import time

start = Time_Utils.Stamp()
time.sleep(1)
elapsed = Time_Utils.Get_term(start)            # timedelta

# 포맷 변환
text = Time_Utils.Make_text_from(start, d_fmt="%Y-%m-%d")
```
