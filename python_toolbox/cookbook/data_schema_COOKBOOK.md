# data_schema — 스키마 기반 데이터 컨테이너 사용 예시

`Data_Schema`는 dataclass에 직렬화(`Serialize`) 및 평탄화 추출(`Extract`) 기능을 부여하는 베이스 클래스입니다.

---

## 기본 사용 (Serialize / Extract)

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

---

## 직렬화(Serialize) 제어 ClassVar

특정 필드를 제외하거나, 출력 키 이름을 바꾸거나, 커스텀 직렬화 함수를 적용할 수 있습니다.

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

---

## 추출(Extract) 제어 ClassVar

Extract 시 특정 필드를 제외하거나, 딕셔너리 필드를 상위 레벨로 평탄화(Unpack)할 수 있습니다.

```python
from typing import ClassVar
from dataclasses import dataclass, field
from python_toolbox import Data_Schema

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

---

## MRO 누적 병합 규칙

부모 클래스의 ClassVar 규칙은 자식에 **자동 누적 병합**됩니다.

```python
from typing import ClassVar
from dataclasses import dataclass
from python_toolbox import Data_Schema

@dataclass
class A(Data_Schema):
    __exclude_serialize__: ClassVar[set] = {"a_secret"}

@dataclass
class B(A):
    __exclude_serialize__: ClassVar[set] = {"b_secret"}

print(B.__exclude_serialize__)   
# {"a_secret", "b_secret"}
```

> **주의:** 병합은 누적 전용입니다. 부모 항목의 제거가 필요하면 조부모 레벨에서 새 분기 클래스를 정의해야 합니다.

---

## 신규 ClassVar 규칙 등록

도메인 특화 ClassVar를 추가하려면 `__merge_specs__`에 등록합니다.

```python
from typing import ClassVar
from dataclasses import dataclass
from python_toolbox import Data_Schema

@dataclass
class Custom_Schema(Data_Schema):
    __merge_specs__: ClassVar[dict] = {"__my_filter__": set}
    __my_filter__: ClassVar[set] = {"some_field"}
```

`__merge_specs__` 자체도 누적 병합되므로 부모 entry는 명시 복사할 필요가 없습니다. container_type은 `update` 메서드를 가진 타입(`set`, `dict` 등)이어야 합니다.
