# registry — 컴포넌트 등록 및 조회 사용 예시

`Registry`는 타입/시그니처 안전성을 보장하는 하이브리드 모듈 레지스트리입니다.
잘못된 타입이나 함수 시그니처가 등록되는 것을 등록(import) 시점에 차단합니다.

---

## 클래스 레지스트리

등록할 대상이 클래스일 경우, `target_type`으로 베이스 클래스를 지정합니다.

```python
from python_toolbox import Registry

class Loss_Base:
    pass

# target_type이 클래스이므로, 등록되는 모든 객체는 Loss_Base의 하위 클래스여야 함.
LOSSES = Registry[type[Loss_Base]]("losses", Loss_Base)

@LOSSES.Register_module("ce_loss")
class CE_Loss(Loss_Base):
    pass

cls = LOSSES.Get("ce_loss")     # CE_Loss 클래스 반환
```

만약 `Loss_Base`를 상속받지 않은 클래스를 등록하려고 시도하면 `TypeError`가 발생합니다.

---

## Callable 레지스트리

등록할 대상이 함수(Callable)일 경우, `target_type`으로 `Callable[[...], R]` 형태를 지정합니다.

```python
from typing import Callable, Any
from python_toolbox import Registry

# target_type이 Callable이므로, 파라미터 개수가 검증됨.
COLLATE_FNS = Registry[Callable[[Any], Any]](
    "collate_fn", Callable[[Any], Any]
)

@COLLATE_FNS.Register_module("default")
def default_collate(batch):
    ...
```

지정된 Callable과 실제 등록하려는 함수의 파라미터 개수가 다르면 (예: `Callable[[Any], Any]`인데 인자를 2개 받는 함수를 등록하려 할 때) `TypeError`가 발생합니다.

---

## 자동 이름 추론 및 유의사항

`Register_module()` 사용 시 이름을 생략하면 객체의 `__name__`을 소문자(lowercase)로 변환하여 등록합니다.

```python
@LOSSES.Register_module()  # "mse_loss"로 자동 등록됨
class MSE_Loss(Loss_Base):
    pass
```

- `lambda` 함수처럼 이름 추론이 불가능한 객체는 이름을 생략할 수 없으며, 반드시 명시해야 합니다.
- 동일한 이름으로 중복 등록을 시도하면 `KeyError`가 발생합니다.
- 클래스 전용 레지스트리에 함수를 등록하거나, 함수 전용 레지스트리에 클래스를 등록하면 즉시 `TypeError`가 발생합니다 (XNOR 검증).
