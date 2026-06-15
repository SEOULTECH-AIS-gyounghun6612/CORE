from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, ClassVar
from dataclasses import dataclass, field

from python_toolbox.project import Base_Config


@dataclass
class Accumulator_Config(Base_Config):
    """단일 Accumulator 생성 설정.

    Attributes:
        config_type: CFGS 레지스트리 조회 키.
        object_type: ACCUMULATORS 레지스트리 조회 키.
        acc_kwargs: Accumulator 생성자에 전달할 추가 인자.
    """

    __unpack_extract__: ClassVar[set[str]] = {"acc_kwargs"}
    __exclude_extract__: ClassVar[set[str]] = {"config_type", "object_type"}

    config_type: str = "Accumulator_Config"
    object_type: str = ""
    acc_kwargs: dict[str, Any] = field(default_factory=dict)


class Accumulator(ABC):
    """minibatch 결과를 스트리밍 누적하는 stateful 객체의 공통 인터페이스.

    Update는 batch마다 호출되며, Finalize는 누적 완료 후 확정값을 반환함.
    연산은 교환·결합 법칙을 만족해야 함(도착 순서 무관).
    """

    @abstractmethod
    def Update(self, **output: Any) -> None:
        """_Forward 출력을 **kwargs로 받아 필요한 키만 선택해 누적함."""

    @abstractmethod
    def Finalize(self) -> Any:
        """누적 state를 확정값으로 반환함. 호출 후 state는 소비된 것으로 간주."""

    @abstractmethod
    def Reset(self) -> None:
        """누적 state를 초기화함."""


@dataclass
class Assemble_Metric_Config(Base_Config):
    """복수 Accumulator를 묶는 Assemble_Metric 생성 설정.

    Attributes:
        sub_metric_meta: {acc_name: acc_config_dict} 매핑. Build_metric에서 Accumulator_Config로 변환된다.
    """

    sub_metric_meta: dict[str, dict[str, Any]] = field(default_factory=dict)


class Assemble_Metric:
    """단일 mode에 속한 Accumulator 집합 관리자.

    mode 라우팅은 상위(Component_Assembler)에서 담당하며,
    본 클래스는 이름으로 묶인 Accumulator 묶음의 Update·Finalize·Reset을 일괄 처리한다.

    Attributes:
        _accs: {acc_name: Accumulator} 내부 저장소.
    """

    def __init__(self, accs: dict[str, Accumulator]) -> None:
        self._accs = accs

    def Update(self, **output: Any) -> None:
        """모든 accumulator에 batch 출력을 전달한다."""
        for _acc in self._accs.values():
            _acc.Update(**output)

    def Finalize(self) -> dict[str, Any]:
        """모든 accumulator의 확정값을 {name: value} 형태로 반환한다."""
        return {_name: _acc.Finalize() for _name, _acc in self._accs.items()}

    def Reset(self) -> None:
        """모든 accumulator의 누적 state를 초기화한다."""
        for _acc in self._accs.values():
            _acc.Reset()

    def __getitem__(self, key: str) -> Accumulator:
        return self._accs[key]

    def __contains__(self, key: str) -> bool:
        return key in self._accs