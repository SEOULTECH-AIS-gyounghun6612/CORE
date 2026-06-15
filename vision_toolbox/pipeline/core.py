import time
from abc import ABC, abstractmethod
from typing import Optional, Iterable, Any, Generic

from vision_toolbox.constants import DEBUG_MODE
from vision_toolbox.pipeline.state import TState


class Base_Step(ABC, Generic[TState]):
    """연산 및 상태 관리 기본 추상 클래스.

    Attributes:
        name: 단계 명칭.
        elapsed: 최근 실행 소요 시간.
        last_run_timestamp: 마지막 실행 시점 타임스탬프.
        metadata: 실행 관련 부가 정보.
    """

    def __init__(self, name: Optional[str] = None):
        """기본 정보 초기화.

        Args:
            name: 단계 이름 (기본값은 클래스명).
        """
        self.name = name or self.__class__.__name__
        self.elapsed: float = 0.0
        self.last_run_timestamp: float = 0.0
        self.metadata: dict[str, Any] = {}

    def __call__(self, state: TState, *args, **kwargs) -> TState:
        """단계 실행 진입점.

        DEBUG_MODE 활성 시 소요 시간을 측정함.

        Args:
            state: 현재 파이프라인 상태.
            *args: 가변 인자.
            **kwargs: 가변 키워드 인자.

        Returns:
            TState: 실행 후 업데이트된 상태.
        """
        if not DEBUG_MODE:
            return self.forward(state, *args, **kwargs)

        # 시작 시간 캡처
        self.last_run_timestamp = time.perf_counter()
        state = self.forward(state, *args, **kwargs)
        # 소요 시간 갱신
        self.elapsed = time.perf_counter() - self.last_run_timestamp
        return state

    @abstractmethod
    def forward(self, state: TState, *args, **kwargs) -> TState:
        """하위 클래스에서 구현할 실제 연산 로직."""
        raise NotImplementedError

    def __repr__(self):
        return f"{self.__class__.__name__}(name='{self.name}')"


class Sequential_Step(Base_Step[TState]):
    """하위 단계들을 순차적으로 실행하는 컨테이너.

    Attributes:
        steps: 실행할 하위 단계 리스트.
    """

    def __init__(
        self, name: str, *steps: Base_Step[TState] | Iterable[Base_Step[TState]]
    ):
        """컨테이너 초기화.

        Args:
            *steps: 가변 개수의 BaseStep 또는 리스트.
            name: 컨테이너 명칭.
        """
        super().__init__(name)
        _holder: list[Base_Step[TState]] = []
        for s in steps:
            if isinstance(s, Iterable):
                _holder.extend(s)
            else:
                _holder.append(s)

        self.steps = _holder

    def forward(
        self, state: TState, *args, **kwargs
    ) -> TState:
        """하위 단계들을 순서대로 실행.

        Args:
            state: 현재 파이프라인 상태.
            *args: 하위 단계로 전달될 인자.
            **kwargs: 하위 단계로 전달될 키워드 인자.

        Returns:
            TState: 모든 단계 실행 후 상태.
        """
        for step in self.steps:
            state = step(state, *args, **kwargs)
        return state

    def __getitem__(self, idx) -> Base_Step[TState]:
        return self.steps[idx]

    def __len__(self) -> int:
        return len(self.steps)

    def __repr__(self):
        steps_repr = "".join([str(s) for s in self.steps])
        return f"{self.name}({steps_repr})"
