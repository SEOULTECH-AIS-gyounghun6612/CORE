from typing import TypeVar
from dataclasses import dataclass


@dataclass
class Base_State:
    """
    파이프라인 상태 관리를 위한 기본 클래스.
    """

# BaseState를 상속받는 모든 상태 클래스를 허용하는 제네릭 타입 변수 정의
TState = TypeVar("TState", bound=Base_State)
