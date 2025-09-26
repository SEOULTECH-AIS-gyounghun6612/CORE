"""타입 및 시그니처 안전성을 보장하는 모듈 레지스트리.

클래스 레지스트리(target_type=Base_Class)와 Callable 레지스트리
(target_type=Callable[[...], R]) 두 모드를 단일 인터페이스로 제공함.
등록 시점에 상속 관계 또는 시그니처 일치를 강제하여 잘못된 등록을 차단함.

Requirement:
    - Python >= 3.10
    - typing, inspect, collections.abc
"""
from __future__ import annotations
from typing import Any, TypeVar, Generic, Callable, get_origin, get_args, cast
import inspect
import collections.abc


T = TypeVar("T")
C = TypeVar("C")


class Registry(Generic[T]):
    """타입 및 시그니처 안전성을 보장하는 하이브리드 모듈 레지스트리.

    target_type 형태에 따라 두 가지 모드로 동작함:
    - 클래스 모드: target_type이 클래스이면 issubclass 검증을 수행함.
    - Callable 모드: target_type이 Callable[[...], R]이면 파라미터 개수
      일치 여부를 런타임 검증함.

    이름은 데코레이터 인자로 명시하거나 객체의 __name__에서 자동 추출됨.
    중복 등록은 KeyError로 차단됨.
    """

    def __init__(self, name: str, target_type: Any) -> None:
        """
        Args:
            name: 레지스트리 식별 이름 (오류 메시지에 사용).
            target_type: 등록 대상 타입 (클래스 또는 Callable[[...], R]).

        Raises:
            TypeError: target_type이 클래스도 Callable도 아닌 경우.
        """
        self.name = name
        self.target_type = target_type
        self._module_dict: dict[str, T] = {}

        # Callable 타겟 vs 클래스 타겟 식별
        self._is_callable_target = (
            get_origin(self.target_type) is collections.abc.Callable
        )
        self._expected_param_count = -1

        # Callable 시그니처 파라미터 개수 캐싱
        if self._is_callable_target:
            _args = get_args(self.target_type)
            if _args and _args[0] is not ...:
                self._expected_param_count = len(_args[0])
        elif not isinstance(self.target_type, type):
            raise TypeError(
                f"[ERROR] 지원하지 않는 target_type: {self.target_type}"
            )

    def Get(self, key: str) -> T:
        """등록된 모듈을 키로 조회함.

        Args:
            key: 등록 시 사용된 이름.

        Returns:
            등록된 객체.

        Raises:
            KeyError: 미등록 키인 경우.
        """
        if key not in self._module_dict:
            raise KeyError(f"'{key}'은(는) {self.name}에 등록되지 않았음.")
        return self._module_dict[key]

    def Register_module(self, name: str | None = None) -> Callable[[C], C]:
        """대상 객체를 레지스트리에 등록하는 데코레이터 팩토리.

        Args:
            name: 등록 키 (생략 시 객체의 __name__에서 lowercase로 추출).

        Returns:
            데코레이터 함수.

        Raises:
            TypeError: 객체 종류가 레지스트리 모드와 불일치하거나,
                상속 관계 또는 시그니처가 어긋난 경우.
            ValueError: 람다 등 이름 추론이 불가능한 경우.
            KeyError: 동일 이름이 이미 등록된 경우.
        """
        def _register(obj: C) -> C:
            _obj_is_type = inspect.isclass(obj)
            _obj_is_callable = callable(obj)

            _for_callable = self._is_callable_target

            # XNOR 검증: 클래스 전용에 함수, 함수 전용에 클래스 차단
            if _obj_is_type == _for_callable:
                _target_name = "Callable" if _for_callable else "Class"
                raise TypeError(
                    f"[ERROR] '{getattr(obj, '__name__', obj)}'은(는) "
                    f"레지스트리 목적({_target_name} 전용)과 일치하지 않음."
                )

            # 클래스 상속 계층 검증
            if _obj_is_type:
                if not issubclass(obj, self.target_type):
                    raise TypeError(
                        f"'{obj.__name__}'은(는) "
                        f"'{self.target_type.__name__}'의 하위 클래스여야 함."
                    )

            # Callable 파라미터 개수 검증
            elif _obj_is_callable:
                if self._expected_param_count >= 0:
                    _sig = inspect.signature(obj)
                    _actual_params = [
                        _p for _p in _sig.parameters.values()
                        if _p.kind in (
                            inspect.Parameter.POSITIONAL_OR_KEYWORD,
                            inspect.Parameter.POSITIONAL_ONLY,
                        )
                    ]
                    if len(_actual_params) != self._expected_param_count:
                        raise TypeError(
                            f"[ERROR] 파라미터 개수 불일치. "
                            f"요구됨: {self._expected_param_count}개, "
                            f"실제: {len(_actual_params)}개."
                        )
            else:
                raise TypeError(
                    f"[ERROR] '{obj}'은(는) 지원하지 않는 객체 타입임."
                )

            # 안전한 식별자 추출 및 등록
            _name = name or getattr(
                obj, "__name__", getattr(obj.__class__, "__name__", "")
            ).lower()
            if _name == "<lambda>" or not _name:
                raise ValueError(
                    "이름을 추론할 수 없거나 람다 함수임. "
                    "name 파라미터를 명시할 것."
                )

            if _name in self._module_dict:
                raise KeyError(
                    f"'{_name}'은(는) 이미 {self.name}에 등록됨."
                )

            self._module_dict[_name] = cast(T, obj)
            return obj

        return _register
