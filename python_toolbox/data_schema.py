"""스키마 기반 데이터 컨테이너 모듈.

dataclass에 직렬화 + 추출 기능을 부여하는 베이스 클래스 Data_Schema를 제공함.
중첩된 Data_Schema에 대한 재귀 처리를 지원하며, __merge_specs__ 메타 등록
패턴으로 ClassVar 규칙을 자식 클래스에 자동 누적함.

Requirement:
    - Python >= 3.10
    - dataclasses
"""
from __future__ import annotations
from dataclasses import dataclass, fields
from typing import Any, Callable, ClassVar


@dataclass
class Data_Schema:
    """직렬화 + 추출 기능을 보유한 dataclass 베이스.

    중첩된 Data_Schema 객체에 대한 재귀 처리를 지원하며, 6종 ClassVar로
    직렬화/추출 동작을 세밀하게 제어함.

    ## 동작 제어 ClassVar
    - **__exclude_serialize__**: Serialize에서 제외할 필드 이름 집합.
    - **__custom_keys__**: Serialize 시 필드명 → 출력 키 리매핑.
    - **__custom_serializers__**: 필드별 커스텀 직렬화 콜백.
    - **__exclude_extract__**: Extract에서 제외할 필드 이름 집합.
    - **__unpack_extract__**: Extract 시 dict로 평탄화할 필드 집합.
    - **__custom_extractors__**: 필드별 커스텀 추출 콜백.

    ## ClassVar 누적 병합 규약
    - 모든 ClassVar는 MRO 역순 누적 병합되어 자식 클래스에 적용됨.
    - 자식 클래스에서 추가한 항목은 부모 항목과 합쳐지며, **제거는 불가함**.
    - 부분 override가 필요하면 조부모 레벨에서 새 분기 클래스 정의 필요.

    ## 신규 ClassVar 등록
    - 자식 스키마에서 새로운 ClassVar 규칙이 필요하면 __merge_specs__에
      `{"__var_name__": container_type}` 형태로 등록만 하면 됨.
    - container_type은 set 또는 dict 등 update 메서드를 가진 타입이어야 함.
    - __merge_specs__ 자체도 누적 병합되므로 부모 entry를 명시 복사할 필요 없음.

    ## Serialize vs Extract
    - **Serialize**: 재귀 dict 변환 + 키 리매핑. 저장/전송 목적의 dict 산출.
    - **Extract**: 평탄화 + 원본 타입 유지. 함수 호출/주입 목적의 dict 산출.
    """

    __exclude_serialize__: ClassVar[set[str]] = set()
    __exclude_extract__: ClassVar[set[str]] = set()
    __unpack_extract__: ClassVar[set[str]] = set()

    __custom_keys__: ClassVar[dict[str, str]] = {}
    __custom_serializers__: ClassVar[dict[str, Callable[[Any], Any]]] = {}
    __custom_extractors__: ClassVar[dict[str, Callable[[Any], Any]]] = {}

    # ClassVar 이름 → 컨테이너 타입 매핑. 메타 자기 등록 패턴으로 자기 자신 포함.
    __merge_specs__: ClassVar[dict[str, type]] = {
        "__merge_specs__": dict,
        "__exclude_serialize__": set,
        "__exclude_extract__": set,
        "__unpack_extract__": set,
        "__custom_keys__": dict,
        "__custom_serializers__": dict,
        "__custom_extractors__": dict,
    }

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """__merge_specs__ 등록 ClassVar를 MRO 역순 누적 병합함.

        2단계 처리: __merge_specs__ 자체를 먼저 병합하여 최신 spec을 확보한 뒤,
        등록된 나머지 ClassVar를 일괄 병합함.
        """
        super().__init_subclass__(**kwargs)

        # 1단계: __merge_specs__ 자체 누적 병합
        _merged_specs: dict[str, type] = {}
        for _base in reversed(cls.__mro__):
            _spec = getattr(_base, "__merge_specs__", None)
            if _spec:
                _merged_specs.update(_spec)
        cls.__merge_specs__ = _merged_specs

        # 2단계: 등록된 ClassVar 일괄 병합
        for _name, _container in _merged_specs.items():
            if _name == "__merge_specs__":
                continue
            _merged = _container()
            for _base in reversed(cls.__mro__):
                _value = getattr(_base, _name, None)
                if _value is not None:
                    _merged.update(_value)
            setattr(cls, _name, _merged)

    def Serialize(self) -> dict[str, Any]:
        """재귀 dict 변환 + 키 리매핑으로 저장/전송용 dict를 산출함.

        - __exclude_serialize__ 등록 필드는 결과에서 제외됨.
        - __custom_serializers__ 등록 필드는 콜백 결과로 대체됨.
        - 그 외 필드는 _serialize_value로 재귀 처리됨.
        - __custom_keys__로 출력 키가 리매핑됨.

        Returns:
            dict[str, Any]: 직렬화된 데이터.
        """
        _res: dict[str, Any] = {}

        _cls = self.__class__
        _exclude = _cls.__exclude_serialize__
        _keys = _cls.__custom_keys__
        _serializers = _cls.__custom_serializers__

        for _f in fields(self):
            if _f.name in _exclude:
                continue

            _value = getattr(self, _f.name)
            _key = _keys.get(_f.name, _f.name)

            _serializer = _serializers.get(_f.name)
            if _serializer is not None:
                _res[_key] = _serializer(_value)
                continue

            _res[_key] = self._serialize_value(_value)

        return _res

    def _serialize_value(self, value: Any) -> Any:
        """단일 값/컨테이너 재귀 직렬화."""
        # 원시 타입 조기 반환
        if value is None or isinstance(value, (str, int, float, bool)):
            return value

        if isinstance(value, Data_Schema):
            return value.Serialize()

        if isinstance(value, dict):
            return {k: self._serialize_value(v) for k, v in value.items()}

        if isinstance(value, (list, tuple, set)):
            return type(value)(self._serialize_value(item) for item in value)

        return value

    def Extract(self) -> dict[str, Any]:
        """평탄화된 데이터 추출 — 함수 호출/주입용 dict 산출.

        - __exclude_extract__ 등록 필드는 결과에서 제외됨.
        - __custom_extractors__ 등록 필드는 콜백 결과로 대체됨.
        - 중첩 Data_Schema 필드는 재귀 Extract 결과로 평탄 병합됨.
        - __unpack_extract__ 등록 dict 필드는 본 dict로 펼쳐짐.
        - 원본 타입은 유지되며 직렬화 변환은 수행하지 않음.

        Returns:
            dict[str, Any]: 추출된 데이터.
        """
        _res: dict[str, Any] = {}

        _cls = self.__class__
        _exclude = _cls.__exclude_extract__
        _extractors = _cls.__custom_extractors__
        _unpack = _cls.__unpack_extract__

        for _f in fields(self):
            if _f.name in _exclude:
                continue

            _value = getattr(self, _f.name)

            _extractor = _extractors.get(_f.name)
            if _extractor is not None:
                _res[_f.name] = _extractor(_value)
                continue

            if isinstance(_value, Data_Schema):
                _res.update(_value.Extract())
                continue

            if _f.name in _unpack and isinstance(_value, dict):
                if _value:  # 빈 dict 병합 방지
                    _res.update(_value)
                continue

            _res[_f.name] = _value

        return _res
