"""Data_Schema 단위 테스트."""
from dataclasses import dataclass, field
from typing import ClassVar, Any, Callable

import pytest

from python_toolbox.data_schema import Data_Schema


# =============================================================================
# 테스트 더블
# =============================================================================

@dataclass
class _Leaf(Data_Schema):
    x: int = 1
    y: str = "hello"


@dataclass
class _Nested(Data_Schema):
    value: int = 10
    child: _Leaf = field(default_factory=_Leaf)


@dataclass
class _ExcludeSer(Data_Schema):
    __exclude_serialize__: ClassVar[set[str]] = {"secret"}
    public: str = "pub"
    secret: str = "hidden"


@dataclass
class _CustomKey(Data_Schema):
    __custom_keys__: ClassVar[dict[str, str]] = {"internal": "external"}
    internal: int = 42


@dataclass
class _UnpackExtract(Data_Schema):
    __unpack_extract__: ClassVar[set[str]] = {"kwargs"}
    name: str = "base"
    kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class _ExcludeExtract(Data_Schema):
    __exclude_extract__: ClassVar[set[str]] = {"skip"}
    keep: int = 1
    skip: int = 2


@dataclass
class _CustomExtractor(Data_Schema):
    __custom_extractors__: ClassVar[dict[str, Callable]] = {
        "raw": lambda v: v * 2
    }
    raw: int = 5


# =============================================================================
# Serialize
# =============================================================================

def test_serialize_flat():
    """기본 직렬화 — 원시 필드 그대로 출력."""
    result = _Leaf(x=7, y="world").Serialize()
    assert result == {"x": 7, "y": "world"}


def test_serialize_excludes_fields():
    """`__exclude_serialize__` 등록 필드 제외."""
    result = _ExcludeSer().Serialize()
    assert "public" in result
    assert "secret" not in result


def test_serialize_custom_key():
    """`__custom_keys__`로 출력 키 리매핑."""
    result = _CustomKey(internal=99).Serialize()
    assert "external" in result and result["external"] == 99
    assert "internal" not in result


def test_serialize_nested_data_schema():
    """중첩 Data_Schema → 재귀 직렬화."""
    result = _Nested(value=5, child=_Leaf(x=3, y="hi")).Serialize()
    assert result["value"] == 5
    assert result["child"] == {"x": 3, "y": "hi"}


def test_serialize_collections():
    """list/tuple/dict/set 내부 값도 재귀 직렬화."""
    @dataclass
    class _Coll(Data_Schema):
        items: list = field(default_factory=list)
        meta: dict = field(default_factory=dict)

    r = _Coll(items=[_Leaf(x=1), 2], meta={"k": _Leaf(x=9)}).Serialize()
    assert r["items"][0] == {"x": 1, "y": "hello"}
    assert r["items"][1] == 2
    assert r["meta"]["k"] == {"x": 9, "y": "hello"}


# =============================================================================
# Extract
# =============================================================================

def test_extract_flat():
    """기본 추출 — 원시 타입 그대로."""
    result = _Leaf(x=3, y="abc").Extract()
    assert result == {"x": 3, "y": "abc"}


def test_extract_excludes_fields():
    """`__exclude_extract__` 등록 필드 제외."""
    result = _ExcludeExtract().Extract()
    assert "keep" in result
    assert "skip" not in result


def test_extract_unpack_dict():
    """`__unpack_extract__` dict 필드 평탄화."""
    result = _UnpackExtract(name="n", kwargs={"a": 1, "b": 2}).Extract()
    assert "kwargs" not in result
    assert result["a"] == 1 and result["b"] == 2
    assert result["name"] == "n"


def test_extract_unpack_empty_dict_skipped():
    """빈 dict는 평탄화하지 않음 (키 오염 방지)."""
    result = _UnpackExtract(kwargs={}).Extract()
    assert "kwargs" not in result


def test_extract_custom_extractor():
    """`__custom_extractors__` 콜백 적용."""
    result = _CustomExtractor(raw=10).Extract()
    assert result["raw"] == 20


def test_extract_nested_data_schema_flattened():
    """중첩 Data_Schema → 재귀 Extract 후 병합."""
    result = _Nested(value=7, child=_Leaf(x=9, y="z")).Extract()
    assert result["value"] == 7
    assert result["x"] == 9 and result["y"] == "z"
    assert "child" not in result


# =============================================================================
# MRO ClassVar 병합
# =============================================================================

def test_classvar_merge_accumulates_from_parents():
    """자식 __exclude_serialize__가 부모 항목을 포함."""
    @dataclass
    class _Parent(Data_Schema):
        __exclude_serialize__: ClassVar[set[str]] = {"p_field"}
        p_field: int = 0
        shared: int = 1

    @dataclass
    class _Child(_Parent):
        __exclude_serialize__: ClassVar[set[str]] = {"c_field"}
        c_field: int = 2

    merged = _Child.__exclude_serialize__
    assert "p_field" in merged
    assert "c_field" in merged


def test_classvar_merge_custom_keys_union():
    """자식 + 부모 __custom_keys__ 합산."""
    @dataclass
    class _P(Data_Schema):
        __custom_keys__: ClassVar[dict[str, str]] = {"a": "A"}
        a: int = 0
        b: int = 0

    @dataclass
    class _C(_P):
        __custom_keys__: ClassVar[dict[str, str]] = {"b": "B"}

    keys = _C.__custom_keys__
    assert keys.get("a") == "A"
    assert keys.get("b") == "B"
