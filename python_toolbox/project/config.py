"""파일 I/O 진입점이 부여된 Data_Schema 특화 모듈.

JSON/YAML 파일 또는 Registry를 통해 설정 객체를 구성하고 파일 저장을
지원하는 Base_Config 및 팩토리 함수를 제공함. python_toolbox.file 의존을
이 모듈로 격리하여 Data_Schema 코어를 stdlib만 의존하도록 유지함.

Requirement:
    - Python >= 3.10
    - pathlib, argparse
    - python_toolbox.data_schema, python_toolbox.file, python_toolbox.registry
"""
from __future__ import annotations
import argparse
import types as _types
from dataclasses import dataclass, fields, MISSING
from pathlib import Path
from typing import Any, ClassVar, TypeVar, Union, get_type_hints, get_origin, get_args

from ..data_schema import Data_Schema
from ..registry import Registry
from ..file import Read_from, Write_to


@dataclass
class Base_Config(Data_Schema):
    config_type: str = ""
    object_type: str = ""
    __exclude_extract__: ClassVar[set[str]] = {"config_type", "object_type"}

    """파일 I/O 진입점을 보유한 Data_Schema 특화 클래스.

    설정/구성 데이터를 JSON/YAML로 저장하거나, Registry와 설정 파일에서
    직접 객체를 구성하는 용도에 적합함. 데이터 스키마로서의 역할
    (Serialize/Extract)은 Data_Schema에서 상속받음.

    ## 사용 패턴
    - `cfg.Write_to(name, dir)` — 현재 상태를 파일로 저장.
    - `Build_sub_config(ctx, Expected, registry, key="path")` — Registry 기반 객체 구성.
    - `Build_parser_from_config(MyConfig)` — 필드 기반 ArgumentParser 생성.
    """

    def Write_to(
        self, name: str, save_dir: str | Path, encoding_type: str = "UTF-8"
    ) -> None:
        """현재 상태를 Serialize 결과로 파일에 저장함.

        Args:
            name: 저장 파일명 (확장자 포함, 예: "config.json").
            save_dir: 저장 디렉토리 경로.
            encoding_type: 파일 인코딩 (기본 UTF-8).
        """
        Write_to(Path(save_dir) / name, self.Serialize(), encoding_type)


C_Type = TypeVar("C_Type", bound=Base_Config)


def Build_sub_config(
    context: dict[str, Any],
    expected: type[C_Type],
    registry: Registry,
    **key_with_file: str | None,
) -> C_Type:
    """Registry와 파일 또는 키를 기반으로 설정 객체를 생성하고 반환함.

    Args:
        context: 설정 객체에 주입할 컨텍스트 데이터.
        expected: 반환될 설정 객체의 상위 타입.
        registry: 설정 클래스가 등록된 레지스트리.
        **key_with_file: 단일 기본 설정 키와 파일 경로 (예: key="path/to/file").

    Returns:
        초기화된 설정(Configuration) 객체 인스턴스.

    Raises:
        ValueError: key_with_file 인자가 비어 있는 경우.
        KeyError: Registry에서 키를 찾을 수 없는 경우.
        TypeError: Registry에서 반환된 타입이 expected와 일치하지 않는 경우.
    """
    if not key_with_file:
        raise ValueError("key_with_file 인자가 비어 있음.")

    _cfg_key, _file_path = next(iter(key_with_file.items()))
    _cfg: type[C_Type] | None = None
    _meta: dict[str, Any] = {}

    if _file_path and (_path := Path(_file_path)).exists():
        _is_ok, _meta = Read_from(_path)
        if _is_ok and isinstance(_meta, dict):
            _type_key = _meta.get("config_type", "")
            if _type_key:
                _cfg = registry.Get(_type_key, expected)

    if _cfg is None:
        _cfg = registry.Get(_cfg_key, expected)
        _meta = {}

    _valid = {f.name for f in fields(_cfg)} & context.keys()
    return _cfg(**(_meta | {k: context[k] for k in _valid}))


def Build_parser_from_config(
    config_type: type[Base_Config],
    parser: argparse.ArgumentParser | None = None,
) -> argparse.ArgumentParser:
    """Base_Config 필드 정보로부터 ArgumentParser를 구성함.

    __exclude_extract__ 필드는 등록에서 제외됨.
    bool → BooleanOptionalAction, list → nargs, dict → str(파일 경로),
    Optional[X] / X | None → 내부 타입으로 unwrap.

    Args:
        config_type: 파싱 기준이 될 Base_Config 자식 클래스.
        parser: 기존 파서에 인자를 추가할 경우 전달. None이면 새로 생성.

    Returns:
        필드 기반으로 인자가 등록된 ArgumentParser.
    """
    if parser is None:
        parser = argparse.ArgumentParser()

    _hints = get_type_hints(config_type)
    _exclude = getattr(config_type, "__exclude_extract__", set())

    for f in fields(config_type):
        if f.name in _exclude:
            continue

        _type = _hints.get(f.name, str)
        _has_default = not (f.default is MISSING and f.default_factory is MISSING)  # type: ignore[misc]
        _default = (
            f.default if f.default is not MISSING
            else f.default_factory() if f.default_factory is not MISSING  # type: ignore[misc]
            else None
        )

        # Unwrap Optional[X] / X | None
        _origin = get_origin(_type)
        if _origin is Union or isinstance(_type, _types.UnionType):
            _inner = [a for a in get_args(_type) if a is not type(None)]
            _type = _inner[0] if _inner else str
            _origin = get_origin(_type)

        kwargs: dict[str, Any] = (
            {"default": _default} if _has_default else {"required": True}
        )

        if _type is bool:
            kwargs["action"] = argparse.BooleanOptionalAction
        elif _origin is list:
            _inner_args = get_args(_type)
            kwargs["type"] = _inner_args[0] if _inner_args else str
            kwargs["nargs"] = "*" if _has_default else "+"
        elif _origin is dict:
            kwargs["type"] = str
        else:
            kwargs["type"] = _type

        parser.add_argument(f"--{f.name}", **kwargs)

    return parser
