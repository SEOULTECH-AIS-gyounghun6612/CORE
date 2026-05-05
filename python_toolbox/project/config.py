"""Config helpers built on top of :class:`python_toolbox.data_schema.Data_Schema`."""
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
    """Schema base class with a file-writing convenience method.

    The class keeps config objects as ordinary :class:`Data_Schema` instances
    and only adds a small persistence entrypoint through :meth:`Write_to`.

    Attributes:
        config_type: Optional registry-facing type key stored with the config.
        object_type: Optional object category string stored with the config.
    """

    config_type: str = ""
    object_type: str = ""
    __exclude_extract__: ClassVar[set[str]] = {"config_type", "object_type"}

    def Write_to(
        self, name: str, save_dir: str | Path, encoding_type: str = "UTF-8"
    ) -> None:
        """Serializes the config and writes it to a file.

        Args:
            name: Output filename including suffix.
            save_dir: Directory that will contain the config file.
            encoding_type: Text encoding forwarded to the file writer.
        """
        Write_to(Path(save_dir) / name, self.Serialize(), encoding_type)


C_Type = TypeVar("C_Type", bound=Base_Config)


def Build_sub_config(
    context: dict[str, Any],
    expected: type[C_Type],
    registry: Registry,
    **key_with_file: str | None,
) -> C_Type:
    """Builds a config instance from registry metadata and runtime context.

    Args:
        context: Runtime values injected into the config constructor.
        expected: Expected base type returned from the registry.
        registry: Registry containing config classes.
        **key_with_file: A single mapping from fallback registry key to config
            file path.

    Returns:
        A config instance of the resolved type.

    Raises:
        ValueError: If no fallback key is provided.
        KeyError: If the resolved key is not registered.
        TypeError: If the resolved type does not match ``expected``.
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
    """Builds an ``ArgumentParser`` from ``Base_Config`` fields.

    Args:
        config_type: Config class used as the parser schema.
        parser: Existing parser to extend. If omitted, a new parser is created.

    Returns:
        An ``ArgumentParser`` populated from dataclass fields.
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
