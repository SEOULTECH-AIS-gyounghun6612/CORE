"""Shared helpers and base classes for file readers and writers."""
from __future__ import annotations
from abc import ABC, abstractmethod
from functools import wraps
from typing import TypeVar, Callable, cast, Any
from pathlib import Path

import json


F = TypeVar("F", bound=Callable)


def Handle_exp(extra_exp: dict[type[Exception], str] | None = None):
    """Builds a decorator that converts exceptions into `(False, None)`.

    Args:
        extra_exp: Mapping from exception type to message prefix.

    Returns:
        A decorator that wraps the target callable.
    """
    extra_exp = extra_exp or {}

    def Checker(func: F) -> F:

        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                _exp = type(e)
                _msg = extra_exp.get(_exp, "알 수 없는 파일 처리 오류 발생:")
                print(f"{_msg} -> {e}")
            return False, None
        return cast(F, wrapper)

    return Checker


BASIC_FILE_ERROR = {
    PermissionError: "파일 권한 부족"
}

JSON_FILE_READ_ERROR = {
    **BASIC_FILE_ERROR,
    json.JSONDecodeError: "JSON 파싱 오류 발생",
    TypeError: "JSON 변환 불가능한 데이터 포함"
}


def Suffix_check(
    path: Path, ext: list[str] | str, is_fix: bool = True
) -> tuple[bool, Path]:
    """Checks whether a path suffix matches the expected suffix set.

    Args:
        path: Path to validate.
        ext: Allowed suffix or suffix list.
        is_fix: Whether to replace a mismatched suffix with the first allowed
            suffix.

    Returns:
        A tuple of `(is_valid, path_or_fixed_path)`.
    """
    _ext = ext if isinstance(ext, list) else [ext]
    if path.suffix in _ext:
        return True, path

    return False, path.with_suffix(_ext[0]) if is_fix else path


class File_Process(ABC):
    """Abstract base class for format-specific file handlers."""

    @classmethod
    def Ensure_dir(cls, path: Path) -> None:
        """Creates the parent directory for a file path if needed.

        Args:
            path: Target file path.
        """
        path.parent.mkdir(parents=True, exist_ok=True)

    @classmethod
    @abstractmethod
    def Read_from(
        cls, *args, file: Path, enc: str = "UTF-8", **kwarg
    ) -> tuple[bool, Any]:
        """Reads data from a file."""
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def Write_to(
        cls, *args, file: Path, data: Any, enc: str = "UTF-8", **kwarg
    ) -> bool:
        """Writes data to a file."""
        raise NotImplementedError
