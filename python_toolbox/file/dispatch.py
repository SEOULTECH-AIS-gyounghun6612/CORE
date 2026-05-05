"""Dispatches file I/O calls to handlers selected by filename suffix."""
from __future__ import annotations
from pathlib import Path
from typing import Any

from ._base import File_Process
from ._text import Text
from ._json import Json
from ._yaml import Yaml


_REGISTRY: dict[str, type[File_Process]] = {
    ".txt": Text,
    ".json": Json,
    ".yaml": Yaml,
}


def Read_from(file: Path, enc: str = "UTF-8", **kwarg) -> tuple[bool, Any]:
    """Reads a file with the handler registered for its suffix.

    Args:
        file: Path to the input file.
        enc: Text encoding passed to the handler.
        **kwarg: Extra keyword arguments forwarded to the handler.

    Returns:
        A tuple of `(is_ok, data)`.

    Raises:
        ValueError: If ``file`` is not a file or its suffix is unsupported.
    """
    if not Path.is_file(file):
        raise ValueError(f"!!! This path {file} is not FILE !!!")

    _processor = _REGISTRY.get(file.suffix.lower())
    if _processor is None:
        raise ValueError(
            f"File extension '{file.suffix}' is not supported"
        )

    return _processor.Read_from(file, enc, **kwarg)


def Write_to(
    file: Path, data: Any, enc: str = "UTF-8", **kwarg
) -> None:
    """Writes data with the handler registered for the target suffix.

    Args:
        file: Output file path.
        data: Data to write.
        enc: Text encoding passed to the handler.
        **kwarg: Extra keyword arguments forwarded to the handler.

    Raises:
        ValueError: If the suffix is unsupported for writing.
    """
    _processor = _REGISTRY.get(file.suffix.lower())
    if _processor is None:
        raise ValueError(
            f"File extension '{file.suffix}' is not supported for writing."
        )

    _processor.Write_to(file, data, enc, **kwarg)
