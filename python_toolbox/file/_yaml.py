"""YAML file reader and writer."""
from __future__ import annotations
from pathlib import Path
from typing import Any

import yaml

from ._base import File_Process, Handle_exp, Suffix_check


class Yaml(File_Process):
    """Handles YAML file persistence."""

    loader = yaml.FullLoader

    @classmethod
    @Handle_exp()
    def Read_from(
        cls, file: Path, enc: str = "UTF-8", **kwarg
    ) -> tuple[bool, Any]:
        """Reads a YAML file into a Python object.

        Args:
            file: Input YAML file path.
            enc: Text encoding.
            **kwarg: Unused compatibility arguments.

        Returns:
            A tuple of ``(is_ok, parsed_object)``.
        """
        _, _file = Suffix_check(file, ".yaml")

        if not _file.exists():
            return False, {}

        with _file.open(encoding=enc) as _f:
            return True, yaml.load(_f, cls.loader)

    @classmethod
    @Handle_exp()
    def Write_to(
        cls, file: Path, data: Any, enc: str = "UTF-8", indent: int = 4
    ) -> bool:
        """Writes data to a YAML file.

        Args:
            file: Output YAML file path.
            data: Data to serialize.
            enc: Text encoding.
            indent: YAML indentation level.

        Returns:
            ``True`` when the write succeeds.
        """
        cls.Ensure_dir(file)
        _, _path = Suffix_check(file, ".yaml", True)

        with _path.open(mode="w", encoding=enc) as _f:
            yaml.dump(data, _f, indent=indent, sort_keys=False)
        return True
