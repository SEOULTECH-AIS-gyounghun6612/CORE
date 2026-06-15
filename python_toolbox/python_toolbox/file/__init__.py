"""Unified file I/O package for text, JSON, and YAML handlers."""
from ._base import (
    Handle_exp,
    File_Process,
    Suffix_check,
    BASIC_FILE_ERROR,
    JSON_FILE_READ_ERROR,
)
from ._text import Text
from ._json import Json
from ._yaml import Yaml
from .dispatch import Read_from, Write_to
from .group import Make_the_file_group


__all__ = [
    "Handle_exp",
    "File_Process",
    "Suffix_check",
    "BASIC_FILE_ERROR",
    "JSON_FILE_READ_ERROR",
    "Text",
    "Json",
    "Yaml",
    "Read_from",
    "Write_to",
    "Make_the_file_group",
]
