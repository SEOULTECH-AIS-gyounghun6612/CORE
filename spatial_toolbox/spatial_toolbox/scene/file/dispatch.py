"""Extension-based dispatch for scene import and export."""
from __future__ import annotations

from pathlib import Path
from typing import Callable

from ._base import Scene_State
from .json import Export_to_json, Import_from_json
from .usd import Export_to_usd, Import_from_usd
from ..node.type._base import Base_Node

Exporter = Callable[[str | Path, Base_Node, float], None]
Progress_Callback = Callable[[int, int, str], None]
Importer = Callable[[str | Path, Progress_Callback | None], Scene_State]

_EXPORT_REGISTRY: dict[str, Exporter] = {
    ".json": Export_to_json,
    ".usd": Export_to_usd,
    ".usda": Export_to_usd,
    ".usdc": Export_to_usd,
}

_IMPORT_REGISTRY: dict[str, Importer] = {
    ".json": Import_from_json,
    ".usd": Import_from_usd,
    ".usda": Import_from_usd,
    ".usdc": Import_from_usd,
}


def Export_to(file: str | Path, root: Base_Node, unit_length: float) -> None:
    """Exports a scene graph by selecting a writer from the file suffix."""
    _path = Path(file)
    _processor = _EXPORT_REGISTRY.get(_path.suffix.lower())
    if _processor is None:
        raise ValueError(f"지원하지 않는 장면 export 형식임: {_path.suffix}")
    _processor(_path, root, unit_length)


def Import_from(
    file: str | Path,
    progress_callback: Progress_Callback | None = None,
) -> Scene_State:
    """Imports a scene graph by selecting a reader from the file suffix."""
    _path = Path(file)
    _processor = _IMPORT_REGISTRY.get(_path.suffix.lower())
    if _processor is None:
        raise ValueError(f"지원하지 않는 장면 import 형식임: {_path.suffix}")
    return _processor(_path, progress_callback)
