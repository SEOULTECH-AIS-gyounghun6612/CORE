"""OBJ asset loading helpers."""
from __future__ import annotations

from pathlib import Path

import trimesh

from ..type.mesh import Mesh


def _Fix_normals(source: trimesh.Geometry) -> None:
    """Repairs normals and vertex references on loaded trimesh data."""
    if isinstance(source, trimesh.Trimesh):
        source.remove_unreferenced_vertices()
        source.merge_vertices()
        source.fix_normals()
        trimesh.repair.fix_inversion(source)
        return

    if isinstance(source, trimesh.Scene):
        for _geo in source.geometry.values():
            _Fix_normals(_geo)


def Read_and_parse_obj(
    file_path: str | Path,
    unit_length: float = 1.0,
) -> list[Mesh]:
    """Loads an OBJ file and returns one or more mesh assets."""
    _path = Path(file_path).resolve()
    _loaded = trimesh.load(_path)
    _Fix_normals(_loaded)

    if isinstance(_loaded, trimesh.Trimesh):
        return [
            Mesh(
                label=_path.stem,
                source_path=str(_path),
                geometry=_loaded,
                unit_length=unit_length,
            )
        ]

    if isinstance(_loaded, trimesh.Scene):
        _assets: list[Mesh] = []
        for _name, _geo in _loaded.geometry.items():
            _assets.append(
                Mesh(
                    label=_name,
                    source_path=f"{_path}#{_name}",
                    geometry=_geo,
                    unit_length=unit_length,
                )
            )
        return _assets

    return []
