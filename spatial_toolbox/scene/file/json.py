"""JSON scene import and export built on top of ``Data_Schema``."""
from __future__ import annotations

from pathlib import Path
from typing import Callable

from python_toolbox.file import Read_from, Write_to

from ._base import Scene_State
from ..node.register import NODE_REGISTRY
from ..node.type._base import Base_Node


def Export_to_json(file_path: str | Path, root: Base_Node, unit_length: float) -> None:
    """Writes a scene graph and unit scale to JSON."""
    Write_to(Path(file_path), {"unit_length": unit_length, "root": root.Serialize()})


def Import_from_json(
    file_path: str | Path,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> Scene_State:
    """Loads a scene graph and unit scale from JSON."""
    if progress_callback is not None:
        progress_callback(1, 2, "씬 JSON 읽는 중...")

    _is_ok, _data = Read_from(Path(file_path))
    if not _is_ok or not isinstance(_data, dict):
        raise ValueError(f"장면 파일 읽기 실패: {file_path}")

    _root_data = _data.get("root")
    if not isinstance(_root_data, dict):
        raise ValueError(f"장면 파일 포맷 오류 (root 누락): {file_path}")

    if progress_callback is not None:
        progress_callback(2, 2, "씬 계층 재구성 중...")

    _root = _Build_node(_root_data, parent=None)
    return Scene_State(root=_root, unit_length=float(_data.get("unit_length", 1.0)))


def _Build_node(data: dict, parent: Base_Node | None) -> Base_Node:
    """Recursively rebuilds a node subtree from serialized JSON data."""
    _args = dict(data)
    _prim_type = _args.pop("prim_type", "Xform")
    _children_data = _args.pop("children", [])

    _node_cls = NODE_REGISTRY.Get(_prim_type)
    _args["prim_type"] = _prim_type
    _args["parent"] = parent
    _node = _node_cls(**_args)

    for _child_data in _children_data:
        _child = _Build_node(_child_data, parent=_node)
        _node.children.append(_child)
    return _node
