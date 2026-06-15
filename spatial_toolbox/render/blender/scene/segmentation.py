"""Segmentation helpers for Blender scene rendering."""
from __future__ import annotations

from pathlib import Path
import re as _re
from typing import Any

import numpy as np

from ....scene import Controller
from ....scene.node.type._base import Base_Node
from .rgb import _Load_pixels


def Sanitize_usd_name(label: str) -> str:
    """Normalizes a label into a Blender/USD-safe object name."""
    _name = _re.sub(r"[^A-Za-z0-9_]", "_", label) or "_node"
    if _name[0].isdigit():
        _name = f"_{_name}"
    return _name


def Assign_segmentation_indices(bpy: Any, scene_controller: Controller) -> dict[int, Base_Node]:
    """Assigns Blender ``pass_index`` values and records their node mapping."""
    _mesh_nodes = scene_controller.Get_render_queue()
    _targets: dict[str, list[Base_Node]] = {}
    for _node in _mesh_nodes:
        _targets.setdefault(Sanitize_usd_name(_node.label), []).append(_node)

    _index_to_node: dict[int, Base_Node] = {}
    _next_index = 1
    for _obj in bpy.data.objects:
        if getattr(_obj, "type", None) != "MESH":
            continue
        _base_name = str(_obj.name).split(".", 1)[0]
        _matched = _targets.get(_base_name)
        if not _matched:
            _obj.pass_index = 0
            continue
        _node = _matched.pop(0)
        _obj.pass_index = _next_index
        _index_to_node[_next_index] = _node
        _next_index += 1
    return _index_to_node


def _Base_name(obj: Any) -> str:
    """Returns a Blender object name without the duplicate numeric suffix."""
    return str(obj.name).split(".", 1)[0]


def _Is_descendant_of(obj: Any, ancestor: Any) -> bool:
    """Checks whether ``obj`` is the same object as ``ancestor`` or a child of it."""
    _cursor = obj
    while _cursor is not None:
        if _cursor is ancestor:
            return True
        _cursor = getattr(_cursor, "parent", None)
    return False


def Resolve_node_mesh_objects(bpy: Any, node: Base_Node) -> list[Any]:
    """Finds imported Blender mesh objects that correspond to one scene node subtree."""
    _target_name = Sanitize_usd_name(node.label)

    _roots = [_obj for _obj in bpy.data.objects if _Base_name(_obj) == _target_name]
    _meshes: list[Any] = []
    if _roots:
        for _obj in bpy.data.objects:
            if getattr(_obj, "type", None) != "MESH":
                continue
            if any(_Is_descendant_of(_obj, _root) for _root in _roots):
                _meshes.append(_obj)

    if _meshes:
        return sorted(_meshes, key=lambda o: str(o.name))

    for _obj in bpy.data.objects:
        if getattr(_obj, "type", None) != "MESH":
            continue
        if _Base_name(_obj) == _target_name:
            return [_obj]
    return []


def Resolve_segmentation_object_name(bpy: Any, node: Base_Node) -> str:
    """Finds Blender mesh object names that correspond to one scene node subtree."""
    _meshes = Resolve_node_mesh_objects(bpy, node)
    if _meshes:
        return ",".join(sorted({str(_obj.name) for _obj in _meshes}))
    raise KeyError(f"Blender object not found for segmentation target '{node.label}'")


def Load_segmentation(bpy: Any, path: Path, target_id: int = 1, threshold: float = 0.5) -> np.ndarray:
    """Loads a cryptomatte-derived matte into an integer ID image."""
    _rgba = _Load_pixels(bpy, path)
    _mask = _rgba[..., 0] > float(threshold)
    _seg = np.zeros(_mask.shape, dtype=np.int32)
    _seg[_mask] = int(target_id)
    return _seg


def Build_segmentation_metadata(index_to_node: dict[int, Base_Node]) -> dict[str, object]:
    """Builds metadata for the rendered segmentation pass."""
    return {
        "id_map": index_to_node,
        "label_map": {k: v.label for k, v in index_to_node.items()},
    }


def Build_target_segmentation_metadata(node: Base_Node, target_id: int = 1) -> dict[str, object]:
    """Builds metadata for a target-only segmentation map."""
    return Build_segmentation_metadata({int(target_id): node})
