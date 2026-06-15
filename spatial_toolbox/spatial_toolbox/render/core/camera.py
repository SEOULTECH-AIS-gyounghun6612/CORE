"""Helpers for resolving camera targets from a shared scene graph."""
from __future__ import annotations

from ...scene import Controller
from ...scene.node.type.camera import Camera
from ...scene.node.utils.traversal import walk_nodes


def Resolve_cameras(scene: Controller, camera_labels: list[str]) -> list[Camera]:
    """Resolves camera labels or camera-group labels into concrete cameras."""
    if not camera_labels:
        raise ValueError("camera_labels must not be empty")

    _node_by_label = {
        _node.label: _node
        for _node in walk_nodes(scene.root, lambda n: True)
    }
    _missing = [_label for _label in camera_labels if _label not in _node_by_label]
    if _missing:
        raise KeyError(f"Unknown camera label(s): {_missing}")

    _resolved: list[Camera] = []
    _seen_labels: set[str] = set()

    def _append_camera(_camera: Camera) -> None:
        if _camera.label in _seen_labels:
            return
        _resolved.append(_camera)
        _seen_labels.add(_camera.label)

    for _label in camera_labels:
        _node = _node_by_label[_label]
        if isinstance(_node, Camera):
            _append_camera(_node)
            continue

        _found = False
        for _desc in walk_nodes(_node, lambda n: isinstance(n, Camera)):
            _append_camera(_desc)
            _found = True
        if not _found:
            raise KeyError(f"Label '{_label}' does not resolve to any camera.")

    return _resolved
