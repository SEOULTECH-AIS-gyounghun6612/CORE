"""Mesh drawing helpers for the OpenGL backend."""
from __future__ import annotations

import weakref
from typing import Any

import numpy as np
from OpenGL.GL import (
    GL_BLEND,
    GL_COLOR_ARRAY,
    GL_ENABLE_BIT,
    GL_FLOAT,
    GL_LIGHTING,
    GL_NORMAL_ARRAY,
    GL_TRIANGLES,
    GL_UNSIGNED_BYTE,
    GL_UNSIGNED_INT,
    GL_VERTEX_ARRAY,
    glColor3ub,
    glColorPointer,
    glDisable,
    glDisableClientState,
    glDrawElements,
    glEnableClientState,
    glNormalPointer,
    glPopAttrib,
    glPushAttrib,
    glVertexPointer,
)

from ..utils import create_mesh_vbos, encode_id_to_color


class Mesh_Drawer:
    """Draws mesh geometry with optional VBO caching and ID colors."""

    def __init__(self) -> None:
        self._vbo_cache: dict[str, dict] = {}
        self._id_map: dict[tuple, weakref.ReferenceType] = {}
        self._node_ids: dict[int, tuple] = {}
        self._id_counter: int = 1

    def _Sync_vbo(self, mesh: Any, key: str | None) -> dict | None:
        """Returns cached mesh VBOs, creating them on first use."""
        if key is None or not (mesh and hasattr(mesh, "vertices")):
            return None
        if key not in self._vbo_cache and (vbos := create_mesh_vbos(mesh)):
            self._vbo_cache[key] = vbos
        return self._vbo_cache.get(key)

    def _Setup_id_pass(self, node: Any) -> None:
        """Configures OpenGL state and color for segmentation drawing."""
        glPushAttrib(GL_ENABLE_BIT)
        glDisable(GL_LIGHTING)
        glDisable(GL_BLEND)

        _node_key = id(node) if node else None
        if _node_key and _node_key in self._node_ids:
            _color = self._node_ids[_node_key]
        else:
            _color = encode_id_to_color(self._id_counter)
            self._id_counter += 1
            if _node_key:
                self._node_ids[_node_key] = _color
                self._id_map[_color] = weakref.ref(node)
        glColor3ub(*_color)

    @staticmethod
    def _Toggle_client_states(enable: bool, use_color: bool, use_normal: bool) -> None:
        """Enables or disables vertex-array client states for a draw call."""
        _act = glEnableClientState if enable else glDisableClientState
        _act(GL_VERTEX_ARRAY)
        if use_color:
            _act(GL_COLOR_ARRAY)
        elif use_normal:
            _act(GL_NORMAL_ARRAY)

    @staticmethod
    def _Execute_vbo_draw(vbos: dict, use_color: bool, use_normal: bool) -> None:
        """Draws a mesh using VBO-backed vertex arrays."""
        vbos["vertices"].bind()
        glVertexPointer(3, GL_FLOAT, 0, vbos["vertices"])

        _aux = None
        if use_color and "normal_colors" in vbos:
            vbos["normal_colors"].bind()
            glColorPointer(3, GL_UNSIGNED_BYTE, 0, vbos["normal_colors"])
            _aux = "normal_colors"
        elif use_normal:
            vbos["normals"].bind()
            glNormalPointer(GL_FLOAT, 0, vbos["normals"])
            _aux = "normals"

        vbos["faces"].bind()
        glDrawElements(GL_TRIANGLES, vbos["face_count"], GL_UNSIGNED_INT, None)
        vbos["faces"].unbind()
        vbos["vertices"].unbind()
        if _aux:
            vbos[_aux].unbind()

    @staticmethod
    def _Execute_fallback_draw(mesh: Any, use_color: bool, use_normal: bool) -> None:
        """Draws a mesh directly from contiguous numpy arrays."""
        _verts = np.ascontiguousarray(mesh.vertices, dtype=np.float32)
        glVertexPointer(3, GL_FLOAT, 0, _verts)

        if use_color:
            _colors = ((mesh.vertex_normals + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
            glColorPointer(3, GL_UNSIGNED_BYTE, 0, np.ascontiguousarray(_colors))
        elif use_normal:
            _normals = np.ascontiguousarray(mesh.vertex_normals, dtype=np.float32)
            glNormalPointer(GL_FLOAT, 0, _normals)

        _faces = np.ascontiguousarray(mesh.faces, dtype=np.uint32)
        glDrawElements(GL_TRIANGLES, len(_faces) * 3, GL_UNSIGNED_INT, _faces)

    def Draw(self, mesh: Any, mode: str = "default", node: Any = None, use_vbo: bool = True) -> None:
        """Draws one mesh in the requested shading mode."""
        if not (mesh and hasattr(mesh, "vertices") and hasattr(mesh, "faces")):
            return

        _is_id = mode == "id_color"
        if _is_id:
            self._Setup_id_pass(node)

        _cache_key = getattr(node, "source_key", None) if node else None
        _vbos = self._Sync_vbo(mesh, _cache_key) if use_vbo else None

        _has_normals = hasattr(mesh, "vertex_normals") and mesh.vertex_normals is not None
        _use_color = mode == "normal_color" and _has_normals
        _use_normal = _has_normals and not _is_id

        self._Toggle_client_states(True, _use_color, _use_normal)
        if _vbos:
            self._Execute_vbo_draw(_vbos, _use_color, _use_normal)
        else:
            self._Execute_fallback_draw(mesh, _use_color, _use_normal)
        self._Toggle_client_states(False, _use_color, _use_normal)

        if _is_id:
            glPopAttrib()

    def Clear_resources(self) -> None:
        """Deletes cached VBO resources."""
        for _vbos in self._vbo_cache.values():
            for _vbo in _vbos.values():
                getattr(_vbo, "delete", lambda: None)()
        self._vbo_cache.clear()

    def Reset_id_state(self) -> None:
        """Clears cached node-color assignments."""
        self._id_map.clear()
        self._node_ids.clear()
        self._id_counter = 1

    @property
    def id_map(self) -> dict[tuple, Any]:
        """Returns live node references keyed by encoded ID color."""
        return {_color: _ref() for _color, _ref in self._id_map.items() if _ref() is not None}
