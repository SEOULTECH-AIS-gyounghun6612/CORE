"""VBO construction helpers for OpenGL mesh drawing."""
from typing import Any

import numpy as np
from OpenGL.GL import GL_ELEMENT_ARRAY_BUFFER
from OpenGL.arrays import vbo


def create_mesh_vbos(mesh: Any) -> dict | None:
    """Builds VBO buffers for a mesh when vertex and face arrays exist."""
    if mesh is None or not hasattr(mesh, "vertices") or not hasattr(mesh, "faces"):
        return None

    _vbos: dict[str, object] = {}
    _vbos["vertices"] = vbo.VBO(np.ascontiguousarray(mesh.vertices, dtype=np.float32))

    if hasattr(mesh, "vertex_normals") and mesh.vertex_normals is not None:
        _vbos["normals"] = vbo.VBO(
            np.ascontiguousarray(mesh.vertex_normals, dtype=np.float32)
        )
        _colors = np.ascontiguousarray(
            ((mesh.vertex_normals + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
        )
        _vbos["normal_colors"] = vbo.VBO(_colors)

    _vbos["faces"] = vbo.VBO(
        np.ascontiguousarray(mesh.faces, dtype=np.uint32),
        target=GL_ELEMENT_ARRAY_BUFFER,
    )
    _vbos["face_count"] = len(mesh.faces) * 3
    return _vbos
