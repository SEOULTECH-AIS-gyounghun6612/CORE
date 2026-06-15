from __future__ import annotations

import numpy as np
import pytest
import trimesh

from spatial_toolbox.scene.asset.cache import ASSET_CACHE
from spatial_toolbox.scene.asset.type.mesh import Mesh as Mesh_Asset
from spatial_toolbox.scene.node.type.camera import Camera
from spatial_toolbox.scene.stage import Controller

_MESH_KEY = "/tmp/render_mesh.obj"
_VERTS = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float64)
_FACES = np.array([[0, 1, 2]], dtype=np.int64)
RENDER_W = 320
RENDER_H = 240


@pytest.fixture
def egl_renderer():
    OpenGL_Renderer = pytest.importorskip(
        "spatial_toolbox.render.openGL.renderer"
    ).OpenGL_Renderer
    _r = OpenGL_Renderer(RENDER_W, RENDER_H)
    with _r:
        yield _r


@pytest.fixture
def scene():
    ASSET_CACHE.Clear()
    _geo = trimesh.Trimesh(vertices=_VERTS.copy(), faces=_FACES)
    ASSET_CACHE.Register(_MESH_KEY, Mesh_Asset(label="mesh", source_path=_MESH_KEY, geometry=_geo))
    _ctrl = Controller()
    _ctrl.Add_node(_ctrl.Build_node_from_cache(_MESH_KEY))
    _cam = Camera(label="main_camera")
    _m = np.eye(4, dtype=np.float32)
    _m[2, 3] = 2.0
    _cam.local_rigid = _m
    _cam.Set_parent(_ctrl.root)
    _ctrl.root.children.append(_cam)
    return _ctrl
