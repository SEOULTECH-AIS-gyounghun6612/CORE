from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from spatial_toolbox.render.core.channel import DEPTH, NORMAL, RGB, SEGMENTATION
from spatial_toolbox.render.core.renderer import Render_Result
from spatial_toolbox.scene.node.type.camera import Camera
from spatial_toolbox.scene.node.type.group import Group
from spatial_toolbox.scene.node.type.mesh import Mesh as Mesh_Node
from spatial_toolbox.scene.stage import Controller

_PATCH_RENDERER = "spatial_toolbox.simulation.blender.engine.Blender_Renderer"
_PATCH_SEED = "spatial_toolbox.simulation.blender.engine.np.random.seed"
_PATCH_SAMPLE = "spatial_toolbox.simulation.blender.engine.Sample_delta_matrix"


def make_renderer_return(camera_labels: list[str]):
    _result = Render_Result(
        images={
            RGB: __import__("numpy").zeros((2, 2, 3), dtype=__import__("numpy").uint8),
            DEPTH: __import__("numpy").zeros((2, 2), dtype=__import__("numpy").float32),
            NORMAL: __import__("numpy").zeros((2, 2, 3), dtype=__import__("numpy").uint8),
            SEGMENTATION: __import__("numpy").zeros((2, 2), dtype=__import__("numpy").int32),
        },
        metadata={},
    )
    return {label: _result for label in camera_labels}


@pytest.fixture
def ctrl_with_target():
    _ctrl = Controller()
    _target = Group(label="target")
    _target.Set_parent(_ctrl.root)
    _ctrl.root.children.append(_target)
    for label in ("obj_a", "obj_b"):
        _node = Mesh_Node(label=label, source_key=f"/tmp/{label}.obj")
        _node.Set_parent(_target)
        _target.children.append(_node)
    _cam = Camera(label="main_camera")
    _cam.Set_parent(_ctrl.root)
    _ctrl.root.children.append(_cam)
    return _ctrl, _target


def _mock_renderer(camera_labels):
    _mock = MagicMock()
    _mock.return_value.__enter__.return_value.Render.return_value = make_renderer_return(camera_labels)
    return _mock
