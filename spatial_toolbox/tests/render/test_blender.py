from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from spatial_toolbox.render.blender.renderer import Blender_Renderer
from spatial_toolbox.render.blender.session import Blender_Session
from spatial_toolbox.render.core.channel import DEPTH, NORMAL, RGB, SEGMENTATION
from spatial_toolbox.render.core.renderer import Render_Request


def test_validate_request_all_known_channels():
    _r = Blender_Renderer()
    _r._Validate_request(Render_Request([RGB, DEPTH, NORMAL, SEGMENTATION]))


def test_validate_request_single_channel():
    _r = Blender_Renderer()
    _r._Validate_request(Render_Request([RGB]))


def test_validate_request_unknown_raises():
    _r = Blender_Renderer()
    with pytest.raises(KeyError):
        _r._Validate_request(Render_Request(["weird"]))


def test_validate_request_mixed_raises():
    _r = Blender_Renderer()
    with pytest.raises(KeyError):
        _r._Validate_request(Render_Request([RGB, "weird"]))


def test_empty_result_contains_requested_channels():
    _res = Blender_Renderer.Empty_result(Render_Request([RGB, DEPTH]))
    assert set(_res.images) == {RGB, DEPTH}


def test_empty_result_arrays_are_empty():
    _res = Blender_Renderer.Empty_result(Render_Request([RGB, NORMAL]))
    assert _res.images[RGB].size == 0
    assert _res.images[NORMAL].size == 0


def test_empty_result_dtype_uint8():
    _res = Blender_Renderer.Empty_result(Render_Request([RGB]))
    assert _res.images[RGB].dtype == np.uint8


def test_empty_result_metadata_empty():
    _res = Blender_Renderer.Empty_result(Render_Request([RGB]))
    assert _res.metadata == {}


def test_camera_object_name_format():
    from spatial_toolbox.render.blender.scene.camera import Camera_object_name
    assert Camera_object_name("main") == "focus_camera__main"


def test_sanitize_usd_name_valid():
    from spatial_toolbox.render.blender.scene.segmentation import Sanitize_usd_name
    assert Sanitize_usd_name("hello") == "hello"


def test_sanitize_usd_name_spaces_replaced():
    from spatial_toolbox.render.blender.scene.segmentation import Sanitize_usd_name
    assert Sanitize_usd_name("hello world") == "hello_world"


def test_sanitize_usd_name_leading_digit_prefixed():
    from spatial_toolbox.render.blender.scene.segmentation import Sanitize_usd_name
    assert Sanitize_usd_name("1abc") == "_1abc"


def test_sanitize_usd_name_special_chars_replaced():
    from spatial_toolbox.render.blender.scene.segmentation import Sanitize_usd_name
    assert Sanitize_usd_name("a:b/c") == "a_b_c"


def test_build_segmentation_metadata_structure():
    from spatial_toolbox.render.blender.scene.segmentation import Build_segmentation_metadata
    from spatial_toolbox.scene.node.type.mesh import Mesh as Mesh_Node
    _node = Mesh_Node(label="m")
    _meta = Build_segmentation_metadata({1: _node})
    assert _meta["id_map"][1] is _node
    assert _meta["label_map"][1] == "m"


def test_build_segmentation_metadata_empty():
    from spatial_toolbox.render.blender.scene.segmentation import Build_segmentation_metadata
    _meta = Build_segmentation_metadata({})
    assert _meta["id_map"] == {}
    assert _meta["label_map"] == {}


def test_build_target_segmentation_metadata_structure():
    from spatial_toolbox.render.blender.scene.segmentation import Build_target_segmentation_metadata
    from spatial_toolbox.scene.node.type.mesh import Mesh as Mesh_Node
    _node = Mesh_Node(label="target")
    _meta = Build_target_segmentation_metadata(_node)
    assert _meta["id_map"][1] is _node
    assert _meta["label_map"][1] == "target"


class _StubObj:
    def __init__(self, name: str, type_: str, parent=None):
        self.name = name
        self.type = type_
        self.parent = parent


def test_resolve_segmentation_object_name_exact_mesh_match():
    from spatial_toolbox.render.blender.scene.segmentation import (
        Resolve_node_mesh_objects,
        Resolve_segmentation_object_name,
    )
    from spatial_toolbox.scene.node.type.mesh import Mesh as Mesh_Node

    _node = Mesh_Node(label="target")
    _bpy = MagicMock()
    _bpy.data.objects = [_StubObj("target", "MESH")]

    assert [o.name for o in Resolve_node_mesh_objects(_bpy, _node)] == ["target"]
    assert Resolve_segmentation_object_name(_bpy, _node) == "target"


def test_resolve_segmentation_object_name_from_parent_xform_descendants():
    from spatial_toolbox.render.blender.scene.segmentation import (
        Resolve_node_mesh_objects,
        Resolve_segmentation_object_name,
    )
    from spatial_toolbox.scene.node.type.mesh import Mesh as Mesh_Node

    _node = Mesh_Node(label="target")
    _root = _StubObj("target", "EMPTY")
    _mesh = _StubObj("proto_mesh", "MESH", parent=_root)
    _bpy = MagicMock()
    _bpy.data.objects = [_root, _mesh]

    assert [o.name for o in Resolve_node_mesh_objects(_bpy, _node)] == ["proto_mesh"]
    assert Resolve_segmentation_object_name(_bpy, _node) == "proto_mesh"


def test_resolve_segmentation_object_name_returns_all_descendant_meshes():
    from spatial_toolbox.render.blender.scene.segmentation import (
        Resolve_node_mesh_objects,
        Resolve_segmentation_object_name,
    )
    from spatial_toolbox.scene.node.type.mesh import Mesh as Mesh_Node

    _node = Mesh_Node(label="target")
    _root = _StubObj("target", "EMPTY")
    _mesh_a = _StubObj("proto_a", "MESH", parent=_root)
    _mesh_b = _StubObj("proto_b", "MESH", parent=_root)
    _bpy = MagicMock()
    _bpy.data.objects = [_root, _mesh_b, _mesh_a]

    assert [o.name for o in Resolve_node_mesh_objects(_bpy, _node)] == ["proto_a", "proto_b"]
    assert Resolve_segmentation_object_name(_bpy, _node) == "proto_a,proto_b"


def test_session_bpy_returns_real_bpy():
    pytest.importorskip("bpy")
    _session = Blender_Session()
    assert _session.bpy is not None


def test_setup_accesses_bpy_without_error():
    _r = Blender_Renderer()
    _r._session._bpy = _make_mock_bpy()
    _r.Setup()


def test_teardown_noop():
    _r = Blender_Renderer()
    assert _r.Teardown() is None


def _make_mock_bpy():
    _m = MagicMock()
    _m.context.scene.render.engine = "BLENDER_EEVEE"
    _m.data.objects = {}
    return _m


def test_require_scene_bridge_returns_instance():
    _r = Blender_Renderer()
    _r._session._bpy = _make_mock_bpy()
    assert _r._Require_scene_bridge() is not None


def test_require_scene_bridge_singleton():
    _r = Blender_Renderer()
    _r._session._bpy = _make_mock_bpy()
    assert _r._Require_scene_bridge() is _r._Require_scene_bridge()


def test_render_validates_channels_before_bridge():
    from spatial_toolbox.scene.stage import Controller
    _r = Blender_Renderer()
    _r._session._bpy = _make_mock_bpy()
    with pytest.raises(KeyError):
        _r.Render(Controller(), [], Render_Request(["weird"]))


def test_bpy_has_context():
    bpy = pytest.importorskip("bpy")
    assert hasattr(bpy, "context")


def test_bpy_has_data():
    bpy = pytest.importorskip("bpy")
    assert hasattr(bpy, "data")


def test_bpy_has_ops():
    bpy = pytest.importorskip("bpy")
    assert hasattr(bpy, "ops")
