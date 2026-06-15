from __future__ import annotations

import numpy as np
import pytest

from spatial_toolbox.render.core.camera import Resolve_cameras
from spatial_toolbox.render.core.channel import DEPTH, NORMAL, RGB, SEGMENTATION
from spatial_toolbox.render.core.renderer import Render_Request, Render_Result
from spatial_toolbox.scene.node.type.camera import Camera
from spatial_toolbox.scene.node.type.group import Group
from spatial_toolbox.scene.stage import Controller


def test_channel_rgb_value():
    assert RGB == "rgb"


def test_channel_depth_value():
    assert DEPTH == "depth"


def test_channel_normal_value():
    assert NORMAL == "normal"


def test_channel_segmentation_value():
    assert SEGMENTATION == "segmentation"


def test_channel_values_distinct():
    assert len({RGB, DEPTH, NORMAL, SEGMENTATION}) == 4


def test_render_request_stores_channels():
    _req = Render_Request([RGB, DEPTH])
    assert _req.channels == [RGB, DEPTH]


def test_render_request_single_channel():
    _req = Render_Request([RGB])
    assert len(_req.channels) == 1


def test_render_request_empty_channels():
    _req = Render_Request([])
    assert _req.channels == []


def test_render_result_default_images_empty():
    assert Render_Result().images == {}


def test_render_result_default_metadata_empty():
    assert Render_Result().metadata == {}


def test_render_result_get_image_returns_array():
    _arr = np.zeros((2, 2, 3), dtype=np.uint8)
    _res = Render_Result(images={RGB: _arr})
    assert _res.Get_image(RGB) is _arr


def test_render_result_get_image_missing_raises():
    with pytest.raises(KeyError):
        Render_Result().Get_image(RGB)


def test_render_result_get_metadata_present():
    _res = Render_Result(metadata={RGB: {"a": 1}})
    assert _res.Get_metadata(RGB) == {"a": 1}


def test_render_result_get_metadata_missing_returns_empty():
    assert Render_Result().Get_metadata(RGB) == {}


@pytest.fixture
def ctrl() -> Controller:
    return Controller()


def _add_camera(parent, label):
    _cam = Camera(label=label)
    _cam.Set_parent(parent)
    parent.children.append(_cam)
    return _cam


def _add_group(parent, label):
    _grp = Group(label=label)
    _grp.Set_parent(parent)
    parent.children.append(_grp)
    return _grp


def test_resolve_cameras_empty_labels_raises(ctrl):
    with pytest.raises(ValueError):
        Resolve_cameras(ctrl, [])


def test_resolve_cameras_unknown_label_raises(ctrl):
    with pytest.raises(KeyError):
        Resolve_cameras(ctrl, ["ghost"])


def test_resolve_cameras_direct_camera(ctrl):
    _add_camera(ctrl.root, "cam")
    _result = Resolve_cameras(ctrl, ["cam"])
    assert len(_result) == 1
    assert _result[0].label == "cam"


def test_resolve_cameras_returns_camera_type(ctrl):
    _add_camera(ctrl.root, "cam")
    _result = Resolve_cameras(ctrl, ["cam"])
    assert isinstance(_result[0], Camera)


def test_resolve_cameras_group_yields_children(ctrl):
    _grp = _add_group(ctrl.root, "grp")
    _add_camera(_grp, "cam_a")
    _add_camera(_grp, "cam_b")
    _result = Resolve_cameras(ctrl, ["grp"])
    assert len(_result) == 2


def test_resolve_cameras_deduplicates_repeated_labels(ctrl):
    _add_camera(ctrl.root, "cam")
    _result = Resolve_cameras(ctrl, ["cam", "cam"])
    assert len(_result) == 1


def test_resolve_cameras_group_no_cameras_raises(ctrl):
    _add_group(ctrl.root, "grp")
    with pytest.raises(KeyError):
        Resolve_cameras(ctrl, ["grp"])


def test_resolve_cameras_preserves_order(ctrl):
    _add_camera(ctrl.root, "cam_a")
    _add_camera(ctrl.root, "cam_b")
    _result = Resolve_cameras(ctrl, ["cam_b", "cam_a"])
    assert [_c.label for _c in _result] == ["cam_b", "cam_a"]
