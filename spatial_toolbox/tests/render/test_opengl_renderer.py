from __future__ import annotations

import numpy as np
import pytest

from spatial_toolbox.render.core.channel import DEPTH, NORMAL, RGB, SEGMENTATION
from spatial_toolbox.render.core.renderer import Render_Request


def test_embedded_context_setup_teardown_noop():
    from spatial_toolbox.render.openGL.context import Embedded_Context
    _ctx = Embedded_Context()
    assert _ctx.Setup() is None
    assert _ctx.Teardown() is None


def test_rgb_output_shape(egl_renderer, scene):
    _res = egl_renderer.Render(scene, ["main_camera"], Render_Request([RGB]))
    assert _res["main_camera"].images[RGB].shape[2] == 3


def test_rgb_output_dtype(egl_renderer, scene):
    _res = egl_renderer.Render(scene, ["main_camera"], Render_Request([RGB]))
    assert _res["main_camera"].images[RGB].dtype == np.uint8


def test_depth_output_shape(egl_renderer, scene):
    _res = egl_renderer.Render(scene, ["main_camera"], Render_Request([DEPTH]))
    assert _res["main_camera"].images[DEPTH].ndim == 2


def test_depth_output_dtype(egl_renderer, scene):
    _res = egl_renderer.Render(scene, ["main_camera"], Render_Request([DEPTH]))
    assert _res["main_camera"].images[DEPTH].dtype == np.float32


def test_normal_output_shape(egl_renderer, scene):
    _res = egl_renderer.Render(scene, ["main_camera"], Render_Request([NORMAL]))
    assert _res["main_camera"].images[NORMAL].shape[2] == 3


def test_normal_output_dtype(egl_renderer, scene):
    _res = egl_renderer.Render(scene, ["main_camera"], Render_Request([NORMAL]))
    assert _res["main_camera"].images[NORMAL].dtype == np.uint8


def test_segmentation_output_shape(egl_renderer, scene):
    _res = egl_renderer.Render(scene, ["main_camera"], Render_Request([SEGMENTATION]))
    assert _res["main_camera"].images[SEGMENTATION].shape[2] == 3


def test_segmentation_output_dtype(egl_renderer, scene):
    _res = egl_renderer.Render(scene, ["main_camera"], Render_Request([SEGMENTATION]))
    assert _res["main_camera"].images[SEGMENTATION].dtype == np.uint8


def test_segmentation_metadata_has_id_map(egl_renderer, scene):
    _res = egl_renderer.Render(scene, ["main_camera"], Render_Request([SEGMENTATION]))
    assert "id_map" in _res["main_camera"].metadata[SEGMENTATION]


def test_multichannel_render_returns_all(egl_renderer, scene):
    _res = egl_renderer.Render(scene, ["main_camera"], Render_Request([RGB, DEPTH]))
    assert set(_res["main_camera"].images) == {RGB, DEPTH}


def test_multicamera_returns_key_per_camera(egl_renderer, scene):
    from spatial_toolbox.scene.node.type.camera import Camera
    _cam = Camera(label="cam_b")
    _cam.Set_parent(scene.root)
    scene.root.children.append(_cam)
    _res = egl_renderer.Render(scene, ["main_camera", "cam_b"], Render_Request([RGB]))
    assert set(_res) == {"main_camera", "cam_b"}
