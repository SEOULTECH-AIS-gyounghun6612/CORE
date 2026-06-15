from __future__ import annotations

from spatial_toolbox.render.core.channel import DEPTH, NORMAL, RGB, SEGMENTATION
from spatial_toolbox.render.openGL.passes import PASS_TYPES
from spatial_toolbox.render.openGL.passes.depth import Depth_Pass
from spatial_toolbox.render.openGL.passes.normal import Normal_Pass
from spatial_toolbox.render.openGL.passes.rgb import RGB_Pass
from spatial_toolbox.render.openGL.passes.segmentation import Segmentation_Pass


def test_pass_types_covers_all_channels():
    assert set(PASS_TYPES) == {RGB, DEPTH, NORMAL, SEGMENTATION}


def test_pass_types_class_mapping():
    assert PASS_TYPES[RGB] is RGB_Pass
    assert PASS_TYPES[DEPTH] is Depth_Pass
    assert PASS_TYPES[NORMAL] is Normal_Pass
    assert PASS_TYPES[SEGMENTATION] is Segmentation_Pass


def test_rgb_pass_name():
    assert RGB_Pass().Name == "rgb"


def test_rgb_pass_uses_lighting():
    assert RGB_Pass()._use_lighting is True


def test_rgb_pass_readback_format():
    assert RGB_Pass()._readback_format == "rgb"


def test_rgb_pass_clear_color():
    assert RGB_Pass()._clear_color == (0.0, 0.0, 0.0, 1.0)


def test_depth_pass_name():
    assert Depth_Pass().Name == "depth"


def test_depth_pass_no_lighting():
    assert Depth_Pass()._use_lighting is False


def test_depth_pass_readback_format():
    assert Depth_Pass()._readback_format == "depth"


def test_normal_pass_name():
    assert Normal_Pass().Name == "normal"


def test_normal_pass_no_lighting():
    assert Normal_Pass()._use_lighting is False


def test_normal_pass_draw_mode():
    assert Normal_Pass()._draw_mode == "normal_color"


def test_normal_pass_clear_color_blue_tinted():
    assert Normal_Pass()._clear_color == (0.5, 0.5, 1.0, 1.0)


def test_segmentation_pass_name():
    assert Segmentation_Pass().Name == "segmentation"


def test_segmentation_pass_no_lighting():
    assert Segmentation_Pass()._use_lighting is False


def test_segmentation_pass_draw_mode():
    assert Segmentation_Pass()._draw_mode == "id_color"


def test_segmentation_pass_readback_format():
    assert Segmentation_Pass()._readback_format == "rgb"


def test_pass_name_property_returns_class_name():
    assert RGB_Pass().Name == RGB_Pass.name
