from __future__ import annotations

import numpy as np
import pytest

from spatial_toolbox.render.openGL.utils.color import encode_id_to_color
from spatial_toolbox.render.openGL.utils.matrix import to_gl_matrix


def test_to_gl_matrix_identity_unchanged():
    assert np.allclose(to_gl_matrix(np.eye(4)), np.eye(4))


def test_to_gl_matrix_dtype_float32():
    assert to_gl_matrix(np.eye(4, dtype=np.float64)).dtype == np.float32


def test_to_gl_matrix_shape_preserved():
    assert to_gl_matrix(np.eye(4)).shape == (4, 4)


def test_to_gl_matrix_c_contiguous():
    assert to_gl_matrix(np.eye(4)).flags.c_contiguous is True


def test_to_gl_matrix_transposes():
    _m = np.arange(16, dtype=np.float32).reshape(4, 4)
    assert np.allclose(to_gl_matrix(_m), _m.T)


def test_to_gl_matrix_translation_in_last_row():
    _m = np.eye(4, dtype=np.float32)
    _m[:3, 3] = [1.0, 2.0, 3.0]
    _gl = to_gl_matrix(_m)
    assert np.allclose(_gl[3, :3], [1.0, 2.0, 3.0])


def test_to_gl_matrix_double_transpose_restores():
    _m = np.arange(16, dtype=np.float32).reshape(4, 4)
    assert np.allclose(to_gl_matrix(to_gl_matrix(_m)).T, _m.T)


def test_encode_id_to_color_zero():
    assert encode_id_to_color(0) == (0, 0, 0)


def test_encode_id_to_color_one():
    assert encode_id_to_color(1) == (1, 0, 0)


def test_encode_id_to_color_max_red_byte():
    assert encode_id_to_color(255) == (255, 0, 0)


def test_encode_id_to_color_green_channel():
    assert encode_id_to_color(256) == (0, 1, 0)


def test_encode_id_to_color_blue_channel():
    assert encode_id_to_color(65536) == (0, 0, 1)


def test_encode_id_to_color_max_24bit():
    assert encode_id_to_color(16777215) == (255, 255, 255)


def test_encode_id_to_color_mixed_channels():
    assert encode_id_to_color(0x123456) == (0x56, 0x34, 0x12)


def test_encode_id_to_color_returns_three_tuple():
    _c = encode_id_to_color(7)
    assert isinstance(_c, tuple)
    assert len(_c) == 3


def test_encode_id_to_color_values_in_byte_range():
    _c = encode_id_to_color(123456)
    assert all(0 <= _v <= 255 for _v in _c)


def _depth_linearize(raw: np.ndarray | float, near: float, far: float):
    _raw = np.asarray(raw, dtype=np.float32)
    _linear = (2.0 * near * far) / (far + near - _raw * (far - near))
    _linear[_raw >= 1.0] = 0.0
    return _linear


def test_depth_background_masked_at_one():
    assert _depth_linearize(np.array([1.0], dtype=np.float32), 0.1, 10.0)[0] == 0.0


def test_depth_background_masked_above_one():
    assert _depth_linearize(np.array([1.1], dtype=np.float32), 0.1, 10.0)[0] == 0.0


def test_depth_formula_near_plane():
    _v = _depth_linearize(np.array([-1.0], dtype=np.float32), 0.1, 10.0)[0]
    assert _v == pytest.approx(0.1)


def test_depth_formula_far_boundary():
    _v = _depth_linearize(np.array([1.0 - 1e-6], dtype=np.float32), 0.1, 10.0)[0]
    assert _v > 0.0


def test_depth_formula_monotonic():
    _a = _depth_linearize(np.array([0.0], dtype=np.float32), 0.1, 10.0)[0]
    _b = _depth_linearize(np.array([0.5], dtype=np.float32), 0.1, 10.0)[0]
    assert _b > _a


def test_depth_formula_output_shape():
    _raw = np.zeros((2, 3), dtype=np.float32)
    assert _depth_linearize(_raw, 0.1, 10.0).shape == (2, 3)


def test_depth_formula_positive_in_valid_range():
    _v = _depth_linearize(np.array([0.0], dtype=np.float32), 0.1, 10.0)[0]
    assert _v > 0.0
