from __future__ import annotations

import numpy as np
import pytest

from spatial_toolbox.simulation.core.config import (
    Physics_Drop_Config,
    Randomize_Range,
    Sim_Config,
    Sample_delta_matrix,
    Sample_translation,
    _Sample_value,
)


def test_randomize_range_defaults_all_zero():
    r = Randomize_Range()
    assert (r.tx, r.ty, r.tz, r.rx, r.ry, r.rz) == (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


def test_randomize_range_fixed_float():
    assert Randomize_Range(tx=1.5).tx == 1.5


def test_randomize_range_list_value():
    assert Randomize_Range(tx=[0.0, 1.0]).tx == [0.0, 1.0]


def test_sim_config_default_target_label():
    assert Sim_Config().target_label == "target"


def test_sim_config_default_camera_labels():
    assert Sim_Config().camera_labels == ["main_camera"]


def test_sim_config_default_num_samples():
    assert Sim_Config().num_samples == 1


def test_sim_config_default_output_layout():
    assert Sim_Config().output_layout == "per_object"


def test_sim_config_default_seed_is_none():
    assert Sim_Config().seed is None


def test_sim_config_default_cam_is_randomize_range():
    assert isinstance(Sim_Config().cam, Randomize_Range)


def test_sim_config_default_cam_overrides_empty():
    assert Sim_Config().cam_overrides == {}


def test_sim_config_default_obj_is_randomize_range():
    assert isinstance(Sim_Config().obj, Randomize_Range)


def test_sim_config_default_physics_drop_is_config():
    assert isinstance(Sim_Config().physics_drop, Physics_Drop_Config)


def test_sim_config_default_physics_drop_disabled():
    assert Sim_Config().physics_drop.enabled is False


def test_sim_config_cam_dict_coerced_to_randomize_range():
    assert isinstance(Sim_Config(cam={"tx": 1.0}).cam, Randomize_Range)


def test_sim_config_obj_dict_coerced_to_randomize_range():
    assert isinstance(Sim_Config(obj={"tx": 1.0}).obj, Randomize_Range)


def test_sim_config_physics_drop_dict_coerced():
    assert isinstance(
        Sim_Config(physics_drop={"enabled": True}).physics_drop,
        Physics_Drop_Config,
    )


def test_sim_config_camera_labels_str_coerced_to_list():
    assert Sim_Config(camera_labels="cam").camera_labels == ["cam"]


def test_sim_config_cam_overrides_dict_coerced():
    cfg = Sim_Config(cam_overrides={"cam": {"tx": 1.0}})
    assert isinstance(cfg.cam_overrides["cam"], Randomize_Range)


def test_sim_config_cam_overrides_range_passthrough():
    rr = Randomize_Range(tx=1.0)
    cfg = Sim_Config(cam_overrides={"cam": rr})
    assert cfg.cam_overrides["cam"] is rr


def test_sim_config_cam_overrides_multiple_keys():
    cfg = Sim_Config(cam_overrides={"a": {"tx": 1.0}, "b": {"ty": 2.0}})
    assert set(cfg.cam_overrides) == {"a", "b"}


def test_sample_delta_matrix_shape():
    assert Sample_delta_matrix(Randomize_Range()).shape == (4, 4)


def test_sample_delta_matrix_dtype():
    assert Sample_delta_matrix(Randomize_Range()).dtype == np.float32


def test_sample_delta_matrix_zero_range_is_identity():
    assert np.allclose(Sample_delta_matrix(Randomize_Range()), np.eye(4, dtype=np.float32))


def test_sample_delta_matrix_nonzero_tx_shifts_translation():
    _m = Sample_delta_matrix(Randomize_Range(tx=2.0))
    assert _m[0, 3] == pytest.approx(2.0)


def test_sample_translation_returns_three_floats():
    _t = Sample_translation(Randomize_Range())
    assert len(_t) == 3
    assert all(isinstance(_v, float) for _v in _t)


def test_sample_translation_zero_range_is_origin():
    assert Sample_translation(Randomize_Range()) == (0.0, 0.0, 0.0)


def test_sample_translation_fixed_values():
    _t = Sample_translation(Randomize_Range(tx=1.0, ty=2.0, tz=3.0))
    assert _t == (1.0, 2.0, 3.0)


def test_sample_value_fixed_float():
    assert _Sample_value(1.25) == pytest.approx(1.25)


def test_sample_value_list_range_within_bounds():
    for _ in range(16):
        _v = _Sample_value([1.0, 2.0])
        assert 1.0 <= _v <= 2.0


def test_sample_value_single_element_list():
    assert _Sample_value([3.5]) == pytest.approx(3.5)
