from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

from spatial_toolbox.scene.node.type.camera import Camera
from spatial_toolbox.scene.node.type.group import Group
from spatial_toolbox.scene.stage import Controller
from spatial_toolbox.simulation.blender.engine import Blender_Capture_Engine
from spatial_toolbox.simulation.core.config import Randomize_Range, Sim_Config

from .conftest import _mock_renderer, make_renderer_return, _PATCH_RENDERER, _PATCH_SAMPLE, _PATCH_SEED


def test_find_target_group_default_label(ctrl_with_target):
    _ctrl, _ = ctrl_with_target
    _node = Blender_Capture_Engine._Find_target_group(_ctrl, "target")
    assert _node.label == "target"


def test_find_target_group_custom_label():
    _ctrl = Controller()
    _grp = Group(label="custom")
    _grp.Set_parent(_ctrl.root)
    _ctrl.root.children.append(_grp)
    _node = Blender_Capture_Engine._Find_target_group(_ctrl, "custom")
    assert _node.label == "custom"


def test_find_target_group_missing_raises(ctrl_with_target):
    _ctrl, _ = ctrl_with_target
    with pytest.raises(ValueError):
        Blender_Capture_Engine._Find_target_group(_ctrl, "ghost")


def test_find_target_group_wrong_prim_type_raises():
    _ctrl = Controller()
    _cam = Camera(label="target")
    _cam.Set_parent(_ctrl.root)
    _ctrl.root.children.append(_cam)
    with pytest.raises(ValueError):
        Blender_Capture_Engine._Find_target_group(_ctrl, "target")


def test_make_exporter_per_object_path_includes_label(tmp_path):
    _config = Sim_Config(output_layout="per_object")
    _exp = Blender_Capture_Engine._Make_exporter(tmp_path, "obj", _config)
    assert _exp._output_dir == tmp_path / "obj"


def test_make_exporter_flat_path_is_output_dir(tmp_path):
    _config = Sim_Config(output_layout="flat")
    _exp = Blender_Capture_Engine._Make_exporter(tmp_path, "obj", _config)
    assert _exp._output_dir == tmp_path


def test_capture_sets_seed_when_configured(tmp_path, ctrl_with_target):
    _ctrl, _ = ctrl_with_target
    _config = Sim_Config(seed=123)
    with patch(_PATCH_SEED) as _mock_seed, patch(_PATCH_RENDERER, _mock_renderer(_config.camera_labels)):
        Blender_Capture_Engine().Capture(_ctrl, _config, tmp_path)
        _mock_seed.assert_called_once_with(123)


def test_capture_no_seed_skips_seed_call(tmp_path, ctrl_with_target):
    _ctrl, _ = ctrl_with_target
    _config = Sim_Config(seed=None)
    with patch(_PATCH_SEED) as _mock_seed, patch(_PATCH_RENDERER, _mock_renderer(_config.camera_labels)):
        Blender_Capture_Engine().Capture(_ctrl, _config, tmp_path)
        _mock_seed.assert_not_called()


def test_capture_restores_visibility_after_completion(tmp_path, ctrl_with_target):
    _ctrl, _target = ctrl_with_target
    _objs = list(_target.children)
    _config = Sim_Config()
    with patch(_PATCH_RENDERER, _mock_renderer(_config.camera_labels)):
        Blender_Capture_Engine().Capture(_ctrl, _config, tmp_path)
    assert all(_obj.visible is False for _obj in _objs)


def test_capture_target_group_visible_after_completion(tmp_path, ctrl_with_target):
    _ctrl, _target = ctrl_with_target
    _config = Sim_Config()
    with patch(_PATCH_RENDERER, _mock_renderer(_config.camera_labels)):
        Blender_Capture_Engine().Capture(_ctrl, _config, tmp_path)
    assert _target.visible is True


def test_capture_progress_callback_called_per_sample(tmp_path, ctrl_with_target):
    _ctrl, _target = ctrl_with_target
    _objs = list(_target.children)
    _config = Sim_Config(num_samples=2)
    _calls = []
    with patch(_PATCH_RENDERER, _mock_renderer(_config.camera_labels)):
        Blender_Capture_Engine().Capture(_ctrl, _config, tmp_path, progress_callback=lambda c, t, m: _calls.append((c, t, m)))
    assert len(_calls) == len(_objs) * 2


def test_capture_progress_callback_total_is_objects_times_samples(tmp_path, ctrl_with_target):
    _ctrl, _target = ctrl_with_target
    _config = Sim_Config(num_samples=3)
    _calls = []
    with patch(_PATCH_RENDERER, _mock_renderer(_config.camera_labels)):
        Blender_Capture_Engine().Capture(_ctrl, _config, tmp_path, progress_callback=lambda c, t, m: _calls.append((c, t, m)))
    assert all(_t == len(_target.children) * 3 for _, _t, _ in _calls)


def test_capture_progress_callback_current_increments(tmp_path, ctrl_with_target):
    _ctrl, _ = ctrl_with_target
    _config = Sim_Config(num_samples=2)
    _calls = []
    with patch(_PATCH_RENDERER, _mock_renderer(_config.camera_labels)):
        Blender_Capture_Engine().Capture(_ctrl, _config, tmp_path, progress_callback=lambda c, t, m: _calls.append((c, t, m)))
    assert [c for c, _, _ in _calls] == list(range(1, len(_calls) + 1))


def test_capture_restores_object_rigid_after_samples(tmp_path, ctrl_with_target):
    _ctrl, _target = ctrl_with_target
    _objs = list(_target.children)
    _orig = [_o.local_rigid.copy() for _o in _objs]
    _config = Sim_Config(num_samples=2, obj=Randomize_Range(tx=1.0))
    with patch(_PATCH_RENDERER, _mock_renderer(_config.camera_labels)):
        Blender_Capture_Engine().Capture(_ctrl, _config, tmp_path)
    for _obj, _o in zip(_objs, _orig):
        assert np.allclose(_obj.local_rigid, _o)


def test_capture_restores_camera_rigid_after_each_sample(tmp_path, ctrl_with_target):
    _ctrl, _ = ctrl_with_target
    _cam = next(_n for _n in _ctrl.root.children if isinstance(_n, Camera))
    _orig = _cam.local_rigid.copy()
    _config = Sim_Config(num_samples=2, cam=Randomize_Range(tx=1.0))
    with patch(_PATCH_RENDERER, _mock_renderer(_config.camera_labels)):
        Blender_Capture_Engine().Capture(_ctrl, _config, tmp_path)
    assert np.allclose(_cam.local_rigid, _orig)


def test_capture_cam_overrides_applied_to_correct_camera(tmp_path, ctrl_with_target):
    _ctrl, _ = ctrl_with_target
    _config = Sim_Config(cam=Randomize_Range(tx=1.0), cam_overrides={"main_camera": Randomize_Range(tx=2.0)})
    _cam_calls = []

    def _capture_arg(arg):
        _cam_calls.append(arg)
        return np.eye(4, dtype=np.float32)

    with patch(_PATCH_SAMPLE, side_effect=_capture_arg), patch(_PATCH_RENDERER, _mock_renderer(_config.camera_labels)):
        Blender_Capture_Engine().Capture(_ctrl, _config, tmp_path)
    assert len(_cam_calls) >= 2


def test_capture_cam_fallback_to_default_when_no_override(tmp_path, ctrl_with_target):
    _ctrl, _ = ctrl_with_target
    _default_cam = Randomize_Range(tx=1.0)
    _config = Sim_Config(cam=_default_cam)
    _cam_calls = []

    def _capture_arg(arg):
        _cam_calls.append(arg)
        return np.eye(4, dtype=np.float32)

    with patch(_PATCH_SAMPLE, side_effect=_capture_arg), patch(_PATCH_RENDERER, _mock_renderer(_config.camera_labels)):
        Blender_Capture_Engine().Capture(_ctrl, _config, tmp_path)
    assert any(_arg is _default_cam for _arg in _cam_calls)


def test_capture_each_render_sees_one_object_visible(tmp_path, ctrl_with_target):
    _ctrl, _target = ctrl_with_target
    _objs = list(_target.children)
    _config = Sim_Config(num_samples=1)
    _visible_snapshots = []

    def _capture_visibility(*_args, **_kwargs):
        _visible_snapshots.append([_o.visible for _o in _objs])
        return make_renderer_return(_config.camera_labels)

    with patch(_PATCH_RENDERER) as _mock:
        _mock.return_value.__enter__.return_value.Render.side_effect = _capture_visibility
        Blender_Capture_Engine().Capture(_ctrl, _config, tmp_path)

    assert len(_visible_snapshots) == len(_objs)
    for _snapshot in _visible_snapshots:
        assert sum(_snapshot) == 1


def test_capture_empty_target_does_not_call_render(tmp_path):
    _ctrl = Controller()
    _target = Group(label="target")
    _target.Set_parent(_ctrl.root)
    _ctrl.root.children.append(_target)
    _cam = Camera(label="main_camera")
    _cam.Set_parent(_ctrl.root)
    _ctrl.root.children.append(_cam)
    _config = Sim_Config()
    _mock = _mock_renderer(_config.camera_labels)
    with patch(_PATCH_RENDERER, _mock):
        Blender_Capture_Engine().Capture(_ctrl, _config, tmp_path)
    _mock.return_value.__enter__.return_value.Render.assert_not_called()


def test_capture_skips_physics_drop_when_disabled(tmp_path, ctrl_with_target):
    _ctrl, _ = ctrl_with_target
    _config = Sim_Config()
    with patch.object(Blender_Capture_Engine, "_Apply_physics_drop") as _mock_drop, patch(
        _PATCH_RENDERER, _mock_renderer(_config.camera_labels)
    ):
        Blender_Capture_Engine().Capture(_ctrl, _config, tmp_path)
    _mock_drop.assert_not_called()


def test_capture_calls_physics_drop_when_enabled(tmp_path, ctrl_with_target):
    _ctrl, _target = ctrl_with_target
    _config = Sim_Config(physics_drop={"enabled": True})
    with patch.object(Blender_Capture_Engine, "_Apply_physics_drop") as _mock_drop, patch(
        _PATCH_RENDERER, _mock_renderer(_config.camera_labels)
    ):
        Blender_Capture_Engine().Capture(_ctrl, _config, tmp_path)
    _mock_drop.assert_called_once()
    _args = _mock_drop.call_args.args
    assert _args[0] is _ctrl
    assert _args[1] == list(_target.children)
    assert _args[2] == _config
