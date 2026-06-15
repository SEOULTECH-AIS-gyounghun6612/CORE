from __future__ import annotations

import json

import numpy as np

from spatial_toolbox.render.core.renderer import Render_Result
from spatial_toolbox.scene.node.type.camera import Camera
from spatial_toolbox.scene.node.type.mesh import Mesh as Mesh_Node
from spatial_toolbox.simulation.core.config import Sim_Config
from spatial_toolbox.simulation.core.exporter import Result_Exporter, _make_serializable


def test_make_serializable_plain_dict_passthrough():
    assert _make_serializable({"a": 1}) == {"a": 1}


def test_make_serializable_int_key_converted_to_str():
    _result = _make_serializable({1: 2})
    assert "1" in _result


def test_make_serializable_node_value_extracts_label():
    _node = Mesh_Node(label="mesh")
    _result = _make_serializable({"a": _node})
    assert _result["a"]["label"] == "mesh"


def test_make_serializable_node_value_includes_prim_path():
    _node = Mesh_Node(label="mesh")
    _result = _make_serializable({"a": _node})
    assert "prim_path" in _result["a"]


def test_make_serializable_non_dict_passthrough():
    assert _make_serializable(3) == 3


def test_make_serializable_nested_dict():
    _result = _make_serializable({"a": {"b": 1}})
    assert _result["a"]["b"] == 1


def test_save_array_uint8_writes_png(tmp_path):
    _exp = Result_Exporter(tmp_path)
    _arr = np.zeros((2, 2, 3), dtype=np.uint8)
    _exp._Save_array(tmp_path / "rgb_000000", _arr)
    assert (tmp_path / "rgb_000000.png").exists()


def test_save_array_float32_writes_npy(tmp_path):
    _exp = Result_Exporter(tmp_path)
    _arr = np.zeros((2, 2), dtype=np.float32)
    _exp._Save_array(tmp_path / "depth_000000", _arr)
    assert (tmp_path / "depth_000000.npy").exists()


def test_save_array_npy_roundtrip(tmp_path):
    _exp = Result_Exporter(tmp_path)
    _arr = np.arange(4, dtype=np.float32).reshape(2, 2)
    _exp._Save_array(tmp_path / "depth_000000", _arr)
    _loaded = np.load(tmp_path / "depth_000000.npy")
    assert np.allclose(_loaded, _arr)


def _make_result_with_rgb():
    return Render_Result(images={"rgb": np.zeros((2, 2, 3), dtype=np.uint8)})


def _make_result_with_depth():
    return Render_Result(images={"depth": np.zeros((2, 2), dtype=np.float32)})


def test_save_creates_png_for_uint8_channel(tmp_path):
    _exp = Result_Exporter(tmp_path)
    _exp.Save(0, _make_result_with_rgb(), Camera(label="cam"), Sim_Config())
    assert (tmp_path / "rgb_000000.png").exists()


def test_save_creates_npy_for_float32_channel(tmp_path):
    _exp = Result_Exporter(tmp_path)
    _exp.Save(0, _make_result_with_depth(), Camera(label="cam"), Sim_Config())
    assert (tmp_path / "depth_000000.npy").exists()


def test_save_creates_metadata_json(tmp_path):
    _exp = Result_Exporter(tmp_path)
    _exp.Save(0, _make_result_with_rgb(), Camera(label="cam"), Sim_Config())
    assert (tmp_path / "metadata_000000.json").exists()


def test_save_frame_id_padded_to_six_digits(tmp_path):
    _exp = Result_Exporter(tmp_path)
    _exp.Save(7, _make_result_with_rgb(), Camera(label="cam"), Sim_Config())
    assert (tmp_path / "rgb_000007.png").exists()


def test_save_metadata_contains_camera_label(tmp_path):
    _exp = Result_Exporter(tmp_path)
    _exp.Save(0, _make_result_with_rgb(), Camera(label="cam"), Sim_Config())
    _meta = json.loads((tmp_path / "metadata_000000.json").read_text())
    assert _meta["camera"]["label"] == "cam"


def test_save_metadata_contains_extrinsic(tmp_path):
    _exp = Result_Exporter(tmp_path)
    _exp.Save(0, _make_result_with_rgb(), Camera(label="cam"), Sim_Config())
    _meta = json.loads((tmp_path / "metadata_000000.json").read_text())
    assert "extrinsic" in _meta["camera"]


def test_save_metadata_extra_meta_merged(tmp_path):
    _exp = Result_Exporter(tmp_path)
    _exp.Save(0, _make_result_with_rgb(), Camera(label="cam"), Sim_Config(), extra_meta={"object": "obj", "sample": 1})
    _meta = json.loads((tmp_path / "metadata_000000.json").read_text())
    assert _meta["object"] == "obj"
    assert _meta["sample"] == 1


def test_save_creates_output_dir_if_missing(tmp_path):
    _subdir = tmp_path / "nested"
    _exp = Result_Exporter(_subdir)
    _exp.Save(0, _make_result_with_rgb(), Camera(label="cam"), Sim_Config())
    assert _subdir.exists()


def test_save_metadata_uses_python_toolbox_write_to(monkeypatch, tmp_path):
    _calls = {}

    def _capture(file, data, enc="UTF-8", **kwargs):
        _calls["file"] = file
        _calls["data"] = data
        _calls["enc"] = enc
        _calls["kwargs"] = kwargs

    monkeypatch.setattr(
        "spatial_toolbox.simulation.core.exporter.Write_to",
        _capture,
    )

    _exp = Result_Exporter(tmp_path)
    _exp.Save(0, _make_result_with_rgb(), Camera(label="cam"), Sim_Config())

    assert _calls["file"] == tmp_path / "metadata_000000.json"
    assert _calls["enc"] == "UTF-8"
    assert _calls["kwargs"]["indent"] == 2
    assert _calls["data"]["camera"]["label"] == "cam"
