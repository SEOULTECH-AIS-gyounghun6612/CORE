from __future__ import annotations

import numpy as np
import pytest
import trimesh

from spatial_toolbox.scene.asset.cache import ASSET_CACHE
from spatial_toolbox.scene.asset.type.mesh import Mesh as Mesh_Asset
from spatial_toolbox.scene.node.type.camera import Camera
from spatial_toolbox.scene.node.type.mesh import Mesh as Mesh_Node
from spatial_toolbox.scene.stage import Controller

_KEY = "/tmp/shared.obj"
_VERTS = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float64)
_FACES = np.array([[0, 1, 2]], dtype=np.int64)


@pytest.fixture(autouse=True)
def _clean_cache():
    ASSET_CACHE.Clear()


@pytest.fixture
def geo():
    return trimesh.Trimesh(vertices=_VERTS.copy(), faces=_FACES)


@pytest.fixture
def ctrl_with_mesh(geo):
    ASSET_CACHE.Register(_KEY, Mesh_Asset(label="mesh", source_path=_KEY, geometry=geo))
    ctrl = Controller()
    ctrl.Add_node(ctrl.Build_node_from_cache(_KEY))
    return ctrl


def test_json_roundtrip_preserves_stage_and_hierarchy(tmp_path):
    _path = tmp_path / "scene.json"
    ctrl = Controller()
    ctrl.Add_node()
    ctrl.Export(str(_path))
    loaded = Controller()
    loaded.Import(str(_path))
    assert loaded.unit_length == pytest.approx(1.0)
    assert len(loaded.root.children) == 1
    assert loaded.root.children[0].label == "new_group"


def test_json_roundtrip_preserves_nested_nodes(tmp_path):
    _path = tmp_path / "scene.json"
    ctrl = Controller()
    ctrl.Add_node()
    parent = ctrl.root.children[0]
    parent.children.append(Mesh_Node(label="mesh", parent=parent))
    ctrl.Export(str(_path))
    loaded = Controller()
    loaded.Import(str(_path))
    assert len(loaded.root.children) == 1
    assert loaded.root.children[0].children[0].label == "mesh"


def test_json_roundtrip_preserves_camera_type_and_intrinsic(tmp_path):
    _path = tmp_path / "scene.json"
    ctrl = Controller()
    _cam = Camera(label="main_camera")
    _cam.intrinsic.fx = 777.0
    ctrl.Add_node(_cam)
    ctrl.Export(str(_path))
    loaded = Controller()
    loaded.Import(str(_path))
    assert isinstance(loaded.root.children[0], Camera)
    assert loaded.root.children[0].intrinsic.fx == pytest.approx(777.0)


def test_usd_export_writes_expected_structure(ctrl_with_mesh, tmp_path):
    Usd = pytest.importorskip("pxr.Usd")
    UsdGeom = pytest.importorskip("pxr.UsdGeom")
    _out = tmp_path / "scene.usda"
    ctrl_with_mesh.Export(str(_out))
    assert _out.exists()
    _stage = Usd.Stage.Open(str(_out))
    assert UsdGeom.GetStageMetersPerUnit(_stage) == pytest.approx(1.0)
    _default = _stage.GetDefaultPrim()
    assert _default.IsValid()
    _inst = list(_default.GetChildren())[0]
    assert _inst.IsInstanceable()
    assert _inst.IsA(UsdGeom.Mesh)
    _custom = _inst.GetCustomData()
    assert _custom["focusNodeLabel"] == ctrl_with_mesh.root.children[0].label
    assert _custom["focusNodePrimType"] == "Mesh"


def test_usd_import_restores_scene_state_and_cache(ctrl_with_mesh, tmp_path):
    pytest.importorskip("pxr.Usd")
    _out = tmp_path / "scene.usda"
    ctrl_with_mesh.Export(str(_out))
    ASSET_CACHE.Clear()
    loaded = Controller()
    loaded.Import(str(_out))
    assert loaded.unit_length == pytest.approx(1.0)
    assert len(loaded.root.children) == 1
    assert ASSET_CACHE.Get_all()


def test_usd_roundtrip_preserves_hierarchy_and_transforms(ctrl_with_mesh, tmp_path):
    pytest.importorskip("pxr.Usd")
    _m = np.eye(4, dtype=np.float32)
    _m[0, 3] = 1.5
    ctrl_with_mesh.root.children[0].local_rigid = _m
    ctrl_with_mesh.root.children[0].scale = np.array([2.0, 2.0, 2.0], dtype=np.float32)
    ctrl_with_mesh.root.children[0].unit_scale = 0.5
    _out = tmp_path / "scene.usda"
    ctrl_with_mesh.Export(str(_out))
    loaded = Controller()
    loaded.Import(str(_out))
    assert len(loaded.root.children) == 1
    assert loaded.root.children[0].source_key
    assert np.allclose(loaded.root.children[0].local_rigid, _m)
    assert loaded.root.children[0].unit_scale == pytest.approx(0.5)


def test_usd_instancing_single_prototype(tmp_path, geo):
    Usd = pytest.importorskip("pxr.Usd")
    _shared_key = "/tmp/shared.obj"
    ASSET_CACHE.Register(_shared_key, Mesh_Asset(label="mesh", source_path=_shared_key, geometry=geo))
    ctrl = Controller()
    for i in range(3):
        ctrl.Add_node(ctrl.Build_node_from_cache(_shared_key, label=f"m{i}"))
    _out = tmp_path / "scene.usda"
    ctrl.Export(str(_out))
    _stage = Usd.Stage.Open(str(_out))
    _protos = list(_stage.GetPrimAtPath("/_Prototypes").GetChildren())
    assert len(_protos) == 1


def test_usd_roundtrip_preserves_camera_type_and_intrinsic(tmp_path):
    pytest.importorskip("pxr.Usd")
    _out = tmp_path / "scene.usda"
    ctrl = Controller()
    _cam = Camera(label="main_camera")
    _cam.intrinsic.fx = 777.0
    _cam.intrinsic.width = 1280
    ctrl.Add_node(_cam)
    ctrl.Export(str(_out))
    loaded = Controller()
    loaded.Import(str(_out))
    assert isinstance(loaded.root.children[0], Camera)
    assert loaded.root.children[0].intrinsic.fx == pytest.approx(777.0)
    assert loaded.root.children[0].intrinsic.width == 1280
