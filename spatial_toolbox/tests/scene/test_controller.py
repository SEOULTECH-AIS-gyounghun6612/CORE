from __future__ import annotations

import pytest

from spatial_toolbox.scene.asset.cache import ASSET_CACHE
from spatial_toolbox.scene.asset.type.mesh import Mesh as Mesh_Asset
from spatial_toolbox.scene.node.type.group import Group
from spatial_toolbox.scene.node.type.mesh import Mesh as Mesh_Node
from spatial_toolbox.scene.stage import Controller

_OBJ_TETRA = """\
v 0 0 0
v 1 0 0
v 0 1 0
f 1 2 3
"""
_CACHE_KEY = "/tmp/mesh.obj"


@pytest.fixture(autouse=True)
def _clean_cache():
    ASSET_CACHE.Clear()


@pytest.fixture
def ctrl() -> Controller:
    return Controller()


@pytest.fixture
def mesh_in_cache():
    ASSET_CACHE.Register(_CACHE_KEY, Mesh_Asset(label="mesh", unit_length=0.001))


def test_add_none_creates_group(ctrl: Controller):
    ctrl.Add_node()
    assert len(ctrl.root.children) == 1
    assert ctrl.root.children[0].label == "new_group"


def test_add_mesh_node_to_root(ctrl: Controller):
    ctrl.Add_node(Mesh_Node(label="m"))
    assert ctrl.root.children[0].label == "m"


def test_add_group_unpacks_children(ctrl: Controller):
    group = Group(label="g")
    child = Mesh_Node(label="c")
    child.Set_parent(group)
    group.children.append(child)
    ctrl.Add_node(group)
    assert len(ctrl.root.children) == 1
    assert ctrl.root.children[0].label == "c"


def test_add_node_to_specified_parent(ctrl: Controller):
    ctrl.Add_node()
    parent_node = ctrl.root.children[0]
    ctrl.Add_node(Mesh_Node(label="m"), parent_node)
    assert parent_node.children[0].label == "m"


def test_move_node(ctrl: Controller):
    ctrl.Add_node()
    group = ctrl.root.children[0]
    node = Mesh_Node(label="m")
    node.Set_parent(ctrl.root)
    ctrl.root.children.append(node)
    assert ctrl.Move_node(node, ctrl.root, group) is True


def test_pop_node(ctrl: Controller):
    node = Mesh_Node(label="m")
    node.Set_parent(ctrl.root)
    ctrl.root.children.append(node)
    popped = ctrl.Pop_node(node, ctrl.root)
    assert popped is node
    assert popped.parent is None


def test_pop_node_nonexistent_returns_none(ctrl: Controller):
    ghost = Mesh_Node(label="g")
    assert ctrl.Pop_node(ghost, ctrl.root) is None


def test_clear_removes_all_children(ctrl: Controller):
    ctrl.Add_node()
    ctrl.Clear()
    assert len(ctrl.root.children) == 0


def test_render_queue_returns_visible_mesh_only(ctrl: Controller):
    _vis = Mesh_Node(label="vis")
    _invis = Mesh_Node(label="invis", visible=False)
    ctrl.Add_node(_vis)
    ctrl.Add_node(_invis)
    queue = ctrl.Get_render_queue()
    assert len(queue) == 1
    assert queue[0].label == "vis"


def test_render_queue_excludes_groups(ctrl: Controller):
    ctrl.Add_node()
    assert ctrl.Get_render_queue() == []


def test_build_node_from_cache_unit_scale(ctrl: Controller, mesh_in_cache):
    node = ctrl.Build_node_from_cache(_CACHE_KEY)
    assert node.unit_scale == pytest.approx(0.001)


def test_build_node_from_cache_raises_on_missing(ctrl: Controller):
    with pytest.raises(ValueError):
        ctrl.Build_node_from_cache("missing.obj")


def test_unit_length_change_recomputes_unit_scale(ctrl: Controller, mesh_in_cache):
    node = ctrl.Build_node_from_cache(_CACHE_KEY)
    ctrl.Add_node(node)
    node = ctrl.root.children[0]
    assert node.unit_scale == pytest.approx(0.001)
    ctrl.unit_length = 0.01
    assert node.unit_scale == pytest.approx(0.1)


def test_register_from_file_unit_length(tmp_path, ctrl: Controller):
    _path = tmp_path / "tri.obj"
    _path.write_text(_OBJ_TETRA)
    keys = ctrl.Register_from_file(str(_path), unit_length=0.01)
    assert len(keys) == 1
    assert ASSET_CACHE.Get(keys[0]).unit_length == pytest.approx(0.01)
