from __future__ import annotations

import numpy as np
import pytest

from spatial_toolbox.scene.node.type._base import Base_Node
from spatial_toolbox.scene.node.type.group import Group
from spatial_toolbox.scene.node.type.mesh import Mesh as Mesh_Node


def test_default_prim_type():
    assert Base_Node().prim_type == "Xform"


def test_default_scale():
    assert np.allclose(Base_Node().scale, np.ones(3))


def test_default_unit_scale():
    assert Base_Node().unit_scale == 1.0


def test_default_visible():
    assert Base_Node().visible is True


def test_default_local_rigid_is_identity():
    assert np.allclose(Base_Node().local_rigid, np.eye(4))


def test_world_matrix_root_identity():
    assert np.allclose(Base_Node().world_matrix, np.eye(4))


def test_world_matrix_scale_applied():
    n = Base_Node(scale=np.array([2.0, 3.0, 4.0], dtype=np.float32))
    wm = n.world_matrix
    assert wm[0, 0] == pytest.approx(2.0)
    assert wm[1, 1] == pytest.approx(3.0)
    assert wm[2, 2] == pytest.approx(4.0)


def test_world_matrix_unit_scale_multiplied():
    n = Base_Node(unit_scale=0.1)
    assert n.world_matrix[0, 0] == pytest.approx(0.1)


def test_world_matrix_parent_child_composition():
    parent = Group()
    child = Base_Node()
    child.Set_parent(parent)
    parent.children.append(child)
    _m = np.eye(4, dtype=np.float32)
    _m[0, 3] = 2.0
    parent.local_rigid = _m
    assert np.allclose(child.world_matrix, _m)


def test_world_matrix_cache_reused():
    n = Base_Node()
    _wm1 = n.world_matrix
    _wm2 = n.world_matrix
    assert _wm1 is _wm2


def test_dirty_initially_true():
    assert Base_Node()._is_dirty is True


def test_dirty_cleared_after_matrix_access():
    n = Base_Node()
    _ = n.world_matrix
    assert n._is_dirty is False


def test_dirty_propagates_to_children():
    parent = Group()
    child = Base_Node()
    child.Set_parent(parent)
    parent.children.append(child)
    _ = child.world_matrix
    parent.local_rigid = np.eye(4, dtype=np.float32)
    assert child._is_dirty is True


def test_dirty_on_scale_change():
    n = Base_Node()
    _ = n.world_matrix
    n.scale = np.array([2.0, 2.0, 2.0], dtype=np.float32)
    assert n._is_dirty is True


def test_dirty_on_unit_scale_change():
    n = Base_Node()
    _ = n.world_matrix
    n.unit_scale = 0.5
    assert n._is_dirty is True


def test_visibility_propagates_to_child():
    parent = Group()
    child = Base_Node()
    parent.children.append(child)
    parent.visible = False
    assert child.visible is False


def test_visibility_restore_does_not_propagate():
    parent = Group()
    child = Base_Node()
    parent.children.append(child)
    parent.visible = False
    parent.visible = True
    assert child.visible is False


def test_is_renderable_reflects_visible():
    n = Base_Node()
    assert n.is_renderable is True
    n.visible = False
    assert n.is_renderable is False


def test_prim_path_root():
    assert Base_Node(label="root").prim_path == "/root"


def test_prim_path_child():
    parent = Base_Node(label="root")
    child = Base_Node(label="child", parent=parent)
    assert child.prim_path == "/root/child"


def test_prim_path_three_levels():
    r = Base_Node(label="r")
    m = Base_Node(label="m", parent=r)
    leaf = Base_Node(label="leaf", parent=m)
    assert leaf.prim_path == "/r/m/leaf"


def test_clone_preserves_label():
    assert Base_Node(label="a").Clone().label == "a"


def test_clone_label_override():
    assert Base_Node(label="a").Clone("b").label == "b"


def test_clone_scale_independent():
    orig = Base_Node(scale=np.array([2.0, 3.0, 4.0], dtype=np.float32))
    cloned = orig.Clone()
    cloned.scale[0] = 9.0
    assert np.allclose(orig.scale, np.array([2.0, 3.0, 4.0], dtype=np.float32))


def test_clone_children_deep_copied():
    parent = Group(label="p")
    child = Base_Node(label="c")
    child.Set_parent(parent)
    parent.children.append(child)
    cloned = parent.Clone()
    assert cloned.children[0] is not child
    assert cloned.children[0].label == "c"


def test_clone_source_key_preserved():
    n = Mesh_Node(label="m", source_key="shared")
    assert n.Clone().source_key == "shared"
