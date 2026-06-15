"""Base scene-node schema used across scene, render, and simulation code."""
from __future__ import annotations
from dataclasses import dataclass, field, InitVar
from typing import ClassVar, Literal, get_args

import numpy as np

from python_toolbox.data_schema import Data_Schema


PrimType = Literal[
    "Xform", "Mesh", "Points", "Material",
    "Shader", "Camera", "PhysicsScene", "SkelRoot", "Empty"
]
PRIM_TYPES = set(get_args(PrimType))


def __Local_rigid_serialize__(m: np.ndarray) -> list:
    """Serializes a local rigid transform matrix into a flat list."""
    return m.flatten().tolist()


def __Scale_serialize__(s: np.ndarray) -> list:
    """Serializes a scale vector into a list."""
    return s.tolist()


@dataclass
class Base_Node(Data_Schema):
    """Defines the shared scene-node state and transform behavior.

    Attributes:
        label: Human-readable node label.
        prim_type: Scene primitive type identifier.
        local_rigid: Local rigid transform matrix.
        local_rigid_meta: Serialized local transform payload used on load.
        scale: Local non-uniform scale vector.
        scale_meta: Serialized scale payload used on load.
        unit_scale: Additional scale derived from scene units.
        source_key: Asset-cache key used by asset-backed nodes.
        children: Child nodes in the scene hierarchy.
        visible: Visibility flag used by render-queue filtering.
        parent: Parent node in the scene hierarchy.
    """
    label: str = "obj"
    prim_type: PrimType = "Xform"
    local_rigid: np.ndarray = field(
        default_factory=lambda: np.eye(4, dtype=np.float32)
    )
    local_rigid_meta: InitVar[list | None] = None
    scale: np.ndarray = field(
        default_factory=lambda: np.ones(3, dtype=np.float32)
    )
    scale_meta: InitVar[list | None] = None
    unit_scale: float = 1.0
    source_key: str | None = None
    children: list["Base_Node"] = field(default_factory=list)
    visible: bool = True
    parent: "Base_Node | None" = field(default=None, repr=False)

    __exclude_serialize__: ClassVar[set[str]] = {
        "parent", "_matrix_cache", "_is_dirty"
    }
    __custom_keys__: ClassVar[dict[str, str]] = {
        "local_rigid": "local_rigid_meta",
        "scale": "scale_meta",
    }
    __custom_serializers__: ClassVar[dict] = {
        "local_rigid": __Local_rigid_serialize__,
        "scale": __Scale_serialize__,
    }

    _matrix_cache: np.ndarray | None = field(default=None, init=False, repr=False)
    _is_dirty: bool = field(default=True, init=False, repr=False)

    def __post_init__(
        self,
        local_rigid_meta: list | None,
        scale_meta: list | None,
    ):
        """Normalizes serialized transform payloads into numpy arrays."""
        if local_rigid_meta is not None:
            self.local_rigid = np.asarray(
                local_rigid_meta, dtype=np.float32
            ).reshape(4, 4)
        if scale_meta is not None:
            self.scale = np.asarray(scale_meta, dtype=np.float32)

    def __setattr__(self, key, value):
        """Invalidates cached transforms and propagates visibility changes."""
        super().__setattr__(key, value)
        if key in ("local_rigid", "scale", "unit_scale", "parent"):
            self._Mark_dirty()
        elif key == "visible":
            self._Propagate_visibility(value)

    def _Mark_dirty(self) -> None:
        """Invalidates cached world matrices for this subtree."""
        self._is_dirty = True
        if hasattr(self, "children"):
            for _child in self.children:
                if hasattr(_child, "_Mark_dirty"):
                    _child._Mark_dirty()

    def _Propagate_visibility(self, v: bool) -> None:
        """Propagates hidden state downward without forcing children visible."""
        if not hasattr(self, "children"):
            return
        if v:
            return
        for _child in self.children:
            if isinstance(_child, Base_Node):
                _child.visible = v

    @property
    def is_renderable(self) -> bool:
        """Returns whether the node should be submitted to renderers."""
        return self.visible

    @property
    def world_matrix(self) -> np.ndarray:
        """Returns the cached world transform for the node."""
        if not self._is_dirty and self._matrix_cache is not None:
            return self._matrix_cache

        _scale = np.diag([
            float(self.scale[0] * self.unit_scale),
            float(self.scale[1] * self.unit_scale),
            float(self.scale[2] * self.unit_scale),
            1.0,
        ]).astype(np.float32)
        _local = self.local_rigid @ _scale

        if self.parent is None:
            self._matrix_cache = _local
        else:
            self._matrix_cache = self.parent.world_matrix @ _local

        self._is_dirty = False
        return self._matrix_cache

    @property
    def prim_path(self) -> str:
        """Returns the slash-separated path from the root to this node."""
        if self.parent is None:
            return f"/{self.label}"
        return f"{self.parent.prim_path}/{self.label}"

    def Set_parent(self, new_parent: "Base_Node | None") -> None:
        """Assigns a new parent and invalidates transform caches."""
        self.parent = new_parent

    def Clone(self, label_name: str | None = None) -> "Base_Node":
        """Clones the node and its full subtree."""
        _new_node = self.__class__(
            label=self.label if label_name is None else label_name,
            prim_type=self.prim_type,
            local_rigid=self.local_rigid.copy(),
            scale=self.scale.copy(),
            unit_scale=self.unit_scale,
            source_key=self.source_key,
            visible=self.visible,
        )
        for _child in self.children:
            _cloned_child = _child.Clone()
            _cloned_child.Set_parent(_new_node)
            _new_node.children.append(_cloned_child)
        return _new_node

__all__ = ["Base_Node", "PrimType"]
