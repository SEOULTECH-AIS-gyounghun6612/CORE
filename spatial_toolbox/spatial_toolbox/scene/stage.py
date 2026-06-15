"""Scene graph controller and persistence entrypoint."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, ClassVar

from python_toolbox.data_schema import Data_Schema

from .asset.cache import ASSET_CACHE
from .asset.file import Load_and_register
from .asset.type._base import Base_Asset
from .node.register import NODE_REGISTRY
from .node.type._base import Base_Node
from .node.type.group import Group
from .node.type.mesh import Mesh
from .node.utils.traversal import walk_nodes
from .file import Export_to, Import_from


@dataclass
class Controller(Data_Schema):
    """Owns the scene graph, asset-backed node creation, and file I/O.

    Structural group nodes are represented internally through
    ``prim_type == "Xform"``. The public ``Group`` type exists only as a more
    explicit constructor for that structural node kind.

    Attributes:
        root: Root scene node that owns the full hierarchy.
        unit_length: Scene-wide unit scale used for asset-backed nodes.
    """

    root: Base_Node = field(
        default_factory=lambda: Base_Node(label="World_Root", prim_type="Xform")
    )
    unit_length: float = 1.0

    __exclude_serialize__: ClassVar[set[str]] = set()

    def __setattr__(self, key, value):
        """Propagates unit-length changes to asset-backed nodes."""
        _should_propagate = (
            key == "unit_length"
            and getattr(self, "unit_length", None) is not None
            and getattr(self, "unit_length", None) != value
        )
        super().__setattr__(key, value)
        if _should_propagate:
            self._Recompute_unit_scales()

    def Add_node(
        self,
        node: Base_Node | None = None,
        parent: Base_Node | None = None,
    ) -> None:
        """Adds a node into the scene graph.

        Args:
            node: Source node to clone into the tree. When omitted, an empty
                structural group is created.
            parent: Parent node that receives the new child. Defaults to
                ``root``.
        """
        _target = parent if parent else self.root

        if node is None:
            _new_group = Group(label="new_group", prim_type="Xform")
            _new_group.Set_parent(_target)
            _target.children.append(_new_group)
            return

        if node.prim_type == "Xform":
            for _child in node.children:
                _cloned_child = _child.Clone()
                self.Add_node(_cloned_child, _target)
            return

        _new_node = node.Clone()
        _new_node.Set_parent(_target)
        _target.children.append(_new_node)

    def Move_node(self, node: Base_Node, old: Base_Node, new: Base_Node) -> bool:
        """Moves a node between parents.

        Args:
            node: Node instance to move.
            old: Current parent expected to contain ``node``.
            new: New parent that receives ``node``.

        Returns:
            ``True`` when the node was moved, otherwise ``False``.
        """
        _moved_node = self.Pop_node(node, old)
        if _moved_node:
            _moved_node.Set_parent(new)
            new.children.append(_moved_node)
            return True
        return False

    def Pop_node(self, node: Base_Node, parent: Base_Node) -> Base_Node | None:
        """Detaches a node from its parent.

        Args:
            node: Node instance to detach.
            parent: Parent expected to contain ``node``.

        Returns:
            The detached node, or ``None`` when it was not found.
        """
        try:
            parent.children.remove(node)
            node.Set_parent(None)
            return node
        except (ValueError, AttributeError):
            return None

    def Clear(self) -> None:
        """Removes all children from the root node."""
        for _child in self.root.children:
            _child.Set_parent(None)
        self.root.children.clear()

    def Get_render_queue(self) -> list[Base_Node]:
        """Returns visible mesh nodes submitted to render backends."""
        return list(
            walk_nodes(self.root, lambda n: isinstance(n, Mesh) and n.is_renderable)
        )

    def Register_from_file(
        self, file_path: str | Path, unit_length: float = 1.0
    ) -> list[str]:
        """Loads an asset file and registers it in the shared asset cache."""
        _path = Path(file_path).resolve()
        return Load_and_register(_path, unit_length)

    def Build_node_from_cache(
        self,
        key: str,
        asset_type: type[Base_Asset] | None = None,
        label: str | None = None,
        editable: bool = True,
    ) -> Base_Node:
        """Builds a scene node from a cached asset entry.

        Args:
            key: Asset-cache key returned by the asset loader.
            asset_type: Optional asset type constraint.
            label: Explicit node label. Defaults to the cache fragment or file
                stem.
            editable: Whether the cache lookup can return editable assets.

        Returns:
            A node instance configured to reference the cached asset.

        Raises:
            ValueError: If the asset key is not registered.
        """
        _asset = ASSET_CACHE.Get(key, asset_type, editable)
        if _asset is None:
            raise ValueError(f"캐시에 등록되지 않은 자산 키임: {key}")

        _label = label or (key.rsplit("#", 1)[-1] if "#" in key else Path(key).stem)
        _name = _asset.__class__.__name__
        _unit_scale = _asset.unit_length / self.unit_length
        return NODE_REGISTRY.Get(_name)(
            label=_label,
            source_key=key,
            prim_type=_name,
            unit_scale=_unit_scale,
        )

    def Export(self, file_name: str) -> None:
        """Serializes the current scene state to JSON or USD."""
        Export_to(file_name, self.root, self.unit_length)

    def Import(self, file_name: str) -> None:
        """Loads a serialized scene and replaces the current graph."""
        _scene_state = Import_from(file_name)
        self._Apply_scene_state(_scene_state.root, _scene_state.unit_length)

    def _Apply_scene_state(self, imported_root: Base_Node, unit_length: float) -> None:
        """Applies an imported scene state after clearing the current graph."""
        self.Clear()
        self.unit_length = unit_length
        for _child in list(imported_root.children):
            _child.Set_parent(self.root)
            self.root.children.append(_child)

    def _Recompute_unit_scales(self) -> None:
        """Recomputes ``unit_scale`` for nodes backed by cached assets."""
        if not hasattr(self, "root") or self.root is None:
            return

        _has_source = lambda n: n.source_key is not None
        for _node in walk_nodes(self.root, _has_source):
            _key = _node.source_key
            if _key is None:
                continue
            _asset = ASSET_CACHE.Get(_key, is_hold=True)
            if _asset is None:
                continue
            _node.unit_scale = _asset.unit_length / self.unit_length
