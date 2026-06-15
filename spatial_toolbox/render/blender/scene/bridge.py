"""Bridge layer that maps shared scene data into Blender operations."""
from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import numpy as np

from ....scene import Controller
from ....scene.node.type._base import Base_Node
from ....scene.node.type.camera import Camera
from ....scene.node.utils.traversal import walk_nodes
from ...core.channel import DEPTH, NORMAL, RGB, SEGMENTATION
from ...core.renderer import Render_Request
from .camera import Apply_camera_intrinsic, Get_or_create_camera_object
from .compositor import Configure_compositor_outputs, Configure_passes, Resolve_output_path
from .depth import Load_depth
from .normal import Load_normal
from .rgb import Load_rgb
from .segmentation import (
    Build_target_segmentation_metadata,
    Build_segmentation_metadata,
    Load_segmentation,
    Resolve_segmentation_object_name,
    Resolve_node_mesh_objects,
    Assign_segmentation_indices,
    Sanitize_usd_name,
)



class Blender_Scene_Bridge:
    """Coordinates scene import, camera setup, and output loading in Blender."""

    def __init__(self, bpy: Any) -> None:
        self._bpy = bpy
        self._temp_dir = TemporaryDirectory(prefix="spatial_toolbox_blender_")
        self._loaded_scene_id: int | None = None
        self._node_mesh_map: dict[int, tuple[Base_Node, list[Any]]] = {}
        self._last_vis: dict[int, bool] = {}
        self._y2z: Any = None
        self._compositor_key: tuple | None = None
        self._compositor_outputs: dict[str, Any] = {}
        self._compositor_crypto_node: Any | None = None
        self._render_engine_set: bool = False
        self._passes_key: tuple = ()
        self._group_obj_map: dict[int, tuple[Base_Node, Any]] = {}
        self._group_nodes_key: tuple[int, ...] = ()
        self._camera_obj_map: dict[str, tuple[Any, Any]] = {}

    @property
    def temp_dir(self) -> Path:
        """Returns the temporary working directory for Blender outputs."""
        return Path(self._temp_dir.name)

    def Clear_scene(self) -> None:
        """Resets the Blender scene to an empty factory state."""
        self._bpy.ops.wm.read_factory_settings(use_empty=True)

    def Export_scene_to_usd(self, scene: Controller) -> Path:
        """Exports the shared scene graph to a temporary USD file."""
        _usd_path = self.temp_dir / "scene.usd"
        scene.Export(_usd_path)
        return _usd_path

    def _Setup_default_lighting(self) -> None:
        """Adds a sun lamp and world ambient after a factory-reset scene."""
        import mathutils
        _bpy = self._bpy
        _scene = _bpy.context.scene
        _scene.render.film_transparent = True
        _world = _bpy.data.worlds.new("__focus_world__")
        _scene.world = _world
        _world.use_nodes = True
        _bg = _world.node_tree.nodes.get("Background")
        if _bg is not None:
            _bg.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)
            _bg.inputs["Strength"].default_value = 0.5
        _light_data = _bpy.data.lights.new(name="__focus_sun__", type="SUN")
        _light_data.energy = 3.0
        _light_obj = _bpy.data.objects.new("__focus_sun__", _light_data)
        _scene.collection.objects.link(_light_obj)
        _light_obj.rotation_euler = mathutils.Euler((0.785, 0.0, 0.785), "XYZ")

    def Load_scene(self, scene: Controller) -> Path:
        """Exports and imports the shared scene into Blender."""
        _usd_path = self.Export_scene_to_usd(scene)
        self.Clear_scene()
        self._bpy.ops.wm.usd_import(filepath=str(_usd_path))
        self._Setup_default_lighting()
        return _usd_path

    def _Build_camera_obj_map(self, scene: Controller) -> None:
        """Creates Blender camera objects and applies intrinsics once per scene load."""
        self._camera_obj_map.clear()
        for _base in walk_nodes(scene.root, lambda n: isinstance(n, Camera)):
            _node: Camera = _base  # type: ignore[assignment]
            if _node.intrinsic is None:
                continue
            _obj = Get_or_create_camera_object(self._bpy, _node)
            Apply_camera_intrinsic(_obj.data, _node.intrinsic)
            _obj.data.clip_start = float(_node.intrinsic.near_clip / scene.unit_length)
            _obj.data.clip_end = float(_node.intrinsic.far_clip / scene.unit_length)
            _scene = self._bpy.context.scene
            _scene.render.resolution_x = int(_node.intrinsic.width)
            _scene.render.resolution_y = int(_node.intrinsic.height)
            _scene.render.resolution_percentage = 100
            self._camera_obj_map[_node.label] = (_node, _obj)

    def _Build_node_mesh_map(self, root: Base_Node) -> None:
        """Builds a mapping from FOCUS node id to Blender mesh objects."""
        self._node_mesh_map.clear()
        for _node in walk_nodes(root, lambda n: getattr(n, "source_key", None) is not None):
            _objs = Resolve_node_mesh_objects(self._bpy, _node)
            if _objs:
                self._node_mesh_map[id(_node)] = (_node, _objs)

    def _Ensure_y2z(self) -> None:
        if self._y2z is None:
            import math
            from mathutils import Matrix
            self._y2z = Matrix.Rotation(math.pi / 2, 4, "X")

    def _Apply_transforms(self, visible_only: bool = False) -> None:
        """Pushes matrix_world for nodes into Blender.

        visible_only=True: only updates visible nodes (lightweight, for Sync_scene).
        visible_only=False: updates all nodes regardless of visibility (for Sync_transforms).
        """
        from mathutils import Matrix
        self._Ensure_y2z()
        for _node, _objs in self._node_mesh_map.values():
            if visible_only and not _node.visible:
                continue
            _objs[0].matrix_world = self._y2z @ Matrix(_node.world_matrix.astype(float).tolist())

    def _Apply_visibility(self) -> None:
        """Flips hide_render only for nodes whose visibility changed; triggers depsgraph."""
        for _nid, (_node, _objs) in self._node_mesh_map.items():
            _vis = _node.visible
            if self._last_vis.get(_nid) != _vis:
                for _obj in _objs:
                    _obj.hide_render = not _vis
                self._last_vis[_nid] = _vis
        self._bpy.context.view_layer.update()

    def _Load_scene_full(self, scene: Controller) -> Path:
        """Exports scene with all geometry visible and imports into Blender."""
        from ....scene.file.usd import Export_to_usd
        _usd_path = self.temp_dir / "scene.usd"
        Export_to_usd(_usd_path, scene.root, scene.unit_length, write_visibility=False)
        self.Clear_scene()
        self._bpy.ops.wm.usd_import(filepath=str(_usd_path))
        self._Setup_default_lighting()
        return _usd_path

    def _Ensure_scene(self, scene: Controller) -> None:
        """Loads scene into Blender if not already loaded."""
        if self._loaded_scene_id != id(scene):
            self._Load_scene_full(scene)
            self._Build_node_mesh_map(scene.root)
            self._Build_camera_obj_map(scene)
            self._loaded_scene_id = id(scene)
            self._last_vis.clear()
            self._compositor_key = None
            self._compositor_crypto_node = None
            self._render_engine_set = False
            self._passes_key = ()
            self._group_obj_map.clear()
            self._group_nodes_key = ()

    def _Build_group_obj_map(self, nodes: list[Base_Node]) -> None:
        """Finds Blender Xform objects matching the given FOCUS group nodes by label."""
        self._group_obj_map.clear()
        for _node in nodes:
            _name = Sanitize_usd_name(_node.label)
            _obj = next(
                (_o for _o in self._bpy.data.objects
                 if str(_o.name).split(".", 1)[0] == _name),
                None,
            )
            if _obj is not None:
                self._group_obj_map[id(_node)] = (_node, _obj)
        self._group_nodes_key = tuple(id(n) for n in nodes)

    def Sync_transforms(self, scene: Controller, group_nodes: list[Base_Node] | None = None) -> None:
        """Pushes transforms to Blender.

        group_nodes: direct children of the target group whose local_rigid changed.
          When provided, updates only those Xform objects — Blender propagates to
          mesh children via parent-child relationship (fewer API calls).
          When None, falls back to updating all mesh-leaf nodes.
        """
        self._Ensure_scene(scene)
        from mathutils import Matrix
        self._Ensure_y2z()
        if group_nodes is not None:
            _key = tuple(id(n) for n in group_nodes)
            if _key != self._group_nodes_key:
                self._Build_group_obj_map(group_nodes)
            for _node, _obj in self._group_obj_map.values():
                _obj.matrix_world = self._y2z @ Matrix(_node.world_matrix.astype(float).tolist())
        else:
            self._Apply_transforms()
        for _node, _obj in self._camera_obj_map.values():
            _obj.matrix_world = self._y2z @ Matrix(_node.world_matrix.astype(float).tolist())
        self._bpy.context.view_layer.update()

    def Sync_visibility(self, scene: Controller) -> None:
        """Applies only visibility changes; assumes transforms already pushed."""
        self._Ensure_scene(scene)
        self._Apply_visibility()

    def Sync_scene(self, scene: Controller) -> Path:
        """Loads scene on first call; applies transforms + visibility on subsequent calls."""
        self._Ensure_scene(scene)
        self._Apply_transforms(visible_only=True)
        self._Apply_visibility()
        return self.temp_dir / "scene.usd"

    @staticmethod
    def _Node_scale_matrix(node: Base_Node) -> np.ndarray:
        """Builds the diagonal scale matrix used by shared scene nodes."""
        return np.diag([
            float(node.scale[0] * node.unit_scale),
            float(node.scale[1] * node.unit_scale),
            float(node.scale[2] * node.unit_scale),
            1.0,
        ]).astype(np.float32)

    @staticmethod
    def _World_to_local_rigid(node: Base_Node, world_matrix: np.ndarray) -> np.ndarray:
        """Converts a world matrix into the node's local rigid matrix."""
        _parent_world = (
            np.eye(4, dtype=np.float32)
            if node.parent is None
            else node.parent.world_matrix.astype(np.float32)
        )
        _local_with_scale = np.linalg.inv(_parent_world) @ world_matrix.astype(np.float32)
        return _local_with_scale @ np.linalg.inv(Blender_Scene_Bridge._Node_scale_matrix(node))

    def _Find_objects_for_nodes(
        self,
        nodes: list[Base_Node],
    ) -> dict[int, tuple[Base_Node, Any]]:
        """Matches shared scene nodes to imported Blender mesh objects."""
        _mapping: dict[int, tuple[Base_Node, Any]] = {}
        for _node in nodes:
            _meshes = Resolve_node_mesh_objects(self._bpy, _node)
            if not _meshes:
                continue
            _mapping[id(_node)] = (_node, _meshes[0])
        return _mapping

    def _Ensure_rigid_body_world(self, physics_cfg: Any) -> None:
        """Creates and configures Blender's rigid-body world."""
        _scene = self._bpy.context.scene
        if getattr(_scene, "rigid_body_world", None) is None:
            self._bpy.ops.rigidbody.world_add()
        _world = _scene.rigid_body_world
        if _world is None:
            raise RuntimeError("Blender rigid body world is unavailable.")

        _world.steps_per_second = int(physics_cfg.steps_per_second)
        _world.solver_iterations = int(physics_cfg.solver_iterations)
        _scene.frame_start = 1
        _scene.frame_end = max(1, int(physics_cfg.settle_frames))

    def _Set_active_object(self, obj: Any) -> None:
        """Marks a Blender object as active and selected."""
        _view_layer = self._bpy.context.view_layer
        try:
            self._bpy.ops.object.select_all(action="DESELECT")
        except Exception:
            pass
        try:
            obj.select_set(True)
        except Exception:
            pass
        _view_layer.objects.active = obj

    def _Configure_rigid_body(
        self,
        obj: Any,
        body_type: str,
        physics_cfg: Any,
    ) -> None:
        """Adds or updates a rigid body on a Blender object."""
        self._Set_active_object(obj)
        if getattr(obj, "rigid_body", None) is None:
            self._bpy.ops.rigidbody.object_add()
        _rb = obj.rigid_body
        if _rb is None:
            raise RuntimeError(f"Rigid body could not be created for '{obj.name}'.")

        _rb.type = body_type
        _rb.friction = float(physics_cfg.friction)
        _rb.restitution = float(physics_cfg.restitution)
        if body_type == "ACTIVE":
            _rb.mass = float(physics_cfg.mass)
            _rb.linear_damping = float(physics_cfg.linear_damping)
            _rb.angular_damping = float(physics_cfg.angular_damping)
            _rb.collision_shape = str(physics_cfg.collision_shape)
        else:
            _rb.collision_shape = "MESH"

    def _Ensure_ground_plane(self, physics_cfg: Any) -> Any | None:
        """Adds a passive ground plane when configured."""
        if not getattr(physics_cfg, "use_ground_plane", False):
            return None

        _obj = self._bpy.data.objects.get("__focus_ground__")
        if _obj is None:
            self._bpy.ops.mesh.primitive_plane_add(
                size=100.0,
                location=(0.0, 0.0, float(physics_cfg.floor_z)),
            )
            _obj = self._bpy.context.active_object
            _obj.name = "__focus_ground__"
        else:
            _obj.location = (0.0, 0.0, float(physics_cfg.floor_z))
        self._Configure_rigid_body(_obj, "PASSIVE", physics_cfg)
        return _obj

    def Simulate_physics_drop(
        self,
        scene: Controller,
        target_nodes: list[Base_Node],
        physics_cfg: Any,
    ) -> None:
        """Runs Blender rigid-body settling and writes poses back to the scene."""
        self.Load_scene(scene)
        self._Ensure_rigid_body_world(physics_cfg)
        _mapping = self._Find_objects_for_nodes(target_nodes)

        if not _mapping:
            raise ValueError("No Blender objects matched the configured target nodes.")

        _target_ids = set(_mapping.keys())
        for _node_id, (_node, _obj) in _mapping.items():
            if getattr(_obj, "type", None) != "MESH":
                continue
            self._Configure_rigid_body(_obj, "ACTIVE", physics_cfg)

        if getattr(physics_cfg, "collide_with_scene", True):
            for _obj in self._bpy.data.objects:
                if getattr(_obj, "type", None) != "MESH":
                    continue
                if any(_obj is _mapped_obj for _, _mapped_obj in _mapping.values()):
                    continue
                self._Configure_rigid_body(_obj, "PASSIVE", physics_cfg)

        self._Ensure_ground_plane(physics_cfg)

        _scene = self._bpy.context.scene
        _scene.frame_set(1)
        for _frame in range(1, max(1, int(physics_cfg.settle_frames)) + 1):
            _scene.frame_set(_frame)

        for _node_id in _target_ids:
            _node, _obj = _mapping[_node_id]
            _world = np.array(_obj.matrix_world, dtype=np.float32)
            _node.local_rigid = self._World_to_local_rigid(_node, _world)

    def _Set_render_engine(self) -> None:
        """Configures GPU render engine. Runs once per scene load.

        Priority: CYCLES+GPU (OPTIX→CUDA→HIP) → EEVEE fallback.
        """
        if self._render_engine_set:
            return
        _bpy = self._bpy
        _scene = _bpy.context.scene

        # CYCLES with GPU
        try:
            _cycles_prefs = _bpy.context.preferences.addons["cycles"].preferences
            for _compute_type in ("OPTIX", "CUDA", "HIP"):
                try:
                    _cycles_prefs.compute_device_type = _compute_type
                    _cycles_prefs.get_devices()
                    _gpu_devs = [d for d in _cycles_prefs.devices if d.type != "CPU"]
                    if not _gpu_devs:
                        continue
                    for _dev in _gpu_devs:
                        _dev.use = True
                    _scene.render.engine = "CYCLES"
                    _scene.cycles.device = "GPU"
                    _scene.cycles.samples = 32
                    _scene.cycles.use_denoising = False
                    self._render_engine_set = True
                    _names = [d.name for d in _gpu_devs]
                    print(f"[bridge] CYCLES GPU ({_compute_type}): {_names}")
                    return
                except Exception:
                    continue
        except Exception:
            pass

        # EEVEE fallback
        for _engine in ("BLENDER_EEVEE_NEXT", "BLENDER_EEVEE", "BLENDER_WORKBENCH", "CYCLES"):
            try:
                _scene.render.engine = _engine
                if hasattr(_scene, "eevee"):
                    _scene.eevee.taa_render_samples = 32
                self._render_engine_set = True
                print(f"[bridge] fallback engine: {_engine} (CPU)")
                return
            except Exception:
                continue
        raise RuntimeError("No supported Blender render engine available.")

    def Render_channels(
        self,
        scene_controller: Controller,
        camera_label: str,
        request: Render_Request,
        segmentation_target: Base_Node | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, dict[str, object]]]:
        """Renders all requested channels for one Blender camera.

        Args:
            scene_controller: Shared scene controller used to derive metadata.
            camera_label: Camera label resolved from the shared scene.
            request: Channel request for the current frame.

        Returns:
            A tuple of rendered images and channel metadata.
        """
        _scene = self._bpy.context.scene
        _cam_entry = self._camera_obj_map.get(camera_label)
        if _cam_entry is None:
            raise KeyError(f"Configured Blender camera not found: {camera_label}")
        _cam_node, _camera_obj = _cam_entry

        self._Set_render_engine()
        _passes_key = tuple(sorted(request.channels))
        if _passes_key != self._passes_key:
            Configure_passes(self._bpy, request)
            self._passes_key = _passes_key
        _scene.camera = _camera_obj
        _scene.render.resolution_x = int(_cam_node.intrinsic.width)
        _scene.render.resolution_y = int(_cam_node.intrinsic.height)
        _scene.frame_current = 1

        _seg_index_map = {}
        _seg_object_name: str | None = None
        if SEGMENTATION in request.channels:
            if segmentation_target is not None:
                _seg_object_name = Resolve_segmentation_object_name(self._bpy, segmentation_target)
            else:
                _seg_index_map = Assign_segmentation_indices(self._bpy, scene_controller)

        _compositor_key = (camera_label, tuple(sorted(request.channels)))
        if _compositor_key != self._compositor_key:
            _outputs, self._compositor_crypto_node = Configure_compositor_outputs(
                bpy=self._bpy,
                temp_dir=self.temp_dir,
                request=request,
                camera_label=camera_label,
                segmentation_object_name=_seg_object_name,
            )
            self._compositor_key = _compositor_key
            self._compositor_outputs = _outputs
        else:
            _outputs, self._compositor_crypto_node = Configure_compositor_outputs(
                bpy=self._bpy,
                temp_dir=self.temp_dir,
                request=request,
                camera_label=camera_label,
                segmentation_object_name=_seg_object_name,
                existing_crypto_node=self._compositor_crypto_node,
            )
            self._compositor_outputs = _outputs
        self._bpy.ops.render.render(write_still=False)

        _images: dict[str, np.ndarray] = {}
        _metadata: dict[str, dict[str, object]] = {}
        for _channel, _path in _outputs.items():
            _resolved = Resolve_output_path(_path)
            if _channel == RGB:
                _images[_channel] = Load_rgb(self._bpy, _resolved)
            elif _channel == DEPTH:
                _images[_channel] = Load_depth(self._bpy, _resolved)
            elif _channel == NORMAL:
                _images[_channel] = Load_normal(self._bpy, _resolved)
            elif _channel == SEGMENTATION:
                _images[_channel] = Load_segmentation(self._bpy, _resolved)
                if segmentation_target is not None:
                    _metadata[_channel] = Build_target_segmentation_metadata(segmentation_target)
                else:
                    _metadata[_channel] = Build_segmentation_metadata(_seg_index_map)
        return _images, _metadata
