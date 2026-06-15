"""Blender-backed simulation capture engine."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ...scene import Controller
from ...scene.node.type.camera import Camera
from ...scene.node.utils.traversal import walk_nodes
from ...render.blender import Blender_Renderer, Blender_Scene_Bridge, Blender_Session
from ...render.core.camera import Resolve_cameras
from ...render.core.channel import DEPTH, NORMAL, RGB, SEGMENTATION
from ...render.core.renderer import Render_Request
from ..core.config import Sim_Config, Sample_delta_matrix
from ..core.engine import Base_Capture_Engine, Progress_Callback
from ..core.exporter import Result_Exporter

_DEFAULT_CHANNELS = [RGB, DEPTH, NORMAL, SEGMENTATION]


class Blender_Capture_Engine(Base_Capture_Engine):
    """Captures per-object datasets through the Blender renderer."""

    def __init__(self, channels: list[str] | None = None) -> None:
        self._channels = channels if channels is not None else _DEFAULT_CHANNELS

    def Capture(
        self,
        scene: Controller,
        config: Sim_Config,
        output_dir: Path,
        progress_callback: Progress_Callback | None = None,
    ) -> None:
        """Captures samples by isolating each target child in sequence."""
        if config.seed is not None:
            np.random.seed(config.seed)

        _target = self._Find_target_group(scene, config.target_label)
        _cameras = Resolve_cameras(scene, config.camera_labels)
        _request = Render_Request(channels=self._channels)
        _children = list(_target.children)

        if config.physics_drop.enabled and _children:
            self._Apply_physics_drop(scene, _children, config)

        _total = len(_children) * config.num_samples
        _frame_id = 0

        with Blender_Renderer() as _renderer:
            _target.visible = False

            for _obj in _children:
                _obj.visible = True
                _exporter = self._Make_exporter(output_dir, _obj.label, config)
                _orig_obj_rigid = _obj.local_rigid.copy()

                for _sample_idx in range(config.num_samples):
                    _obj.local_rigid = _orig_obj_rigid @ Sample_delta_matrix(config.obj)

                    _cam_restores: list[tuple[Camera, np.ndarray]] = []
                    for _cam in _cameras:
                        _orig_cam = _cam.local_rigid.copy()
                        _range = config.cam_overrides.get(_cam.label, config.cam)
                        _cam.local_rigid = _orig_cam @ Sample_delta_matrix(_range)
                        _cam_restores.append((_cam, _orig_cam))

                    _results = _renderer.Render(
                        scene,
                        config.camera_labels,
                        _request,
                        segmentation_target=_obj,
                    )

                    for _cam in _cameras:
                        _res = _results.get(_cam.label)
                        if _res is None:
                            continue
                        _exporter.Save(
                            frame_id=_frame_id,
                            result=_res,
                            camera_node=_cam,
                            config=config,
                            extra_meta={
                                "object": _obj.label,
                                "sample": _sample_idx,
                            },
                        )

                    for _cam, _orig_cam in _cam_restores:
                        _cam.local_rigid = _orig_cam

                    _frame_id += 1
                    if progress_callback:
                        progress_callback(
                            _frame_id,
                            _total,
                            f"{_obj.label} [{_sample_idx + 1}/{config.num_samples}]",
                        )

                _obj.local_rigid = _orig_obj_rigid
                _obj.visible = False

            _target.visible = True

    @staticmethod
    def _Find_target_group(scene: Controller, target_label: str):
        """Finds the structural target group configured for capture."""
        for _node in walk_nodes(
            scene.root,
            lambda n: n.label == target_label and n.prim_type == "Xform",
        ):
            return _node
        raise ValueError(f"씬에 label='{target_label}' Xform 그룹이 없음")

    @staticmethod
    def _Make_exporter(
        output_dir: Path,
        object_label: str,
        config: Sim_Config,
    ) -> Result_Exporter:
        """Builds an exporter using the configured output layout."""
        if config.output_layout == "per_object":
            return Result_Exporter(output_dir / object_label)
        return Result_Exporter(output_dir)

    @staticmethod
    def _Apply_physics_drop(
        scene: Controller,
        target_nodes: list,
        config: Sim_Config,
    ) -> None:
        """Settles target objects in Blender before capture."""
        _bridge = Blender_Scene_Bridge(Blender_Session().bpy)
        _bridge.Simulate_physics_drop(scene, target_nodes, config.physics_drop)
