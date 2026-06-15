"""OpenGL-backed simulation capture engine."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ...scene import Controller
from ...scene.node.type.camera import Camera
from ...scene.node.utils.traversal import walk_nodes
from ...render.core.camera import Resolve_cameras
from ...render.core.channel import DEPTH, NORMAL, RGB, SEGMENTATION
from ...render.core.renderer import Render_Request
from ...render.openGL import OpenGL_Renderer
from ...render.openGL.context import Context_Type
from ..core.config import Sim_Config, Sample_delta_matrix
from ..core.engine import Base_Capture_Engine, Progress_Callback
from ..core.exporter import Result_Exporter

_DEFAULT_CHANNELS = [RGB, DEPTH, NORMAL, SEGMENTATION]


class OpenGL_Capture_Engine(Base_Capture_Engine):
    """Captures per-object datasets through the OpenGL renderer.

    Uses visibility toggling to isolate each target child per frame.
    Resolution must match the camera intrinsic parameters in the scene.
    """

    def __init__(
        self,
        width: int = 640,
        height: int = 480,
        channels: list[str] | None = None,
        context_type: Context_Type = "auto",
    ) -> None:
        self._width = width
        self._height = height
        self._channels = channels if channels is not None else _DEFAULT_CHANNELS
        self._context_type = context_type

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
        _cam_by_label: dict[str, Camera] = {_c.label: _c for _c in _cameras}
        _request = Render_Request(channels=self._channels)
        _children = list(_target.children)
        _total = len(_children) * config.num_samples
        _frame_id = 0

        with OpenGL_Renderer(self._width, self._height, self._context_type) as _renderer:
            self._Set_subtree_visible(_target, False)

            for _obj in _children:
                self._Set_subtree_visible(_obj, True)
                _exporter = self._Make_exporter(output_dir, _obj.label, config)
                _orig_obj_rigid = _obj.local_rigid.copy()

                for _sample_idx in range(config.num_samples):
                    _obj.local_rigid = _orig_obj_rigid @ Sample_delta_matrix(config.obj)

                    _cam_restores: list[tuple[Camera, np.ndarray]] = []
                    for _cam in _cameras:
                        _orig = _cam.local_rigid.copy()
                        _range = config.cam_overrides.get(_cam.label, config.cam)
                        _cam.local_rigid = _orig @ Sample_delta_matrix(_range)
                        _cam_restores.append((_cam, _orig))

                    _results = _renderer.Render(scene, config.camera_labels, _request)

                    for _cam_label, _res in _results.items():
                        _cam_node = _cam_by_label.get(_cam_label)
                        if _cam_node is None:
                            continue
                        _exporter.Save(
                            frame_id=_frame_id,
                            result=_res,
                            camera_node=_cam_node,
                            config=config,
                            extra_meta={
                                "object": _obj.label,
                                "sample": _sample_idx,
                            },
                        )

                    for _cam, _orig in _cam_restores:
                        _cam.local_rigid = _orig

                    _frame_id += 1
                    if progress_callback:
                        progress_callback(
                            _frame_id,
                            _total,
                            f"{_obj.label} [{_sample_idx + 1}/{config.num_samples}]",
                        )

                _obj.local_rigid = _orig_obj_rigid
                self._Set_subtree_visible(_obj, False)

            self._Set_subtree_visible(_target, True)

    @staticmethod
    def _Set_subtree_visible(node, v: bool) -> None:
        """Sets visible on node and all descendants.

        _Propagate_visibility only propagates False, so True must be walked manually.
        """
        node.visible = v
        if v:
            for _child in node.children:
                OpenGL_Capture_Engine._Set_subtree_visible(_child, v)

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
