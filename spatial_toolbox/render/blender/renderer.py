"""Blender-backed renderer implementation."""
from __future__ import annotations

import numpy as np

from ...scene import Controller
from ...scene.node.type._base import Base_Node
from ..core.camera import Resolve_cameras
from ..core.channel import DEPTH, NORMAL, RGB, SEGMENTATION
from ..core.renderer import Render_Request, Render_Result, Renderer
from .scene import Blender_Scene_Bridge
from .session import Blender_Session


class Blender_Renderer(Renderer):
    """Renders channels through a headless Blender session.

    Blender handles a full render request by configuring one scene and running
    its compositor outputs, rather than exposing per-channel pass objects.
    """

    _SUPPORTED_CHANNELS = {RGB, DEPTH, NORMAL, SEGMENTATION}

    def __init__(self) -> None:
        self._session = Blender_Session()
        self._scene_bridge: Blender_Scene_Bridge | None = None

    def Setup(self) -> None:
        """Ensures the Blender Python session is available."""
        _ = self._session.bpy

    def Teardown(self) -> None:
        """Keeps teardown as a no-op for the managed Blender session."""
        return None

    def _Require_scene_bridge(self) -> Blender_Scene_Bridge:
        """Lazily creates the Blender scene bridge."""
        if self._scene_bridge is None:
            self._scene_bridge = Blender_Scene_Bridge(self._session.bpy)
        return self._scene_bridge

    def _Validate_request(self, request: Render_Request) -> None:
        """Validates that all requested channels are supported by Blender."""
        _unknown = [c for c in request.channels if c not in self._SUPPORTED_CHANNELS]
        if _unknown:
            raise KeyError(f"Unsupported render channel(s): {_unknown}")

    def Sync_transforms(self, scene: Controller, group_nodes: list | None = None) -> None:
        """Pre-applies transforms to Blender. Call once per sample before inner render loop."""
        self._Require_scene_bridge().Sync_transforms(scene, group_nodes=group_nodes)

    def Render_scene(
        self,
        scene_controller: Controller,
        camera_labels: list[str],
        request: Render_Request,
        segmentation_target: Base_Node | None = None,
        _visibility_only: bool = False,
    ) -> dict[str, Render_Result]:
        """Renders the requested channels using the Blender scene bridge."""
        self._Validate_request(request)
        _bridge = self._Require_scene_bridge()
        if _visibility_only:
            _bridge.Sync_visibility(scene_controller)
        else:
            _bridge.Sync_scene(scene_controller)
        _cameras = Resolve_cameras(scene_controller, camera_labels)

        _results: dict[str, Render_Result] = {}
        for _camera in _cameras:
            _images, _metadata = _bridge.Render_channels(
                scene_controller=scene_controller,
                camera_label=_camera.label,
                request=request,
                segmentation_target=segmentation_target,
            )
            _results[_camera.label] = Render_Result(images=_images, metadata=_metadata)
        return _results

    def Render(
        self,
        scene: Controller,
        camera_labels: list[str],
        request: Render_Request,
        segmentation_target: Base_Node | None = None,
        _visibility_only: bool = False,
    ) -> dict[str, Render_Result]:
        """Implements the backend-agnostic renderer entrypoint."""
        return self.Render_scene(
            scene, camera_labels, request,
            segmentation_target=segmentation_target,
            _visibility_only=_visibility_only,
        )

    @staticmethod
    def Empty_result(request: Render_Request) -> Render_Result:
        """Builds an empty result payload for the requested channels."""
        return Render_Result(
            images={c: np.empty((0, 0), dtype=np.uint8) for c in request.channels},
            metadata={},
        )
