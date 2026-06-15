"""OpenGL renderer implementation for the shared render contract."""
from __future__ import annotations

from typing import Any

import numpy as np
from OpenGL.GL import (
    GL_COLOR_BUFFER_BIT,
    GL_DEPTH_BUFFER_BIT,
    GL_DEPTH_TEST,
    GL_LIGHTING,
    GL_MODELVIEW,
    GL_PROJECTION,
    glClear,
    glClearColor,
    glDisable,
    glEnable,
    glLoadIdentity,
    glLoadMatrixf,
    glMatrixMode,
    glViewport,
)

from ...scene import Controller
from ...scene.node.type.camera import Camera
from ..core.camera import Resolve_cameras
from ..core.channel import SEGMENTATION
from ..core.renderer import Render_Request, Render_Result, Renderer
from .context import Base_GL_Context, Build_GL_Context, Context_Type
from .draw import Draw_Dispatcher
from .passes import PASS_TYPES
from .utils.matrix import to_gl_matrix, Build_gl_projection
from .passes._base import OpenGL_Base_Pass
from .shading import (
    DEFAULT_LIGHT_AMBIENT,
    DEFAULT_LIGHT_DIFFUSE,
    DEFAULT_LIGHT_POSITION,
    DEFAULT_LIGHT_SPECULAR,
    DEFAULT_MATERIAL_SHININESS,
    DEFAULT_MATERIAL_SPECULAR,
    Apply_lighting_state,
)


class OpenGL_Renderer(Renderer):
    """Renders scene channels through an OpenGL context.

    The renderer owns the OpenGL context, a draw dispatcher, and per-channel
    render-pass instances. Each render call resolves camera nodes first and
    then executes the requested channels for every camera.

    Attributes:
        width: Output image width in pixels.
        height: Output image height in pixels.
        light_position: World-space light position passed to the shading state.
        light_diffuse: Diffuse light color.
        light_ambient: Ambient light color.
        light_specular: Specular light color.
        material_specular: Specular material response.
        material_shininess: Material shininess exponent.
    """
    _PASS_TYPES: dict[str, type[OpenGL_Base_Pass]] = dict(PASS_TYPES)

    def __init__(self, w: int, h: int, context_type: Context_Type = "auto"):
        self.width = w
        self.height = h
        self._context: Base_GL_Context = Build_GL_Context(w, h, context_type)
        self.light_position = list(DEFAULT_LIGHT_POSITION)
        self.light_diffuse = list(DEFAULT_LIGHT_DIFFUSE)
        self.light_ambient = list(DEFAULT_LIGHT_AMBIENT)
        self.light_specular = list(DEFAULT_LIGHT_SPECULAR)
        self.material_specular = list(DEFAULT_MATERIAL_SPECULAR)
        self.material_shininess = float(DEFAULT_MATERIAL_SHININESS)
        self._drawer = Draw_Dispatcher()
        self._pass_cache: dict[str, OpenGL_Base_Pass] = {}

    def Setup(self) -> None:
        """Initializes the owned OpenGL context."""
        self._context.Setup()

    def Teardown(self) -> None:
        """Releases renderer resources and tears down the context."""
        self.Clear_resources()
        self._context.Teardown()

    def Configure_lighting(self, cfg: Any) -> None:
        """Copies lighting values from a backend config object.

        Args:
            cfg: Config object exposing the lighting attributes used by the
                renderer.
        """
        for attr in ("position", "diffuse", "ambient", "specular"):
            setattr(self, f"light_{attr}", list(getattr(cfg, f"light_{attr}")))
        self.material_specular = list(cfg.material_specular)
        self.material_shininess = float(cfg.material_shininess)

    def Render(
        self,
        scene: Controller,
        camera_labels: list[str],
        request: Render_Request,
    ) -> dict[str, Render_Result]:
        """Renders the requested channels for the resolved camera set.

        Args:
            scene: Scene controller providing drawables and node lookup.
            camera_labels: Camera labels or group labels resolved into cameras.
            request: Channel request shared by all cameras.

        Returns:
            A mapping from camera label to channel images and metadata.
        """
        _render_queue = scene.Get_render_queue()
        _cameras = Resolve_cameras(scene, camera_labels)
        _results: dict[str, Render_Result] = {}
        for _camera in _cameras:
            self._Setup_camera(_camera, scene.unit_length)
            _result = Render_Result()
            for _channel in request.channels:
                self._Clear_buffer(_channel)
                _image, _meta = self._Render_channel(_channel, _render_queue, _camera)
                _result.images[_channel] = _image
                if _meta:
                    _result.metadata[_channel] = _meta
            _results[_camera.label] = _result
        return _results

    def Draw(
        self,
        primitive: str = "mesh",
        data: Any = None,
        mode: str = "default",
        node: Any = None,
        use_vbo: bool = True,
    ) -> None:
        """Delegates primitive drawing to the shared draw dispatcher."""
        self._drawer.Draw(
            primitive=primitive,
            data=data,
            mode=mode,
            node=node,
            use_vbo=use_vbo,
        )

    def Clear_resources(self) -> None:
        """Releases cached draw resources such as VBOs."""
        self._drawer.Clear_resources()

    def Reset_id_state(self) -> None:
        """Clears segmentation ID allocation state before a new pass."""
        self._drawer.Reset_id_state()

    @property
    def id_map(self) -> dict[tuple, Any]:
        """Returns the current segmentation color-to-node mapping."""
        return self._drawer.id_map

    def Apply_lighting(self) -> None:
        """Applies the cached lighting state to the OpenGL context."""
        Apply_lighting_state(
            light_position=self.light_position,
            light_diffuse=self.light_diffuse,
            light_ambient=self.light_ambient,
            light_specular=self.light_specular,
            material_specular=self.material_specular,
            material_shininess=self.material_shininess,
        )

    def _Setup_camera(self, camera: Camera, unit_length: float = 1.0) -> None:
        """Loads camera projection and view matrices into OpenGL.

        Args:
            camera: Camera node that provides intrinsic and world transforms.
            unit_length: Scene unit scale forwarded to the projection builder.

        Raises:
            ValueError: If the output resolution is invalid or the camera lacks
                intrinsic parameters.
        """
        if self.width <= 0 or self.height <= 0:
            raise ValueError("Invalid resolution")
        if camera.intrinsic is None:
            raise ValueError("intrinsic is None")
        glViewport(0, 0, self.width, self.height)
        glMatrixMode(GL_PROJECTION)
        glLoadMatrixf(Build_gl_projection(camera.intrinsic, unit_length))
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        glLoadMatrixf(to_gl_matrix(np.linalg.inv(camera.world_matrix)))
        self.Apply_lighting()

    def _Get_pass(self, channel: str) -> OpenGL_Base_Pass:
        """Returns a cached pass instance for a render channel."""
        if channel not in self._PASS_TYPES:
            raise KeyError(f"Unsupported render channel: {channel}")
        if channel not in self._pass_cache:
            self._pass_cache[channel] = self._PASS_TYPES[channel]()
        return self._pass_cache[channel]

    def _Clear_buffer(self, channel: str) -> None:
        """Resets the framebuffer using pass-specific clear state."""
        _pass = self._Get_pass(channel)
        glClearColor(*getattr(_pass, "_clear_color", (0.0, 0.0, 0.0, 1.0)))
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glEnable(GL_DEPTH_TEST)
        if not getattr(_pass, "_use_lighting", False):
            glDisable(GL_LIGHTING)

    def _Render_channel(
        self,
        channel: str,
        render_queue: list[Any],
        camera_node: Camera,
    ) -> tuple[np.ndarray, dict[str, object]]:
        """Executes a single channel pass and reads back its image."""
        _pass = self._Get_pass(channel)
        if channel == SEGMENTATION:
            self.Reset_id_state()
        _pass.Execute(self, render_queue)
        _image = _pass.Readback(self.width, self.height, camera_node)
        return _image, _pass.Build_metadata()
