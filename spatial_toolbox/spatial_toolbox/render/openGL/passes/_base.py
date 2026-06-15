"""Base pass implementation shared by OpenGL render channels."""
from __future__ import annotations

from abc import ABC
from typing import Any

import numpy as np
from OpenGL.GL import (
    GL_DEPTH_COMPONENT,
    GL_FLOAT,
    GL_LIGHTING,
    GL_MODELVIEW,
    GL_RGB,
    GL_UNSIGNED_BYTE,
    GLfloat,
    GLubyte,
    glDisable,
    glEnable,
    glFlush,
    glMatrixMode,
    glMultMatrixf,
    glPopMatrix,
    glPushMatrix,
    glReadPixels,
)

from ....scene import ASSET_CACHE
from ....scene.node import Base_Node
from ....scene.node.type.camera import Camera
from ....scene.node.type.mesh import Mesh
from ..utils import to_gl_matrix


class OpenGL_Base_Pass(ABC):
    """Defines the execution template for an OpenGL render pass."""
    _clear_color: tuple[float, ...] = (0.0, 0.0, 0.0, 1.0)
    _readback_format: str = "rgb"
    _use_lighting: bool = False
    _draw_mode: str = "default"

    def Execute(self, renderer: Any, render_queue: list[Base_Node]) -> None:
        """Runs the pass lifecycle against a prepared renderer."""
        self._renderer = renderer
        self._On_setup()
        self._On_post_camera()
        self._On_draw(render_queue)
        glFlush()
        self._On_cleanup()

    def Readback(self, width: int, height: int, camera_node: Camera) -> np.ndarray:
        """Reads the rendered buffer and flips it into image coordinates."""
        _raw = self._On_readback(width, height, camera_node=camera_node)
        return np.ascontiguousarray(np.flipud(_raw))

    @property
    def Name(self) -> str:
        """Returns the public channel name of the pass."""
        return self.name

    def Build_metadata(self) -> dict[str, object]:
        """Builds per-channel metadata for the current pass."""
        return {}

    def _On_setup(self) -> None:
        """Applies pass-specific OpenGL state before drawing."""
        if self._use_lighting:
            glEnable(GL_LIGHTING)
        else:
            glDisable(GL_LIGHTING)

    def _On_draw(self, render_queue: list[Base_Node]) -> None:
        """Draws the scene content targeted by the pass."""
        self._Draw_scene(render_queue)

    def _On_readback(self, width: int, height: int, **kwargs) -> np.ndarray:
        """Dispatches framebuffer readback based on the configured format."""
        if self._readback_format == "rgb":
            return self._Read_rgb(width, height)
        if self._readback_format == "depth":
            return self._Read_depth(width, height)
        raise NotImplementedError(f"Unsupported readback format: {self._readback_format}")

    def _On_post_camera(self) -> None:
        """Hook executed after camera matrices are loaded."""
        return None

    def _On_cleanup(self) -> None:
        """Hook executed after drawing and readback."""
        return None

    def _Draw_scene(self, render_queue: list[Base_Node]) -> None:
        """Draws all queued nodes accepted by the pass."""
        for _node in render_queue:
            self._Draw_one_mesh(_node)

    def _Draw_one_mesh(self, node: Base_Node) -> None:
        """Draws one mesh node if it resolves to cached geometry."""
        if not isinstance(node, Mesh) or node.source_key is None:
            return
        _asset = ASSET_CACHE.Get(node.source_key, is_hold=True)
        if _asset is None or getattr(_asset, "geometry", None) is None:
            return

        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glMultMatrixf(to_gl_matrix(node.world_matrix))
        self._renderer.Draw("mesh", _asset.geometry, mode=self._draw_mode, node=node)
        glPopMatrix()

    @staticmethod
    def _Read_rgb(width: int, height: int) -> np.ndarray:
        """Reads an RGB framebuffer into a ``uint8`` array."""
        _buf = (GLubyte * (width * height * 3))()
        glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE, _buf)
        return np.frombuffer(_buf, dtype=np.uint8).reshape(height, width, 3)

    @staticmethod
    def _Read_depth(width: int, height: int) -> np.ndarray:
        """Reads a depth framebuffer into a ``float32`` array."""
        _buf = (GLfloat * (width * height))()
        glReadPixels(0, 0, width, height, GL_DEPTH_COMPONENT, GL_FLOAT, _buf)
        return np.frombuffer(_buf, dtype=np.float32).reshape(height, width)
