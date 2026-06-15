"""Normal pass for the OpenGL renderer."""
from __future__ import annotations

from OpenGL.GL import GL_DITHER, glDisable, glEnable

from ._base import OpenGL_Base_Pass


class Normal_Pass(OpenGL_Base_Pass):
    """Renders normals encoded into RGB colors."""
    name = "normal"
    _clear_color = (0.5, 0.5, 1.0, 1.0)
    _readback_format = "rgb"
    _use_lighting = False
    _draw_mode = "normal_color"

    def _On_setup(self) -> None:
        """Disables dithering to preserve encoded normal colors."""
        super()._On_setup()
        glDisable(GL_DITHER)

    def _On_cleanup(self) -> None:
        """Restores dithering after the pass completes."""
        glEnable(GL_DITHER)
