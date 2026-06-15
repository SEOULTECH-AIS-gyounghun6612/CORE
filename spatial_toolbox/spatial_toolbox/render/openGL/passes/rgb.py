"""RGB color pass for the OpenGL renderer."""
from __future__ import annotations

from OpenGL.GL import glColor3f

from ._base import OpenGL_Base_Pass


class RGB_Pass(OpenGL_Base_Pass):
    """Renders lit RGB output."""
    name = "rgb"
    _clear_color = (0.0, 0.0, 0.0, 1.0)
    _readback_format = "rgb"
    _use_lighting = True

    def _On_setup(self) -> None:
        """Applies the base setup and a neutral default draw color."""
        super()._On_setup()
        glColor3f(0.7, 0.7, 0.7)
