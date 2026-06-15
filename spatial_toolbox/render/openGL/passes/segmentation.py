"""Segmentation pass that renders node IDs as colors."""
from __future__ import annotations

from OpenGL.GL import GL_DITHER, glDisable, glEnable

from ....scene.node import Base_Node
from ._base import OpenGL_Base_Pass


class Segmentation_Pass(OpenGL_Base_Pass):
    """Renders segmentation colors and exposes the color-to-node map."""
    name = "segmentation"
    _readback_format = "rgb"
    _use_lighting = False
    _draw_mode = "id_color"

    @property
    def last_id_map(self) -> dict[tuple[int, int, int], Base_Node]:
        """Returns the latest color-to-node mapping built by the renderer."""
        return self._renderer.id_map

    def Build_metadata(self) -> dict[str, object]:
        """Exports the current segmentation lookup table."""
        return {"id_map": self.last_id_map}

    def _On_setup(self) -> None:
        """Disables dithering to keep ID colors exact."""
        super()._On_setup()
        glDisable(GL_DITHER)

    def _On_cleanup(self) -> None:
        """Restores dithering after the pass completes."""
        glEnable(GL_DITHER)
