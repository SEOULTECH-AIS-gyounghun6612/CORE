"""Primitive dispatch layer for OpenGL drawing."""
from __future__ import annotations

from typing import Any

from .mesh import Mesh_Drawer


class Draw_Dispatcher:
    """Routes draw calls to primitive-specific drawers."""

    def __init__(self) -> None:
        self._mesh = Mesh_Drawer()

    def Draw(
        self,
        primitive: str,
        data: Any,
        mode: str = "default",
        node: Any = None,
        use_vbo: bool = True,
    ) -> None:
        """Draws one primitive using the matching drawer implementation."""
        if primitive == "mesh":
            self._mesh.Draw(data, mode=mode, node=node, use_vbo=use_vbo)
            return
        raise KeyError(f"Unsupported primitive: {primitive}")

    def Clear_resources(self) -> None:
        """Releases cached primitive resources."""
        self._mesh.Clear_resources()

    def Reset_id_state(self) -> None:
        """Clears per-node ID allocation used by segmentation drawing."""
        self._mesh.Reset_id_state()

    @property
    def id_map(self) -> dict[tuple, Any]:
        """Returns the current color-to-node mapping."""
        return self._mesh.id_map
