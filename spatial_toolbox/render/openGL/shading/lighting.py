"""Lighting defaults and state application for OpenGL rendering."""
from __future__ import annotations

from typing import Sequence

from OpenGL.GL import (
    GL_AMBIENT,
    GL_COLOR_MATERIAL,
    GL_DIFFUSE,
    GL_FRONT_AND_BACK,
    GL_LIGHT0,
    GL_LIGHTING,
    GL_LIGHT_MODEL_TWO_SIDE,
    GL_NORMALIZE,
    GL_POSITION,
    GL_SHININESS,
    GL_SPECULAR,
    glEnable,
    glLightfv,
    glLightModeli,
    glMaterialf,
    glMaterialfv,
)

DEFAULT_LIGHT_POSITION: list[float] = [10.0, 10.0, 10.0, 1.0]
DEFAULT_LIGHT_DIFFUSE: list[float] = [0.9, 0.9, 0.9, 1.0]
DEFAULT_LIGHT_AMBIENT: list[float] = [0.2, 0.2, 0.2, 1.0]
DEFAULT_LIGHT_SPECULAR: list[float] = [0.1, 0.1, 0.1, 1.0]
DEFAULT_MATERIAL_SPECULAR: list[float] = [0.15, 0.15, 0.15, 1.0]
DEFAULT_MATERIAL_SHININESS: float = 8.0


def Apply_lighting_state(
    light_position: Sequence[float],
    light_diffuse: Sequence[float],
    light_ambient: Sequence[float],
    light_specular: Sequence[float],
    material_specular: Sequence[float],
    material_shininess: float,
) -> None:
    """Applies a basic single-light material setup to OpenGL."""
    for _flag in (GL_LIGHTING, GL_LIGHT0, GL_COLOR_MATERIAL, GL_NORMALIZE):
        glEnable(_flag)
    glLightModeli(GL_LIGHT_MODEL_TWO_SIDE, 1)
    glLightfv(GL_LIGHT0, GL_POSITION, light_position)
    glLightfv(GL_LIGHT0, GL_DIFFUSE, light_diffuse)
    glLightfv(GL_LIGHT0, GL_AMBIENT, light_ambient)
    glLightfv(GL_LIGHT0, GL_SPECULAR, light_specular)
    glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, material_specular)
    glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, material_shininess)
