from typing import TypeVar

from .register import ASSET_REGISTRY
from .type._base import Base_Asset
from .type.mesh import Mesh
from .type.points import Point_Cloud


Asset = TypeVar("Asset", bound=Base_Asset)

__all__ = [
    "ASSET_REGISTRY",
    "Base_Asset",
    "Mesh",
    "Point_Cloud",
]
