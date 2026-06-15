from python_toolbox.registry import Registry

from .type._base import Base_Asset


ASSET_REGISTRY = Registry[type[Base_Asset]]("Asset", Base_Asset)

__all__ = ["ASSET_REGISTRY"]
