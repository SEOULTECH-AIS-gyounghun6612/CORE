"""Registry for scene-node classes."""
from python_toolbox.registry import Registry

from .type._base import Base_Node


NODE_REGISTRY = Registry[type[Base_Node]]("Node", Base_Node)

__all__ = ["NODE_REGISTRY"]
