"""Scene-node traversal helpers."""
from __future__ import annotations

from typing import Callable, Iterator

from ..type._base import Base_Node


def walk_nodes(
    root: Base_Node,
    predicate: Callable[[Base_Node], bool]
) -> Iterator[Base_Node]:
    """Recursively yields nodes that match the predicate."""
    if predicate(root):
        yield root
    for _child in root.children:
        yield from walk_nodes(_child, predicate)
