"""Backend-agnostic render request, result, and lifecycle contracts."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Self

import numpy as np

from ...scene import Controller


@dataclass(slots=True)
class Render_Request:
    """Describes a single render job shared across backends.

    Attributes:
        channels: Render channels requested for every resolved camera.
    """

    channels: list[str]


@dataclass(slots=True)
class Render_Result:
    """Collects images and metadata produced for one camera.

    Attributes:
        images: Channel-to-image mapping.
        metadata: Channel-to-metadata mapping.
    """

    images: dict[str, np.ndarray] = field(default_factory=dict)
    metadata: dict[str, dict[str, object]] = field(default_factory=dict)

    def Get_image(self, channel: str) -> np.ndarray:
        """Returns the image stored for a render channel."""
        return self.images[channel]

    def Get_metadata(self, channel: str) -> dict[str, object]:
        """Returns metadata stored for a render channel."""
        return self.metadata.get(channel, {})


class Renderer(ABC):
    """Defines the backend lifecycle and render entrypoint.

    Subclasses are expected to own the backend context and any cached
    resources. The interface can be used directly or through a context
    manager.
    """

    @abstractmethod
    def Setup(self) -> None:
        """Initializes backend resources before rendering."""
        ...

    @abstractmethod
    def Teardown(self) -> None:
        """Releases backend resources after rendering."""
        ...

    @abstractmethod
    def Render(
        self,
        scene: Controller,
        camera_labels: list[str],
        request: Render_Request,
    ) -> dict[str, Render_Result]:
        """Renders the requested channels for the resolved camera set."""
        ...

    def __enter__(self) -> Self:
        """Enters the renderer lifecycle as a context manager."""
        self.Setup()
        return self

    def __exit__(self, *_: object) -> None:
        """Leaves the renderer lifecycle as a context manager."""
        self.Teardown()
