"""Shared capture-engine contracts for simulation backends."""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable

from ...scene import Controller
from .config import Sim_Config

Progress_Callback = Callable[[int, int, str], None]


class Base_Capture_Engine(ABC):
    """Abstract interface for dataset capture backends."""

    @abstractmethod
    def Capture(
        self,
        scene: Controller,
        config: Sim_Config,
        output_dir: Path,
        progress_callback: Progress_Callback | None = None,
    ) -> None:
        """Captures a dataset from the shared scene and simulation config."""
        ...
