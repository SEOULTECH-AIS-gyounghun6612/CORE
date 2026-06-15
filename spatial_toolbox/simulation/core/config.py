"""Simulation configuration schemas and randomization helpers."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from python_toolbox import Data_Schema
from python_toolbox.project import Base_Config

from ...scene.node.utils.transform import Build_transform

Output_Layout = Literal["per_object", "flat"]
Randomizable = float | list[float]


@dataclass
class Randomize_Range(Data_Schema):
    """Defines randomization ranges for translation and rotation deltas.

    Attributes:
        tx: Translation delta on the X axis.
        ty: Translation delta on the Y axis.
        tz: Translation delta on the Z axis.
        rx: Rotation delta on the X axis in degrees.
        ry: Rotation delta on the Y axis in degrees.
        rz: Rotation delta on the Z axis in degrees.
    """

    tx: Randomizable = 0.0
    ty: Randomizable = 0.0
    tz: Randomizable = 0.0
    rx: Randomizable = 0.0
    ry: Randomizable = 0.0
    rz: Randomizable = 0.0


@dataclass
class Physics_Drop_Config(Data_Schema):
    """Configures an optional Blender rigid-body settling pass.

    Attributes:
        enabled: Whether to run physics settling before capture.
        floor_z: World-space height of the synthetic ground plane.
        use_ground_plane: Whether to add a passive ground plane.
        settle_frames: Number of simulation frames to advance.
        steps_per_second: Physics update rate for the rigid-body world.
        solver_iterations: Solver iteration count for the rigid-body world.
        mass: Active rigid-body mass assigned to target objects.
        friction: Shared friction used for active and passive bodies.
        restitution: Bounce coefficient used for active and passive bodies.
        linear_damping: Linear damping applied to target objects.
        angular_damping: Angular damping applied to target objects.
        collision_shape: Blender rigid-body collision shape for targets.
        collide_with_scene: Whether non-target mesh objects become passive
            colliders.
    """

    enabled: bool = False
    floor_z: float = 0.0
    use_ground_plane: bool = True
    settle_frames: int = 24
    steps_per_second: int = 120
    solver_iterations: int = 20
    mass: float = 1.0
    friction: float = 0.5
    restitution: float = 0.0
    linear_damping: float = 0.04
    angular_damping: float = 0.1
    collision_shape: Literal["MESH", "CONVEX_HULL", "BOX"] = "CONVEX_HULL"
    collide_with_scene: bool = True


@dataclass
class Sim_Config(Base_Config):
    """Defines backend-agnostic simulation capture settings.

    Attributes:
        scene_path: Scene JSON or USD file path.
        target_label: Label of the structural group whose children are sampled.
        camera_labels: Camera labels or camera-group labels used for capture.
        num_samples: Number of captures generated per target object.
        output_layout: Output directory layout policy.
        seed: Optional random seed for deterministic sampling.
        cam: Default camera transform perturbation.
        cam_overrides: Per-camera transform perturbation overrides.
        obj: Target-object local transform perturbation.
        physics_drop: Optional rigid-body settling pass run before capture.
    """

    scene_path: str = ""
    target_label: str = "target"
    camera_labels: list[str] = field(default_factory=lambda: ["main_camera"])
    num_samples: int = 1
    output_layout: Output_Layout = "per_object"
    seed: int | None = None
    cam: Randomize_Range = field(default_factory=Randomize_Range)
    cam_overrides: dict[str, Randomize_Range] = field(default_factory=dict)
    obj: Randomize_Range = field(default_factory=Randomize_Range)
    physics_drop: Physics_Drop_Config = field(default_factory=Physics_Drop_Config)

    def __post_init__(self) -> None:
        """Normalizes nested dict inputs into schema instances."""
        if isinstance(self.cam, dict):
            self.cam = Randomize_Range(**self.cam)
        if isinstance(self.obj, dict):
            self.obj = Randomize_Range(**self.obj)
        if isinstance(self.physics_drop, dict):
            self.physics_drop = Physics_Drop_Config(**self.physics_drop)
        if isinstance(self.camera_labels, str):
            self.camera_labels = [self.camera_labels]
        self.cam_overrides = {
            _k: Randomize_Range(**_v) if isinstance(_v, dict) else _v
            for _k, _v in self.cam_overrides.items()
        }


def Sample_delta_matrix(r: Randomize_Range) -> np.ndarray:
    """Samples a rigid transform matrix from a randomization range."""
    return Build_transform(
        tx=_Sample_value(r.tx),
        ty=_Sample_value(r.ty),
        tz=_Sample_value(r.tz),
        rx=_Sample_value(r.rx),
        ry=_Sample_value(r.ry),
        rz=_Sample_value(r.rz),
    )


def Sample_translation(r: Randomize_Range) -> tuple[float, float, float]:
    """Samples only the translation part of a randomization range."""
    return (
        _Sample_value(r.tx),
        _Sample_value(r.ty),
        _Sample_value(r.tz),
    )


def _Sample_value(v: Randomizable) -> float:
    """Samples a float from a fixed value or ``[min, max]`` range."""
    if isinstance(v, list):
        if len(v) >= 2:
            return float(np.random.uniform(v[0], v[1]))
        return float(v[0])
    return float(v)
