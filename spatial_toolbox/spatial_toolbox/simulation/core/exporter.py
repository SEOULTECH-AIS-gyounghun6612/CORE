"""Persistence helpers for simulation render results."""
from __future__ import annotations

from dataclasses import asdict, is_dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from python_toolbox.file import Write_to

from ...scene.node.type.camera import Camera
from ...render.core.renderer import Render_Result
from .config import Sim_Config


class Result_Exporter:
    """Writes render outputs and frame metadata to disk.

    The exporter stores image arrays per channel and emits one metadata file
    per frame next to them.
    """

    def __init__(self, output_dir: str | Path) -> None:
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)

    def Save(
        self,
        frame_id: int,
        result: Render_Result,
        camera_node: Camera,
        config: Sim_Config,
        extra_meta: dict | None = None,
    ) -> None:
        """Saves all channel images and metadata for one frame.

        Args:
            frame_id: Frame index used in output filenames.
            result: Render result for a single camera.
            camera_node: Camera used for this capture.
            config: Simulation config associated with the capture.
            extra_meta: Optional metadata merged into the top-level metadata
                document.
        """
        for _channel, _image in result.images.items():
            self._Save_array(
                self._output_dir / f"{_channel}_{frame_id:06d}",
                _image,
            )
        self._Save_metadata(frame_id, result, camera_node, config, extra_meta)

    def _Save_array(self, base_path: Path, data: np.ndarray) -> None:
        """Persists a channel array using a format suitable for its dtype."""
        if data.dtype == np.float32:
            np.save(
                str(base_path.with_suffix(".npy")),
                np.ascontiguousarray(data),
            )
            return
        if data.ndim == 2 and np.issubdtype(data.dtype, np.integer):
            np.save(
                str(base_path.with_suffix(".npy")),
                np.ascontiguousarray(data),
            )
            return
        Image.fromarray(data).save(str(base_path.with_suffix(".png")))

    def _Save_metadata(
        self,
        frame_id: int,
        result: Render_Result,
        camera_node: Camera,
        config: Sim_Config,
        extra_meta: dict | None = None,
    ) -> None:
        """Writes the per-frame metadata document."""
        _intrinsic = camera_node.intrinsic
        _channel_meta = {
            _ch: _make_serializable(_m)
            for _ch, _m in result.metadata.items()
        }
        _meta = {
            "camera": {
                "label": camera_node.label,
                "intrinsic": _intrinsic.Serialize() if _intrinsic else {},
                "extrinsic": camera_node.world_matrix.tolist(),
            },
            "render": _make_serializable(config.Serialize()),
            "channels": _channel_meta,
        }
        if extra_meta:
            _meta.update(extra_meta)
        Write_to(
            self._output_dir / f"metadata_{frame_id:06d}.json",
            _meta,
            indent=2,
        )


def _make_serializable(obj: object) -> object:
    """Normalizes nested metadata into file-writable primitives."""
    if isinstance(obj, dict):
        _result = {}
        for _k, _v in obj.items():
            _key = _k if isinstance(_k, str) else str(_k)
            if hasattr(_v, "label"):
                _result[_key] = {
                    "label": _v.label,
                    "prim_path": getattr(_v, "prim_path", ""),
                }
                continue
            _result[_key] = _make_serializable(_v)
        return _result
    if isinstance(obj, list):
        return [_make_serializable(_v) for _v in obj]
    if is_dataclass(obj):
        return _make_serializable(asdict(obj))
    if hasattr(obj, "Serialize"):
        return _make_serializable(obj.Serialize())
    return obj
