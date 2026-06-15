"""Raw asset loading entrypoint."""
from __future__ import annotations

from pathlib import Path
from typing import Callable

from ..type.mesh import Mesh
from .obj import Read_and_parse_obj

_LOADERS: dict[str, Callable[[Path, float], list[Mesh]]] = {
    ".obj": Read_and_parse_obj,
}

Progress_Callback = Callable[[int, int, str], None]


def Load_and_register(
    file_path: str | Path,
    unit_length: float = 1.0,
    progress_callback: Progress_Callback | None = None,
) -> list[str]:
    """Loads an asset file, registers the parsed assets, and returns the keys."""
    from ..cache import ASSET_CACHE

    _path = Path(file_path).resolve()
    if not _path.exists():
        raise FileNotFoundError(f"경로를 찾을 수 없음: {_path}")

    _loader = _LOADERS.get(_path.suffix.lower())
    if _loader is None:
        raise ValueError(f"지원하지 않는 파일 형식임: {_path.suffix}")

    if progress_callback is not None:
        progress_callback(0, 1, f"{_path.name} mesh 읽는 중...")

    _assets = _loader(_path, unit_length)
    _keys: list[str] = []
    _total = max(len(_assets), 1)
    for _idx, _asset in enumerate(_assets, start=1):
        _key = str(_asset.source_path)
        ASSET_CACHE.Register(_key, _asset)
        _keys.append(_key)
        if progress_callback is not None:
            progress_callback(_idx, _total, f"{_path.name} mesh 등록 중...")
    return _keys
