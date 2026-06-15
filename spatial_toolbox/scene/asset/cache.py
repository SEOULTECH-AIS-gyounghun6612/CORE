from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from copy import deepcopy
from typing import ClassVar

from python_toolbox.data_schema import Data_Schema

from .file import Load_and_register
from .type._base import Base_Asset


def __Cache_serialize__(cache: dict[type[Base_Asset], dict[str, Base_Asset]]):
    """Serializes the typed cache buckets into plain dictionaries."""
    return {
        _class_key.__name__: {
            _key: _asset.Serialize() for _key, _asset in _bucket.items()
        }
        for _class_key, _bucket in cache.items()
    }


def _Normalize_key(key: str | Path) -> str:
    """Normalizes cache keys while preserving fragment-style identifiers."""
    _s = str(key)
    if "#" in _s:
        return _s
    return str(Path(_s).resolve())


def Parse_key(key: str | Path) -> tuple[str, str | None]:
    """Splits a cache key into its absolute base path and optional fragment."""
    _s = str(key)
    if "#" in _s:
        _base, _frag = _s.rsplit("#", 1)
        return str(Path(_base).resolve()), _frag
    return str(Path(_s).resolve()), None


@dataclass
class Asset_Cache(Data_Schema):
    """Caches loaded assets by normalized key and concrete asset type."""
    cache: dict[type[Base_Asset], dict[str, Base_Asset]] = field(
        default_factory=dict
    )

    __custom_serializers__: ClassVar[dict] = {
        "cache": __Cache_serialize__
    }

    def Get(
        self,
        file_path: str | Path,
        asset_type: type[Base_Asset] | None = None,
        is_hold: bool = False,
    ) -> Base_Asset | None:
        """Returns a cached asset, optionally constrained by asset type."""
        _key = _Normalize_key(file_path)

        _buckets = (
            [self.cache.get(asset_type, {})]
            if asset_type is not None
            else self.cache.values()
        )
        for _bucket in _buckets:
            _asset = _bucket.get(_key)
            if _asset is not None:
                return _asset if is_hold else deepcopy(_asset)

        return None

    def Get_all(self) -> list[Base_Asset]:
        """Returns every cached asset."""
        _all: list[Base_Asset] = []
        for _bucket in self.cache.values():
            _all.extend(_bucket.values())
        return _all

    def Get_paths(self) -> list[str]:
        """Returns all registered cache keys."""
        _paths: set[str] = set()
        for _bucket in self.cache.values():
            _paths.update(_bucket.keys())
        return sorted(_paths)

    def Get_by_type(self, asset_type: type[Base_Asset]) -> list[Base_Asset]:
        """Returns all cached assets for one concrete type."""
        _bucket = self.cache.get(asset_type)
        if _bucket is None:
            return []
        return list(_bucket.values())

    def Register(self, file_path: str | Path, asset: Base_Asset) -> None:
        """Registers one asset under a normalized cache key."""
        self._Put(_Normalize_key(file_path), asset)

    def Add_from_file(
        self, file_path: str | Path, unit_length: float = 1.0
    ) -> list[str]:
        """Loads assets from a file and registers every returned cache entry."""
        _path = Path(file_path).resolve()
        return Load_and_register(_path, unit_length)

    def _Put(self, key: str, asset: Base_Asset) -> None:
        """Stores a normalized asset entry directly into the cache."""
        self.cache.setdefault(type(asset), {})[key] = deepcopy(asset)

    def Remove(self, file_path: str | Path) -> bool:
        """Removes an asset key from every type bucket."""
        _key = _Normalize_key(file_path)
        _removed = False

        for _bucket in self.cache.values():
            if _key in _bucket:
                del _bucket[_key]
                _removed = True

        return _removed

    def Clear(self) -> None:
        """Clears the full asset cache."""
        self.cache.clear()


# Shared process-level cache used by loaders and render paths.
ASSET_CACHE = Asset_Cache()
