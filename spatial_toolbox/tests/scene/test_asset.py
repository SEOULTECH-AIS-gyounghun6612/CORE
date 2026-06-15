from __future__ import annotations

from dataclasses import dataclass

import pytest

from spatial_toolbox.scene.asset.cache import ASSET_CACHE, Asset_Cache, Parse_key
from spatial_toolbox.scene.asset.file.loader import Load_and_register
from spatial_toolbox.scene.asset.file.obj import Read_and_parse_obj
from spatial_toolbox.scene.asset.type._base import Base_Asset
from spatial_toolbox.scene.asset.type.mesh import Mesh as Mesh_Asset

_OBJ_TETRA = """\
v 0 0 0
v 1 0 0
v 0 1 0
v 0 0 1
f 1 2 3
f 1 2 4
f 1 3 4
f 2 3 4
"""


@dataclass
class _Stub(Base_Asset):
    value: int = 0


@pytest.fixture(autouse=True)
def _clean_cache():
    ASSET_CACHE.Clear()


@pytest.fixture
def cache() -> Asset_Cache:
    return Asset_Cache()


@pytest.fixture
def obj_file(tmp_path):
    _p = tmp_path / "tetra.obj"
    _p.write_text(_OBJ_TETRA)
    return _p


def test_parse_key_no_fragment():
    _base, _frag = Parse_key("a/b/c.obj")
    assert _base.endswith("a/b/c.obj")
    assert _frag is None


def test_parse_key_with_fragment():
    _base, _frag = Parse_key("a/b/c.obj#geo")
    assert _base.endswith("a/b/c.obj")
    assert _frag == "geo"


def test_parse_key_same_string_deterministic():
    assert Parse_key("a.obj#x") == Parse_key("a.obj#x")


def test_parse_key_fragment_separator_last():
    _base, _frag = Parse_key("a#b#c")
    assert _base.endswith("a#b")
    assert _frag == "c"


def test_register_and_get(cache: Asset_Cache):
    cache.Register("a.obj", _Stub(value=7))
    result = cache.Get("a.obj")
    assert isinstance(result, _Stub)
    assert result.value == 7


def test_get_returns_deepcopy_by_default(cache: Asset_Cache):
    cache.Register("a.obj", _Stub(value=7))
    copy = cache.Get("a.obj")
    assert copy is not None
    copy.value = 9
    assert cache.Get("a.obj", is_hold=True).value == 7


def test_get_is_hold_returns_reference(cache: Asset_Cache):
    _a = _Stub(value=1)
    cache.Register("a.obj", _a)
    assert cache.Get("a.obj", is_hold=True) is not None


def test_get_nonexistent_returns_none(cache: Asset_Cache):
    assert cache.Get("missing.obj") is None


def test_get_with_matching_type(cache: Asset_Cache):
    _a = Mesh_Asset(label="m")
    cache.Register("a.obj", _a)
    assert isinstance(cache.Get("a.obj", Mesh_Asset), Mesh_Asset)


def test_get_with_mismatched_type_returns_none(cache: Asset_Cache):
    cache.Register("a.obj", _Stub(value=1))
    assert cache.Get("a.obj", Mesh_Asset) is None


def test_get_by_type_filters_correctly(cache: Asset_Cache):
    cache.Register("a.obj", _Stub(label="a"))
    cache.Register("b.obj", Mesh_Asset(label="b"))
    stubs = cache.Get_by_type(_Stub)
    assert len(stubs) == 1
    assert stubs[0].label == "a"


def test_get_all_spans_all_types(cache: Asset_Cache):
    cache.Register("a.obj", _Stub(label="a"))
    cache.Register("b.obj", Mesh_Asset(label="b"))
    assert len(cache.Get_all()) == 2


def test_get_paths_returns_all_keys(cache: Asset_Cache):
    cache.Register("a.obj", _Stub(label="a"))
    cache.Register("b.obj", _Stub(label="b"))
    paths = cache.Get_paths()
    assert any(_p.endswith("a.obj") for _p in paths)
    assert any(_p.endswith("b.obj") for _p in paths)


def test_remove_existing_key(cache: Asset_Cache):
    cache.Register("a.obj", _Stub())
    assert cache.Remove("a.obj") is True
    assert cache.Get("a.obj") is None


def test_remove_nonexistent_key(cache: Asset_Cache):
    assert cache.Remove("ghost.obj") is False


def test_clear_empties_all_buckets(cache: Asset_Cache):
    cache.Register("a.obj", _Stub())
    cache.Clear()
    assert cache.Get_all() == []


def test_overwrite_same_key(cache: Asset_Cache):
    cache.Register("a.obj", _Stub(value=1))
    cache.Register("a.obj", _Stub(value=2))
    assert cache.Get("a.obj", is_hold=True).value == 2


def test_parse_obj_returns_mesh_asset(obj_file):
    assets = Read_and_parse_obj(obj_file)
    assert len(assets) == 1
    assert isinstance(assets[0], Mesh_Asset)


def test_parse_obj_source_path_set(obj_file):
    assets = Read_and_parse_obj(obj_file)
    assert assets[0].source_path == str(obj_file.resolve())


def test_parse_obj_default_unit_length(obj_file):
    assets = Read_and_parse_obj(obj_file)
    assert assets[0].unit_length == 1.0


def test_parse_obj_custom_unit_length(obj_file):
    assets = Read_and_parse_obj(obj_file, unit_length=0.001)
    assert assets[0].unit_length == pytest.approx(0.001)


def test_parse_obj_geometry_not_none(obj_file):
    assets = Read_and_parse_obj(obj_file)
    assert assets[0].geometry is not None


def test_load_and_register_returns_key_list(obj_file):
    keys = Load_and_register(obj_file)
    assert len(keys) == 1
    assert all(isinstance(_k, str) for _k in keys)


def test_load_and_register_populates_cache(obj_file):
    keys = Load_and_register(obj_file)
    assert ASSET_CACHE.Get(keys[0]) is not None


def test_load_and_register_unit_length_propagated(obj_file):
    keys = Load_and_register(obj_file, unit_length=0.01)
    asset = ASSET_CACHE.Get(keys[0])
    assert asset.unit_length == pytest.approx(0.01)


def test_load_and_register_default_unit_length(obj_file):
    keys = Load_and_register(obj_file)
    asset = ASSET_CACHE.Get(keys[0])
    assert asset.unit_length == 1.0


def test_load_and_register_unsupported_format_raises(tmp_path):
    _p = tmp_path / "a.stl"
    _p.write_text("solid x\nendsolid x\n")
    with pytest.raises(ValueError):
        Load_and_register(_p)


def test_load_and_register_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        Load_and_register(tmp_path / "missing.obj")
