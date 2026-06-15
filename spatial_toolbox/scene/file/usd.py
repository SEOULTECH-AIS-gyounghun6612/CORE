"""USD scene import/export."""
from __future__ import annotations

from pathlib import Path
import re as _re
from typing import Callable

import numpy as np
from pxr import Gf, Usd, UsdGeom

from ._base import Scene_State
from ..asset.cache import ASSET_CACHE
from ..asset.type.mesh import Mesh as Mesh_Asset
from ..node.type._base import Base_Node
from ..node.type.camera import Camera as Camera_Node, Camera_Intrinsic
from ..node.type.group import Group
from ..node.type.mesh import Mesh as Mesh_Node
from ..node.utils.traversal import walk_nodes


def Export_to_usd(file_path: str | Path, root: Base_Node, unit_length: float, *, write_visibility: bool = True) -> None:
    """Exports a scene graph to USD using shared prototype meshes.

    Mesh geometry is stored under ``/_Prototypes`` and scene nodes reference
    those prims internally instead of baking duplicate geometry.
    """
    _out = str(Path(file_path).resolve())
    _stage = Usd.Stage.CreateNew(_out)
    UsdGeom.SetStageMetersPerUnit(_stage, unit_length)
    UsdGeom.SetStageUpAxis(_stage, UsdGeom.Tokens.y)

    _proto_scope = _stage.DefinePrim("/_Prototypes", "Scope")
    UsdGeom.Imageable(_proto_scope).MakeInvisible()
    _proto_map: dict[str, str] = {}

    for _node in walk_nodes(root, lambda n: n.source_key is not None):
        _key = _node.source_key
        if _key in _proto_map:
            continue
        _asset = ASSET_CACHE.Get(_key, is_hold=True)
        if not isinstance(_asset, Mesh_Asset) or _asset.geometry is None:
            continue

        _proto_path = f"/_Prototypes/proto_{len(_proto_map)}"
        _proto_map[_key] = _proto_path

        _geo = _asset.geometry
        _mesh = UsdGeom.Mesh.Define(_stage, _proto_path)
        _mesh.CreatePointsAttr(
            [Gf.Vec3f(float(v[0]), float(v[1]), float(v[2])) for v in _geo.vertices]
        )
        _mesh.CreateFaceVertexCountsAttr([3] * len(_geo.faces))
        _mesh.CreateFaceVertexIndicesAttr(_geo.faces.flatten().tolist())
        _mesh.GetPrim().SetCustomData({"focusSourceKey": _key})

    _Write_usd_hierarchy(_stage, root, _proto_map, "", write_visibility=write_visibility)
    _root_name = _re.sub(r"[^A-Za-z0-9_]", "_", root.label) or "_root"
    _root_prim = _stage.GetPrimAtPath(f"/{_root_name}")
    if _root_prim.IsValid():
        _stage.SetDefaultPrim(_root_prim)
    _stage.GetRootLayer().Save()


def Import_from_usd(
    file_path: str | Path,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> Scene_State:
    """Imports a scene graph and prototype meshes from USD.

    Prototype meshes are re-registered in ``ASSET_CACHE`` before the scene
    hierarchy is rebuilt.
    """
    import trimesh

    _path = Path(file_path).resolve()
    if progress_callback is not None:
        progress_callback(1, 2, f"{_path.name} USD stage 여는 중...")

    _stage = Usd.Stage.Open(str(_path))
    if not _stage:
        raise ValueError(f"USD 파일 열기 실패: {file_path}")

    _meters = float(UsdGeom.GetStageMetersPerUnit(_stage))
    _proto_key_map: dict[str, str] = {}
    _proto_scope = _stage.GetPrimAtPath("/_Prototypes")
    if _proto_scope.IsValid():
        _mesh_prims = [_child for _child in _proto_scope.GetChildren() if _child.IsA(UsdGeom.Mesh)]
        _total = len(_mesh_prims) + 2
        for _idx, _child in enumerate(_mesh_prims, start=1):
            if progress_callback is not None:
                progress_callback(_idx + 1, _total, f"{_path.name} mesh 불러오는 중...")
            if not _child.IsA(UsdGeom.Mesh):
                continue
            _proto_usd_path = str(_child.GetPath())
            _mesh_api = UsdGeom.Mesh(_child)
            _pts_attr = _mesh_api.GetPointsAttr()
            _idx_attr = _mesh_api.GetFaceVertexIndicesAttr()
            if not _pts_attr.IsValid() or not _idx_attr.IsValid():
                continue
            _pts_raw = _pts_attr.Get()
            _idx_raw = _idx_attr.Get()
            if _pts_raw is None or _idx_raw is None:
                continue
            _pts = np.array([[p[0], p[1], p[2]] for p in _pts_raw], dtype=np.float32)
            _idx = np.array(list(_idx_raw), dtype=np.int32).reshape(-1, 3)
            _geo = trimesh.Trimesh(vertices=_pts, faces=_idx, process=False)
            _custom = _child.GetCustomData()
            _orig_key = _custom.get("focusSourceKey") if _custom else None
            _cache_key = _orig_key or f"{_path}#{_child.GetName()}"
            ASSET_CACHE.Register(
                _cache_key,
                Mesh_Asset(label=_child.GetName(), source_path=_cache_key, geometry=_geo),
            )
            _proto_key_map[_proto_usd_path] = _cache_key
    else:
        _total = 2

    if progress_callback is not None:
        progress_callback(_total, _total, "씬 계층 재구성 중...")

    _default_prim = _stage.GetDefaultPrim()
    if not _default_prim.IsValid():
        _candidates = [
            p for p in _stage.GetPseudoRoot().GetChildren()
            if not str(p.GetPath()).startswith("/_")
        ]
        if not _candidates:
            raise ValueError("USD 씬에 유효한 루트 prim 없음")
        _default_prim = _candidates[0]

    _root = _Read_usd_hierarchy(_default_prim, None, _proto_key_map)
    return Scene_State(root=_root, unit_length=_meters)


def _Write_usd_hierarchy(
    stage,
    node: Base_Node,
    proto_map: dict[str, str],
    parent_usd_path: str,
    *,
    write_visibility: bool = True,
) -> None:
    """Recursively writes one scene subtree into USD prims."""
    _name = _re.sub(r"[^A-Za-z0-9_]", "_", node.label) or "_node"
    if _name[0].isdigit():
        _name = f"_{_name}"
    _usd_path = f"{parent_usd_path}/{_name}" if parent_usd_path else f"/{_name}"

    _custom_data = {
        "focusNodeLabel": node.label,
        "focusNodePrimType": node.prim_type,
    }

    if isinstance(node, Camera_Node):
        _prim = UsdGeom.Camera.Define(stage, _usd_path).GetPrim()
        _custom_data["focusCameraIntrinsic"] = node.intrinsic.Serialize()
        _prim.SetCustomData(_custom_data)
    elif node.source_key and node.source_key in proto_map:
        _prim = UsdGeom.Mesh.Define(stage, _usd_path).GetPrim()
        _custom_data["focusProtoRef"] = proto_map[node.source_key]
        _prim.SetCustomData(_custom_data)
    else:
        _prim = stage.DefinePrim(_usd_path, "Xform")
        _prim.SetCustomData(_custom_data)
    if node.source_key and node.source_key in proto_map:
        _prim.GetReferences().AddInternalReference(proto_map[node.source_key])

    if write_visibility and not node.visible and node.source_key is not None:
        UsdGeom.Imageable(_prim).MakeInvisible()

    _xf = UsdGeom.Xformable(_prim)
    _scale_op = _xf.AddScaleOp()
    _scale_op.Set(Gf.Vec3f(float(node.scale[0]), float(node.scale[1]), float(node.scale[2])))
    _u = float(node.unit_scale)
    _unitfix_op = _xf.AddScaleOp(opSuffix="unitFix")
    _unitfix_op.Set(Gf.Vec3f(_u, _u, _u))
    _m = node.local_rigid.T.astype(float)
    _transform_op = _xf.AddTransformOp()
    _transform_op.Set(Gf.Matrix4d(*_m.flatten().tolist()))

    for _child in node.children:
        _Write_usd_hierarchy(stage, _child, proto_map, _usd_path, write_visibility=write_visibility)


def _Read_usd_hierarchy(prim, parent: Base_Node | None, proto_key_map: dict[str, str]) -> Base_Node:
    """Recursively rebuilds one scene subtree from USD prims."""
    _label = prim.GetName()
    _local_rigid = np.eye(4, dtype=np.float32)
    _scale = np.ones(3, dtype=np.float32)
    _unit_scale = 1.0

    _xf = UsdGeom.Xformable(prim)
    if _xf:
        for _op in _xf.GetOrderedXformOps():
            _op_name = _op.GetOpName()
            _val = _op.Get()
            if _val is None:
                continue
            if _op_name == "xformOp:transform":
                _m_rows = [[_val[i][j] for j in range(4)] for i in range(4)]
                _local_rigid = np.array(_m_rows, dtype=np.float32).T
            elif _op_name == "xformOp:scale":
                _scale = np.array([float(_val[0]), float(_val[1]), float(_val[2])], dtype=np.float32)
            elif _op_name == "xformOp:scale:unitFix":
                _unit_scale = float(_val[0])

    _source_key: str | None = None
    _custom = prim.GetCustomData()
    if _custom:
        _orig_label = _custom.get("focusNodeLabel")
        if _orig_label:
            _label = _orig_label
        _proto_ref = _custom.get("focusProtoRef")
        if _proto_ref and _proto_ref in proto_key_map:
            _source_key = proto_key_map[_proto_ref]

    if _source_key:
        _node: Base_Node = Mesh_Node(
            label=_label,
            prim_type="Mesh",
            local_rigid=_local_rigid,
            scale=_scale,
            unit_scale=_unit_scale,
            source_key=_source_key,
            parent=parent,
        )
    elif prim.IsA(UsdGeom.Camera):
        _intrinsic_data = None
        if _custom:
            _intrinsic_data = _custom.get("focusCameraIntrinsic")
        _node = Camera_Node(
            label=_label,
            prim_type="Camera",
            local_rigid=_local_rigid,
            scale=_scale,
            unit_scale=_unit_scale,
            parent=parent,
            intrinsic=Camera_Intrinsic(**_intrinsic_data) if isinstance(_intrinsic_data, dict) else Camera_Intrinsic(),
        )
    else:
        _node = Group(
            label=_label,
            prim_type="Xform",
            local_rigid=_local_rigid,
            scale=_scale,
            unit_scale=_unit_scale,
            parent=parent,
        )

    for _child_prim in prim.GetChildren():
        _child_node = _Read_usd_hierarchy(_child_prim, _node, proto_key_map)
        _node.children.append(_child_node)
    return _node
