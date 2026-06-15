"""Geometry similarity helpers for mesh assets."""
from __future__ import annotations

import numpy as np
import trimesh
from scipy.spatial import KDTree


def Is_exact_match(
    geo_a: trimesh.Trimesh, geo_b: trimesh.Trimesh, tol: float = 1e-5
) -> bool:
    """Checks whether two meshes match exactly within a tolerance."""
    if len(geo_a.vertices) != len(geo_b.vertices):
        return False

    if len(geo_a.faces) != len(geo_b.faces):
        return False

    if not np.allclose(geo_a.extents, geo_b.extents, atol=tol):
        return False

    _vertices = np.allclose(geo_a.vertices, geo_b.vertices, atol=tol)
    _face = np.array_equal(geo_a.faces, geo_b.faces)

    return _vertices and _face


def Calculate_match_rate(
    geo_a: trimesh.Trimesh, geo_b: trimesh.Trimesh,
    num_samples: int = 5000, threshold: float = 0.01
) -> float:
    """Estimates geometric overlap by bidirectional surface sampling."""
    _pts_a = trimesh.sample.sample_surface(geo_a, num_samples)[0]
    _pts_b = trimesh.sample.sample_surface(geo_b, num_samples)[0]

    _dist_a_to_b = geo_b.nearest.on_surface(_pts_a)[1]
    _dist_b_to_a = geo_a.nearest.on_surface(_pts_b)[1]

    _match_a = np.sum(_dist_a_to_b < threshold) / num_samples
    _match_b = np.sum(_dist_b_to_a < threshold) / num_samples

    return float((_match_a + _match_b) / 2.0)


def Calculate_scan_match_rate(
    geo_a: trimesh.Trimesh, geo_b: trimesh.Trimesh,
    num_samples: int = 5000, threshold: float = 0.02,
    icp_iterations: int = 30, icp_tol: float = 1e-6
) -> float:
    """Estimates scan similarity with normalization and lightweight ICP."""
    _pts_a = _Normalize_to_unit(
        trimesh.sample.sample_surface(geo_a, num_samples)[0]
    )
    _pts_b = _Normalize_to_unit(
        trimesh.sample.sample_surface(geo_b, num_samples)[0]
    )

    _pts_b_aligned = _ICP(
        _pts_b, _pts_a,
        max_iterations=icp_iterations, tol=icp_tol
    )

    _tree_a = KDTree(_pts_a)
    _tree_b = KDTree(_pts_b_aligned)

    _dist_a_to_b, _ = _tree_b.query(_pts_a)
    _dist_b_to_a, _ = _tree_a.query(_pts_b_aligned)

    _match_a = np.sum(_dist_a_to_b < threshold) / num_samples
    _match_b = np.sum(_dist_b_to_a < threshold) / num_samples

    return float(max(_match_a, _match_b))


def _Normalize_to_unit(points: np.ndarray) -> np.ndarray:
    """Centers points at the origin and normalizes by bounding-box diagonal."""
    _center = (points.max(axis=0) + points.min(axis=0)) * 0.5
    _centered = points - _center

    _diag = np.linalg.norm(points.max(axis=0) - points.min(axis=0))
    if _diag < 1e-12:
        return _centered

    return _centered / _diag


def _ICP(
    source: np.ndarray, target: np.ndarray,
    max_iterations: int = 30, tol: float = 1e-6
) -> np.ndarray:
    """Aligns source points to target points with a simple ICP loop."""
    _src = source.copy()
    _tree = KDTree(target)
    _prev_error = float("inf")

    for _ in range(max_iterations):
        _dists, _indices = _tree.query(_src)
        _matched = target[_indices]

        _mean_error = float(np.mean(_dists))
        if abs(_prev_error - _mean_error) < tol:
            break
        _prev_error = _mean_error

        _centroid_src = _src.mean(axis=0)
        _centroid_tgt = _matched.mean(axis=0)

        _H = (_src - _centroid_src).T @ (_matched - _centroid_tgt)
        _U, _, _Vt = np.linalg.svd(_H)
        _R = _Vt.T @ _U.T

        if np.linalg.det(_R) < 0:
            _Vt[-1, :] *= -1
            _R = _Vt.T @ _U.T

        _t = _centroid_tgt - _R @ _centroid_src

        _src = (_R @ _src.T).T + _t

    return _src
