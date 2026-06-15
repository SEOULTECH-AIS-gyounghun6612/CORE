"""회전 변환 및 좌표계 연산 functional 모듈."""

from typing import Literal, Union, Tuple
from scipy.spatial.transform import Rotation as R
import numpy as np

from vision_toolbox.typing.images import (
    M_ROT, V_4D, V_3D, V_2D, V_1D, M_TF
)

__all__ = [
    "Rotation_Converter", "Coordinate", "Pose", "Gaussian_Math"
]


class Rotation_Converter:
    """회전 변환 관련 유틸리티 클래스."""

    @staticmethod
    def M2Q(mat: M_ROT, w_first: bool = True) -> V_4D:
        """회전 행렬 -> 쿼터니언."""
        return R.from_matrix(mat).as_quat(scalar_first=w_first)

    @staticmethod
    def Q2M(quat: V_4D, w_first: bool = True) -> M_ROT:
        """쿼터니언 -> 회전 행렬."""
        return R.from_quat(quat, scalar_first=w_first).as_matrix()

    @staticmethod
    def M2R(mat: M_ROT, is_degrees: bool = True) -> V_3D:
        """회전 행렬 -> 회전 벡터."""
        return R.from_matrix(mat).as_rotvec(degrees=is_degrees)

    @staticmethod
    def R2M(r_vec: V_3D, is_degrees: bool = True) -> M_ROT:
        """회전 벡터 -> 회전 행렬."""
        return R.from_rotvec(r_vec, degrees=is_degrees).as_matrix()

    @staticmethod
    def M2E(
        mat: M_ROT, seq: str, is_degrees: bool = True
    ) -> Union[V_3D, V_2D, V_1D]:
        """회전 행렬 -> 오일러 각도."""
        return R.from_matrix(mat).as_euler(seq, degrees=is_degrees)

    @staticmethod
    def E2M(
        seq: str, angles: Union[V_3D, V_2D, V_1D], is_degrees: bool = True
    ) -> M_ROT:
        """오일러 각도 -> 회전 행렬."""
        return R.from_euler(seq, angles, degrees=is_degrees).as_matrix()


class Coordinate:
    """좌표계 변환 관련 유틸리티 클래스."""

    L_TO_R = np.array([  # 좌표계 변환(Left-to-Right Handed) 상수
        [0.0, 1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0],
        [-0.0, -0.0, -1.0, -0.0], [0.0, 0.0, 0.0, 1.0]
    ], dtype=np.float32)

    @staticmethod
    def To_homogeneous(pts: Union[V_2D, V_3D]) -> Union[V_3D, V_4D]:
        """포인트 -> 동차 좌표(Homogeneous coordinates)."""
        _ones = np.ones((pts.shape[0], 1), dtype=pts.dtype)
        return np.c_[pts, _ones]

    @staticmethod
    def Change_handedness(
        obj: Union[V_3D, V_4D, M_TF], mode: Literal["L2R", "R2L"] = "L2R"
    ) -> Union[V_3D, V_4D, M_TF]:
        """좌표계(Handedness) 변환."""
        if mode != "L2R":
            raise NotImplementedError("R2L 모드는 아직 구현되지 않음.")

        if obj.ndim == 2:  # 포인트인 경우
            _pts_h = Coordinate.To_homogeneous(
                obj) if obj.shape[1] == 3 else obj
            return (Coordinate.L_TO_R @ _pts_h.T).T[:, :3]

        return Coordinate.L_TO_R @ obj  # 변환 행렬인 경우


class Pose:
    """포즈(Extrinsic) 변환 및 처리 유틸리티 클래스."""

    @staticmethod
    def Compose_extrinsic_matrix(q: V_4D, t: V_3D) -> M_TF:
        """쿼터니언과 이동 벡터로 외부 파라미터 행렬을 구성합니다."""
        _mat = Rotation_Converter.Q2M(q)
        if q.ndim >= 2:
            _ext = np.tile(np.eye(4, dtype=np.float32), [q.shape[0], 1, 1])
            _ext[:, :3, :3], _ext[:, :3, 3] = _mat, t
        else:
            _ext = np.eye(4, dtype=np.float32)
            _ext[:3, :3], _ext[:3, 3] = _mat, t
        return _ext

    @staticmethod
    def Extract_extrinsic_params(ext: M_TF) -> Tuple[V_4D, V_3D]:
        """외부 파라미터 행렬에서 쿼터니언과 이동 벡터를 추출합니다."""
        if ext.ndim >= 3:
            _mat, _t = ext[:, :3, :3], ext[:, :3, 3]
        else:
            _mat, _t = ext[:3, :3], ext[:3, 3]
        _q = Rotation_Converter.M2Q(_mat)
        return _q, _t

    @staticmethod
    def Get_median_extrinsic(from_tfs: M_TF, to_tfs: M_TF) -> M_TF:
        """변환 행렬들 사이의 중간값에 해당하는 변환 행렬을 계산합니다."""
        assert len(from_tfs) == len(to_tfs)
        _rel_tfs = to_tfs @ np.linalg.inv(from_tfs)
        _tr = np.eye(4, dtype=np.float32)
        _r_vecs = Rotation_Converter.M2R(_rel_tfs[:, :3, :3])
        _tr[:3, :3] = Rotation_Converter.R2M(_r_vecs.mean(axis=0))
        _tr[:3, 3] = _rel_tfs[:, :3, 3].mean(axis=0)
        return _tr

    @staticmethod
    def Apply_extrinsic_transform(
        pts: Union[V_3D, V_4D], ext: M_TF, inv: bool = False
    ) -> V_3D:
        """포인트에 외부 파라미터 변환을 적용합니다."""
        _ext = np.linalg.inv(ext) if inv else ext
        _ext = _ext[..., :3, :]
        _pts_h = Coordinate.To_homogeneous(pts) if pts.shape[-1] == 3 else pts
        _transformed_pts = np.einsum("...ij,...j->...i", _ext, _pts_h)
        return _transformed_pts

    @staticmethod
    def Remove_duplicate_poses(
        data: M_TF, precision: int
    ) -> Tuple[M_TF, np.ndarray]:
        """중복되는 포즈들을 제거합니다."""
        _scale: int = 10 ** precision
        _flat_data = data.reshape(data.shape[0], -1)
        _scaled = (_flat_data * _scale).round().astype(np.int64)
        _, _indices = np.unique(_scaled, axis=0, return_index=True)
        return data[_indices], _indices

    @staticmethod
    def Align_trajectory_scale(
        trajectory: M_TF, constraints: list[tuple[int, float]]
    ) -> Tuple[M_TF, Tuple[float, float]]:
        """기준 포즈들과의 거리를 이용해 궤적의 스케일을 조정합니다."""
        _traj = trajectory.copy()

        if not constraints:
            return _traj, (1.0, 0.0)

        _indices, _real_dists = zip(*constraints)
        _indices = np.array(_indices, dtype=int)
        _real_dists = np.array(_real_dists, dtype=float)

        _deltas = _traj[_indices, :3, 3] - _traj[0, :3, 3]
        _current_dists = np.linalg.norm(_deltas, axis=1)

        _valid = _current_dists > 1e-9
        if not np.any(_valid):
            return _traj, (1.0, 0.0)

        _scales: np.ndarray = _real_dists[_valid] / _current_dists[_valid]
        _mean = np.mean(_scales).item()
        _cv = (np.std(_scales).item() / _mean) if _mean > 1e-9 else 0.0

        _traj[:, :3, 3] *= _mean
        return _traj, (_mean, _cv)

    @staticmethod
    def Normalize_trajectory_origin(trajectory: M_TF) -> M_TF:
        """궤적의 첫 번째 포즈를 원점 및 정방향(Identity)으로 이동시킵니다."""
        if trajectory.shape[0] == 0:
            return trajectory.copy()

        _inv_first = np.linalg.inv(trajectory[0])
        return _inv_first @ trajectory


class Gaussian_Math:
    """3D Gaussian 관련 수학 연산 유틸리티 클래스."""

    @staticmethod
    def Compute_3d_covariance(scales: V_3D, rot: V_4D) -> Tuple[V_3D, V_3D]:
        """3D 가우시안의 공분산 행렬의 상삼각행렬 성분을 계산합니다."""
        _norm = np.linalg.norm(rot, axis=1, keepdims=True)
        _norm_rot = rot / _norm
        _mat = Rotation_Converter.Q2M(_norm_rot, False)

        _s = np.zeros((scales.shape[0], 3, 3), dtype=np.float32)
        _s[:, 0, 0] = scales[:, 0]
        _s[:, 1, 1] = scales[:, 1]
        _s[:, 2, 2] = scales[:, 2]

        _m = _mat @ _s
        _sigma = _m @ np.transpose(_m, (0, 2, 1))

        _covA = np.stack([
            _sigma[:, 0, 0], _sigma[:, 0, 1], _sigma[:, 0, 2]
        ], axis=1)
        _covB = np.stack([
            _sigma[:, 1, 1], _sigma[:, 1, 2], _sigma[:, 2, 2]
        ], axis=1)

        return _covA, _covB
