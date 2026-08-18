from __future__ import annotations
from dataclasses import dataclass
import math
from typing import Any, NamedTuple

import torch
from torch import Tensor

from .... import CFGS
from ... import MODELS
from ...definition import Composable_Config
from ...model.definition import Trainable_Model

"""정준 좌표계(frame) 산출 — **파라미터만 낸다. raster를 돌리지 않는다.**

기존 :func:`dataloader.classification.mask_sdf_edge.align_pca` 는 PCA 주축각을 구한 뒤
``cv2.warpAffine`` 으로 마스크를 실제로 회전시켰다. 이 모듈은 각도를 **값으로만** 내고,
회전은 하류(:class:`~torch_toolbox.modules.transform.mask.polar.Polar_Raster`)에서 theta축 roll 로 소비한다.

그 결과:
    - ``warpAffine`` 이 기하 분기에서 사라진다 (NEAREST 리샘플 손실·cv2/torch 불일치 제거)
    - ``_recenter_to_centroid`` 의 정수 시프트 + **클리핑**(밀려난 픽셀을 테두리에 쌓아
      없는 경계를 만들던 결함)이 사라진다
    - 극좌표와 해석적 측정이 **같은 원점**을 쓴다 (기존엔 극좌표=정수 반올림 centroid,
      chirality/moments=float centroid 로 원점이 둘이었다)

원점은 **정수로 양자화**한다. 이 결정이 하류의 bilinear 가중치를 상수로 만들어
``GridSample`` 없이 ``Gather``/``Mul``/``Add`` 만으로 극좌표 리샘플이 되게 한다.
sub-pixel 원점으로 바꾸면 그 상수성이 깨진다.
"""


def _Abs2(re: Tensor, im: Tensor) -> Tensor:
    """복소수 크기 |re + i·im|. ``torch.hypot`` 대신 쓴다.

    ONNX exporter 에 ``prims.hypot`` 디컴포지션이 없어 export 가 실패하기 때문이다.
    ``clamp_min`` 은 0 에서 sqrt 의 기울기가 무한대인 것을 막는다(회전대칭 형상에서
    실제로 0 이 나온다).
    """
    return (re * re + im * im).clamp_min(1e-24).sqrt()


def _Grid(h: int, w: int, device: torch.device) -> tuple[Tensor, Tensor]:
    """캔버스 기하중심 기준 픽셀 좌표 ``(dy (1,H,1), dx (1,1,W))``.

    build 상수가 아니라 입력 크기에서 만든다 — 캔버스가 자유로워야 크롭 없이 원본
    프레임을 넣을 수 있다 (``Polar_Raster._sample`` 과 같은 규약).
    """
    _ys = torch.arange(h, dtype=torch.float32, device=device) - (h - 1) / 2.0
    _xs = torch.arange(w, dtype=torch.float32, device=device) - (w - 1) / 2.0
    return _ys.view(1, h, 1), _xs.view(1, 1, w)


class Frame(NamedTuple):
    """정준 좌표계 파라미터.

    ``angle`` 은 **신뢰도와 한 벌로 읽어야 한다.** 두 신뢰도가 0 에 가까우면 각도는 노이즈이며,
    소비처는 정렬을 믿는 대신 회전 불변 거리로 내려가야 한다
    (:mod:`torch_toolbox.metric.functional.vector.circular`).

    ``anisotropy`` / ``flip_margin`` 은 무차원 **비율**이라 그 자체로는 절대 크기를 잃는다.
    그래서 절대량 ``scale`` 을 같이 낸다 — 셋을 합치면 원래 2차 모멘트가 **그대로 복원된다**::

        Sxx + Syy = 2 * scale**2
        |Z2|      = anisotropy * 2 * scale**2
        lambda_+- = scale**2 * (1 +- anisotropy)        # 주축/부축 분산 (px^2)

    Attributes:
        center: (B, 2) float — centroid ``(cx, cy)``. 원본 픽셀 좌표, sub-pixel.
        origin: (B, 2) long  — 정수 양자화한 극좌표 원점 ``(col, row)``.
        angle:  (B,)  float  — 주축각(rad). 180° 모호성 확정 완료.
        scale:  (B,)  float  — 회전반경(gyration radius) ``sqrt((Sxx+Syy)/2)``, **px 절대값**.
            회전 불변이며 정규화되지 않은 유일한 크기량이다.
        anisotropy:  (B,) float [0, 1] — k=2 harmonic 상대 크기. **``angle`` 자체의 신뢰도.**
            0 이면 주축이 없다(원환·n>=3 회전대칭) — 그 경우 ``angle`` 은 노이즈가 정한 값이다.
        flip_margin: (B,) float [0, 1] — k=3 harmonic 상대 크기 ``|Z3| / Σ m·r³``.
            **180° 확정의 신뢰도.** 2회 회전대칭 형상에서 정확히 0 이고, 0 에 가까우면 그
            확정이 픽셀 노이즈로 뒤집힌다. ``anisotropy`` 와 달리 절대량 ``scale`` 로 원본
            3차 모멘트를 복원할 수는 없다 — 분모가 ``Σ m·r³`` 라 2차 모멘트와 무관하다.
    """

    center: Tensor
    origin: Tensor
    angle:  Tensor
    scale:  Tensor
    anisotropy:  Tensor
    flip_margin: Tensor


_FRAME_NAME = "centroid_frame"
_FRAME_CFG  = f"{_FRAME_NAME}_Config"
_COORD_NAME = "frame_coords"
_COORD_CFG  = f"{_COORD_NAME}_Config"


@CFGS.Register_module(_FRAME_CFG)
@dataclass
class Centroid_Frame_Config(Composable_Config):
    """정준 좌표계 산출 설정.

    Attributes:
        sampling_size: **샘플링 기준 캔버스** ``(H, W)`` — 입력 캔버스가 아니다.
            여기서 뽑는 것은 3차 모멘트의 FP16 범위뿐이고, 출력값에서는 약분된다.
        flip_phase_deg: 180° 확정에 쓸 ``w = Z3·e^{-i3a}`` 의 위상(0 또는 90). 형상이
            정하는 상수라 **부품마다 표가 들고 있다** — 프레임마다 값을 보고 고르면
            판정 경계에 걸린 형상이 회전마다 선택을 뒤집어 더 나빠진다(실측: 어떤
            임계를 써도 고정 0° 보다 나빴다).
    """
    config_type: str = _FRAME_CFG
    object_type: str = _FRAME_NAME
    trainable: bool = False
    sampling_size: tuple[int, int] = (224, 224)
    flip_phase_deg: float = 0.0


@MODELS.Register_module(_FRAME_NAME)
class Centroid_Frame(Trainable_Model):
    """이진 마스크 → centroid + PCA 주축각 + 신뢰도.

    주축각은 2x2 공분산의 고유분해 닫힌 해로 구한다 (``eigh`` 불필요)::

        angle = 0.5 * atan2(2 * Sxy, Sxx - Syy)

    **이 식이 실제로 재는 것은 각도 질량분포의 k=2 Fourier harmonic 위상이다.** 극좌표로 풀면
    ``Sxx - Syy ∝ Σ m·r²·cos2θ``, ``2·Sxy ∝ Σ m·r²·sin2θ`` 이므로 ``Z2 = Σ m·r²·e^{i2θ}`` 이고
    ``angle = arg(Z2)/2`` 다. 즉 **2차 모멘트는 harmonic 2 하나만 본다** — 형상 차이가
    k=3,4,5… 에 들어있으면 여기엔 전혀 잡히지 않는다.

    그 귀결로 **n-fold 회전대칭(n>=3)은 ``Z2 = 0`` 이 정확히 성립한다** (대칭 2x2 텐서의
    deviatoric 성분은 spin-2 라 ``R(2π/n)`` 불변이려면 ``n | 2``). 원환·3홀 플랜지·6홀 플랜지·
    기어는 형상이 전부 달라도 똑같이 "축 없음"을 낸다. 부분대칭(링 + 노치 하나)도 링 본체의
    등방 질량이 지배해 유효신호가 수 % 로 떨어진다. 호출자가 이걸 알 수 있도록 ``Z2`` 의
    상대 크기를 :attr:`Frame.anisotropy` 로 함께 낸다.

    180° 모호성은 **k=3 harmonic ``Z3`` 의 위상**으로 가른다 (``forward`` 주석에 근거).

    그 전에는 여러 종류의 **반쪽 분할**을 시도했다. 자기일관성(같은 마스크를 K 방향으로
    돌려 같은 방향으로 되돌리는지)만 재면 반경가중 분할이 가장 좋았다 — 1400 마스크 x
    8 회전, 혼합 클래스:

    ========================  =========  ==========
    tiebreak                  정상정렬   실패 샘플
    ========================  =========  ==========
    BR 사분면 질량             99.19%     -
    u축 skewness              99.35%     27/1400
    면적 분할 (r⁰)             99.33%     26/1400
    반경가중 분할 (r¹)          99.79%      8/1400
    ========================  =========  ==========

    셋을 섞으면 오히려 나빠진다(19~27/1400) — 표를 늘리는 문제가 아니라 통계 자체의 우열이다.

    **그런데 이 표는 자기일관성만 잰다.** 하류가 실제로 요구하는 것은 **표본 간 정합** —
    같은 부품의 서로 다른 인스턴스가 한 방향으로 모이는가다. 단일 클래스로 둘을 같이 재면
    순위가 뒤집힌다 (10N042000NT9, 332 객체 x 8 회전):

    ========================  ==========  ===========  ==================
    tiebreak                  자기일관성  표본간 정합  flip_margin 중앙값
    ========================  ==========  ===========  ==================
    반경가중 분할 (r¹)          94.92%      72.9%        0.0003
    **Z3 위상**                **98.76%**  **100%**     **0.153**
    ========================  ==========  ===========  ==================

    반경가중 분할이 이 부품에서 무너지는 이유는 margin 이 통째로 노이즈 수준이기 때문이다
    (5분위 실패율 58% → 17% → 26% → 8% → 1.5% 로 단조 감소 — 실패가 낮은 margin 에 몰린다).
    ``Z3`` 는 같은 형상에서 margin 이 두 자릿수 크고(0.085~0.29) 그 값이 **진단력을 갖는다**.
    속도는 동률이다 — GPU 는 ±2% 이내, CPU 는 Z3 가 다소 빠르다.

    Note:
        **어떤 tiebreak 도 2회 대칭 형상을 안정시키지는 못한다.** 결정론적 함수가 형상 자체의
        대칭을 깰 수는 없다 — 정확히 대칭이면 ``Z3 = 0`` 이 성립하고, 근대칭이면 노이즈가
        방향을 정한다. 통계의 역할은 결정이 아니라 **그 미결정성을 매끄럽게 보고하는 것**이고,
        ``flip_margin`` 이 그 보고다. 반쪽 분할은 ``sign()`` 때문에 피적분함수가 불연속이라
        이 보고가 무너졌다.

        입력 마스크는 fill/최대연결성분 처리를 **하지 않은** 원본이어야 한다.
        관통 구멍은 형상 정보이므로 메우지 않는다.
    """

    def Out_channels(self) -> list[int]:
        """center (2) / origin (2) / angle (1) / scale (1) / anisotropy (1) / flip_margin (1)."""
        return [2, 2, 1, 1, 1, 1]

    def Build(self, sampling_size: tuple[int, int] = (224, 224),
              flip_phase_deg: float = 0.0, **kwargs: Any) -> None:
        _h, _w = int(sampling_size[0]), int(sampling_size[1])
        self.sampling_size = (_h, _w)
        self.flip_phase_deg = float(flip_phase_deg)
        # 3차 모멘트용 무차원화 상수. u^3 을 픽셀 단위로 만들면 112^3 ~= 1.4e6 으로
        # 단일 항이 FP16 최대값(65504)을 넘는다 (:class:`Frame_Coords` 와 같은 이유).
        # 출력값에서는 약분되므로(flip_margin 의 분자·분모 모두 3차) 입력 캔버스와 무관.
        self.norm = math.hypot((_h - 1) / 2.0, (_w - 1) / 2.0)

    def forward(self, mask: Tensor) -> Frame:
        """
        Args:
            mask: (B, 1, H, W) float. 전경 1, 배경 0. 캔버스 크기는 자유.

        Returns:
            :class:`Frame`.
        """
        _m = mask[:, 0]                                        # (B, H, W)
        # int() 를 쓰지 않는다 — 심볼릭 shape 이 그대로 흘러야 동적 축으로 export 된다.
        _h, _w = mask.shape[-2], mask.shape[-1]
        _gy, _gx = _Grid(_h, _w, mask.device)

        # **FP16 계약 — 모든 리덕션은 무차원 좌표로, sum 이 아니라 mean 으로.**
        # px 로 누산하면 224 캔버스에서 이미 Sum(m·dx²) = 3.1e6 으로 FP16 최대(65504)를
        # 넘어 TRT 가 0 을 낸다(실측: 64 통과 / 224 이상 전부 0). 아래 모멘트는 전부
        # `_nm` 으로 나뉘어 H·W 가 약분되므로, mean 으로 바꿔도 **값이 바뀌지 않는다**.
        _ux, _uy = _gx / self.norm, _gy / self.norm
        # 빈 마스크 가드. FP16 정규수 하한(6.1e-5)보다 커야 해서 1e-4 다 — 이보다 작은
        # 전경 비율은 seg 단계 `min_area` 가 이미 걸러낸다.
        _nm = _m.mean(dim=(1, 2)).clamp_min(1e-4)              # (B,) 전경 화소 비율

        # centroid — 캔버스 기하중심 기준 오프셋 (무차원)
        _cx = (_m * _ux).mean(dim=(1, 2)) / _nm                # (B,)
        _cy = (_m * _uy).mean(dim=(1, 2)) / _nm

        # centroid 기준 중심 모멘트 (2차). 좌표가 무차원이라 `norm²` 만큼 작다.
        _x = _ux - _cx.view(-1, 1, 1)                          # (B, H, W) 무차원
        _y = _uy - _cy.view(-1, 1, 1)
        _sxx = (_m * _x * _x).mean(dim=(1, 2)) / _nm
        _syy = (_m * _y * _y).mean(dim=(1, 2)) / _nm
        _sxy = (_m * _x * _y).mean(dim=(1, 2)) / _nm

        # Z2 = Σ m·r²·e^{i2θ} 의 실/허부. 주축각은 그 위상의 절반이다.
        _z_re = _sxx - _syy
        _z_im = 2.0 * _sxy
        _angle = 0.5 * torch.atan2(_z_im, _z_re)               # (B,) **주축(major) 각** (선, 180° 주기)

        # 회전반경(px) — 정규화되지 않은 **절대** 크기량. 아래 두 비율의 분모를 여기 남겨
        # 두어야 Sxx+Syy = 2*scale², |Z2| = anisotropy*2*scale² 로 원본이 복원된다.
        # 모멘트가 무차원이므로 여기서 `norm` 을 곱해 px 로 되돌린다 — **출력 계약**이다.
        _trace = _sxx + _syy
        _scale = (_trace * 0.5).clamp_min(0.0).sqrt() * self.norm

        # |Z2| 를 전체 관성으로 정규화 — k=2 harmonic 이 얼마나 실재하는지. 0 이면 주축이 없고
        # (원환·n>=3 회전대칭에서 정확히 0) 위 atan2 는 노이즈의 위상을 낸 것이다.
        # torch.hypot 을 쓰지 않는다 — ONNX exporter 에 prims.hypot 디컴포지션이 없어
        # export 가 실패한다. 여기 값들은 정규화된 모멘트라 제곱해도 넘치지 않는다.
        _aniso = _Abs2(_z_re, _z_im) / _trace.clamp_min(1e-12)

        # 2-fold 확정: **홀수 harmonic** Z3 = Σ m·r³·e^{i3θ} = Σ m·(dx + i·dy)³ 를 쓴다.
        # π 회전에서 e^{i3π} = -1 이라 Z3 가 통째로 부호를 뒤집는다 — 정확히 필요한 한 비트다.
        # 짝수 harmonic 은 어느 것도 못 쓴다(Z2 를 포함해 π 회전에 불변).
        #
        # **원점을 옮겨서는 이 비트를 못 만든다.** 원점을 복소수 d 만큼 옮기면 평행축 정리가
        # Z2' = Z2 + d² 인데, π 회전은 d → -d 라 d² 가 불변이다. 편심·최대밀도점 등 어떤
        # 원점 규칙도 2차 모멘트로는 180° 를 가를 수 없다 — 홀수차 통계라야 한다.
        #
        # 결정은 주축 방향으로의 투영 s = Re(Z3·e^{-i3a}), a = arg(Z2)/2 다. s < 0 이면 π 더한다.
        # 반쪽 분할(sign 가중)을 버린 이유는 실측 정확도이자 구조다 — sign() 은 피적분함수를
        # 불연속으로 만들어 근대칭 형상에서 margin 이 노이즈에 잠긴다. Z3 는 좌표 다항식이라
        # 형상이 연속 변형되면 값도 연속으로 움직인다.
        # 좌표는 위에서 이미 무차원이다(`_x`, `_y`).
        _xx = _x * _x
        _yy = _y * _y
        _z3_re = (_m * _x * (_xx - 3.0 * _yy)).mean(dim=(1, 2))    # Re Z3 = mean m·(x³ - 3xy²)
        _z3_im = (_m * _y * (3.0 * _xx - _yy)).mean(dim=(1, 2))    # Im Z3 = mean m·(3x²y - y³)

        _c3, _s3 = torch.cos(3.0 * _angle), torch.sin(3.0 * _angle)
        _proj = _z3_re * _c3 + _z3_im * _s3                        # Re(w) — 0°

        # |Z3| 를 Σ m·r³ 로 정규화 — k=3 harmonic 이 얼마나 실재하는지. 2회 대칭에서 정확히 0.
        # r³ 는 pow(1.5) 대신 r²·sqrt(r²) 로 둔다 (일반 거듭제곱보다 빠르다).
        _r2 = _xx + _yy
        # 분자 `_z3_*` 와 같은 mean 이라 H·W 가 약분된다 — 비율은 그대로다.
        _r3 = (_m * _r2 * _r2.sqrt()).mean(dim=(1, 2)).clamp_min(1e-12)
        _flip = _Abs2(_z3_re, _z3_im) / _r3                    # (B,) [0, 1]

        # ``w = Z3·e^{-i3a}`` 는 **형상 상수**다 — 물체를 돌리면 ``Z3`` 와 ``e^{-i3a}`` 가
        # 상쇄돼 ``w`` 는 그대로고, π 회전에서만 부호가 뒤집힌다. 그래서 ``w`` 의 어느
        # 방향 성분을 봐도 필요한 한 비트를 담는다.
        #
        # 위 ``_proj`` 는 실수부(0°)다. ``proj(φ) = |w|·cos(arg w − φ)`` 를 φ 로 미분하면
        # φ=0 에서 그 값이 곧 ``Im(w)`` 이고 ``proj² + (dproj/dφ)² = |w|²`` 가 항등이다 —
        # **90° 성분은 별개 통계가 아니라 위상에 대한 기울기**다. 그래서 0° 가 못 쓰는
        # 형상(값 0, 기울기 최대)은 90° 로 보면 최대가 된다.
        #
        # 실측: ``20G031``/``20G032`` 는 ``|Z3|/Σm·r³ = 0.58`` 로 크지만 위상이 −89° 라
        # 실수부가 0 을 지나며 36각도 중 13~14회 뒤집힌다. 허수부로 보면 0회다.
        #
        # **위상은 인자로 받는다. 여기서 값을 보고 고르지 않는다.** 어느 쪽을 쓸지는
        # 형상이 정하는 상수이고, 프레임마다 값을 보고 고르면 그 판정 경계에 걸린 형상이
        # 회전마다 선택을 뒤집어 더 나빠진다(실측: 어떤 임계를 써도 고정 0° 보다 나빴다 —
        # 5 → 8~21). 표를 만들 때 부품마다 한 번 정하고, 여기서는 받은 값을 쓰기만 한다.
        # 261개 중 259개가 0°/90° 중 하나로 해결되고, 남는 2개는 2회 대칭에 가까워 어떤
        # 위상으로도 안 된다 — 그건 ``flip_margin`` 이 보고한다.
        if self.flip_phase_deg == 90.0:
            _proj = _z3_im * _c3 - _z3_re * _s3                    # Im(w) = dproj/dφ|₀

        _angle = _angle + (_proj < 0).to(_angle.dtype) * torch.pi

        # centroid 를 px 로 되돌려 캔버스 절대 좌표로 낸다 — **출력 계약**이다.
        _center = torch.stack([_cx * self.norm + (_w - 1) / 2.0,
                               _cy * self.norm + (_h - 1) / 2.0], dim=1)
        _origin = _center.round().long()                       # 정수 양자화 (col, row)
        return Frame(
            center=_center, origin=_origin, angle=_angle, scale=_scale,
            anisotropy=_aniso, flip_margin=_flip,
        )


@CFGS.Register_module(_COORD_CFG)
@dataclass
class Frame_Coords_Config(Composable_Config):
    """정렬 좌표 ``(u, v)`` 산출 설정.

    Attributes:
        sampling_size: **샘플링 기준 캔버스** ``(H, W)`` — 입력 캔버스가 아니다. 여기서
            길이 단위를 뽑는다. :class:`~torch_toolbox.modules.transform.mask.geometry.region.Region_Scalars`
            와 **같아야 한다** — 그쪽이 이 값을 곱해 px 로 되돌린다.
    """
    config_type: str = _COORD_CFG
    object_type: str = _COORD_NAME
    trainable: bool = False
    sampling_size: tuple[int, int] = (224, 224)


@MODELS.Register_module(_COORD_NAME)
class Frame_Coords(Trainable_Model):
    """centroid 원점·주축 정렬 좌표 ``(u, v)`` 를 낸다 — raster 회전 없이 해석적으로.

    ``u`` 는 주축(장축) 방향, ``v`` 는 그 수직 방향. 기존 구현은 마스크를 실제로 회전시킨 뒤
    image frame 에서 측정했지만, 여기서는 좌표만 돌린다. NEAREST 리샘플 손실이 없고
    극좌표 분기와 **같은 원점(centroid)** 을 쓴다.

    좌표는 ``norm`` 으로 나눠 [-1, 1] 근방에 두고 낸다. 이것이 FP16 안전성의 핵심이다 —
    정규화 없이 3차 모멘트를 만들면 ``dx^3 = 112^3 ~= 1.4e6`` 으로 FP16 최대값(65504)을
    단일 항에서 넘긴다.
    """

    def Out_channels(self) -> list[int]:
        """u, v 두 텐서."""
        return [1, 1]

    def Build(self, sampling_size: tuple[int, int] = (224, 224), **kwargs: Any) -> None:
        _h, _w = int(sampling_size[0]), int(sampling_size[1])
        self.sampling_size = (_h, _w)
        self.norm = math.hypot((_h - 1) / 2.0, (_w - 1) / 2.0)

    def forward(self, mask: Tensor, frame: Frame) -> tuple[Tensor, Tensor]:
        """
        Args:
            mask: (B, 1, H, W) float — 좌표 격자 크기만 쓴다.
            frame: :class:`Frame`.

        Returns:
            ``(u, v)`` — 각각 (B, H, W) float. ``norm`` 으로 나눈 무차원 좌표.
        """
        # int() 를 쓰지 않는다 — 심볼릭 shape 이 그대로 흘러야 동적 축으로 export 된다.
        _h, _w = mask.shape[-2], mask.shape[-1]
        _gy, _gx = _Grid(_h, _w, mask.device)
        _cx = frame.center[:, 0].view(-1, 1, 1) - (_w - 1) / 2.0
        _cy = frame.center[:, 1].view(-1, 1, 1) - (_h - 1) / 2.0
        _ddx = (_gx - _cx) / self.norm
        _ddy = (_gy - _cy) / self.norm

        _cos = torch.cos(frame.angle).view(-1, 1, 1)
        _sin = torch.sin(frame.angle).view(-1, 1, 1)
        return _ddx * _cos + _ddy * _sin, -_ddx * _sin + _ddy * _cos
