from dataclasses import dataclass

import numpy as np

from vision_toolbox.typing.images import IMG_1C, IMG_3C
from vision_toolbox.typing.components import BBox
from vision_toolbox.pipeline.state import Base_State


@dataclass
class ImageVisionState(Base_State):
    """파이프라인 데이터 상태 관리 객체.

    모든 내부 좌표(ROI, Mask)는 현재 offset 기준의 상대 좌표임.

    Attributes:
        image: 현재 처리 중인 뷰 이미지.
        mask: 현재 영역의 이진 마스크.
        roi: 현재 관심 영역 좌표 (x, y, w, h).
        offset: 원본 이미지 대비 현재 뷰의 누적 시작점.
        is_done: 파이프라인 중단 플래그.
    """

    image: IMG_1C | IMG_3C

    offset : tuple[int, int] = (0, 0)
    mask: IMG_1C | None = None
    roi: BBox | None = None

    def crop(self, window: BBox, strict: bool = True):
        """지정 영역으로 이미지 및 상태 크롭.

        Args:
            window: 크롭할 영역 (x, y, w, h).
            strict: 데이터 잘림 시 에러 발생 여부.

        Raises:
            ValueError: 윈도우가 이미지 범위를 벗어나거나 데이터가 잘릴 때.
        """
        bx, by, bw, bh = window
        ih, iw = self.image.shape[:2]

        # 이미지 경계 검사
        if bx < 0 or by < 0 or bx + bw > iw or by + bh > ih:
            msg = f"Crop window {window} exceeds image bounds ({iw}x{ih})"
            raise ValueError(msg)

        # ROI 업데이트 및 검증
        new_roi = None
        if self.roi is not None:
            rx, ry, rw, rh = self.roi
            # 잘림 여부 확인
            is_cut = (rx < bx or ry < by or
                      rx + rw > bx + bw or ry + rh > by + bh)

            if strict and is_cut:
                msg = (f"Crop window {window} truncates ROI {self.roi}. "
                       f"Use strict=False to allow.")
                raise ValueError(msg)

            # 교집합 계산
            nx1, ny1 = max(rx, bx), max(ry, by)
            nx2, ny2 = min(rx + rw, bx + bw), min(ry + rh, by + bh)
            nw, nh = max(0, nx2 - nx1), max(0, ny2 - ny1)
            # 상대 좌표 변환
            new_roi = (nx1 - bx, ny1 - by, nw, nh) if nw > 0 and nh > 0 else None

        # 슬라이싱 객체 생성
        sy, sx = slice(by, by + bh), slice(bx, bx + bw)

        # 마스크 검증 및 크롭
        _new_mask = None
        _mask = self.mask
        if _mask is not None:
            if strict:
                # 윈도우 외부 마스크 존재 여부 체크
                is_truncated = (
                    np.any(_mask[:by, :]) or     # 상단 구역
                    np.any(_mask[by + bh:, :]) or # 하단 구역
                    np.any(_mask[sy, :bx]) or     # 좌측 구역
                    np.any(_mask[sy, bx + bw:])   # 우측 구역
                )
                if is_truncated:
                    msg = (f"Crop window {window} truncates mask. "
                           f"Use strict=False to allow.")
                    raise ValueError(msg)

            _new_mask = _mask[sy, sx]

        # 상태 최종 업데이트
        self.image = self.image[sy, sx]
        self.roi = new_roi
        self.mask = _new_mask
        self.offset = (self.offset[0] + bx, self.offset[1] + by)

    def masking(self, is_positive: bool = True):
        """마스크 영역 기반 이미지 필터링.

        Args:
            is_positive: True면 마스크 영역 유지, False면 제외.
        """
        if self.mask is None or self.image is None:
            return

        _mask = self.mask
        # 다채널 이미지 대응
        if self.image.ndim == 3:
            _mask = _mask[:, :, np.newaxis]

        # 마스킹 적용
        _cond = _mask == 0 if is_positive else _mask > 0
        self.image[_cond] = 0
