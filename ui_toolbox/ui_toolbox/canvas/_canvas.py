"""캔버스 계약 - 포인터를 도메인 좌표로 내는 화면.

무슨 좌표인지는 구현이 안다 - 라스터면 원본 픽셀, 3D 장면이면 ray hit. 계약은 좌표의 뜻을 묻지
않으므로 2D 와 3D 가 같은 편집 골격에 붙는다.

좌표를 실수로 내는 이유 - 픽셀 격자는 정수지만 ray hit 은 아니다. 정수 격자를 쓰는 쪽이 자기
자리에서 자른다.
"""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QWidget


class Canvas(QWidget):
    """포인터 · 보기 계약. 구현이 좌표계와 그리기를 든다.

    Attributes:
        mouse_pressed: 좌클릭 ``(x, y)``.
        mouse_moved: 포인터 이동 ``(x, y)``.
        mouse_released: 좌클릭 해제 ``(x, y)``.
        mouse_right_pressed: 우클릭 ``(x, y)``. 그리던 것 취소용.
        zoom_changed: 유효 배율. ``0.0`` 은 fit.
    """

    mouse_pressed       = Signal(float, float)
    mouse_moved         = Signal(float, float)
    mouse_released      = Signal(float, float)
    mouse_right_pressed = Signal(float, float)
    zoom_changed        = Signal(float)

    def set_interactive(self, on: bool) -> None:
        """포인터 신호 방출을 켜고 끈다.

        Args:
            on: True 면 포인터를 도메인 좌표로 환산해 신호로 낸다.
        """
        raise NotImplementedError
