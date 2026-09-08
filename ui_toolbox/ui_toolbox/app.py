"""앱 수명 - `QApplication` 을 세우고 창을 띄우고 루프를 돈다.

- 무엇을 띄우는지는 모름. 본문을 만드는 콜러블을 받음
- 팔레트도 시트도 [`style`](style.py) 의 토큰 하나에서 나옴
- 창에 시트를 한 번 걺. 아래는 상속
- 아무것도 안 봄 - `style` 하나만
"""

from __future__ import annotations

import sys
from collections.abc import Callable

from PySide6.QtWidgets import QApplication, QMainWindow, QWidget

from .style import Palette, Sheet, Style, Use

__all__ = ["Run"]


def Run(build: Callable[[], QWidget], title: str = "",
        size: tuple[int, int] = (1280, 860), style: Style | None = None) -> int:
    """앱을 세우고 본문을 띄운 뒤 루프를 돈다.

    Args:
        build: 본문 위젯을 만드는 콜러블. 스타일이 선 뒤에 불린다.
        title: 창 제목.
        size: 창 크기 `(폭, 높이)`.
        style: 쓸 토큰. 없으면 기본.

    Returns:
        프로세스 종료 코드.
    """
    _style = style or Style()
    Use(_style)                          # 코드가 치수를 읽는 자리

    _app = QApplication(sys.argv)
    _app.setStyle("Fusion")              # 플랫폼 테마가 팔레트를 덮지 않게
    _app.setPalette(Palette(_style))

    _win = QMainWindow()
    _win.setWindowTitle(title)
    _win.resize(*size)
    _win.setStyleSheet(Sheet(_style))    # 창에 한 번, 아래는 상속
    _win.setCentralWidget(build())
    _win.show()
    return _app.exec()
