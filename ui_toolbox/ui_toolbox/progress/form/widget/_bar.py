"""진행의 막대 표현 — 도는 동안 얼마나 왔는지 보이고, 편집을 막는다.

무슨 일이 도는지는 모른다. [`_task.Runner`](_task.py) 에 붙여 그 신호만 받는다.
"""

from __future__ import annotations

from PySide6.QtWidgets import QProgressBar, QWidget

from ..._task import Runner


class Progress_bar(QProgressBar):
    """진행 막대 — 전체가 미확정이면 busy(불확정), 확정되면 채워지는 막대.

    긴 일이 도는 동안 관련 UI 를 비활성해 데이터가 바뀌는 사이 편집이 끼어들지 못하게 한다
    (``guard``). 대량이 아니어도 늘 그렇게 한다 — 도는 중 편집은 무엇이 반영됐는지 알 수 없다.
    """

    def __init__(self, idle: str = "대기", parent: QWidget | None = None) -> None:
        """Args:
        idle: 아무것도 안 돌 때 보일 문구.
        parent: 부모 위젯.
        """
        super().__init__(parent)
        self._idle = idle
        self._guarded: list[QWidget] = []
        self.reset_idle()

    # ── 붙이기 ────────────────────────────────────────────────────────────────
    def bind(self, runner: Runner) -> None:
        """``Runner`` 의 진행을 이 막대가 받는다 (여럿 붙여도 된다)."""
        runner.started.connect(self._on_started)
        runner.progress.connect(self.report)
        runner.done.connect(self._on_done)

    def guard(self, *widgets: QWidget) -> None:
        """일이 도는 동안 비활성할 위젯을 등록한다."""
        self._guarded.extend(widgets)

    # ── 표시 ──────────────────────────────────────────────────────────────────
    def report(self, done: int, total: int, label: str = "") -> None:
        """진행을 반영한다.

        Args:
            done: 지금까지 처리한 수.
            total: 전체 수. ``0`` 이하면 아직 세는 중이라 불확정 막대로 둔다.
            label: 무엇을 하는 중인지.
        """
        if total > 0:
            self.setRange(0, total)
            self.setValue(done)
            self.setFormat(f"{label} : %v / %m" if label else "%v / %m")
        else:
            self.setRange(0, 0)                  # busy — 아직 세는 중
            self.setFormat(f"{label} … ({done}개)" if label else "…")

    def reset_idle(self) -> None:
        """대기 상태로 되돌린다."""
        self.setRange(0, 1)
        self.setValue(0)
        self.setFormat(self._idle)

    def _on_started(self) -> None:
        """일이 시작됨 — 등록된 위젯을 잠그고 불확정 막대로 둔다 (첫 진행 tick 전)."""
        self._set_enabled(False)
        self.setRange(0, 0)
        self.setFormat("시작하는 중…")

    def _on_done(self, ok: bool, _payload: object) -> None:
        """일이 끝남 — 잠금을 풀고 결과를 문구로 남긴다 (막대는 그 자리에 둔다)."""
        self._set_enabled(True)
        if ok:
            self.setRange(0, 1)
            self.setValue(1)
            self.setFormat("완료")
        else:
            self.setRange(0, 1)
            self.setValue(0)
            self.setFormat("실패")

    def _set_enabled(self, on: bool) -> None:
        for _w in self._guarded:
            _w.setEnabled(on)
