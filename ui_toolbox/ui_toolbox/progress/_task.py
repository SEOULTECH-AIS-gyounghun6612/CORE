"""진행의 개념 — 백그라운드에서 도는 일 하나와, 그것을 스레드에 올리는 자리.

일이 무엇인지는 모른다. 아는 것은 셋뿐 — ``얼마나 왔나`` · ``무엇을 하는 중인가`` · ``끝났나``.
그래서 표현([`_bar`](_bar.py))도 소비처도 일의 정체를 안 물어도 된다.
"""

from __future__ import annotations

import traceback
from typing import Callable

from PySide6.QtCore import QObject, QThread, Signal


class Task(QObject):
    """백그라운드에서 도는 일 하나.

    ``run`` 은 워커 스레드에서 불린다 — 그 안에서 Qt 위젯을 건드리지 않는다.

    Attributes:
        progress: ``(done, total, label)`` — ``total <= 0`` 은 전체 미확정(세는 중).
        finished: ``(성공 여부, 결과 또는 traceback 문자열)``.
    """

    progress = Signal(int, int, str)
    finished = Signal(bool, object)

    def run(self) -> None:
        """일을 수행한다 (서브클래스 구현). 진행은 ``progress`` 로 중계한다."""
        raise NotImplementedError


class Call_task(Task):
    """콜러블 하나를 일로 감싼다 — 예외는 삼키지 않고 ``finished(False, traceback)`` 로 드러낸다."""

    def __init__(self, call: Callable[[Callable[[int, int, str], None]], object]) -> None:
        """Args:
        call: ``report(done, total, label)`` 콜백을 받아 결과를 돌려주는 콜러블.
        """
        super().__init__()
        self._call = call

    def run(self) -> None:
        """콜러블을 부르고 결과를 ``finished`` 로 알린다."""
        try:
            _result = self._call(
                lambda _done, _total, _label="": self.progress.emit(_done, _total, _label))
        except Exception:
            self.finished.emit(False, traceback.format_exc())
            return
        self.finished.emit(True, _result)


class Runner(QObject):
    """일을 스레드에 올려 돌린다 — 한 번에 하나, 스레드 수명은 여기가 든다.

    도는 중에 다시 ``start`` 하면 **조용히 갈아타지 않고 거절한다**(``False``) — 앞의 일이 데이터를
    고치는 중일 수 있다. 막는 것은 부르는 쪽이 아니라 여기다.

    Attributes:
        started: 일이 시작됨.
        progress: 도는 일의 진행을 그대로 중계 ``(done, total, label)``.
        done: ``(성공 여부, 결과 또는 traceback)`` — 스레드 정리까지 끝난 뒤.
    """

    started  = Signal()
    progress = Signal(int, int, str)
    done     = Signal(bool, object)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._thread: QThread | None = None
        self._task: Task | None = None

    def busy(self) -> bool:
        """지금 무엇인가 도는 중인가."""
        return self._thread is not None

    def start(self, task: Task) -> bool:
        """일을 시작한다.

        Args:
            task: 돌릴 일. 이 함수가 소유권을 가져가 스레드에 올린다.

        Returns:
            시작했으면 True, 이미 도는 중이라 거절했으면 False.
        """
        if self.busy():
            return False
        self._task = task
        self._thread = QThread()
        task.moveToThread(self._thread)
        self._thread.started.connect(task.run)
        task.progress.connect(self.progress)
        task.finished.connect(self._on_finished)
        self._thread.start()
        self.started.emit()
        return True

    def _on_finished(self, ok: bool, payload: object) -> None:
        """워커가 끝났다 — 스레드를 접고 결과를 알린다 (메인 스레드)."""
        if self._thread is not None:
            self._thread.quit()
            self._thread.wait()
        self._thread = None
        self._task = None
        self.done.emit(ok, payload)
