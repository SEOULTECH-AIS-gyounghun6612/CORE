"""표 - 보이는 글자는 선언이, 순서는 값이 정함. 행을 붙여도 모델이 안 풀림.

행이 몇 만씩 붙는 소비처가 있음. 붙일 때마다 리셋이면 고른 자리가 풀리고 비용이 행 수를 따라감.
"""

from __future__ import annotations

import pytest
from PySide6.QtCore import Qt

from ui_toolbox.field import Field, Rows, Table_view


def _hash(value) -> str:
    """글자 순서가 값 순서와 어긋나게 - 글자로는 `#10` 이 `#9` 앞."""
    return f"#{value}"


_FIELDS = [Field("이름", str, editable=False),
           Field("수", int, editable=False, display=_hash)]


def _table(rows: list[dict]) -> Table_view:
    return Table_view(Rows(_FIELDS, rows), add_label="")


def _names(table: Table_view) -> list[str]:
    """보이는 순서의 첫 칸."""
    _m = table._model
    return [_m.data(_m.index(_r, 0)) for _r in range(_m.rowCount())]


# ── 보이는 글자 ───────────────────────────────────────────────────────────────
def test_display_role_uses_the_declaration():
    _m = _table([{"이름": "a", "수": 9}])._model
    _at = _m.index(0, 1)
    assert _m.data(_at, Qt.ItemDataRole.DisplayRole) == "#9"
    assert _m.data(_at, Qt.ItemDataRole.EditRole) == "9"


def test_filter_reads_the_shown_text():
    _t = _table([{"이름": "a", "수": 9}, {"이름": "b", "수": 10}])
    _t._model.filter("#9")
    assert _names(_t) == ["a"]


def test_sort_reads_the_value():
    """글자로 정렬하면 `#10` 이 앞. 수는 수로."""
    _t = _table([{"이름": "a", "수": 10}, {"이름": "b", "수": 9}])
    _t._model.sort(1)
    assert _names(_t) == ["b", "a"]


# ── 붙이기 ────────────────────────────────────────────────────────────────────
def test_extend_inserts_without_reset():
    _t = _table([{"이름": "a", "수": 1}])
    _resets, _inserts, _edits = [], [], []
    _t._model.modelReset.connect(lambda: _resets.append(True))
    _t._model.rowsInserted.connect(lambda *_a: _inserts.append(tuple(_a[1:])))
    _t.edited.connect(lambda: _edits.append(True))

    _t.extend([{"이름": "b", "수": 2}, {"이름": "c", "수": 3}])

    assert (_resets, _inserts, _edits) == ([], [(1, 2)], [])
    assert _names(_t) == ["a", "b", "c"]


def test_extend_keeps_the_pick():
    """앞에 끼어도 고른 것이 그대로. 리셋이면 여기서 풀림."""
    _t = _table([{"이름": "a", "수": 5}, {"이름": "b", "수": 1}])
    _t._model.sort(1)
    _t.select(0)
    _t.extend([{"이름": "c", "수": 0}])
    assert _t.current() == 0


def test_extend_hides_what_the_filter_drops():
    _t = _table([{"이름": "a", "수": 1}])
    _t._model.filter("a")
    _t.extend([{"이름": "b", "수": 2}, {"이름": "aa", "수": 3}])
    assert (_names(_t), len(_t.value())) == (["a", "aa"], 3)


@pytest.mark.parametrize("order", [Qt.SortOrder.AscendingOrder,
                                   Qt.SortOrder.DescendingOrder])
def test_extend_under_sort_lands_where_a_rebuild_would(order):
    """같은 값끼리는 원본 순서. 안정 정렬과 같은 답이어야 다시 정렬해도 안 튐."""
    _t = _table([{"이름": _n, "수": _v} for _n, _v in [("a", 2), ("b", 1), ("c", 2)]])
    _t._model.sort(1, order)
    _t.extend([{"이름": _n, "수": _v}
               for _n, _v in [("d", 2), ("e", 0), ("f", 3), ("g", 1), ("h", None)]])

    _inserted = list(_t._model._order)
    _t._model._rebuild()
    assert _inserted == _t._model._order
