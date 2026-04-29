"""파일 I/O 단위 테스트 (Handle_exp, Suffix_check, Json, Yaml, Text, dispatch)."""
import json
from pathlib import Path

import pytest
import yaml

from python_toolbox.file._base import Handle_exp, Suffix_check
from python_toolbox.file._json import Json
from python_toolbox.file._yaml import Yaml
from python_toolbox.file._text import Text
from python_toolbox.file.dispatch import Read_from, Write_to


# =============================================================================
# Handle_exp
# =============================================================================

def test_handle_exp_success():
    """정상 함수 → 반환값 그대로 전달."""
    @Handle_exp()
    def _ok():
        return True, {"key": "val"}

    assert _ok() == (True, {"key": "val"})


def test_handle_exp_catches_exception(capsys):
    """예외 발생 → (False, None) 반환, 메시지 출력."""
    @Handle_exp({ValueError: "값 오류"})
    def _bad():
        raise ValueError("bad input")

    result = _bad()
    assert result == (False, None)
    captured = capsys.readouterr()
    assert "값 오류" in captured.out


def test_handle_exp_unknown_exception(capsys):
    """등록되지 않은 예외 → 기본 메시지로 처리."""
    @Handle_exp()
    def _unk():
        raise RuntimeError("unknown")

    result = _unk()
    assert result == (False, None)
    captured = capsys.readouterr()
    assert "알 수 없는" in captured.out


# =============================================================================
# Suffix_check
# =============================================================================

def test_suffix_check_match():
    """확장자 일치 → (True, 원본 경로)."""
    p = Path("file.json")
    ok, result = Suffix_check(p, ".json")
    assert ok is True and result == p


def test_suffix_check_mismatch_with_fix():
    """불일치 + is_fix=True → (False, 수정된 경로)."""
    p = Path("file.txt")
    ok, result = Suffix_check(p, ".json", is_fix=True)
    assert ok is False and result.suffix == ".json"


def test_suffix_check_mismatch_no_fix():
    """불일치 + is_fix=False → (False, 원본 경로 유지)."""
    p = Path("file.txt")
    ok, result = Suffix_check(p, ".json", is_fix=False)
    assert ok is False and result == p


def test_suffix_check_list_ext():
    """확장자 리스트 중 하나와 일치 → (True, 원본)."""
    p = Path("data.yaml")
    ok, result = Suffix_check(p, [".json", ".yaml"])
    assert ok is True and result == p


# =============================================================================
# Json 읽기/쓰기
# =============================================================================

def test_json_write_and_read(tmp_path):
    """JSON 저장 후 재읽기 → 데이터 일치."""
    _path = tmp_path / "data.json"
    data = {"key": "value", "num": 42}
    Json.Write_to(_path, data)
    ok, loaded = Json.Read_from(_path)
    assert ok is True and loaded == data


def test_json_read_missing_file(tmp_path):
    """존재하지 않는 파일 → (False, {}) 반환."""
    ok, result = Json.Read_from(tmp_path / "missing.json")
    assert ok is False and result == {}


def test_json_read_invalid_json(tmp_path):
    """잘못된 JSON → (False, None) 반환."""
    _path = tmp_path / "bad.json"
    _path.write_text("not json content {{{", encoding="UTF-8")
    ok, result = Json.Read_from(_path)
    assert ok is False


# =============================================================================
# Yaml 읽기/쓰기
# =============================================================================

def test_yaml_write_and_read(tmp_path):
    """YAML 저장 후 재읽기 → 데이터 일치."""
    _path = tmp_path / "data.yaml"
    data = {"model": "resnet", "lr": 0.001}
    Yaml.Write_to(_path, data)
    ok, loaded = Yaml.Read_from(_path)
    assert ok is True and loaded == data


def test_yaml_read_missing_file(tmp_path):
    """존재하지 않는 파일 → (False, {}) 반환."""
    ok, result = Yaml.Read_from(tmp_path / "missing.yaml")
    assert ok is False and result == {}


# =============================================================================
# Text 읽기/쓰기
# =============================================================================

def test_text_write_and_read(tmp_path):
    """텍스트 저장 후 재읽기 → 라인 리스트 일치."""
    _path = tmp_path / "data.txt"
    Text.Write_to(_path, ["line1", "line2", "line3"])
    ok, lines = Text.Read_from(_path)
    assert ok is True and lines == ["line1", "line2", "line3"]


def test_text_write_with_annotation(tmp_path):
    """주석 포함 저장 → 첫 줄에 주석 추가됨."""
    _path = tmp_path / "ann.txt"
    Text.Write_to(_path, ["data"], anno="# header")
    ok, lines = Text.Read_from(_path)
    assert ok is True and lines[0] == "# header"


def test_text_read_missing_file(tmp_path):
    """존재하지 않는 파일 → (False, []) 반환."""
    ok, result = Text.Read_from(tmp_path / "missing.txt")
    assert ok is False and result == []


def test_text_read_with_start_offset(tmp_path):
    """start 인덱스 이후 라인만 반환."""
    _path = tmp_path / "offset.txt"
    Text.Write_to(_path, ["a", "b", "c"])
    ok, lines = Text.Read_from(_path, start=1)
    assert ok and lines == ["b", "c"]


# =============================================================================
# dispatch — Read_from / Write_to
# =============================================================================

def test_dispatch_read_from_json(tmp_path):
    """dispatch Read_from → JSON 파일 읽기."""
    _path = tmp_path / "d.json"
    _path.write_text('{"x": 1}', encoding="UTF-8")
    ok, data = Read_from(_path)
    assert ok and data["x"] == 1


def test_dispatch_read_from_yaml(tmp_path):
    """dispatch Read_from → YAML 파일 읽기."""
    _path = tmp_path / "d.yaml"
    _path.write_text("x: 2\n", encoding="UTF-8")
    ok, data = Read_from(_path)
    assert ok and data["x"] == 2


def test_dispatch_read_from_non_file_raises(tmp_path):
    """디렉토리 경로 → ValueError."""
    with pytest.raises(ValueError, match="not FILE"):
        Read_from(tmp_path)


def test_dispatch_read_from_unsupported_ext_raises(tmp_path):
    """미지원 확장자 → ValueError."""
    _path = tmp_path / "file.csv"
    _path.write_text("a,b,c")
    with pytest.raises(ValueError, match="not supported"):
        Read_from(_path)


def test_dispatch_write_to_json(tmp_path):
    """dispatch Write_to → JSON 저장."""
    _path = tmp_path / "out.json"
    Write_to(_path, {"a": 1})
    assert json.loads(_path.read_text())["a"] == 1


def test_dispatch_write_to_unsupported_ext_raises(tmp_path):
    """미지원 확장자 쓰기 → ValueError."""
    with pytest.raises(ValueError, match="not supported"):
        Write_to(tmp_path / "out.csv", {"a": 1})
