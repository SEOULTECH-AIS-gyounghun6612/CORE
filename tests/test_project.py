"""Base_Config, Build_from_args, Read_from_file, Project_Template 단위 테스트."""
import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import pytest
import yaml

from python_toolbox.project import Base_Config, Build_from_args, Read_from_file
from python_toolbox.project.template import Project_Template


# =============================================================================
# 테스트 더블
# =============================================================================

@dataclass
class _AppCfg(Base_Config):
    config_type: str = "app"
    object_type: str = "app_obj"
    lr: float = 0.01
    epochs: int = 10


# =============================================================================
# Build_from_args
# =============================================================================

def test_build_from_args_dict():
    """dict 인자 → 올바른 config 객체."""
    cfg = Build_from_args(_AppCfg, {"lr": 0.001, "epochs": 50})
    assert isinstance(cfg, _AppCfg)
    assert cfg.lr == 0.001 and cfg.epochs == 50


def test_build_from_args_namespace():
    """argparse.Namespace → 올바른 config 객체."""
    ns = argparse.Namespace(lr=0.1, epochs=5)
    cfg = Build_from_args(_AppCfg, ns)
    assert cfg.lr == 0.1 and cfg.epochs == 5


def test_build_from_args_invalid_key_raises():
    """존재하지 않는 필드 → ValueError."""
    with pytest.raises(ValueError, match="_AppCfg"):
        Build_from_args(_AppCfg, {"lr": 0.01, "unknown_field": 99})


# =============================================================================
# Read_from_file
# =============================================================================

def test_read_from_file_json(tmp_path):
    """JSON 파일 → config 객체."""
    _path = tmp_path / "cfg.json"
    _path.write_text(json.dumps({"lr": 0.005, "epochs": 20}), encoding="UTF-8")
    cfg = Read_from_file(_AppCfg, _path)
    assert cfg.lr == 0.005 and cfg.epochs == 20


def test_read_from_file_yaml(tmp_path):
    """YAML 파일 → config 객체."""
    _path = tmp_path / "cfg.yaml"
    _path.write_text(yaml.dump({"lr": 0.002, "epochs": 30}), encoding="UTF-8")
    cfg = Read_from_file(_AppCfg, _path)
    assert cfg.lr == 0.002 and cfg.epochs == 30


def test_read_from_file_unsupported_format_raises(tmp_path):
    """미지원 확장자 → ValueError."""
    _path = tmp_path / "cfg.toml"
    _path.write_text("lr = 0.01")
    with pytest.raises(ValueError, match="Unsupported"):
        Read_from_file(_AppCfg, _path)


def test_read_from_file_parse_failure_raises(tmp_path):
    """파싱 불가 파일 → ValueError."""
    _path = tmp_path / "bad.json"
    _path.write_text("not valid json {{{", encoding="UTF-8")
    with pytest.raises(ValueError):
        Read_from_file(_AppCfg, _path)


# =============================================================================
# Project_Template
# =============================================================================

def test_project_template_empty_name_raises():
    """빈 project_name → ValueError."""
    with pytest.raises(ValueError):
        Project_Template("")


def test_project_template_workspace_unique():
    """두 인스턴스의 workspace가 서로 다름."""
    p1 = Project_Template("proj")
    p2 = Project_Template("proj")
    assert p1.workspace != p2.workspace


def test_project_template_workspace_contains_name():
    """workspace 경로에 project_name이 포함됨."""
    p = Project_Template("my_project")
    assert "my_project" in str(p.workspace)


def test_project_template_setup_idempotent(tmp_path, monkeypatch):
    """_Setup 중복 호출 시 두 번째는 True 반환 (멱등성)."""
    p = Project_Template("idempotent")
    monkeypatch.setattr(p, "workspace", tmp_path / "ws")
    assert p._Setup() is False
    assert p._Setup() is True


def test_project_template_setup_creates_workspace(tmp_path, monkeypatch):
    """_Setup 호출 후 workspace 디렉토리 생성됨."""
    p = Project_Template("create_test")
    _ws = tmp_path / "new_workspace"
    monkeypatch.setattr(p, "workspace", _ws)
    p._Setup()
    assert _ws.exists()
