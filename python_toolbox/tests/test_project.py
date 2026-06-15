"""Tests for ``python_toolbox.project``."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import pytest

from python_toolbox.file import Read_from
from python_toolbox.project import Base_Config, Build_parser_from_config, Build_sub_config
from python_toolbox.project.template import Project_Template
from python_toolbox.registry import Registry


@dataclass
class _AppCfg(Base_Config):
    config_type: str = "app"
    object_type: str = "app_obj"
    lr: float = 0.01
    epochs: int = 10


@dataclass
class _OtherCfg(Base_Config):
    config_type: str = "other"
    object_type: str = "other_obj"
    momentum: float = 0.9


@pytest.fixture
def cfg_registry():
    _registry = Registry("config_registry", Base_Config)
    _registry.Register_module("app")(_AppCfg)
    _registry.Register_module("other")(_OtherCfg)
    return _registry


def test_base_config_write_to_json(tmp_path):
    cfg = _AppCfg(lr=0.005, epochs=20)

    cfg.Write_to("cfg.json", tmp_path)

    ok, loaded = Read_from(tmp_path / "cfg.json")
    assert ok is True
    assert loaded["lr"] == 0.005
    assert loaded["epochs"] == 20


def test_base_config_write_to_yaml(tmp_path):
    cfg = _AppCfg(lr=0.002, epochs=30)

    cfg.Write_to("cfg.yaml", tmp_path)

    ok, loaded = Read_from(tmp_path / "cfg.yaml")
    assert ok is True
    assert loaded["lr"] == 0.002
    assert loaded["epochs"] == 30


def test_build_sub_config_uses_fallback_key_when_file_missing(cfg_registry):
    cfg = Build_sub_config(
        context={"lr": 0.25, "epochs": 50},
        expected=Base_Config,
        registry=cfg_registry,
        app=None,
    )

    assert isinstance(cfg, _AppCfg)
    assert cfg.lr == 0.25
    assert cfg.epochs == 50


def test_build_sub_config_uses_config_type_from_file(tmp_path, cfg_registry):
    _path = tmp_path / "cfg.json"
    _path.write_text('{"config_type":"other","momentum":0.75}', encoding="utf-8")

    cfg = Build_sub_config(
        context={},
        expected=Base_Config,
        registry=cfg_registry,
        app=str(_path),
    )

    assert isinstance(cfg, _OtherCfg)
    assert cfg.momentum == 0.75


def test_build_sub_config_merges_file_data_and_context(tmp_path, cfg_registry):
    _path = tmp_path / "cfg.json"
    _path.write_text(
        '{"config_type":"app","lr":0.005,"epochs":20,"object_type":"from_file"}',
        encoding="utf-8",
    )

    cfg = Build_sub_config(
        context={"epochs": 99},
        expected=Base_Config,
        registry=cfg_registry,
        app=str(_path),
    )

    assert isinstance(cfg, _AppCfg)
    assert cfg.lr == 0.005
    assert cfg.epochs == 99
    assert cfg.object_type == "from_file"


def test_build_sub_config_empty_key_with_file_raises(cfg_registry):
    with pytest.raises(ValueError):
        Build_sub_config(
            context={},
            expected=Base_Config,
            registry=cfg_registry,
        )


def test_build_parser_from_config_builds_typed_arguments():
    parser = Build_parser_from_config(_AppCfg)
    args = parser.parse_args(["--lr", "0.2", "--epochs", "50"])

    assert args.lr == 0.2
    assert args.epochs == 50


def test_build_parser_from_config_supports_bool_optional_action():
    @dataclass
    class _BoolCfg(Base_Config):
        enabled: bool = True

    parser = Build_parser_from_config(_BoolCfg)
    args = parser.parse_args(["--no-enabled"])

    assert args.enabled is False


def test_build_parser_from_config_supports_list_arguments():
    @dataclass
    class _ListCfg(Base_Config):
        names: list[str] = None  # type: ignore[assignment]

    parser = Build_parser_from_config(_ListCfg)
    args = parser.parse_args(["--names", "a", "b", "c"])

    assert args.names == ["a", "b", "c"]


def test_build_parser_from_config_supports_dict_arguments_as_text():
    @dataclass
    class _DictCfg(Base_Config):
        options: dict[str, int] | None = None

    parser = Build_parser_from_config(_DictCfg)
    args = parser.parse_args(["--options", '{"size": 3}'])

    assert args.options == '{"size": 3}'


def test_project_template_empty_name_raises():
    with pytest.raises(ValueError):
        Project_Template("")


def test_project_template_workspace_unique():
    p1 = Project_Template("proj")
    p2 = Project_Template("proj")
    assert p1.workspace != p2.workspace


def test_project_template_workspace_contains_name():
    p = Project_Template("my_project")
    assert "my_project" in str(p.workspace)


def test_project_template_setup_idempotent(tmp_path, monkeypatch):
    p = Project_Template("idempotent")
    monkeypatch.setattr(p, "workspace", tmp_path / "ws")
    assert p._Setup() is False
    assert p._Setup() is True


def test_project_template_setup_creates_workspace(tmp_path, monkeypatch):
    p = Project_Template("create_test")
    _ws = tmp_path / "new_workspace"
    monkeypatch.setattr(p, "workspace", _ws)
    p._Setup()
    assert _ws.exists()
