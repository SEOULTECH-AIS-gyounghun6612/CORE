from __future__ import annotations
import argparse
from typing import Any
from dataclasses import fields
from pathlib import Path

from python_toolbox.file import Read_from


def Resolve_config(value: Any) -> Any:
    """config 값을 재귀적으로 정규화한다.

    .yaml 문자열이면 파일을 읽어 재귀 적용하고,
    dict이면 각 값에 재귀 적용하며, 그 외는 그대로 반환한다.

    Args:
        value: 파일 경로 문자열, 인라인 dict, 또는 기타 스칼라 값.

    Returns:
        정규화된 값.
    """
    if isinstance(value, str) and value.endswith((".yaml", ".yml")):
        return Resolve_config(Read_from(Path(value))[1])
    if isinstance(value, dict):
        return {_k: Resolve_config(_v) for _k, _v in value.items()}
    return value


def Build_runner_parser() -> argparse.ArgumentParser:
    """Base_Runner 공통 CLI template을 반환한다.

    run.py 등 진입점에서 이 parser를 기반으로 runner-specific 인자를 추가할 수 있다.

    Returns:
        공통 runner 인자가 등록된 ArgumentParser.
    """
    _p = argparse.ArgumentParser()
    _p.add_argument("--config",         type=str, default="conf/runtime.yaml")
    _p.add_argument("--test",           action="store_true")
    _p.add_argument("--assembler_meta", type=str, default=None)
    _p.add_argument("--project_name",  type=str, default=None)
    _p.add_argument("--max_iters",     type=int, default=None)
    _p.add_argument("--save_interval", type=int, default=None)
    _p.add_argument("--gpus",          type=int, nargs="*", default=None)
    _p.add_argument("--resume_path",   type=str, default=None)
    _p.add_argument("--weight_path",   type=str, default=None)
    _p.add_argument("--start_iter",    type=int, default=None)
    return _p


def Runtime_init(
    config_path: str | Path,
    runner_cls: type,
    assembler_cls: type,
    **hub_override: Any,
) -> Any:
    """hub config를 읽어 runner 인스턴스를 직접 생성한다.

    hub의 최상위 키 중 runner 필드명과 일치하는 것은 runner kwargs로 분리하고,
    assembler_meta 키 하위 값들은 Resolve_config를 통해 파일 경로를 dict로 변환한 뒤
    assembler에 전달한다. hub_override로 전달된 non-None 값은 YAML 값보다 우선한다.

    Args:
        config_path: hub config YAML 파일 경로.
        runner_cls: 생성할 runner 클래스.
        assembler_cls: 생성할 assembler 클래스.
        **hub_override: YAML 값을 덮어쓸 CLI 인자. None이면 무시된다.

    Returns:
        runner_cls 인스턴스.
    """
    _, _hub = Read_from(Path(config_path))
    _hub.update({_k: _v for _k, _v in hub_override.items() if _v is not None})

    _runner_field_names = {f.name for f in fields(runner_cls) if f.init}
    _runner_kwargs = {_k: _v for _k, _v in _hub.items() if _k in _runner_field_names}

    _assembler_meta = {
        _k: Resolve_config(_v)
        for _k, _v in Resolve_config(_hub.get("assembler_meta", {})).items()
    }

    return runner_cls(assembler=assembler_cls(**_assembler_meta), **_runner_kwargs)
