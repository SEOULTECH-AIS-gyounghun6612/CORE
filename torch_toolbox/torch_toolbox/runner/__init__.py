from __future__ import annotations
import argparse
from typing import Any
from dataclasses import fields
from pathlib import Path

from python_toolbox.file import Make_dict_from

from .runtime import Base_Runner


def Resolve_config(value: Any) -> Any:
    """config 값을 재귀적으로 정규화한다.

    dict이면 각 값에 재귀 적용하고, .yaml 문자열이면 파일을 읽어 **dict로 로드만** 한다
    (로드된 config 내용까지는 재귀하지 않는다 — model/loader/metric 등은 자기완결적 config이고,
    그 안의 .yaml 값은 id_map_path 같은 데이터 경로라 문자열로 보존해야 한다). 그 외는 그대로 반환.

    Args:
        value: 인라인 dict, config 파일 경로(.yaml) 문자열, 또는 기타 스칼라 값.

    Returns:
        정규화된 값.
    """
    if isinstance(value, dict):
        return {_k: Resolve_config(_v) for _k, _v in value.items()}
    if isinstance(value, str) and value.endswith((".yaml", ".yml")):
        return Make_dict_from(Path(value))[1]          # 파일 참조는 로드만, 내용은 재귀 안 함
    return value


def Build_runner_parser() -> argparse.ArgumentParser:
    """Base_Runner 공통 CLI template을 반환한다.

    run.py 등 진입점에서 이 parser를 기반으로 runner-specific 인자를 추가할 수 있다.

    Returns:
        공통 runner 인자가 등록된 ArgumentParser.
    """
    _p = argparse.ArgumentParser()
    _p.add_argument("--config_file",   type=str, default="conf/runtime.yaml")
    _p.add_argument("--test",          action="store_true")
    _p.add_argument("--assembler_meta", type=str, default=None)
    _p.add_argument("--project_name",  type=str, default=None)
    _p.add_argument("--max_iters",     type=int, default=None)
    _p.add_argument("--save_interval", type=int, default=None)
    _p.add_argument("--gpus",          type=int, nargs="*", default=None)
    _p.add_argument("--resume_path",   type=str, default=None)
    _p.add_argument("--weight_path",   type=str, default=None)
    _p.add_argument("--start_iter",    type=int, default=None)
    # Export는 Base_Runner의 기능이므로 CLI도 여기 둔다 (--test와 같은 계층).
    # 저장 위치는 workspace(체크포인트가 있는 run 디렉터리) — 산출물이 출처와 함께 남는다.
    # 값은 TensorRT 추론 정밀도 = export 산출물의 속성이므로 --export에 병합했다
    # (precision 단독으로는 학습/추론에 아무 의미가 없어 dead flag가 된다).
    _p.add_argument("--export",        type=str, nargs="?", const="FP32", default=None,
                    choices=["FP32", "FP16", "INT8"],
                    help="학습/추론 대신 ONNX export 수행 (값=TensorRT 추론 정밀도, "
                         "생략 시 FP32). 산출물은 workspace/<project>_<precision>.onnx")
    return _p


def Runtime_init(
    runner_cls: type,
    assembler_cls: type,
    *,
    config_file: str | Path,
    **hub_override: Any,
) -> Base_Runner:
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
    _, _hub = Make_dict_from(Path(config_file))
    _hub.update({_k: _v for _k, _v in hub_override.items() if _v is not None})

    _runner_field_names = {f.name for f in fields(runner_cls) if f.init}
    _runner_kwargs = {_k: _v for _k, _v in _hub.items() if _k in _runner_field_names}

    _am = _hub.get("assembler_meta", {})
    if isinstance(_am, str):                            # 최상위 assembler 파일은 내용을 정규화해야 함
        _, _am = Make_dict_from(Path(_am))
    _assembler_meta = Resolve_config(_am)

    return runner_cls(assembler=assembler_cls(**_assembler_meta), **_runner_kwargs)
