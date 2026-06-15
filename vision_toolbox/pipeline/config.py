from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass, field, InitVar
from typing import Any, List, Dict, Tuple
import json
import hashlib

from python_toolbox.project import Config


@dataclass
class Step_Config(Config.Basement):
    """개별 처리 단계(Step) 또는 단계의 컨테이너를 정의하는 설정 클래스."""
    name: str
    step_type: str
    annotations: str = ""
    params: Dict[str, Any] = field(default_factory=dict)

    steps_data : InitVar[List[Dict[str, Any]]] = field(default_factory=list)
    steps: List[Step_Config] = field(init=False, default_factory=list)

    def __post_init__(self, steps_data: List[Dict[str, Any]]):
        _list = []

        for _data in steps_data:
            if not isinstance(_data, dict):
                raise TypeError(f"하위 단계 설정은 딕셔너리여야 합니다: {_data}")
            _list.append(Step_Config(**_data))

        self.steps = _list

    def Config_to_dict(self) -> Dict[str, Any]:
        """객체를 딕셔너리로 재귀적으로 변환합니다."""
        _dict: dict[str, Any] = {
            "name": self.name,
            "step_type": self.step_type,
        }

        if self.params:
            _dict["params"] = self.params
        if self.steps:
            _dict["steps_data"] = [s.Config_to_dict() for s in self.steps]

        return _dict


def get_pipeline_summary(pipeline: List[Step_Config]) -> Tuple[str, str]:
    """파이프라인 구성 요소들의 이름을 조합하여 요약 문자열과 해시를 생성합니다.
    
    Args:
        pipeline: Step_Config 객체들의 리스트.
        
    Returns:
        Tuple[str, str]: (파이프라인_이름_요약, 파이프라인_구조_해시)
    """
    def _get_name(step: Step_Config) -> str:
        if step.steps:
            _names = "_".join(_get_name(_child) for _child in step.steps)
            return f"SQ_[{_names}]"
        return step.name

    _name = "_".join(_get_name(step) for step in pipeline)
    _hash = hashlib.md5(_name.encode('utf-8')).hexdigest()[:32]
    return _name, _hash


def get_config_hash(pipeline: List[Step_Config]) -> str:
    """파이프라인 설정 전체(파라미터 포함)에 대한 고유 해시를 생성합니다.
    
    Args:
        pipeline: Step_Config 객체들의 리스트.
        
    Returns:
        str: 설정 기반 MD5 해시 (앞 6자리).
    """
    _cfg_list = [step.Config_to_dict() for step in pipeline]
    return hashlib.md5(
        json.dumps(_cfg_list, sort_keys=True).encode('utf-8')
    ).hexdigest()[:6]

@dataclass
class Engine_Config(Config.Basement):
    project_name: str
    pipeline_steps: List[Dict[str, Any]]
    result_path: str = ""
    max_iter: int = 1

    def Get_args(self) -> dict[str, Any]:
        """Engine_Template 초기화에 필요한 인자를 반환합니다.
        주의: pipeline 객체 자체는 외부에서 생성하여 주입해야 합니다.
        """
        return {
            "project_name": self.project_name,
            "result_path": Path(self.result_path) if self.result_path else Path("./results"),
            "max_iter": self.max_iter,
        }
