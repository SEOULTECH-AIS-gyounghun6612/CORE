from __future__ import annotations

from . import ACCUMULATORS
from .definition import Accumulator_Config, Assemble_Metric_Config, Assemble_Metric


def Build_metric(cfg: Assemble_Metric_Config) -> Assemble_Metric:
    """Assemble_Metric_Config로부터 Assemble_Metric을 조립한다.

    cfg의 sub_metric_meta를 순회하며 Accumulator_Config로 변환하고
    ACCUMULATORS 레지스트리에서 각 Accumulator를 생성하여 이름으로 묶어 반환한다.

    Args:
        cfg: Accumulator 목록 설정.

    Returns:
        조립된 Assemble_Metric.
    """
    _accs = {}
    for _name, _meta in cfg.sub_metric_meta.items():
        _acc_cfg = Accumulator_Config(**_meta)
        _accs[_name] = ACCUMULATORS.Get(_acc_cfg.object_type)(**_acc_cfg.Extract())
    return Assemble_Metric(_accs)
