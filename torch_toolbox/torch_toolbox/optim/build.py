from __future__ import annotations
from typing import Any

import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.amp.grad_scaler import GradScaler

from ..modules.model.definition import Trainable_Model
from ..registry import SCHEDULER
from .definition import Optim_Node_Config


# lr·weight_decay를 정수 공간으로 올려 param_group 키로 사용 (부동소수점 해시 불안정 회피)
_SCALED = 10 ** 10


def Build_optim(
    config: Optim_Node_Config,
    model: Trainable_Model | nn.parallel.DistributedDataParallel,
    use_amp: bool = True,
) -> tuple[optim.Optimizer, lr_scheduler.LRScheduler | None, GradScaler]:
    """Config와 모델로부터 옵티마이저·스케줄러·GradScaler를 조립한다.

    모델의 Get_group_map으로 파라미터 그룹을 수집하고,
    각 그룹에 (lr, weight_decay) 쌍을 할당하여 옵티마이저를 초기화한다.
    스케줄러는 torch.optim.lr_scheduler → SCHEDULER 레지스트리 순으로 탐색한다.

    Args:
        config: 옵티마이저·스케줄러 설정.
        model: 파라미터를 제공할 모델. DDP 래퍼도 허용.
        use_amp: True이면 GradScaler를 활성화.

    Returns:
        tuple: (optimizer, scheduler | None, scaler)

    Raises:
        ValueError: optim_name이 없거나 스케줄러를 찾을 수 없는 경우.
    """
    if config.optim_name is None:
        raise ValueError("optim_name이 설정되지 않음")

    # DDP 래퍼 벗기기: param_group은 원본 모듈에서 수집
    _group_map: dict[tuple[int, int], list[Any]] = {}
    _core = model.module if isinstance(model, nn.parallel.DistributedDataParallel) else model
    # _SCALED 단위로 lr·wd를 정수화 → dict 키로 안전하게 사용
    _core.Get_group_map(config.base_lr, config.base_weight_decay, _group_map, _SCALED)

    _optim_cls = getattr(optim, config.optim_name, None)
    if _optim_cls is None:
        raise ValueError(f"옵티마이저 누락: {config.optim_name}")

    # param_group별 lr·wd를 _SCALED로 역정규화하여 실제 값으로 복원
    _optimizer: optim.Optimizer = _optim_cls(
        [
            {"params": _p, "lr": _k_lr / _SCALED, "weight_decay": _k_wd / _SCALED}
            for (_k_lr, _k_wd), _p in _group_map.items()
        ],
        **config.optim_kwargs,
    )

    _scheduler = None
    if config.scheduler_name:
        # torch 내장 스케줄러 우선, 없으면 SCHEDULER 레지스트리에서 탐색
        _sched_cls = getattr(lr_scheduler, config.scheduler_name, None)
        if _sched_cls is None:
            _sched_cls = SCHEDULER.Get(config.scheduler_name)
        if _sched_cls is None:
            raise ValueError(f"스케줄러 누락: {config.scheduler_name}")
        _scheduler = _sched_cls(optimizer=_optimizer, **config.scheduler_kwargs)

    return _optimizer, _scheduler, GradScaler(enabled=use_amp)
