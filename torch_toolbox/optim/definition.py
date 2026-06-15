from __future__ import annotations
from typing import Any
from dataclasses import dataclass, field

from python_toolbox.project import Base_Config


@dataclass
class Optim_Node_Config(Base_Config):
    """옵티마이저·스케줄러 공동 설정.

    Attributes:
        optim_name: torch.optim 클래스명. None이면 빌드 시 오류.
        base_lr: 기본 학습률. 모듈별 lr는 Get_group_map으로 조정된다.
        base_weight_decay: 기본 weight decay.
        optim_kwargs: lr·weight_decay를 제외한 옵티마이저 추가 인자.
        scheduler_name: torch.optim.lr_scheduler 또는 SCHEDULER 등록명. None이면 미사용.
        scheduler_kwargs: optimizer를 제외한 스케줄러 추가 인자.
    """

    optim_name: str | None = None
    base_lr: float = 1e-4
    base_weight_decay: float = 1e-4
    optim_kwargs: dict[str, Any] = field(default_factory=dict)

    scheduler_name: str | None = None
    scheduler_kwargs: dict[str, Any] = field(default_factory=dict)
