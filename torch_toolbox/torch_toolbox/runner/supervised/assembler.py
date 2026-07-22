from __future__ import annotations
from typing import Any, cast
from dataclasses import dataclass, field, InitVar

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.amp.grad_scaler import GradScaler

from python_toolbox.project import Build_config
from pathlib import Path

from ...modules.build import Build_from_registry
from ...modules.model.definition import Trainable_Model, Trainable_Model_Config
from ...modules.loss.definition import Assemble_Loss_Config
from ... import CFGS
from ...modules import MODELS, LOSSES
from ..assembler import Component_Assembler
from ...optim.definition import Optim_Node_Config
from ...optim.build import Build_optim


# ── Assembler ──────────────────────────────────────────────────────────────


@dataclass
class Supervised_Assembler(
    Component_Assembler[
        optim.Optimizer,
        optim.lr_scheduler.LRScheduler | None,
        Trainable_Model,
    ]
):
    """지도학습 전용 단일 모델·옵티마이저 파이프라인 팩토리.

    model·loss·optim 각각의 meta dict를 받아 Config를 생성하고,
    _Build에서 mode_data(dataset·dataloader·metric)와 함께 조립한다.

    Attributes:
        model_cfg: 조립된 모델 설정.
        loss_cfg: 조립된 loss 설정.
        optim_cfg: 조립된 옵티마이저·스케줄러 설정.
    """

    mode_meta: InitVar[dict[str, dict[str, Any]] | None] = None
    model_meta: InitVar[dict[str, Any] | None] = None
    loss_meta: InitVar[dict[str, Any] | None] = None
    optim_meta: InitVar[dict[str, Any] | None] = None

    model_cfg: Trainable_Model_Config = field(init=False)
    loss_cfg: Assemble_Loss_Config = field(init=False)
    optim_cfg: Optim_Node_Config = field(init=False)

    def __post_init__(
        self,
        mode_meta: dict[str, dict[str, Any]] | None,
        model_meta: dict[str, Any] | None,
        loss_meta: dict[str, Any] | None,
        optim_meta: dict[str, Any] | None,
    ) -> None:
        # 부모의 mode_cfg 초기화가 먼저 실행되어야 shared가 확정됨
        super().__post_init__(mode_meta)
        # meta dict → CFGS 레지스트리 → Config 인스턴스화
        self.model_cfg = Build_config(
            Trainable_Model_Config, CFGS, (model_meta or {}), **self.shared
        )
        self.loss_cfg = Build_config(
            Assemble_Loss_Config, CFGS, (loss_meta or {}), **self.shared
        )
        self.optim_cfg = Build_config(
            Optim_Node_Config, CFGS, (optim_meta or {}), **self.shared
        )

    def Save_config(self, save_dir: str | Path, **kwarg) -> None:
        """model·loss·optim config를 직렬화하여 부모 Save_config에 병합한다.

        Args:
            save_dir: 저장 디렉터리 경로.
            **kwarg: 추가 직렬화 항목.
        """
        _extra = {
            "model_cfg": self.model_cfg.Serialize(),
            "loss_cfg": self.loss_cfg.Serialize(),
            "optim_cfg": self.optim_cfg.Serialize(),
            **kwarg
        }
        super().Save_config(save_dir, **_extra)

    def _Build(
        self, device: torch.device, world_size: int, rank: int,
        is_test: bool = False,
    ) -> dict[str, Any]:
        """model·mode_data를 조립하고 is_test이면 optim을 제외하여 반환한다.

        Args:
            device: 타깃 디바이스.
            world_size: 분산 프로세스 수.
            rank: 현재 프로세스 rank.
            is_test: True이면 loss·optim·scaler 없이 반환.

        Returns:
            Runner의 _Iter_hook이 **components로 수신하는 컴포넌트 dict.
        """
        _datasets, _dataloaders, _metric = self._Build_mode_data(is_test, world_size, rank)
        _model = self._Build_model(device, world_size)

        if is_test:
            # 추론 시에는 loss·optim·scaler 불필요
            return {
                "model": _model,
                "datasets": _datasets,
                "dataloaders": _dataloaders,
                "metric": _metric,
            }

        _loss_fn = self._Build_loss(device)
        _optim, _scheduler, _scaler = self._Build_optim_and_scheduler(_model)
        return {
            "model": _model,
            "datasets": _datasets,
            "dataloaders": _dataloaders,
            "loss_fn": _loss_fn,
            "optimizer": _optim,
            "scheduler": _scheduler,
            "scaler": _scaler,
            "metric": _metric,
        }

    def _Build_model(
        self, device: torch.device, world_size: int,
    ) -> Trainable_Model:
        """model_cfg로 모델을 조립하고 DDP 필요 시 래핑한다.

        Args:
            device: 타깃 디바이스.
            world_size: 분산 프로세스 수. 1 초과면 DDP 적용.

        Returns:
            조립된 Trainable_Model (DDP 래퍼 포함 가능).
        """
        _model = Build_from_registry(self.model_cfg, MODELS).to(device)
        if world_size > 1:
            # device_ids는 set_device 이후 current_device()로 결정
            _local = torch.cuda.current_device()
            _model = nn.parallel.DistributedDataParallel(
                _model, device_ids=[_local], output_device=_local
            )
        return cast(Trainable_Model, _model)

    def _Build_loss(self, device: torch.device) -> nn.Module:
        """loss_cfg로 loss 모듈을 조립하고 디바이스로 이동한다.

        Args:
            device: 타깃 디바이스.

        Returns:
            조립된 loss 모듈.
        """
        return Build_from_registry(self.loss_cfg, LOSSES).to(device)

    def _Build_optim_and_scheduler(
        self, model: Trainable_Model,
    ) -> tuple[optim.Optimizer, lr_scheduler.LRScheduler | None, GradScaler]:
        """optim_cfg와 모델로 옵티마이저·스케줄러·스케일러를 조립한다.

        Args:
            model: 파라미터 그룹을 제공할 모델.

        Returns:
            tuple: (optimizer, scheduler | None, scaler)
        """
        return Build_optim(self.optim_cfg, model, self.use_amp)

    def _Load_checkpoint(self, path: str) -> dict | None:
        """체크포인트 파일을 로드하고 표준 dict 형태로 반환한다.

        순수 state_dict(dict에 model_state 키 없음)가 들어오면
        ``{"model_state": data}`` 로 래핑하여 통일된 형태로 반환한다.

        Args:
            path: 체크포인트 파일 경로.

        Returns:
            표준화된 체크포인트 dict.
        """
        _data = torch.load(path, map_location="cpu", weights_only=False)
        # 순수 state_dict이면 model_state 키로 래핑
        if not isinstance(_data, dict) or "model_state" not in _data:
            return {"model_state": _data}
        return _data

    def _Load_model_weights(
        self, weight_path: str | None, is_resume: bool,
        model: Trainable_Model, *args, **kwargs: Any,
    ) -> int:
        """가중치를 로드하고 시작 이터레이션을 반환한다.

        resume이면 체크포인트의 iter + 1을 반환하고,
        fine-tune이면 0을 반환하여 처음부터 카운트한다.

        Args:
            weight_path: 가중치 파일 경로. None이면 scratch.
            is_resume: True이면 이터레이션 카운트를 체크포인트에서 복원.
            model: 가중치를 로드할 모델. DDP 래퍼도 허용.
            *args, **kwargs: Component_Assembler 시그니처 호환용 (무시됨).

        Returns:
            다음 시작 이터레이션 인덱스.
        """
        if weight_path is None:
            return 0
        _ckpt = self._Load_checkpoint(weight_path)
        if _ckpt is None:
            return 0
        # DDP 래퍼 벗기기: state_dict는 원본 모듈에 로드
        _target = model.module if isinstance(model, nn.parallel.DistributedDataParallel) else model
        _target.load_state_dict(_ckpt["model_state"], strict=False)
        # resume이면 저장된 iter 다음부터, fine-tune이면 0부터 시작
        return _ckpt.get("iter", 0) + 1 if is_resume else 0

    def _Restore_train_states(
        self, weight_path: str | None, start_iter: int,
        optimizer: optim.Optimizer,
        scheduler: lr_scheduler.LRScheduler | None,
        scaler: GradScaler,
        *args, **kwargs: Any,
    ) -> None:
        """체크포인트에서 옵티마이저·스케줄러·스케일러 상태를 복원한다.

        각 상태 키가 체크포인트에 없으면 silently skip된다.

        Args:
            weight_path: 체크포인트 파일 경로. None이면 즉시 반환.
            start_iter: 복원 후 시작 이터레이션 (미사용, 시그니처 통일용).
            optimizer: 상태를 복원할 옵티마이저.
            scheduler: 상태를 복원할 스케줄러. None이면 skip.
            scaler: 상태를 복원할 GradScaler.
            *args, **kwargs: Component_Assembler 시그니처 호환용 (무시됨).
        """
        if weight_path is None:
            return
        _ckpt = self._Load_checkpoint(weight_path)
        if _ckpt is None:
            return
        if "optim_state" in _ckpt:
            optimizer.load_state_dict(_ckpt["optim_state"])
        if scheduler is not None and "scheduler_state" in _ckpt:
            scheduler.load_state_dict(_ckpt["scheduler_state"])
        if "scaler_state" in _ckpt:
            scaler.load_state_dict(_ckpt["scaler_state"])
