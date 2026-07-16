from __future__ import annotations
from typing import Any, TypeVar, Generic
from dataclasses import dataclass, field
from pathlib import Path
from contextlib import contextmanager
from datetime import timedelta

import torch
import torch.nn as nn
import torch.onnx
import torch.distributed as dist
from torch import Tensor
from torch.multiprocessing.spawn import spawn

from python_toolbox.project import Project_Template
from python_toolbox.file import Write_to

from .utils.weight import Resolve_weight_path

from .assembler import Component_Assembler

ASSEMBLER = TypeVar("ASSEMBLER", bound=Component_Assembler)
LOSS = TypeVar("LOSS")


@dataclass
class Base_Runner(Project_Template, Generic[ASSEMBLER, LOSS]):
    """분산 실행 인프라 및 파이프라인 생명주기를 관리하는 베이스 러너.

    컴포넌트 생성은 주입된 Assembler에 위임하며,
    본 클래스는 분산 환경 셋업·프로세스 스폰·템플릿 루프 제어만 담당한다.

    Attributes:
        project_name: 워크스페이스 디렉터리명으로 사용.
        assembler: 컴포넌트 조립 팩토리.
        max_iters: 최대 학습 이터레이션 수.
        save_interval: 체크포인트 저장 주기 (iter 단위).
        gpus: 사용할 GPU 인덱스 목록. 2개 이상이면 DDP 모드.
        world_size: 전체 분산 프로세스 수 (멀티 노드 지원용).
        node_rank_offset: 멀티 노드 시 현재 노드의 rank 시작 오프셋.
        resume_path: resume 시 기존 워크스페이스 경로.
        weight_path: 특정 가중치 파일 경로.
        start_iter: 시작 이터레이션 직접 지정. None이면 체크포인트에서 결정.
    """

    project_name: str
    assembler: ASSEMBLER
    max_iters: int = 50
    save_interval: int = 1
    gpus: list[int] = field(default_factory=lambda: [0])
    host_name: str = "localhost"
    port_num: int = 12355
    world_size: int = 1
    node_rank_offset: int = 0
    resume_path: str | None = None
    weight_path: str | None = None
    start_iter: int | None = None
    is_multi_gpu: bool = field(init=False)

    def __post_init__(self):
        super().__init__(self.project_name)
        self.is_multi_gpu = len(self.gpus) > 1

    # --- Private (코어 인프라: 오버라이딩 금지) ---

    @contextmanager
    def _Process_context(self, p_id: int, is_test: bool = False):
        """프로세스 초기화 → yield → 정리의 생명주기를 보장하는 컨텍스트.

        Args:
            p_id: spawn으로 할당된 로컬 프로세스 인덱스.
            is_test: True이면 추론 전용 컨텍스트.

        Yields:
            tuple: (rank, device, iters, components, stop_tensor)
        """
        _rank = self.node_rank_offset + p_id if self.is_multi_gpu else 0
        _normal_exit = False
        try:
            # 디바이스 설정: set_device가 먼저여야 current_device()가 올바른 값을 반환
            _device_idx = self.gpus[p_id] if self.is_multi_gpu else self.gpus[0]
            _device = torch.device(f"cuda:{_device_idx}")
            torch.cuda.set_device(_device)

            if self.is_multi_gpu:
                _num_of_gpu = len(self.gpus)
                if self.world_size < _num_of_gpu:
                    raise ValueError(
                        f"[ERROR] 설정된 world_size({self.world_size})가 "
                        f"할당된 로컬 GPU 개수({_num_of_gpu})보다 작을 수 없음.")
                _rank = self.node_rank_offset + p_id
                dist.init_process_group(
                    backend="nccl",
                    init_method=f"tcp://{self.host_name}:{self.port_num}",
                    world_size=self.world_size, rank=_rank,
                    timeout=timedelta(minutes=30))
                print(f"[INFO] Distributed Env: Rank {_rank}/{self.world_size} on {_device}")
                _w_size = self.world_size
            else:
                print(f"[INFO] Single-Process: Ready on {_device}")
                _w_size = 1

            # 가중치 경로 결정: resume_path → weight_path → start_iter 우선순위
            _resolved_path = Resolve_weight_path(
                self.resume_path, self.weight_path, self.workspace, self.start_iter
            )
            # Assembler 호출 → 컴포넌트 조립 + 가중치 로드 → start_iter 반환
            _start_iter, _components = self.assembler(
                _device, _w_size, _rank, is_test=is_test,
                weight_path=_resolved_path,
                is_resume=self.resume_path is not None)
            _iters = (
                range(_start_iter - 1, _start_iter)
                if is_test
                else range(_start_iter, self.max_iters)
            )
            # DDP에서만 조기 종료 신호를 텐서로 broadcast; 단일 GPU는 None
            _stop_tensor = (
                torch.zeros(1, dtype=torch.int32, device=_device)
                if self.is_multi_gpu else None
            )
            yield _rank, _device, _iters, _components, _stop_tensor
            _normal_exit = True
        except Exception as e:
            if _rank == 0:
                print(f"[ERROR] Process {_rank} failed: {e}")
            raise
        finally:
            # 정상 종료 시에만 barrier: 비정상 종료면 deadlock 방지를 위해 건너뜀
            if self.is_multi_gpu and dist.is_initialized():
                if _normal_exit:
                    dist.barrier()
                dist.destroy_process_group()
                if _rank == 0:
                    print("[INFO] Distributed process group destroyed.")

    def __Process(self, p_id: int, is_test: bool = False):
        """통합 프로세스 루프. is_test 여부로 학습/추론을 분기한다.

        spawn 대상 함수이며 직접 호출하지 않는다.

        Args:
            p_id: 로컬 프로세스 인덱스.
            is_test: True이면 추론 전용.
        """
        with self._Process_context(p_id, is_test=is_test) as (
            _rank, _device, _iters, _components, _stop_tensor
        ):
            for _iter in _iters:
                self._Iter_hook(
                    _iter, _device, rank=_rank, is_test=is_test, **_components)

                # iter 종료 후 metric 초기화: 다음 iter와 누적값이 섞이지 않도록
                for _metric in _components["metric"].values():
                    _metric.Reset()

                if is_test:
                    continue

                if _rank == 0 and _iter % self.save_interval == 0:
                    self._Save_checkpoint(_iter, rank=_rank, **_components)

                # 조기 종료 판단: DDP는 rank 0이 결정하고 broadcast로 동기화
                if _stop_tensor is not None:
                    if _rank == 0:
                        _stop_tensor.fill_(
                            int(self._Should_stop(_iter, _components["metric"])))
                    dist.broadcast(_stop_tensor, src=0)
                    _should_stop = bool(_stop_tensor.item() == 1)
                else:
                    _should_stop = self._Should_stop(_iter, _components["metric"])

                if _should_stop:
                    if _rank == 0:
                        print(f"[INFO] Early stopping at iter {_iter}.")
                    break

    # --- Public ---

    def Run(self, is_test: bool = False):
        """파이프라인 진입점. is_test로 학습/추론을 분기한다.

        Args:
            is_test: True이면 추론 전용 실행.
        """
        self._Setup()
        _n_size = len(self.gpus)
        if not _n_size:
            raise ValueError("[ERROR] CPU 전용 실행은 지원하지 않습니다.")
        if self.is_multi_gpu:
            spawn(self.__Process, args=(is_test,), nprocs=_n_size, join=True)
        else:
            self.__Process(0, is_test)

    def Export(
        self, save_path: str | Path | None = None,
        opset_version: int = 21, do_constant_folding: bool = True,
        precision: str = "FP32", size_mb: int = 4096,
        external_data: bool = False,
        **kwargs: Any
    ):
        """설정된 파이프라인을 기반으로 ONNX 모델을 추출한다.

        단일 GPU 모드로 강제 전환하여 export 후 원래 상태를 복원한다.

        Args:
            save_path: ONNX 파일 저장 디렉터리. None이면 **workspace**에 저장한다
                (resume 시 해당 run 디렉터리) — 산출물이 출처가 된 체크포인트와 함께 남는다.
            opset_version: ONNX opset 버전.
            do_constant_folding: 상수 폴딩 최적화 여부.
            precision: TensorRT 추론 정밀도 (FP32 / FP16 / INT8).
            size_mb: TensorRT workspace 크기 (MB).
            external_data: 가중치를 .onnx.data로 분리할지 여부. 기본 False = **단일 파일**.
                분리되면 .onnx 안에 data 파일명이 박혀 둘을 항상 같이 옮겨야 한다.
                False여도 모델이 protobuf 한계(2GB)를 넘으면 torch가 자동으로 분리하므로
                안전하다.
            **kwargs: torch.onnx.export 추가 인자.
        """
        self._Setup()

        if opset_version >= 25:
            print(
                f"warning. TensorRT 10.x 공식 지원은 opset_version 25 미만입니다. "
                f"현재: {opset_version}"
            )

        # export는 항상 단일 프로세스로 실행: is_multi_gpu를 일시 비활성화
        _was_multi, self.is_multi_gpu = self.is_multi_gpu, False
        try:
            with self._Process_context(0, is_test=True) as (
                _, _device, _, _components, _
            ):
                # 미지정이면 workspace(resume 시 해당 run 디렉터리)에 저장
                _save_path = Path(save_path) if save_path is not None else Path(self.workspace)
                _save_path.mkdir(exist_ok=True, parents=True)
                print(f"[INFO] ONNX Export 시작: {_device}")

                (
                    _model, _inputs, _name, _onnx_cfg, _rt_cfg
                ) = self._Prepare_export_artifacts(
                    _device, _save_path,
                    opset_version=opset_version,
                    do_constant_folding=do_constant_folding,
                    precision=precision, size_mb=size_mb,
                    external_data=external_data,
                    **_components, **kwargs,
                )

                torch.onnx.export(_model, _inputs, **_onnx_cfg)
                Write_to(_save_path / f"{_name}_rt_cfg.yaml", _rt_cfg)
                print(f"[INFO] ONNX Export 완료: {save_path}")
        finally:
            self.is_multi_gpu = _was_multi

    # --- Protected Hooks ---

    def _Setup(self) -> bool:
        """워크스페이스 초기화 및 assembler config 저장."""
        if self._is_setup_done:
            return True

        if self.resume_path is not None:
            self.workspace = self.workspace.parent / self.resume_path

        if super()._Setup():
            return True
        self.assembler.Save_config(self.workspace)
        return False

    def _Iter_hook(
        self, current_iter: int, device: torch.device, *,
        rank: int = 0, is_test: bool = False, **components: Any
    ) -> None:
        raise NotImplementedError

    def _Forward(
        self, batch: dict[str, Any], device: torch.device, is_train: bool,
        model: Any, **kwargs: Any
    ) -> tuple[LOSS, int, dict[str, Any]]:
        raise NotImplementedError

    def _Should_stop(self, current_iter: int, metric: Any) -> bool:
        return False

    def _Save_checkpoint(
        self, current_iter: int, *, rank: int = 0, **components: Any
    ):
        raise NotImplementedError

    def _Prepare_export_artifacts(
        self, device: torch.device, save_path: Path, *,
        opset_version: int, do_constant_folding: bool,
        precision: str, size_mb: int, external_data: bool=False,
        **components: Any,
    ) -> tuple[
        nn.Module, tuple[Tensor, ...], str, dict[str, Any], dict[str, Any]
    ]:
        """ONNX export에 필요한 모델·입력·설정을 준비한다.

        각 구체 Runner가 직접 구현해야 한다.

        Args:
            device: 타깃 디바이스.
            save_path: ONNX 파일 저장 디렉터리.
            external_data: 가중치를 .onnx.data로 분리할지 여부(기본 False = 단일 파일).
                onnx_cfg에 그대로 넣어주면 된다.
            opset_version: ONNX opset 버전.
            do_constant_folding: 상수 폴딩 최적화 여부.
            precision: TensorRT 추론 정밀도 (FP32 / FP16 / INT8).
            size_mb: TensorRT workspace 크기 (MB).
            **components: Assembler.__call__()이 반환한 컴포넌트 dict.

        Returns:
            tuple:
                - export_model: export할 nn.Module. 전처리 레이어 융합 포함 가능.
                - dummy_inputs: torch.onnx.export에 전달할 더미 입력 텐서 튜플.
                - name: ONNX 파일명 기반 (확장자 제외).
                - onnx_cfg: torch.onnx.export에 전달할 키워드 인자 dict.
                  ``f``, ``export_params``, ``opset_version``, ``do_constant_folding``,
                  ``input_names``, ``output_names``, ``dynamic_shapes`` 등을 포함한다.
                - rt_cfg: ``{name}_rt_cfg.yaml``로 저장되는 TensorRT 런타임 설정 dict.
                  반드시 아래 키를 포함해야 한다::

                      {
                          "onnx_file":        str,
                          "precision":        str,
                          "workspace_size_mb": int,
                          "input_profiles": [
                              {
                                  "name":      str,
                                  "dtype":     str,
                                  "min_shape": list[int],
                                  "opt_shape": list[int],
                                  "max_shape": list[int],
                              }, ...
                          ],
                          "output_profiles": [
                              {
                                  "name":      str,
                                  "dtype":     str,
                                  "min_shape": list[int],
                                  "opt_shape": list[int],
                                  "max_shape": list[int],
                              }, ...
                          ],
                      }
        """
        raise NotImplementedError
