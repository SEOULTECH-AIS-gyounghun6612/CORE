from __future__ import annotations
from typing import cast

from torch.utils.data import DataLoader, DistributedSampler

from ..typing import Mode
from ..registry import CFGS, DATASETS, DATALOADER_FN
from .definition import Custom_Dataset, Dataset_Config, Dataloader_Config
from .functional import PK_Batch_Sampler
from .template import Classification_Dataset, Classification_Dataset_Config


def _Get_collate_func(name: str | None):
    if name is None:
        return None
    return DATALOADER_FN.Get(name)


def Build_dataset(
    config: Dataset_Config | Classification_Dataset_Config,
    mode: Mode,
) -> Custom_Dataset | Classification_Dataset:
    """Config로부터 데이터셋 인스턴스를 생성한다.

    Args:
        config: 데이터셋 설정.
        mode: 데이터셋이 사용될 실행 mode.

    Returns:
        생성된 Custom_Dataset 인스턴스.

    Raises:
        ValueError: object_type이 DATASETS에 등록되지 않은 경우.
    """
    _cls = DATASETS.Get(config.object_type)
    if _cls is None:
        raise ValueError(f"'{config.object_type}'가 DATASETS에 미등록.")
    return _cls(mode=mode, **config.Extract())


def Build_dataloader(
    dataloader_cfg: Dataloader_Config,
    mode: Mode,
    world_size: int = 1,
    rank: int = 0,
) -> tuple[Custom_Dataset | Classification_Dataset, DataLoader]:
    """dataset_meta에서 dataset을 생성하고 DataLoader를 구성한다.

    실행 환경에 따라 세 가지 경로로 분기한다:
    - pk_sampler 설정 + TRAIN + 단일 GPU + Classification_Dataset: PK 배치 샘플러 적용.
    - world_size >= 2: DistributedSampler 적용, shuffle 무효화.
    - 그 외: 표준 DataLoader.

    Args:
        dataloader_cfg: DataLoader 설정. dataset_meta에서 dataset을 생성한다.
        mode: 실행 mode. TRAIN 외 mode는 pk_sampler를 적용하지 않는다.
        world_size: 전체 프로세스 수. 1이면 단일 GPU.
        rank: 현재 프로세스의 글로벌 rank.

    Returns:
        tuple: (생성된 Custom_Dataset, 구성된 DataLoader).
    """
    # dataset_meta → Dataset_Config → Custom_Dataset 순서로 생성
    _meta = dataloader_cfg.dataset_meta
    _ds_cfg = cast(Dataset_Config, CFGS.Get(_meta["config_type"])(**_meta))
    _dataset = Build_dataset(_ds_cfg, mode)

    _collate_fn = _Get_collate_func(dataloader_cfg.collate_fn)

    if (
        dataloader_cfg.pk_sampler is not None
        and mode == Mode.TRAIN
        and world_size < 2
        and isinstance(_dataset, Classification_Dataset)
    ):
        # PK 샘플러는 DistributedSampler와 호환되지 않아 단일 GPU TRAIN에만 적용
        _pk = dataloader_cfg.pk_sampler
        _batch_sampler = PK_Batch_Sampler(
            class_ids=_dataset.class_ids,
            P=int(_pk["P"]),
            K=int(_pk["K"]),
            num_batches=_pk.get("num_batches"),
            seed=_pk.get("seed"),
        )
        return _dataset, DataLoader(
            _dataset,
            batch_sampler=_batch_sampler,
            num_workers=dataloader_cfg.num_workers,
            pin_memory=dataloader_cfg.pin_memory,
            collate_fn=_collate_fn,
        )

    if world_size >= 2:
        # DDP: DistributedSampler가 셔플을 제어하므로 DataLoader shuffle 비활성화
        _sampler = DistributedSampler(
            _dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=dataloader_cfg.shuffle if mode == Mode.TRAIN else False,
        )
        _shuffle = False
    else:
        _sampler = None
        _shuffle = dataloader_cfg.shuffle

    return _dataset, DataLoader(
        _dataset,
        batch_size=dataloader_cfg.batch_size,
        num_workers=dataloader_cfg.num_workers,
        shuffle=_shuffle,
        drop_last=dataloader_cfg.drop_last,
        pin_memory=dataloader_cfg.pin_memory,
        collate_fn=_collate_fn,
        sampler=_sampler,
    )
