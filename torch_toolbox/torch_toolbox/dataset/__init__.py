""" ###

------------------------------------------------------------------------
### Requirement
    -

### Structure
    -

"""
from __future__ import annotations
from typing import Any, Callable, Literal
from dataclasses import dataclass, field

from torch.utils.data import Dataset, DataLoader, DistributedSampler

from python_ex.project import Config


@dataclass
class Data_Config(Config.Basement):
    name: str = "no_data"
    process: str = "custom"
    data_dir: str = "./datasets"
    additional: dict = field(default_factory=dict)


class Paser():
    # TODO: A template class needs to be created for upcoming data parsing tasks.
    ...


class Custom_Dataset(Dataset):
    def __init__(
        self,
        mode: str, cfg: Data_Config
    ):
        self.dataset_cfg = cfg
        self.dataset_block = self.Get_data_block(
            mode, cfg.name, cfg.data_dir, cfg.process, **cfg.additional)

    def Get_data_block(
        self,
        mode: str,
        name: str, data_dir: str, process: str, **kwarg
    ) -> dict[int, Any]:
        """ ### 지정된 모드와 설정에 따라 데이터를 로드하거나 구성하는 추상 메소드

        ------------------------------------------------------------------
        ### Args
        - mode: 데이터셋 모드. -> "train", "validation", "test"
        - name: 데이터셋의 이름.
        - data_dir: 데이터가 저장된 디렉토리 경로.
        - process: 데이터 전처리 방법 또는 식별자.
        - **kwarg: 추가적인 설정 값들 (Data_Config의 additional).

        ### Returns
        - Dict[int, Any]: 데이터 인덱스를 키로, 해당 데이터를 값으로 하는 딕셔너리.
                           (예: {0: (이미지_데이터, 레이블), 1: (이미지_데이터, 레이블), ...})

        ### Raises
        - NotImplementedError: 이 메소드가 하위 클래스에서 구현되지 않았을 경우 발생.

        """
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, index) -> Any:
        raise NotImplementedError


@dataclass
class Dataloader_Config(Config.Basement):
    batch_size: int = 1
    shuffle: bool = True
    num_workers: int = 0
    drop_last: bool = False


def Build_loader(
    dataloader_cfg: Dataloader_Config,
    dataset: Dataset,
    collect_func: Callable | None = None,
    is_multi_process: bool = False,
    world_size: int = 1,
    rank: int = 0
):
    _dataloader_cfg = dataloader_cfg.Config_to_dict()
    if is_multi_process:
        _smapler = DistributedSampler(
            dataset, world_size, rank, _dataloader_cfg["shuffle"]
        )
        _dataloader_cfg["shuffle"] = False

    else:
        _smapler = None

    _dataloder = DataLoader(
        dataset, sampler=_smapler, collate_fn=collect_func,
        **dataloader_cfg.Config_to_dict()
    )

    return _dataloder, _smapler
