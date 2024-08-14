""" ###

------------------------------------------------------------------------
### Requirement
    -

### Structure
    -

"""
from __future__ import annotations
from typing import (
    Type, TypeVar, Any
)

from torch.utils.data import Dataset


class Parser():
    def __init__(self, data_dir: str, **kwarg) -> None:
        self.data_info: dict[str, Any]
        self.data_block: dict[str, list] = {}

        self.Get_data_from(data_dir, **kwarg)

    def Get_data_from(self, data_dir: str, **kwarg):
        raise NotImplementedError


PARSER = TypeVar("PARSER", bound=Parser)


class Dataset_Basement(Dataset):
    def __init__(
        self,
        data_dir: str, data_parser: Type[PARSER] = Parser,
        **parser_kwarg
    ) -> None:
        _parser = data_parser(data_dir, **parser_kwarg)
        self.data_info = _parser.data_info
        self.data_block = _parser.data_block

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, index) -> Any:
        raise NotImplementedError
