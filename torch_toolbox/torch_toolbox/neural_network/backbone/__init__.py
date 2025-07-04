from typing import Any

from pathlib import Path
from torch.nn import Module

from torch import hub


class Backbone_Module(Module):
    def __init__(
        self,
        is_pretrained: bool, is_trainable: bool, save_dir: str | Path | None,
        **kwarg
    ):
        super().__init__()
        self.is_pretrained = is_pretrained
        self.is_trainable = is_trainable

        self.save_path = save_dir if save_dir is None else Path(save_dir)
        self.backbone = None

        self.__modele_init__(**kwarg)

    def __modele_init__(self, **kwarg):
        raise NotImplementedError


class Backbone_Module_with_Hub(Module):
    def __init__(
        self,
        is_pretrained: bool, is_trainable: bool,
        **kwarg
    ):
        super().__init__()
        self.is_pretrained = is_pretrained
        self.is_trainable = is_trainable

        self.__modele_init__(**kwarg)

    def __Get_model_from_hub__(self, location: str, model_name: str) -> Any:
        _model_list = hub.list(location)

        if model_name in _model_list:
            return hub.load(
                location, model_name
            )

        raise ValueError(f"This model {model_name} is not exist in {location}")

    def __modele_init__(self, **kwarg):
        raise NotImplementedError