from enum import auto
from python_ex.system import String

from . import Backbone_Module_with_Hub

from ..submodules.dino_v2.dinov2.models.vision_transformer import (
    DinoVisionTransformer)


LAYER_IDX = {
    'vits': [2, 5, 8, 11],
    'vitb': [2, 5, 8, 11], 
    'vitl': [4, 11, 17, 23], 
    'vitg': [9, 19, 29, 39]
}


class MODEL_TYPE(String.String_Enum):
    VITS = auto()
    VITB = auto()
    VITL = auto()
    VITG = auto()


class DINO_V2(Backbone_Module_with_Hub):
    def __init__(
        self,
        is_pretrained: bool,
        is_trainable: bool,
        model_type: MODEL_TYPE,
        with_registers: bool = False,
        is_reshape: bool = True,
    ):
        super().__init__(
            is_pretrained, is_trainable,
            model_type=model_type,
            with_registers=with_registers,
        )

        self.is_reshape = is_reshape

    def __modele_init__(
        self, model_type: MODEL_TYPE, with_registers: bool,
    ):
        _location = "facebookresearch/dinov2"

        _model_name = f"dinov2_{str(model_type)}14"
        _model_name = f"{_model_name}{'_reg' if with_registers else ''}"

        self.backbone: DinoVisionTransformer = self.__Get_model_from_hub__(
            _location, _model_name
        )
        self.idx = LAYER_IDX[str(model_type)]

    def forward(self, x):
        return self.backbone.get_intermediate_layers(
            x, self.idx, return_class_token=True
        )