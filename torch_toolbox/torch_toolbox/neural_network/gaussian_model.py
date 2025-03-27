# import numpy as np
from torch import empty

from . import Custom_Model
from .utils.viewpoints import Viewpoints_Tenser


class Gaussian_Model(Custom_Model):
    def __init__(self, model_name: str, device: str = "cuda") -> None:
        super().__init__(model_name)

        self.xyz = empty(0, device=device)
        self.features_dc = empty(0, device=device)
        self.features_rest = empty(0, device=device)
        self.scaling = empty(0, device=device)
        self.rotation = empty(0, device=device)
        self.opacity = empty(0, device=device)

    def _Extend(self, **new_parameter):
        """
        """
        


        ...

    def _Clone(self):
        ...

    def _Prune(self):
        ...

    def _Split(self):
        ...

    def forward(self, input: Viewpoints_Tenser.Scene):
        ...


class Splatting_Render():
    @staticmethod
    def Render(
        gaussian: Gaussian_Model,
        veiwpoint: Viewpoints_Tenser.Scene
    ):
        ...