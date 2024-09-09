# import numpy as np


from torch import empty

from . import Model_Basement
from .utils.viewpoints import Viewpoints_Tenser


class Gaussian_Splatting(Model_Basement):
    def __init__(self, model_name: str) -> None:
        super().__init__(model_name)

        self.xyz = empty(0, device="cuda")
        self.features_dc = empty(0, device="cuda")
        self.features_rest = empty(0, device="cuda")
        self.scaling = empty(0, device="cuda")
        self.rotation = empty(0, device="cuda")
        self.opacity = empty(0, device="cuda")

    def _Add_gaussian(self, input: Viewpoints_Tenser.Scene):
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
        gaussian: Gaussian_Splatting,
        veiwpoint: Viewpoints_Tenser.Scene
    ):
        ...