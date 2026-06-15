from typing import Any, Callable

from python_toolbox.registry import Registry

from .definition import Module_Config_Template, Composable_Config, Composable_Module
from .model.definition import Trainable_Model

MODELS = Registry[type[Trainable_Model]]("models", Trainable_Model)
LOSSES = Registry[type[Composable_Module]]("losses", Composable_Module)
