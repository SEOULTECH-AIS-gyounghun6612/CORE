
if __package__ == "":
    # if this file in local project
    from torch_ex._torch_base import Learning_Mode, Log_Config, Torch_Utils, Debug
    from torch_ex._dataloader import Dataset_Config
    from torch_ex._layer import Custom_Model_Config
    from torch_ex._optimizer import Scheduler_Config
else:
    # if this file in package folder
    from ._torch_base import Learning_Mode, Log_Config, Torch_Utils, Debug
    from ._dataloader import Dataset_Config
    from ._layer import Custom_Model_Config
    from ._optimizer import Scheduler_Config


# -- DEFINE CONFIG -- #


# -- Mation Function -- #
class Set_Up():
    ...
