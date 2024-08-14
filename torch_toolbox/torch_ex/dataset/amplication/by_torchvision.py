""" ### torchvision을 사용한 데이터 증폭 처리 모듈

------------------------------------------------------------------------
### Requirement
    - torchvision

### Structure
    - FromTorchvision:

"""
from typing import Dict, Any

try:
    from torchvision.transforms import Compose
except Exception as e:
    _error_masage: str = "\n".join([
        "This process, that from Torchvision, is not suport in this env.",
        "Please change the augment process"
    ])

    raise ValueError(_error_masage) from e

from .basement import __Basement__


class FromTorchvision(__Basement__):
    """ ### torchvision을 사용한 데이터 증폭 처리 구조

    ---------------------------------------------------------------------
    ### Args
    - None

    ### Attributes
    - None

    ### Structure
    - None

    """
    def Resize(self):
        """ ###

        ------------------------------------------------------------------
        ### Args
        -

        ### Returns
        -

        ### Raises
        -

        """
        return super().Resize()

    def Config_to_compose(self, process_params: Dict[str, Dict[str, Any]]):
        """ ###

        ------------------------------------------------------------------
        ### Args
        -

        ### Returns
        -

        ### Raises
        -

        """
        _process_list = []

        for _name, _kwarg in process_params.items():
            _process_list.append(self.__class__.__dict__[_name](**_kwarg))

        return Compose(_process_list)
