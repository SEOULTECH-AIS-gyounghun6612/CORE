from typing import Dict, Any


def Build(augment_name: str, augment_config: Dict[str, Any], process_config: Dict[str, Dict[str, Any]]):
    if augment_name.lower() == "torchvision":
        from .From_Torchvision import FromTorchvision
        return FromTorchvision(**augment_config).Config_to_compose(process_config)

    _error_text = f"This augment process, that from {augment_name.lower()}, is not suport in this module version.\n"
    _error_text = f"{_error_text}Please change the augment process or module version\n"
    raise ValueError(_error_text)
