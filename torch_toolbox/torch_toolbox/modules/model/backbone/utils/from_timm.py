import torch.nn as nn
import timm


def load_timm_backbone(
    model_name: str,
    out_indices: list[int] | None = None,
    **timm_kwargs
) -> nn.Module:
    out_indices = out_indices if out_indices is not None else [-1]
    return timm.create_model(
        model_name,
        features_only=True,
        out_indices=out_indices,
        **timm_kwargs
    )
