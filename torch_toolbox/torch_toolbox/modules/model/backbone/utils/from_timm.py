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


class Timm_Feature_Backbone:
    """``load_timm_backbone`` 으로 만든 백본의 ``Out_channels`` 공통 구현.

    ``features_only=True`` 로 만들어진 timm 모델은 ``feature_info`` 에 선택된
    ``out_indices`` 단계별 채널 수를 들고 있다. 그대로 노출한다.

    Note:
        ``timm.create_model`` 을 직접 쓰는 백본(예: DINO)은 ``feature_info`` 가 없거나
        의미가 달라 각자 오버라이드한다.
    """

    backbone: nn.Module

    def Out_channels(self) -> list[int]:
        return [int(_c) for _c in self.backbone.feature_info.channels()]
