from . import repvgg, dino, clip, convnext, efficientnet, resnet, swin


BACKBONES = (
    repvgg.RepVGG | dino.DINO | clip.CLIP_Vision
    | convnext.ConvNeXt | efficientnet.EfficientNet | resnet.ResNet | swin.Swin
)

BACKBONE_CFGS = (
    repvgg.RepVGG_Config | dino.DINO_Config | clip.CLIP_Vision_Config
    | convnext.ConvNeXt_Config | efficientnet.EfficientNet_Config
    | resnet.ResNet_Config | swin.Swin_Config
)