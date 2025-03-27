from typing import Literal, Any

from torch import Tensor, hub
from torch.nn import Module, functional as F


class Backbone():
    class Backbone_Module(Module):
        def __init__(
            self,
            is_pretrained: bool,
            is_trainable: bool
        ):
            super().__init__()
            self.is_pretrained = is_pretrained
            self.is_trainable = is_trainable

            self.output_channel = []

        def Average_pooling(self, ouput: Tensor) -> Tensor:
            raise NotImplementedError

    class VGG(Backbone_Module):
        def __init__(
            self, model_type: int, is_pretrained: bool, is_trainable: bool
        ):
            super().__init__(is_pretrained, is_trainable)
            self.output_channel = [64, 128, 256, 512, 512]
            if model_type == 11:
                from torchvision.models import vgg11, VGG11_Weights
                _model = vgg11(
                    weights=VGG11_Weights.DEFAULT if is_pretrained else None)
            if model_type == 13:
                from torchvision.models import vgg13, VGG13_Weights
                _model = vgg13(
                    weights=VGG13_Weights.DEFAULT if is_pretrained else None)
            else:
                from torchvision.models import vgg16, VGG16_Weights
                _model = vgg16(
                    weights=VGG16_Weights.DEFAULT if is_pretrained else None)

            self.conv = _model.features
            self.avgpool = _model.avgpool

            for _parameter in self.conv.parameters():
                _parameter.requires_grad = is_trainable

        def forward(self, x: Tensor):
            return self.conv(x)  # retrun shape : batch_size, 512, h/32, w/32

        def Average_pooling(self, ouput: Tensor) -> Tensor:
            return self.avgpool(ouput)

    class ResNet(Backbone_Module):
        def __init__(
            self, model_type: int, is_pretrained: bool, is_trainable: bool
        ):
            super().__init__(is_pretrained, is_trainable)
            self.output_channel = [64, 256, 512, 1024, 2048]
            if model_type == 101:
                from torchvision.models import resnet101, ResNet101_Weights
                _weight = ResNet101_Weights.DEFAULT if is_pretrained else None
                _model = resnet101(weights=_weight)
            else:
                from torchvision.models import resnet50, ResNet50_Weights
                _weight = ResNet50_Weights.DEFAULT if is_pretrained else None
                _model = resnet50(weights=_weight)

            # features parameters doesn't train
            for _parameters in _model.parameters():
                _parameters.requires_grad = is_trainable

            self.conv1 = _model.conv1
            self.bn1 = _model.bn1
            self.relu = _model.relu
            self.maxpool = _model.maxpool
            self.layer1 = _model.layer1
            self.layer2 = _model.layer2
            self.layer3 = _model.layer3
            self.layer4 = _model.layer4
            self.avgpool = _model.avgpool

        def forward(self, x: Tensor):
            _x = self.conv1(x)
            _x = self.bn1(_x)

            _out_conv1 = self.relu(_x)
            # _out_conv1 shape : batch_size, 64, h/2, w/2

            _x = self.maxpool(_out_conv1)
            _out_conv2 = self.layer1(_x)
            # _out_conv2 shape : batch_size, 256, h/4, w/4
            _out_conv3 = self.layer2(_out_conv2)
            # _out_conv3 shape : batch_size, 512, h/8, w/8
            _out_conv4 = self.layer3(_out_conv3)
            # _out_conv4 shape : batch_size, 1024, h/16, w/16
            _out_conv5 = self.layer4(_out_conv4)
            # _out_conv5 shape : batch_size, 2048, h/32, w/32

            return [_out_conv1, _out_conv2, _out_conv3, _out_conv4, _out_conv5]

        def Average_pooling(self, ouput: Tensor) -> Tensor:
            return self.avgpool(ouput)

    class DINO_V2(Backbone_Module):
        def __init__(
            self,
            model_type: Literal["vits", "vitb", "vitl", "vitg"] = "vits",
            with_registers: bool = False,
            is_reshape: bool = True
        ):
            super().__init__(True, False)
            _name_arg = [model_type, '_reg' if with_registers else '']
            _model: Any = hub.load(
                "facebookresearch/dinov2",
                "dinov2_{}14{}".format(*_name_arg)
            )

            for _parameter in _model.parameters():
                _parameter.requires_grad = False

            self.is_reshape = is_reshape
            self.model: Any = _model
            """
            "vits": [2, 5, 8, 11],
            "vitb": [2, 5, 8, 11],
            "vitl": [4, 11, 17, 23],
            """

        def forward(self, x: Tensor, *intermediate_layer_ids: int):
            _b, _, _h, _w = x.shape
            _b_h, _b_w = _h // 14, _w // 14
            _r_h, _r_w = _b_h * 14, _b_w * 14
            _r_x = F.interpolate(
                x, (_r_h, _r_w), mode="bilinear", align_corners=True
            )

            _outputs: list[Tensor] = self.model.get_intermediate_layers(
                _r_x, list(intermediate_layer_ids), return_class_token=False
            )

            if self.is_reshape:
                return [
                    _o
                    .reshape(_b, _b_h, _b_w, -1)
                    .permute(0, 3, 1, 2).contiguous() for _o in _outputs
                ]
            return list(_outputs)

        def Average_pooling(self, ouput: Tensor) -> Tensor:
            raise NotImplementedError

    @staticmethod
    def Build_backbone(
        model_name: str,
        model_type: int,
        is_pretrained: bool,
        is_trainable: bool
    ):
        _backbone_dict = Backbone.__dict__

        if model_name in _backbone_dict:
            return _backbone_dict[model_name](
                model_type, is_pretrained, is_trainable
            )

        raise ValueError(f"{model_name} is not supported. Please check it")
