from __future__ import annotations

from enum import Enum
from typing import List, Tuple, Type

from python_ex._Base import TYPE_NUMBER
from python_ex._Numpy import Random_Process

from torch import Tensor
import math

# layer utils
from torchsummary import summary as ModelSummary

# modules
from torch.nn import Module, ModuleList, Sequential, Dropout
from torch.nn import parameter
from torch.nn import Linear, Conv1d, Conv2d, Upsample, MultiheadAttention
from torch.nn import LayerNorm

# optim
from torch import optim
from torch.optim.lr_scheduler import _LRScheduler

from ._Base import Tensor_Process


class Model(Module):
    def __init__(self, model_name: str, **model_parameter):
        super(Model, self).__init__()
        self._model_name = model_name

    # Freeze function
    def _Sumarry(self, input_shape: List[int]):
        ModelSummary(self, input_shape)

    # Un-Freeze function
    def forward(self, x):
        raise NotImplementedError


class Model_Componant():
    @staticmethod
    def _Make_module_list(module_list: List) -> ModuleList:
        return ModuleList(module_list)

    @staticmethod
    def _Make_sequential(componant_list: List) -> Sequential:
        return Sequential(*componant_list)

    @staticmethod
    def _Make_weight(size: int | List[int], value: TYPE_NUMBER | List[TYPE_NUMBER], rand_opt: Random_Process = Random_Process.UNIFORM, dtype: Type | None = None):
        return parameter.Parameter(Tensor_Process._Make_tensor(size, value, rand_opt, dtype))

    class Linear(Module):
        def __init__(
            self,
            input_size: int,
            output_size: int,
            is_bias: bool = True,
            normization: Module | None = None,
            activate: Type | None = None
        ):
            super(Model_Componant.Linear, self).__init__()

            self._linear = Linear(input_size, output_size, is_bias)
            self._norm = normization
            self._active = activate

        def forward(self, x: Tensor) -> Tensor:
            _x = self._linear(x)
            _x = self._norm(_x) if self._norm is not None else _x
            _x = self._active(_x) if self._active is not None else _x
            return _x

    class Conv1d(Module):
        def __init__(
            self,
            input_size: int,
            output_size: int,
            kernel: int = 1,
            stride: int = 1,
            padding: int = 0,
            padding_mode: str = 'zeros',
            dilation: int = 1,
            groups: int = 1,
            is_bias: bool = True,
            normization: Module | None = None,
            activate: Type | None = None
        ):
            super(Model_Componant.Conv1d, self).__init__()

            self._conv1D = Conv1d(input_size, output_size, kernel, stride, padding, dilation, groups, is_bias, padding_mode)
            self._norm = normization
            self._activate = activate

        def forward(self, x: Tensor) -> Tensor:
            _x = self._conv1D(x)
            _x = self._norm(_x) if self._norm is not None else _x
            _x = self._activate(_x) if self._activate is not None else _x
            return _x

    class Conv2d(Module):
        def __init__(
            self,
            input_size: int,
            output_size: int,
            kernel: int | Tuple[int, int] = 1,
            stride: int | Tuple[int, int] = 1,
            padding: int | Tuple[int, int] = 0,
            padding_mode: str = 'zeros',
            dilation: int | Tuple[int, int] = 1,
            groups: int = 1,
            is_bias: bool = True,
            normization: Module | None = None,
            activate: Type | None = None
        ):
            super(Model_Componant.Conv2d, self).__init__()

            self._conv2D = Conv2d(input_size, output_size, kernel, stride, padding, dilation, groups, is_bias, padding_mode)
            self._norm = normization
            self._activate = activate

        def forward(self, x: Tensor) -> Tensor:
            _x = self._conv2D(x)
            _x = self._norm(_x) if self._norm is not None else _x
            _x = self._activate(_x) if self._activate is not None else _x
            return _x

    class Decoder():
        class Skip_Upconv2d(Module):
            ...

        class PUP(Module):
            def __init__(
                self,
                input_size: int,
                output_size: int,
                scale_factor: int = 2,
                normization: Module | None = None,
                activate: Type | None = None
            ):
                super().__init__()

                self._conv_module = Model_Componant.Conv2d(input_size, output_size, 3, 1, 1, normization=normization, activate=activate)
                self._sampling = Upsample(scale_factor=scale_factor, mode="bilinear")

            def forward(self, x: Tensor) -> Tensor:
                _x = self._conv_module(x)
                _x = self._sampling(_x)
                return _x

    class Position_Embeder():
        class Supported(Enum):
            TRIGONOMETRIC = "Trigonometric"
            GAUSSIAN = "Gaussian"
            TRAINABLE = "Trainable"

        class Embeder_Module(Module):
            def __init__(self, num_of_data: int, temperature: int = 10000, normalize=True, scale=None):
                super().__init__()
                self._is_normalize = normalize

                # make dimention term
                self._dim_term = temperature ** (2 * (Tensor_Process._Arange(num_of_data) // 2) / num_of_data)
                self._dim_term.requires_grad = False

        class Trigonometric(Embeder_Module):
            def __init__(self, num_of_data: int, temperature: int = 10000, normalize=True, scale=None):
                super().__init__(num_of_data, temperature, normalize, scale)

                if scale is None:
                    scale = 2 * math.pi
                self.scale = scale

            def forward(self, x: Tensor):
                _tensor_shape = x.shape
                assert len(_tensor_shape) == 3, f"Default Position_Embeder {self.__class__.__name__} is supported 3 demention data (batch channel data)."

                _pos = Tensor_Process._Make_tensor([_tensor_shape[0], _tensor_shape[2]], 0).to(device=x.device)
                _pos = _pos[:, :, None] / self._dim_term.to(device=x.device)
                _pos = Tensor_Process._stack([_pos[:, :, 0::2].sin(), _pos[:, :, 1::2].cos()], dim=3).flatten(2).permute(0, 2, 1)

                return _pos

        class Gaussian(Embeder_Module):
            def __init__(self, num_of_data: int, temperature: int = 10000, normalize=True, scale=None):
                super().__init__(num_of_data, temperature, normalize, scale)

        class Trainable(Embeder_Module):
            def __init__(self, num_of_data: int, temperature: int = 10000, normalize=True, scale=None):
                super().__init__(num_of_data, temperature, normalize, scale)

        @staticmethod
        def _Build(Type: Supported, num_of_data: int, temperature: int = 10000, normalize=True, scale=None):
            return Model_Componant.Position_Embeder.__dict__[Type.value](num_of_data, temperature, normalize, scale)

    class Attention():
        class Muiltihead(Module):
            def __init__(self, input_dim: int, head_count: int, drop_rate: float) -> None:
                super().__init__()
                self.attention = MultiheadAttention(input_dim, head_count, drop_rate)
                self.dropout = Dropout(drop_rate)

            def _Make_QKV(self, x: Tuple[Tensor, Tensor], **additional_parm) -> List[Tensor]:
                return [x[0], x[1], x[1]]

            def forward(self, x: Tuple[Tensor, Tensor], mask: Tensor | None = None, key_padding_mask: Tensor | None = None, **additional_parm):
                _q, _k, _v = self._Make_QKV(x, **additional_parm)
                _x, _map = self.attention(query=_q, key=_k, value=_v, attn_mask=mask, key_padding_mask=key_padding_mask)
                _x = x[0] + self.dropout(_x)

                return _x, _map

        class Self_Muiltihead(Muiltihead):
            def _Make_QKV(self, x: Tensor, **additional_parm) -> List[Tensor]:
                return [x, x, x]

            def forward(self, x: Tensor, mask: Tensor | None = None, key_padding_mask: Tensor | None = None, **additional_parm):
                _q, _k, _v = self._Make_QKV(x, **additional_parm)
                _x, _map = self.attention(query=_q, key=_k, value=_v, attn_mask=mask, key_padding_mask=key_padding_mask)
                _x = x + self.dropout(_x)

                return _x, _map

    class Transformer():
        class Encoder(Module):
            def __init__(
                self,
                input_dim: int,
                head_count: int,
                feadforward_dim: int,
                drop_rate: float,
                attention_method: Type[Model_Componant.Attention.Self_Muiltihead],
                activation: Type,
                normalize_before: bool = False
            ):
                super().__init__()
                self._normalize_before = normalize_before

                self._front_norm = LayerNorm(input_dim)
                self._attention = attention_method(input_dim, head_count, drop_rate)
                self._back_norm = LayerNorm(input_dim)

                self._linear_block = Model_Componant._Make_sequential([
                    Model_Componant.Linear(input_dim, feadforward_dim, activate=activation),
                    Dropout(drop_rate),
                    Model_Componant.Linear(feadforward_dim, input_dim),
                    Dropout(drop_rate)
                ])

            def forward(self, x: Tensor, mask: Tensor | None = None, key_padding_mask: Tensor | None = None, **additional_parm):
                if self._normalize_before:
                    # attention
                    _x = self._front_norm(x)
                    _attention_out, _map = self._attention(_x, mask, key_padding_mask, **additional_parm)

                    # fc
                    _x = self._back_norm(_attention_out)
                    _x = _attention_out + self._linear_block(_x)
                else:
                    # attention
                    _attention_out, _map = self._attention(x, mask, key_padding_mask, **additional_parm)
                    _attention_out = self._front_norm(_attention_out)

                    # fc
                    _x = _attention_out + self._linear_block(_attention_out)
                    _x = self._back_norm(_x)
                return _x, _map

        class Decoder(Module):
            def __init__(
                self,
                input_dim: int,
                head_count: int,
                feadforward_dim: int,
                drop_rate: float,
                attention_method: Tuple[Type[Model_Componant.Attention.Self_Muiltihead], Type[Model_Componant.Attention.Muiltihead]],
                activation: Type,
                normalize_before: bool = False
            ):
                super().__init__()
                self._normalize_before = normalize_before

                self._front_norm = LayerNorm(input_dim)
                self._self_attention = attention_method[0](input_dim, head_count, drop_rate)
                self._mid_norm = LayerNorm(input_dim)
                self._multi_attention = attention_method[1](input_dim, head_count, drop_rate)
                self._back_norm = LayerNorm(input_dim)

                self._linear_block = Model_Componant._Make_sequential([
                    Model_Componant.Linear(input_dim, feadforward_dim, activate=activation),
                    Dropout(drop_rate),
                    Model_Componant.Linear(feadforward_dim, input_dim),
                    Dropout(drop_rate)
                ])

            def forward(
                    self,
                    x: Tuple[Tensor, Tensor],
                    mask: Tuple[Tensor | None, Tensor | None] = (None, None),
                    key_padding_mask: Tuple[Tensor | None, Tensor | None] = (None, None),
                    **additional_parm
            ):
                if self._normalize_before:
                    # attention
                    _x = self._front_norm(x[0])
                    _attention_out, _self_attention_map = self._self_attention(_x, mask[0], key_padding_mask[0], **additional_parm)
                    _x = self._mid_norm(_attention_out)
                    _attention_out, _multi_attention_map = self._multi_attention([_x, x[1]], mask[1], key_padding_mask[1], **additional_parm)

                    # fc
                    _x = self._back_norm(_attention_out)
                    _x = _attention_out + self._linear_block(_x)
                else:
                    # attention
                    _attention_out, _self_attention_map = self._self_attention(x[0], mask[0], key_padding_mask[0], **additional_parm)
                    _x = self._front_norm(_attention_out)
                    _attention_out, _multi_attention_map = self._multi_attention([_x, x[1]], mask[1], key_padding_mask[1], **additional_parm)
                    _x = self._mid_norm(_attention_out)

                    # fc
                    _x = _attention_out + self._linear_block(_x)
                    _x = self._back_norm(_x)
                return _x, _self_attention_map, _multi_attention_map

    # class PerceiverIO(Module):
    #     def __init__(self):
    #         super().__init__()

    #         self.latent = torch_utils._tensor.make_tensor()

    #         self.encode = module._Attention._cross_dot()
    #         self.decode = module._Attention._cross_dot()

    #         _process = []
    #         for _ct in range():
    #             _process.append(module._Attention._self_dot())

    #         self.process = module.make_module_list(_process)

    #     def forward(self, input, ouput_query):
    #         _output = self.encode(Q_source=self.latent, KV_source=input)

    #         for _p in self.process:
    #             _output = _p(_output)

    #         _output = self.decode(ouput_query, _output)
    #         return _output

    # class GatedFusion(Module):
    #     def __init__(self, feature_channel: int, fusion_target_num: int) -> None:
    #         super().__init__()
    #         self.fusion_target_num = fusion_target_num

    #         layer_list = []
    #         for _ in range(fusion_target_num):
    #             layer_list.append(Linear(feature_channel, fusion_target_num))
    #         self.fusion_weight_layer = ModuleList(layer_list)

    #     def forward(self, x):
    #         reshape_tensor = False
    #         if x[0].dim() == 4:
    #             reshape_tensor = True
    #             B, C, H, W = x[0].shape
    #             x_copy = []
    #             for x_ele in x:
    #                 x_copy.append(rearrange(x_ele, 'b c h w -> b (h w) c'))
    #         else:
    #             x_copy = x.copy()

    #         weight_tensor = 0
    #         for i, fusion_layer in enumerate(self.fusion_weight_layer):
    #             weight_tensor += fusion_layer(x_copy[i])
    #         softmax_weight = functional.softmax(weight_tensor, dim=-1)
    #         split_weight = softmax_weight.chunk(self.fusion_target_num, dim=-1)
    #         y = 0
    #         for i, x_ele in enumerate(x_copy):
    #             y += (x_ele * split_weight[i])

    #         if reshape_tensor:
    #             y = rearrange(y, 'b (h w) c -> b c h w', h=H, w=W)
    #         return y

            # x1 = torch.rand((3, 64, 480, 640))
            # x2 = torch.rand((3, 64, 480, 640))
            # x3 = torch.rand((3, 64, 480, 640))

            # fusion_layer = GatedFusion(64, 3)
            # y = fusion_layer([x1, x2, x3])
            # print(y)

    class Backbone():
        class Supported(Enum):
            # from torchvision.models import vgg11, vgg13, vgg16, convnext_base, convnext_large, convnext_small, convnext_tiny
            # classification
            VGG_11 = ("VGG", 11)
            VGG_13 = ("VGG", 13)
            VGG_16 = ("VGG", 16)
            ResNet_50 = ("ResNet", 50)
            ResNet_101 = ("ResNet", 101)
            ConvNext_tiny = ("ConvNext", "tiny")
            ConvNext_small = ("ConvNext", "small")
            ConvNext_base = ("ConvNext", "base")
            ConvNext_large = ("ConvNext", "large")

            # segmentation
            FCN_50 = ("FCN", 50)
            FCN_101 = ("FCN", 101)

        class Backbone_Module(Module):
            def __init__(self, model_type: int, is_pretrained: bool, is_trainable: bool):
                super(Model_Componant.Backbone.Backbone_Module, self).__init__()
                self._output_channel = []

            def _Average_pooling(self, ouput: Tensor):
                raise NotImplementedError

        class VGG(Backbone_Module):
            def __init__(self, model_type: int, is_pretrained: bool, is_trainable: bool):
                super(Model_Componant.Backbone.VGG, self).__init__(model_type, is_pretrained, is_trainable)
                self._output_channel = [64, 128, 256, 512, 512]
                if model_type == 11:
                    from torchvision.models import vgg11, VGG11_Weights
                    _model = vgg11(weights=VGG11_Weights.DEFAULT if is_pretrained else None)
                if model_type == 13:
                    from torchvision.models import vgg13, VGG13_Weights
                    _model = vgg13(weights=VGG13_Weights.DEFAULT if is_pretrained else None)
                else:
                    from torchvision.models import vgg16, VGG16_Weights
                    _model = vgg16(weights=VGG16_Weights.DEFAULT if is_pretrained else None)

                self._conv = _model.features
                self._avgpool = _model.avgpool

                for _parameter in self._conv.parameters():
                    _parameter.requires_grad = is_trainable

            def forward(self, x: Tensor):
                return self._conv(x)  # retrun shape : batch_size, 512, h/32, w/32

            def _Average_pooling(self, ouput: Tensor):
                return self._avgpool(ouput)

        class ResNet(Backbone_Module):
            def __init__(self, model_type: int, is_pretrained: bool, is_trainable: bool):
                super(Model_Componant.Backbone.ResNet, self).__init__(model_type, is_pretrained, is_trainable)
                self._output_channel = [64, 256, 512, 1024, 2048]
                if model_type == 101:
                    from torchvision.models import resnet101, ResNet101_Weights
                    _model = resnet101(weights=ResNet101_Weights.DEFAULT if is_pretrained else None)
                else:
                    from torchvision.models import resnet50, ResNet50_Weights
                    _model = resnet50(weights=ResNet50_Weights.DEFAULT if is_pretrained else None)

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
                _out_conv1 = self.relu(_x)              # x shape : batch_size, 64, h/2, w/2
                _x = self.maxpool(_out_conv1)
                _out_conv2 = self.layer1(_x)            # x shape : batch_size, 256, h/4, w/4
                _out_conv3 = self.layer2(_out_conv2)    # x shape : batch_size, 512, h/8, w/8
                _out_conv4 = self.layer3(_out_conv3)    # x shape : batch_size, 1024, h/16, w/16
                _out_conv5 = self.layer4(_out_conv4)    # x shape : batch_size, 2048, h/32, w/32

                return [_out_conv1, _out_conv2, _out_conv3, _out_conv4, _out_conv5]

            def _Average_pooling(self, ouput: Tensor):
                return Tensor_Process._Flatten(self.avgpool(ouput))

        class ConvNext(Backbone_Module):
            def __init__(self, model_type: int, is_pretrained: bool, is_trainable: bool):
                super(Model_Componant.Backbone.ConvNext, self).__init__(model_type, is_pretrained, is_trainable)
                self._output_channel = [96, 96, 192, 384, 768]
                if model_type == "tiny":
                    from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
                    _model = convnext_tiny(weights=ConvNeXt_Tiny_Weights.DEFAULT if is_pretrained else None)
                elif model_type == "small":
                    from torchvision.models import convnext_small, ConvNeXt_Small_Weights
                    _model = convnext_small(weights=ConvNeXt_Small_Weights.DEFAULT if is_pretrained else None)
                    self._output_channel = [64, 256, 512, 1024, 2048]
                elif model_type == "base":
                    from torchvision.models import convnext_base, ConvNeXt_Base_Weights
                    _model = convnext_base(weights=ConvNeXt_Base_Weights.DEFAULT if is_pretrained else None)
                    self._output_channel = [64, 256, 512, 1024, 2048]
                else:  # large
                    from torchvision.models import convnext_large, ConvNeXt_Large_Weights
                    _model = convnext_large(weights=ConvNeXt_Large_Weights.DEFAULT if is_pretrained else None)
                    self._output_channel = [64, 256, 512, 1024, 2048]

                self._conv = _model.features
                self._avgpool = _model.avgpool

                # features parameters doesn't train
                for _parameters in _model.parameters():
                    _parameters.requires_grad = is_trainable

            def forward(self, x: Tensor):
                return self._conv(x)  # retrun shape : batch_size, 768, h/32, w/32

            def _Average_pooling(self, ouput: Tensor):
                return self._avgpool(ouput)

        class FCN(Backbone_Module):
            def __init__(self, model_type: int, is_pretrained: bool, is_trainable: bool):
                super(Model_Componant.Backbone.FCN, self).__init__(model_type, is_pretrained, is_trainable)
                if model_type == 101:
                    from torchvision.models.segmentation import fcn_resnet101, FCN_ResNet101_Weights
                    self._line = fcn_resnet101(weights=FCN_ResNet101_Weights.DEFAULT if is_pretrained else None)
                else:
                    from torchvision.models.segmentation import fcn_resnet50, FCN_ResNet50_Weights
                    self._line = fcn_resnet50(weights=FCN_ResNet50_Weights.DEFAULT if is_pretrained else None)

                for _module in self._line.parameters():
                    _module.requires_grad = is_trainable

            def forward(self, x):
                return self._line(x)

            def _Average_pooling(self, ouput: Tensor):
                raise NotImplementedError

        @staticmethod
        def _Build(model_info: Supported, is_pretrained: bool, is_trainable: bool) -> Backbone_Module:
            _name = model_info.value[0]
            _type = model_info.value[1]

            return Model_Componant.Backbone.__dict__[_name](_type, is_pretrained, is_trainable)


class Optim():
    class Supported(Enum):
        Adam = "Adam"

    class Scheduler():
        class Supported(Enum):
            Cosin_Annealing = "Cosin_Annealing"

        class Basement(_LRScheduler):
            def __init__(
                    self,
                    optimizer: optim.Optimizer,
                    term: int | List[int],
                    term_amp: float,
                    maximum: float,
                    minimum: float,
                    decay: float,
                    last_epoch: int = -1) -> None:
                self._Cycle: int = 0
                self._Term = term  # int -> fixed term list[int] -> milestone
                self._Term_amp = term_amp

                self._Maximum = maximum
                self._Minimum = minimum
                self._Decay = decay

                self._This_count: int = last_epoch
                self._This_term: int = self._get_next_term()

                super().__init__(optimizer, last_epoch)

            # Freeze function
            def _get_next_term(self):
                if isinstance(self._Term, list):
                    return self._Term[-1] if self._Cycle >= len(self._Term) else self._Term[self._Cycle]
                else:
                    return round(self._Term * (self._Term_amp ** self._Cycle))

            def step(self, epoch: int | None = None):
                if epoch is None:  # go to next epoch
                    self.last_epoch += 1
                    self._This_count += 1

                    if self._This_count >= self._This_term:
                        self._This_count = 0
                        self._Cycle += 1
                        self._This_term = self._get_next_term()

                else:  # restore session
                    while epoch >= self._This_term:
                        epoch -= self._This_term
                        self._Cycle += 1
                        self._This_term = self._get_next_term()

                    self._This_count = epoch

                for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):  # type: ignore
                    param_group['lr'] = lr

            # Un-Freeze function
            def get_lr(self):
                return [self._Maximum for _ in self.base_lrs]  # type: ignore

        class Cosin_Annealing(Basement):
            def get_lr(self):
                _amp = (1 + math.cos(math.pi * (self._This_count) / (self._This_term))) / 2
                _value = self._Minimum + (self._Maximum - self._Minimum) * _amp
                return [_value for _ in self.base_lrs]  # type: ignore

    @staticmethod
    def _build(
        optim_name: Supported,
        model: Module,
        initial_lr: float,
        schedule_name: Scheduler.Supported | None,
        last_epoch: int = -1,
        **additional_parameter
    ) -> Tuple[optim.Optimizer, _LRScheduler | None]:

        _optim = optim.__dict__[optim_name.value](model.parameters(), initial_lr)
        _scheduler = Optim.Scheduler.__dict__[schedule_name.value](_optim, last_epoch=last_epoch, **additional_parameter) if schedule_name is not None else None

        return _optim, _scheduler
