from __future__ import annotations

from enum import Enum
from typing import List, Tuple, Type

from python_ex._Base import TYPE_NUMBER
from python_ex._Numpy import Random_Process

from torch import Tensor
from torch import exp, cos, sin, matmul
import math

# layer utils
from torchsummary import summary as ModelSummary

# modules
from torch.nn.common_types import _size_2_t
from torch.nn import Module, ModuleList, Sequential, Dropout
from torch.nn import parameter, init
from torch.nn import Linear, Conv2d, Upsample
from torch.nn import LayerNorm
from torch.nn.functional import softmax, gelu

from torchvision.models import resnet101, resnet50, vgg11, vgg13, vgg16, vgg19
from torchvision.models.segmentation import fcn_resnet101, fcn_resnet50

from einops import rearrange

# optim
from torch import optim
from torch.optim.lr_scheduler import _LRScheduler

from ._Base import Tensor_Process, Data_Type


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
    def _Make_weight(size: int | List[int], value: TYPE_NUMBER | List[TYPE_NUMBER], rand_opt: Random_Process = Random_Process.UNIFORM, dtype: Data_Type | None = None):
        return parameter.Parameter(Tensor_Process._Make_tensor(size, value, rand_opt, dtype))

    class Linear(Module):
        def __init__(
            self,
            input_size: int,
            output_size: int,
            is_bias: bool = True,
            normization: Module | None = None,
            activate: Module | Type | None = None
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

    class Conv2d(Module):
        def __init__(
            self,
            input_size: int,
            output_size: int,
            kernel: _size_2_t = 1,
            stride: _size_2_t = 1,
            padding: _size_2_t = 0,
            padding_mode: str = 'zeros',
            dilation: _size_2_t = 1,
            groups: int = 1,
            is_bias: bool = True,
            normization: Module | None = None,
            activate: Module | None = None
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
                activate: Module | None = None
            ):
                super().__init__()

                self._conv_module = Model_Componant.Conv2d(input_size, output_size, 3, 1, 1, normization=normization, activate=activate)
                self._sampling = Upsample(scale_factor=scale_factor, mode="bilinear")

            def forward(self, x: Tensor) -> Tensor:
                _x = self._conv_module(x)
                _x = self._sampling(_x)
                return _x

    class Position_Embeder():
        class Trigonometric(Module):
            def __init__(self, num_of_data: int, max_token_size: int = 1000):
                super().__init__()
                _pe = Tensor_Process._Make_tensor([max_token_size, num_of_data], value=0)
                _position = Tensor_Process._Arange(max_token_size, dtype=Data_Type.FLOAT).unsqueeze(1)
                _div_term = exp(Tensor_Process._Arange(num_of_data, step=2, dtype=Data_Type.FLOAT) * (-math.log(10000.0) / num_of_data))
                _pe[:, 0::2] = sin(_position * _div_term)
                _pe[:, 1::2] = cos(_position * _div_term)
                _pe = _pe.unsqueeze(0)

                self.register_buffer("_Position_value", _pe, persistent=False)

            def forward(self, x: Tensor):
                if isinstance(self._Position_value, Tensor):
                    return x + self._Position_value[:, : x.size(1)]
                else:
                    raise TypeError(f"Parameter '_Position_value' in {self.__class__.__name__} type incorrect")

        class Gaussian(Module):
            ...

    class Attention():
        class Supported(Enum):
            Dot_Attention = "Dot_Attention"

        class Base(Module):
            def __init__(self, input_dim: int, output_dim: int, head_count: int):
                super().__init__()
                _output_dim = output_dim + (output_dim % head_count) if (output_dim % head_count) else output_dim

                self._head_count = head_count
                self._head_dim = _output_dim // head_count

                self.q_maker = Model_Componant.Linear(input_dim, output_dim)
                init.xavier_uniform_(self.q_maker._linear.weight)
                self.q_maker._linear.bias.data.fill_(0)

                self.k_maker = Model_Componant.Linear(input_dim, output_dim)
                init.xavier_uniform_(self.k_maker._linear.weight)
                self.k_maker._linear.bias.data.fill_(0)

                self.v_maker = Model_Componant.Linear(input_dim, output_dim)
                init.xavier_uniform_(self.v_maker._linear.weight)
                self.v_maker._linear.bias.data.fill_(0)

                self.o_maker = Model_Componant.Linear(output_dim, output_dim)
                init.xavier_uniform_(self.o_maker._linear.weight)
                self.o_maker._linear.bias.data.fill_(0)

            def forward(
                self,
                Q_source: Tensor,
                K_source: Tensor,
                V_source: Tensor,
                mask: Tensor | None = None
            ) -> Tuple[Tensor, Tensor]:
                raise NotImplementedError

        class Dot_Attention(Base):
            def __init__(self, input_dim: int, output_dim: int, head_count: int):
                super().__init__(input_dim, output_dim, head_count)

            def _Dot_product(self, Q, K, V, mask=None):  # dot_product
                _logits = matmul(Q, rearrange(K, 'batch head_num seq head_dim -> batch head_num head_dim seq'))  # -> batch head_num seq seq
                _logits = _logits / math.sqrt(self._head_dim)

                if mask is not None:
                    _logits = _logits.masked_fill(mask == 0, -9e15)

                _attention = softmax(_logits, dim=-1)

                return matmul(_attention, V), _attention

            def forward(
                self,
                Q_source: Tensor,
                K_source: Tensor,
                V_source: Tensor,
                mask: Tensor | None = None
            ) -> Tuple[Tensor, Tensor]:
                _q = self.q_maker(Q_source)
                _q = rearrange(_q, 'batch seq (head_dim head_num) -> batch head_num seq head_dim', head_dim=self._head_dim, head_num=self._head_count)

                _k = self.k_maker(K_source)
                _k = rearrange(_k, 'batch seq (head_dim head_num) -> batch head_num seq head_dim', head_dim=self._head_dim, head_num=self._head_count)

                _v = self.v_maker(V_source)
                _v = rearrange(_v, 'batch seq (head_dim head_num) -> batch head_num seq head_dim', head_dim=self._head_dim, head_num=self._head_count)

                _value, _attention = self._Dot_product(_q, _k, _v, mask)  # value -> batch head_num seq head_dim

                _value = rearrange(_value, 'batch head_num seq head_dim -> batch seq (head_dim head_num)')
                _outpot = self.o_maker(_value)
                return _outpot, _attention

    class Transformer(Module):
        def __init__(self, input_dim: int, output_dim: int, head_count: int, hidden_rate: int, drop_rate: float, attention_method: Model_Componant.Attention.Supported):
            super().__init__()

            _output_dim = output_dim + (output_dim % head_count) if (output_dim % head_count) else output_dim
            self._head_count = head_count
            self._head_dim = _output_dim // head_count

            self._front_norm = LayerNorm(_output_dim)
            self._attention = Model_Componant.Attention.__dict__[attention_method.value](input_dim, _output_dim, head_count)
            self._back_norm = LayerNorm(_output_dim)

            self.linear_block = Model_Componant._Make_sequential([
                Model_Componant.Linear(_output_dim, _output_dim * hidden_rate, activate=gelu),
                Dropout(drop_rate),
                Model_Componant.Linear(_output_dim * hidden_rate, _output_dim)
            ])

        def forward(self, x) -> Tensor:
            _x = self._front_norm(x + self._attention(x, x, x))
            _x = self._back_norm(_x + self.linear_block(_x))

            return _x

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
            ResNet_50 = ("ResNet", 50)
            ResNet_101 = ("ResNet", 101)
            VGG_11 = ("VGG", 11)
            VGG_13 = ("VGG", 13)
            VGG_16 = ("VGG", 16)
            VGG_19 = ("VGG", 19)
            FCN_50 = ("FCN", 50)
            FCN_101 = ("FCN", 101)

        class Backbone_Base(Module):
            _output_channel: List[int]

            def _Average_pooling(self, ouput: Tensor) -> Tensor:
                raise NotImplementedError

        class ResNet(Backbone_Base):
            def __init__(self, model_type: int, is_pretrained: bool, is_trainable: bool):
                super(Model_Componant.Backbone.ResNet, self).__init__()
                if model_type == 101:
                    _model = resnet101(pretrained=is_pretrained)  # [64, 256, 512, 1024, 2048]
                    self._output_channel = [64, 256, 512, 1024, 2048]
                else:
                    _model = resnet50(pretrained=is_pretrained)  # [64, 256, 512, 1024, 2048]
                    self._output_channel = [64, 256, 512, 1024, 2048]

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
                _out_conv1 = self.relu(_x)              # x shape : batch_size, channel #0, h/2, w/2
                _x = self.maxpool(_out_conv1)
                _out_conv2 = self.layer1(_x)            # x shape : batch_size, channel #1, h/4, w/4
                _out_conv3 = self.layer2(_out_conv2)    # x shape : batch_size, channel #2, h/8, w/8
                _out_conv4 = self.layer3(_out_conv3)    # x shape : batch_size, channel #3, h/16, w/16
                _out_conv5 = self.layer4(_out_conv4)    # x shape : batch_size, channel #4, h/32, w/32

                return [_out_conv1, _out_conv2, _out_conv3, _out_conv4, _out_conv5]

            def _Average_pooling(self, ouput: Tensor):
                return Tensor_Process._Flatten(self.avgpool(ouput))

        class VGG(Backbone_Base):
            def __init__(self, model_type: int, is_pretrained: bool, is_trainable: bool):
                super().__init__()
                if model_type == 11:
                    _line = vgg11(pretrained=is_pretrained)
                if model_type == 13:
                    _line = vgg13(pretrained=is_pretrained)
                elif model_type == 16:
                    _line = vgg16(pretrained=is_pretrained)
                else:
                    _line = vgg19(pretrained=is_pretrained)

                self._conv = _line.features
                self._avgpool = _line.avgpool

                for _parameter in self._conv.parameters():
                    _parameter.requires_grad = is_trainable

            def forward(self, x: Tensor):
                return self._conv(x)  # x shape : batch_size, 512, 7, 7

            def _Average_pooling(self, ouput: Tensor):
                return self._avgpool(ouput)

        class FCN(Backbone_Base):
            def __init__(self, model_type: int, is_pretrained: bool, is_trainable: bool):
                super(Model_Componant.Backbone.FCN, self).__init__()
                if model_type == 101:
                    self._line = fcn_resnet101(pretrained=is_pretrained)
                else:
                    self._line = fcn_resnet50(pretrained=is_pretrained)

                for _module in self._line.parameters():
                    _module.requires_grad = is_trainable

            def forward(self, x):
                return self._line(x)

        @staticmethod
        def _Build(model_info: Supported, is_pretrained: bool, is_trainable: bool) -> Backbone_Base:
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
                        epoch = epoch - self._This_term
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
