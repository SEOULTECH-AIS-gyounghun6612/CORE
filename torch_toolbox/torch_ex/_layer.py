from enum import Enum
from typing import List, Union, Optional, Tuple
from math import sqrt, log

from python_ex._base import NUMBER
from python_ex._numpy import Random_Process

from torch import Tensor
from torch import exp, sin, cos, mean, matmul

# layer utils
from torch.nn import Module, init, functional, ModuleList, Sequential, parameter
from torch.nn.common_types import _size_2_t
from torchsummary import summary as ModelSummary
from einops import rearrange

# modules
from torch.nn import Linear, Conv2d, Upsample
import torchvision.models as models  # for backbone

# loss
from torch.nn import MSELoss, CrossEntropyLoss

if __package__ == "":
    # if this file in local project
    from _torch_base import Tensor_Process, Data_Type

else:
    # if this file in package folder
    from ._torch_base import Tensor_Process, Data_Type


# -- DEFINE CONSTNAT -- #
class Suport_Padding(Enum):
    ZEROS = 'zeros'
    REFLECT = 'reflect'
    REPLICATE = 'replicate'
    CIRCULAR = 'circular'


class Suport_Backbone(Enum):
    ResNet = "ResNet"
    VGG = "VGG"
    FCN = "FCN"


class Suport_Attention(Enum):
    Dot_Attention = "Dot_Attention"


# -- Mation Function -- #
class Custom_Model(Module):
    def __init__(self, model_name: str, **model_parameter):
        super(Custom_Model, self).__init__()
        self._model_name = model_name

    # Freeze function
    def _Sumarry(self, input_shape: List[int]):
        ModelSummary(self, input_shape)

    # Un-Freeze function
    def forward(self, x):
        raise NotImplementedError


class Module_Componant():
    @staticmethod
    def _Make_module_list(module_list: List) -> ModuleList:
        return ModuleList(module_list)

    @staticmethod
    def _Make_sequential(componant_list: List) -> Sequential:
        return Sequential(*componant_list)

    @staticmethod
    def _Make_weight(size: Union[int, List[int]], value: Union[NUMBER, List[NUMBER]], rand_opt: Random_Process = Random_Process.NORM, dtype: Optional[Data_Type] = None):
        return parameter.Parameter(Tensor_Process._Make_tensor(size, value, rand_opt, dtype))

    class Linear(Module):
        def __init__(
            self,
            input: int,
            output: int,
            is_bias: bool = True,
            normization: Optional[Module] = None,
            activate: Optional[Module] = None
        ):
            super(Module_Componant.Linear, self).__init__()

            self._linear = Linear(input, output, is_bias)
            self._norm = normization
            self._active = activate

        def forward(self, x: Tensor) -> Tensor:
            _x = self._linear(x)
            _x = self._norm(_x) if self._norm is not None else _x
            _x = self._active(_x) if self._active is not None else _x
            return _x

    class Conv2D(Module):
        def __init__(
            self,
            input: int,
            output: int,
            kernel: _size_2_t = 1,
            stride: _size_2_t = 1,
            padding: _size_2_t = 0,
            padding_mode: Suport_Padding = Suport_Padding.ZEROS,
            dilation: _size_2_t = 1,
            groups: int = 1,
            is_bias: bool = True,
            normization: Optional[Module] = None,
            activate: Optional[Module] = None
        ):
            super(Module_Componant.Conv2D, self).__init__()

            self._conv2D = Conv2d(input, output, kernel, stride, padding, dilation, groups, is_bias, padding_mode.value)
            self._norm = normization
            self._activate = activate

        def forward(self, x: Tensor) -> Tensor:
            _x = self._conv2D(x)
            _x = self._norm(_x) if self._norm is not None else _x
            _x = self._activate(_x) if self._activate is not None else _x
            return _x

    class Decoder():
        class PUP(Module):
            def __init__(
                self,
                input: int,
                output: int,
                scale_factor: int = 2,
                normization: Optional[Module] = None,
                activate: Optional[Module] = None
            ):
                super().__init__()

                self._conv_module = Module_Componant.Conv2D(input, output, 3, 1, 1, normization=normization, activate=activate)
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
                _div_term = exp(Tensor_Process._Arange(num_of_data, step=2, dtype=Data_Type.FLOAT) * (-log(10000.0) / num_of_data))
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
        class Base(Module):
            def __init__(self, input_dim: int, output_dim: int, head_count: int):
                super().__init__()
                _output_dim = output_dim + (output_dim % head_count) if (output_dim % head_count) else output_dim

                self._head_count = head_count
                self._head_dim = _output_dim // head_count

                self.q_maker = Module_Componant.Linear(input_dim, output_dim)
                init.xavier_uniform_(self.q_maker._linear.weight)
                self.q_maker._linear.bias.data.fill_(0)

                self.k_maker = Module_Componant.Linear(input_dim, output_dim)
                init.xavier_uniform_(self.k_maker._linear.weight)
                self.k_maker._linear.bias.data.fill_(0)

                self.v_maker = Module_Componant.Linear(input_dim, output_dim)
                init.xavier_uniform_(self.v_maker._linear.weight)
                self.v_maker._linear.bias.data.fill_(0)

                self.o_maker = Module_Componant.Linear(output_dim, output_dim)
                init.xavier_uniform_(self.o_maker._linear.weight)
                self.o_maker._linear.bias.data.fill_(0)

            def forward(
                self,
                Q_source: Tensor,
                K_source: Tensor,
                V_source: Tensor,
                mask: Optional[Tensor] = None,
                is_get_map: bool = False
            ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
                raise NotImplementedError

        class Dot_Attention(Base):
            def __init__(self, input_dim: int, output_dim: int, head_count: int):
                super().__init__(input_dim, output_dim, head_count)

            def _Dot_product(self, Q, K, V, mask=None):  # dot_product
                _logits = matmul(Q, rearrange(K, 'batch head_num seq head_dim -> batch head_num head_dim seq'))  # -> batch head_num seq seq
                _logits /= sqrt(self._head_dim)

                if mask is not None:
                    _logits = _logits.masked_fill(mask == 0, -9e15)

                _attention = functional.softmax(_logits, dim=-1)

                return matmul(_attention, V), _attention

            def forward(
                self,
                Q_source: Tensor,
                K_source: Tensor,
                V_source: Tensor,
                mask: Optional[Tensor] = None,
                is_get_map: bool = False
            ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
                _q = self.q_maker(Q_source)
                _q = rearrange(_q, 'batch seq (head_dim head_num) -> batch head_num seq head_dim', head_dim=self._head_dim, head_num=self._head_count)

                _k = self.k_maker(K_source)
                _k = rearrange(_k, 'batch seq (head_dim head_num) -> batch head_num seq head_dim', head_dim=self._head_dim, head_num=self._head_count)

                _v = self.v_maker(V_source)
                _v = rearrange(_v, 'batch seq (head_dim head_num) -> batch head_num seq head_dim', head_dim=self._head_dim, head_num=self._head_count)

                _value, _attention = self._Dot_product(_q, _k, _v, mask)  # value -> batch head_num seq head_dim

                _value = rearrange(_value, 'batch head_num seq head_dim -> batch seq (head_dim head_num)')
                _outpot = self.o_maker(_value)
                return (_outpot, _attention) if is_get_map else _outpot

    # class Transformer():
    #     def __init__(self, input_dim: int, output_dim: int, head_count: int, hidden_rate: int = 2, drop_rate: float = 0.5):
    #         super().__init__()

    #         _output_dim = output_dim + (output_dim % head_count) if (output_dim % head_count) else output_dim
    #         self._head_count = head_count
    #         self._head_dim = _output_dim // head_count

    #         self._front_norm = LayerNorm(_output_dim)
    #         self._attention = Module_Componant.Attention.Dot_Attention(input_dim, _output_dim, head_count)
    #         self._back_norm = LayerNorm(_output_dim)

    #         self.linear_block = Module_Componant._Make_sequential([
    #             Module_Componant.Linear(_output_dim, _output_dim * hidden_rate, activate=GELU()),
    #             Dropout(drop_rate),
    #             Module_Componant.Linear(_output_dim * hidden_rate, _output_dim)
    #         ])

    #     def forward(self, x) -> Tensor:
    #         _x = self._front_norm(x + self._attention(x, x, x))
    #         _x = self._back_norm(_x + self.linear_block(_x))

    #         return _x

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
        class Backbone_Base(Module):
            _output_channel: List[int]

            def _Average_pooling(self, ouput: Tensor):
                raise NotImplementedError

        class ResNet(Backbone_Base):
            def __init__(self, model_type: int, is_pretrained: bool, is_trainable: bool):
                super(Module_Componant.Backbone.ResNet, self).__init__()
                if model_type == 101:
                    _model = models.resnet101(pretrained=is_pretrained)  # [64, 256, 512, 1024, 2048]
                    self._output_channel = [64, 256, 512, 1024, 2048]
                else:
                    _model = models.resnet50(pretrained=is_pretrained)  # [64, 256, 512, 1024, 2048]
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
                return self.avgpool(ouput)

        class VGG(Backbone_Base):
            def __init__(self, model_type: int, is_pretrained: bool, is_trainable: bool):
                super().__init__()
                if model_type == 11:
                    _line = models.vgg11(pretrained=is_pretrained)
                if model_type == 13:
                    _line = models.vgg13(pretrained=is_pretrained)
                elif model_type == 16:
                    _line = models.vgg16(pretrained=is_pretrained)
                else:
                    _line = models.vgg19(pretrained=is_pretrained)

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
                super(Module_Componant.Backbone.FCN, self).__init__()
                if model_type == 101:
                    self._line = models.segmentation.fcn_resnet101(pretrained=is_pretrained)
                else:
                    self._line = models.segmentation.fcn_resnet50(pretrained=is_pretrained)

                for _module in self._line.parameters():
                    _module.requires_grad = is_trainable

            def forward(self, x):
                return self._line(x)

        @staticmethod
        def _Build(name: Suport_Backbone, model_type: int, is_pretrained: bool, is_trainable: bool) -> Backbone_Base:
            return Module_Componant.Backbone.__dict__[name.value](model_type, is_pretrained, is_trainable)


class Loss_Function():
    @staticmethod
    def _Mean_squared_error(output, target) -> Tensor:
        """
        Args:
            output: [batch, c, h, w]
            target: [batch, c, h, w]
        Return:
            loss
        """
        return MSELoss()(output, target)

    @staticmethod
    def _Mean_absolute_error(output, target) -> Tensor:
        """
        Args:
            output: [batch, c, h, w]
            target: [batch, c, h, w]
        Return:
            loss
        """
        return mean(output - target)

    @staticmethod
    def _Cross_Entropy(output, target, ignore_index=-100) -> Tensor:
        """
        Args:
            output: [batch, class_num, h, w]
            target: [batch, h, w]
        Return:
            loss value
        """
        return CrossEntropyLoss(ignore_index=ignore_index)(output, target)

    @staticmethod
    def _Mean(output, target) -> Tensor:
        return mean(output * target)
