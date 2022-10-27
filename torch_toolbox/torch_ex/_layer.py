from dataclasses import dataclass
from enum import Enum
from typing import List, Dict, Any, Union
from math import sqrt, log

from python_ex._base import Utils

# from torch.nn import ReLU, Softmax, parameter, ModuleList
from torch import Tensor
from torch import zeros, mean, matmul, sin, cos, exp, arange

# layer utils
from torch.nn import Module, init, functional, ModuleList, Sequential, parameter
from torchsummary import summary as ModelSummary
from einops import rearrange

# modules
from torch.nn import Linear, Conv2d, BatchNorm1d, BatchNorm2d, SyncBatchNorm, LayerNorm
from torch.nn import Dropout, Upsample
import torchvision.models as models  # for backbone

# active
from torch.nn import ReLU, LeakyReLU, Tanh, Sigmoid, GELU

# loss
from torch.nn import MSELoss, CrossEntropyLoss

if __package__ == "":
    # if this file in local project
    from _torch_base import Torch_Utils

else:
    # if this file in package folder
    from ._torch_base import Torch_Utils


# -- DEFINE CONSTNAT -- #
class Suported_Active(Enum):
    ReLU = 0
    LeakyReLU = 1
    Tanh = 2
    Sigmoid = 3
    GELU = 4


class Suported_Norm(Enum):
    BatchNorm = 0
    SyncBatchNorm = 1
    LayerNorm = 2


class Suported_Backbone(Enum):
    ResNet = 0
    VGG = 1
    FCN = 2


# -- DEFINE CONFIG -- #
class Layer_Config():
    @dataclass
    class Linear(Utils.Config):
        _In_features: int
        _Out_features: int
        _Bias: bool = True

        def _convert_to_dict(self) -> Dict[str, Any]:
            return super()._convert_to_dict()

        def _restore_from_dict(self, data: Dict[str, Any]):
            return super()._restore_from_dict(data)

        def _make_parameter(self):
            return {
                "in_features": self._In_features,
                "out_features": self._Out_features,
                "bias": self._Bias}

    @dataclass
    class Conv2D(Utils.Config):
        _In_channels: int
        _Out_channels: int
        _Kernel_size: Union[int, List[int]]
        _Stride: Union[int, List[int]] = 1
        _Padding: Union[int, List[int]] = 0
        _Dilation: Union[int, List[int]] = 1
        _Groups: int = 1
        _Bias: bool = True

        def _convert_to_dict(self) -> Dict[str, Any]:
            return super()._convert_to_dict()

        def _restore_from_dict(self, data: Dict[str, Any]):
            return super()._restore_from_dict(data)

        def _make_parameter(self):
            return {
                "in_channels": self._In_channels,
                "out_channels": self._Out_channels,
                "kernel_size": self._Kernel_size,
                "stride": self._Stride,
                "padding": self._Padding,
                "dilation": self._Dilation,
                "groups": self._Groups,
                "bias": self._Bias}

    @dataclass
    class Norm(Utils.Config):
        _Type: Suported_Norm
        _Out_features: int

        _Eps: float = 1e-5
        _Momentum: float = 0.1
        _Affine: bool = True
        _Track_running_stats: bool = True

        def _convert_to_dict(self) -> Dict[str, Any]:
            return super()._convert_to_dict()

        def _restore_from_dict(self, data: Dict[str, Any]):
            return super()._restore_from_dict(data)

        def _make_parameter(self) -> Module:
            # normalization setting
            if self._Type == Suported_Norm.BatchNorm:
                return {
                    "num_features": self._Out_features,
                    "eps": self._Eps,
                    "momentum": self._Momentum,
                    "affine": self._Affine,
                    "track_running_stats": self._Track_running_stats}
            if self._Type == Suported_Norm.LayerNorm:
                return {
                    "nomalized_shape": self._Out_features,
                    "eps": self._Eps,
                    "elementwise_affine": self._Affine}

    @dataclass
    class Activate(Utils.Config):
        _Type: Suported_Active
        _Inplace: bool = True
        _Negative_slope: float = 0.01

        def _convert_to_dict(self) -> Dict[str, Any]:
            return super()._convert_to_dict()

        def _restore_from_dict(self, data: Dict[str, Any]):
            return super()._restore_from_dict(data)

        def _make_parameter(self) -> Module:
            # normalization setting
            if self._Type == Suported_Active.ReLU:
                return {"inplace": self._Inplace}
            elif self._Type == Suported_Active.LeakyReLU:
                return {"inplace": self._Inplace, "negative_slope": self._Negative_slope}
            else:  # Tanh, Sigmoid, GELU
                return {}


class Module_Config():
    @dataclass
    class Linear(Utils.Config):
        _Linear_config: Layer_Config.Linear
        _Norm_type: Suported_Norm = None
        _Activate_type: Suported_Active = None

        def _convert_to_dict(self) -> Dict[str, Any]:
            return super()._convert_to_dict()

        def _restore_from_dict(self, data: Dict[str, Any]):
            return super()._restore_from_dict(data)

        def _make_norm_config(self):
            return Layer_Config.Norm(self._Norm_type, self._Linear_config._Out_features)

        def _make_active_config(self):
            return Layer_Config.Activate(self._Activate_type)

    @dataclass
    class Conv2D(Utils.Config):
        _Conv2D_config: Layer_Config.Conv2D
        _Norm_type: Suported_Norm = None
        _Activate_type: Suported_Active = None

        def _convert_to_dict(self) -> Dict[str, Any]:
            return super()._convert_to_dict()

        def _restore_from_dict(self, data: Dict[str, Any]):
            return super()._restore_from_dict(data)

        def _make_norm_config(self):
            return Layer_Config.Norm(self._Norm_type, self._Conv2D_config._Out_channels)

        def _make_active_config(self):
            return Layer_Config.Activate(self._Activate_type)

    @dataclass
    class PUP(Utils.Config):
        _In_channels: int
        _Out_channels: int
        _Norm_type: Suported_Norm = Suported_Norm.BatchNorm
        _Activate_type: Suported_Active = Suported_Active.ReLU

        _Scale_factor: int = 2
        _Scale_mode: str = "bilinear"

        def _convert_to_dict(self) -> Dict[str, Any]:
            return super()._convert_to_dict()

        def _restore_from_dict(self, data: Dict[str, Any]):
            return super()._restore_from_dict(data)

        def _make_conv_config(self):
            __front = Layer_Config.Conv2D(self._In_channels, self._In_channels, 1)
            __backend = Layer_Config.Conv2D(self._In_channels, self._Out_channels, 1)

            return __front, __backend

        def _make_norm_config(self):
            return Layer_Config.Norm(self._Norm_type, self._In_channels)

        def _make_active_config(self):
            return Layer_Config.Activate(self._Activate_type)

    @dataclass
    class Attention(Utils.Config):
        _Attention_type: str
        _Input_dim: int
        _Output_dim: int
        _Head_count: int

        def __init__(self, _Attention_type: str, _Input_dim: int, _Output_dim: int, _Head_count: int):
            self._Attention_type = _Attention_type
            self._Input_dim = _Input_dim
            self._Head_count = _Head_count
            self._Output_dim = _Output_dim + (_Output_dim % self._Head_count) if (_Output_dim % _Head_count) else _Output_dim

        def _convert_to_dict(self) -> Dict[str, Any]:
            return super()._convert_to_dict()

        def _restore_from_dict(self, data: Dict[str, Any]):
            return super()._restore_from_dict(data)

        def _get_head_dim(self):
            return self._Output_dim // self._Head_count

        def _get_linear_config(self):
            __q_config = Module_Config.Linear(Layer_Config.Linear(self._Input_dim, self._Output_dim))
            __k_config = Module_Config.Linear(Layer_Config.Linear(self._Input_dim, self._Output_dim))
            __v_config = Module_Config.Linear(Layer_Config.Linear(self._Input_dim, self._Output_dim))
            __o_config = Module_Config.Linear(Layer_Config.Linear(self._Output_dim, self._Output_dim))

            return __q_config, __k_config, __v_config, __o_config

    @dataclass
    class Transformer(Attention):
        _Hidden_rate: int = 2
        _Dropout_rate: float = 0.5

        def _convert_to_dict(self) -> Dict[str, Any]:
            return super()._convert_to_dict()

        def _restore_from_dict(self, data: Dict[str, Any]):
            return super()._restore_from_dict(data)

        def _get_linear_config(self):
            __linear_01 = Module_Config.Linear(
                Layer_Config.Linear(self._Output_dim, self._Output_dim * self._Hidden_rate), _Activate_tpye=Suported_Active.GELU)
            __linear_02 = Module_Config.Linear(
                Layer_Config.Linear(self._Output_dim * self._Hidden_rate, self._Output_dim))

            return __linear_01, __linear_02

    @dataclass
    class Backbone(Utils.Config):
        _Name: Suported_Backbone
        _Type: int

        _Is_pretrained: bool = True
        _Is_trainable: bool = False

        def _convert_to_dict(self) -> Dict[str, Any]:
            return super()._convert_to_dict()

        def _restore_from_dict(self, data: Dict[str, Any]):
            return super()._restore_from_dict(data)


# -- Mation Function -- #
class Layer():
    @ staticmethod
    def _make_module_list(list: List[Module]) -> ModuleList:
        return ModuleList(list)

    @ staticmethod
    def _make_sequential(list: List[Module]) -> Sequential:
        return Sequential(*list)

    @ staticmethod
    def _make_weight(size, value_range: List[float]):
        return parameter.Parameter(Torch_Utils.Tensor._make_tensor(size, value=value_range, dtype="float32"))

    @staticmethod
    def _make_norm_layer(config: Layer_Config.Norm, dimension: int) -> Module:
        if config._Type is None:
            return None
        elif config._Type == Suported_Norm.BatchNorm:
            if dimension == 1:
                return BatchNorm1d(**config._make_parameter())
            elif dimension == 2:
                return BatchNorm2d(**config._make_parameter())
        elif config._Type == Suported_Norm.SyncBatchNorm:
            return SyncBatchNorm(**config._make_parameter())
        elif config._Type == Suported_Norm.LayerNorm:
            return LayerNorm(**config._make_parameter())

    @staticmethod
    def _make_activate_layer(active_opt: Layer_Config.Activate) -> Module:
        # activation setting
        if active_opt is None:
            return None
        elif active_opt._Type == Suported_Active.ReLU:
            return ReLU(**active_opt._make_parameter())
        elif active_opt._Type == Suported_Active.LeakyReLU:
            return LeakyReLU(**active_opt._make_parameter())
        elif active_opt._Type == Suported_Active.Tanh:
            return Tanh(**active_opt._make_parameter())
        elif active_opt._Type == Suported_Active.Sigmoid:
            return Sigmoid(**active_opt._make_parameter())
        elif active_opt._Type == Suported_Active.GELU:
            return GELU(**active_opt._make_parameter())


class Custom_Module():
    class Model(Module):
        def __init__(self, model_name: str):
            super(Custom_Module.Model, self).__init__()
            self.model_name = model_name

        # Freeze function
        def _sumarry(self, input_shape):
            ModelSummary(self, input_shape)

        # Un-Freeze function
        def forward(self, x):
            return x

    class Linear(Module):
        def __init__(self, config: Module_Config.Linear):
            super(Custom_Module.Linear, self).__init__()

            self._Linear = Linear(**config._Linear_config._make_parameter())
            self._Norm = Layer._make_norm_layer(config._make_norm_config(), 1)
            self._Activate = Layer._make_activate_layer(config._make_active_config())

        def forward(self, x: Tensor) -> Tensor:
            __x = self._Linear(x)
            __x = self._Norm(__x) if self._Norm is not None else __x
            __x = self._Activate(__x) if self._Activate is not None else __x
            return __x

    class Conv2D(Module):
        def __init__(self, config: Module_Config.Conv2D):
            super(Custom_Module.Conv2D, self).__init__()

            self._Conv2D = Conv2d(**config._Conv2D_config._make_parameter())
            self._Norm = Layer._make_norm_layer(config._make_norm_config(), 2)
            self._Activate = Layer._make_activate_layer(config._make_active_config())

        def forward(self, x: Tensor) -> Tensor:
            __x = self._Conv2D(x)
            __x = self._Norm(__x) if self._Norm is not None else __x
            __x = self._Activate(__x) if self._Activate is not None else __x
            return __x

    class PUP(Module):
        def __init__(self, config: Module_Config.PUP):
            super().__init__()

            __front_conv, __backend_conv = config._make_conv_config()

            self._Front = Conv2d(**__front_conv._make_parameter())
            self._Norm = Layer._make_norm_layer(config._make_norm_config(), 2)
            self._Activate = Layer._make_activate_layer(config._make_active_config())
            self._Back = Conv2d(**__backend_conv._make_parameter())
            self._Samppling = Upsample(scale_factor=config._Scale_factor, mode=config._Scale_mode)

        def forward(self, x: Tensor) -> Tensor:
            __x = self._Front(x)
            __x = self._Norm(__x) if self._Norm is not None else __x
            __x = self._Activate(__x) if self._Activate is not None else __x
            __x = self._Back(__x)
            __x = self._Samppling(__x)
            return __x

    class Position_Encoding():
        class Trigonometric(Module):
            def __init__(self, num_of_data: int, max_token_size: int = 1000):
                super().__init__()
                __pe = zeros(max_token_size, num_of_data)
                __position = arange(0, max_token_size, dtype=float).unsqueeze(1)
                __div_term = exp(arange(0, num_of_data, 2).float() * (-log(10000.0) / num_of_data))
                __pe[:, 0::2] = sin(__position * __div_term)
                __pe[:, 1::2] = cos(__position * __div_term)
                __pe = __pe.unsqueeze(0)

                self.register_buffer("_Position_value", __pe, persistent=False)

            def forward(self, x: Tensor):
                return x + self._Position_value[:, : x.size(1)]

        class Gaussian(Module):
            ...

    class Attention():
        class Base(Module):
            def __init__(self, config: Module_Config.Attention) -> None:
                super().__init__()
                self._Option = config
                self._Head_dim = config._get_head_dim()

                __maker_config = config._get_linear_config()

                self.q_maker = Custom_Module.Linear(__maker_config[0])
                init.xavier_uniform_(self.q_maker._Linear.weight)
                self.q_maker._Linear.bias.data.fill_(0)

                self.k_maker = Custom_Module.Linear(__maker_config[0])
                init.xavier_uniform_(self.k_maker._Linear.weight)
                self.k_maker._Linear.bias.data.fill_(0)

                self.v_maker = Custom_Module.Linear(__maker_config[0])
                init.xavier_uniform_(self.v_maker._Linear.weight)
                self.v_maker._Linear.bias.data.fill_(0)

                self.o_maker = Custom_Module.Linear(__maker_config[0])
                init.xavier_uniform_(self.o_maker._Linear.weight)
                self.o_maker._Linear.bias.data.fill_(0)

            def _dot_product(self, Q, K, V, mask=None):  # dot_product
                __logits = matmul(Q, rearrange(K, 'batch head_num seq head_dim -> batch head_num head_dim seq'))  # -> batch head_num seq seq
                __logits /= sqrt(self._Head_dim)

                if mask is not None:
                    __logits = __logits.masked_fill(mask == 0, -9e15)

                __attention = functional.softmax(__logits, dim=-1)

                return matmul(__attention, V), __attention

        class Dot_Attention(Base):
            def __init__(self, config: Module_Config.Attention) -> None:
                super().__init__(config)

            def forward(self, Q_source: Tensor, K_source: Tensor, V_source: Tensor, mask: Tensor = None, return_attention_map: bool = False) -> Tensor:
                __q = self.q_maker(Q_source)
                __q = rearrange(__q, 'batch seq (head_dim head_num) -> batch head_num seq head_dim', head_dim=self._Head_dim, head_num=self._Option._Head_count)

                __k = self.k_maker(K_source)
                __k = rearrange(__k, 'batch seq (head_dim head_num) -> batch head_num seq head_dim', head_dim=self._Head_dim, head_num=self._Option._Head_count)

                __v = self.v_maker(V_source)
                __v = rearrange(__v, 'batch seq (head_dim head_num) -> batch head_num seq head_dim', head_dim=self._Head_dim, head_num=self._Option._Head_count)

                __value, _attention = self._dot_product(__q, __k, __v, mask)  # value -> batch head_num seq head_dim

                __value = rearrange(__value, 'batch head_num seq head_dim -> batch seq (head_dim head_num)')
                __outpot = self.o_maker(__value)
                return (__outpot, _attention) if return_attention_map else __outpot

    class Transformer():
        class Base(Module):
            def __init__(self, config: Module_Config.Transformer):
                super().__init__()
                self._Layer_norm_01 = LayerNorm(config._Output_dim)
                self._Attention = Custom_Module.Attention.Dot_Attention(config)
                self._Layer_norm_02 = LayerNorm(config._Output_dim)

                [__linear_config_01, __linear_config_02] = config._get_linear_config()

                __linear = [
                    Custom_Module.Linear(__linear_config_01),
                    Dropout(config._Dropout_rate),
                    Custom_Module.Linear(__linear_config_02)]

                self.linear_block = Layer._make_sequential(__linear)

            def forward(self, x):
                __x = self._Layer_norm_01(x + self._Attention(x))
                __x = self._Layer_norm_02(__x + self.linear_block(__x))

                return __x

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

    class Backbone():
        class ResNet(Module):
            def __init__(self, config: Module_Config.Backbone):
                """
                args:
                """
                super(Custom_Module.Backbone.ResNet, self).__init__()
                if config._Type == 101:
                    self._Model = models.resnet101(pretrained=config._Is_pretrained)
                    self._Output_channel = [64, 256, 512, 1024, 2048]
                else:
                    self._Model = models.resnet50(pretrained=config._Is_pretrained)
                    self._Output_channel = [64, 256, 512, 1024, 2048]

                # features parameters doesn't train
                for __parameters in self._Model.parameters():
                    __parameters.requires_grad = config._Is_trainable

                # delete classfication module
                self._Model.fc = None

            def forward(self, x):
                __x = self._Model.conv1(x)
                __x = self._Model.bn1(__x)
                __out_conv1 = self._Model.relu(__x)              # x shape : batch_size, 2048, h/2, w/2
                __x = self._Model.maxpool(__out_conv1)
                __out_conv2 = self._Model.layer1(__x)            # x shape : batch_size, 2048, h/4, w/4
                __out_conv3 = self._Model.layer2(__out_conv2)    # x shape : batch_size, 2048, h/8, w/8
                __out_conv4 = self._Model.layer3(__out_conv3)    # x shape : batch_size, 2048, h/16, w/16
                __out_conv5 = self._Model.layer4(__out_conv4)    # x shape : batch_size, 2048, h/32, w/32

                return [__out_conv1, __out_conv2, __out_conv3, __out_conv4, __out_conv5]

            def _average_pooling(self, ouput: Tensor):
                return self._Model.avgpool(ouput)

            def _sumarry(self, input_shape):
                ModelSummary(self, input_shape)

            def _get_out_shape(self, input_size: List[int]):
                __h, __w = input_size[:2]
                return [
                    [__c, __h / 2 ** (__d + 1), __w / 2 ** (__d + 1)] for [__d, __c] in enumerate(self._Output_channel)]

        class VGG(Module):
            def __init__(self, config: Module_Config.Backbone):
                super(Custom_Module.Backbone.VGG, self).__init__()
                if config._Type == 11:
                    _line = models.vgg11(pretrained=config._Is_pretrained)
                if config._Type == 13:
                    _line = models.vgg13(pretrained=config._Is_pretrained)
                elif config._Type == 16:
                    _line = models.vgg16(pretrained=config._Is_pretrained)
                else:
                    _line = models.vgg19(pretrained=config._Is_pretrained)

                self._conv = _line.features
                self._avgpool = _line.avgpool

                for _parameter in self._conv.parameters():
                    _parameter.requires_grad = config._Is_trainable

            def forward(self, x: Tensor):
                return self._conv(x)  # x shape : batch_size, 512, 7, 7

            def _average_pooling(self, ouput: Tensor):
                return self._avgpool(ouput)

            def _sumarry(self, input_shape):
                ModelSummary(self, input_shape)

        class FCN(Module):
            def __init__(self, config: Module_Config.Backbone) -> None:
                super(Custom_Module.Backbone.FCN, self).__init__()
                if config._Type == 101:
                    self._line = models.segmentation.fcn_resnet101(pretrained=config._Is_pretrained)
                else:
                    self._line = models.segmentation.fcn_resnet50(pretrained=config._Is_pretrained)

                for _module in self._line.parameters():
                    _module.requires_grad = config._Is_trainable

            def forward(self, x):
                return self._line(x)

            def _sumarry(self, input_shape):
                ModelSummary(self, input_shape)

        @staticmethod
        def _build(config: Module_Config.Backbone):
            if config._Name == Suported_Backbone.ResNet:
                return Custom_Module.Backbone.ResNet(config)
            elif config._Name == Suported_Backbone.VGG:
                return Custom_Module.Backbone.VGG(config)
            elif config._Name == Suported_Backbone.FCN:
                return Custom_Module.Backbone.FCN(config)
            else:
                return None


class Loss_Function():
    @staticmethod
    def _mse_loss(output, target) -> Tensor:
        """
        Args:
            output: [batch, c, h, w]
            target: [batch, c, h, w]
        Return:
            loss
        """
        return MSELoss()(output, target)

    @staticmethod
    def _cross_entropy_loss(output, target, ignore_index=-100) -> Tensor:
        """
        Args:
            output: [batch, class_num, h, w]
            target: [batch, h, w]
        Return:
            loss value
        """
        return CrossEntropyLoss(ignore_index=ignore_index)(output, target)

    @staticmethod
    def _mean_loss(output, target) -> Tensor:
        return mean(output * target)
