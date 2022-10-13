from dataclasses import dataclass, asdict
from typing import List
from math import sqrt, log

import torch

# from torch.nn import ReLU, Softmax, parameter, ModuleList
from torch.nn import Module, init, functional, ModuleList, Sequential
from torch.nn import parameter
from torch.nn import Linear, Conv2d, ConvTranspose2d, BatchNorm1d, BatchNorm2d  # , LayerNorm
from torch.nn import ReLU, LeakyReLU, Tanh, Sigmoid  # , GELU
from torch.nn import MSELoss, CrossEntropyLoss

from torch.optim import Optimizer, Adam

import torchvision.models as models
from torchsummary import summary as ModelSummary

from einops import rearrange

from python_ex import _error as _e

_error = _e.Custom_error(
    module_name="torch_custom_utils_v 1.x",
    file_name="_model_part.py")


class opt():
    @dataclass
    class backbone():
        type: int

        is_pretrained: bool = True
        is_trainable: bool = False

        use_flat: bool = False
        use_avg_pooling: bool = False

    @dataclass
    class fc():
        in_features: int
        out_features: int
        bias: bool = True

    @dataclass
    class conv2d():
        in_channels: int
        out_channels: int
        kernel_size: int
        stride: int = 1
        padding: int = 0
        dilation: int = 1
        groups: int = 1
        bias: bool = True

    @dataclass
    class attention():
        attention_type: str
        input_dim: int
        output_dim: int
        head_count: int

        def __init__(self, attention_type: str, input_dim: int, output_dim: int, num_head: int):
            self.attention_type = attention_type
            self.input_dim = input_dim
            self.head_count = num_head
            self.output_dim = output_dim + (output_dim % self.head_count) if (output_dim % num_head) else output_dim

        def get_head_dim(self):
            return self.output_dim // self.head_count

    @dataclass
    class postion_encoding():
        encoding_type: str
        max_len: int

        def __init__(self, attention_type: str, input_dim: int, output_dim: int, num_head: int):
            ...

    @dataclass
    class norm2d():
        norm_type: str

        num_features: int
        eps: float = 1e-5
        momentum: float = 0.1
        affine: bool = True
        track_running_stats: bool = True

        def to_parameters(self):
            if self.norm_type == "BatchNorm":
                return {"num_features": self.num_features, "eps": self.eps, "momentum": self.momentum, "affine": self.affine, "track_running_stats": self.track_running_stats}
            else:
                return {}

    @dataclass
    class active_function():
        active_type: str

        # ReLU - basement
        inplace: bool = True

        # LeakyReLU
        negative_slope: float = 0.01

        # Tanh, Sigmoid
        # empty

        def to_parameters(self):
            if self.active_type == "ReLU":
                return {"inplace": self.inplace}
            elif self.active_type == "LeakyReLU":
                return {"inplace": self.inplace, "negative_slope": self.negative_slope}
            else:  # Tanh, Sigmoid
                return {}

    @dataclass
    class optim():
        optimize_type: str

        def make(self, model: Module, Learning_rate: float) -> Optimizer:
            if self.optimize_type == "Adam":
                return Adam(model.parameters(), Learning_rate)


class loss():
    @staticmethod
    def mse_loss(output, target) -> torch.Tensor:
        """
        Args:
            output: [batch, c, h, w]
            target: [batch, c, h, w]
        Return:
            loss
        """
        return MSELoss()(output, target)

    @staticmethod
    def cross_entropy_loss(output, target, ignore_index=-100) -> torch.Tensor:
        """
        Args:
            output: [batch, class_num, h, w]
            target: [batch, h, w]
        Return:
            loss
        """
        return CrossEntropyLoss(ignore_index=ignore_index)(output, target)

    @staticmethod
    def mean_loss(output, target) -> torch.Tensor:
        return torch.mean(output * target)


class module():
    class custom_module(Module):
        def __init__(self, model_name):
            super(module.custom_module, self).__init__()
            self.model_name = model_name

        def sumarry(self, input_shape):
            ModelSummary(self, input_shape)

        def _save_to(self, save_dir, epoch, optim: Optimizer = None):
            save_dic = {"epoch": epoch,
                        "model_state_dict": self.state_dict(),
                        "optimizer_state_dict": optim.state_dict() if optim is not None else None}

            torch.save(save_dic, save_dir + self.model_name + ".h5")

        def _load_from(self, model_file):
            checkpoint = torch.load(model_file)
            self.load_state_dict(checkpoint["model_state_dict"])

            return checkpoint  # if restore train sasseion

        def forward(self, x):
            return x

        def set_layers(self, **parameters):
            ...

    @staticmethod
    def make_norm_layer(out_features: int, dimension: int, norm_opt: opt.norm2d) -> Module:
        # normalization setting
        if norm_opt is None:
            return None
        elif norm_opt.norm_type == "BatchNorm":
            if dimension == 1:
                return BatchNorm1d(out_features, **norm_opt.to_parameters())

            elif dimension == 2:
                return BatchNorm2d(out_features, **norm_opt.to_parameters())

    @staticmethod
    def make_activate_layer(active_opt: opt.active_function) -> Module:
        # activation setting
        if active_opt is None:
            return None
        elif active_opt.active_type == "ReLU":
            return ReLU(**active_opt.to_parameters())
        elif active_opt.active_type == "LeakyReLU":
            return LeakyReLU(**active_opt.to_parameters())
        elif active_opt.active_type == "Tanh":
            return Tanh(**active_opt.to_parameters())
        elif active_opt.active_type == "Sigmoid":
            return Sigmoid(**active_opt.to_parameters())

    @ staticmethod
    def make_module_list(list: List[Module]) -> ModuleList:
        return ModuleList(list)

    @ staticmethod
    def make_sequential(list: List[Module]) -> Sequential:
        return Sequential(*list)

    @ staticmethod
    def make_weight(size):

        return parameter()

    class _Fc(Module):
        def __init__(self, layer_opt: opt.fc, norm_opt: opt.norm2d = None, active_opt: opt.active_function = None):
            super(module._Fc, self).__init__()

            self.liner = Linear(**asdict(layer_opt))
            self.norm = module.make_norm_layer(layer_opt.out_features, 1, norm_opt)
            self.activate = module.make_activate_layer(active_opt)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.liner(x)
            x = self.norm(x) if self.norm is not None else x
            x = self.activate(x) if self.activate is not None else x
            return x

    class _Conv2D(Module):
        def __init__(self, layer_opt: opt.conv2d, norm_opt: opt.norm2d = None, active_opt: opt.active_function = None):
            super(module._Conv2D, self).__init__()

            self.liner = Conv2d(**asdict(layer_opt))
            self.norm = module.make_norm_layer(layer_opt.out_channels, 2, norm_opt)
            self.activate = module.make_activate_layer(active_opt)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.liner(x)
            x = self.norm(x) if self.norm is not None else x
            x = self.activate(x) if self.activate is not None else x
            return x

    class _UpConv2D(Module):
        def __init__(self, layer_opt: opt.conv2d, norm_opt: opt.norm2d = None, active_opt: opt.active_function = None):
            super(module._Conv2D, self).__init__()

            self.liner = ConvTranspose2d(**asdict(layer_opt))
            self.norm = module.make_norm_layer(layer_opt.out_channels, 2, norm_opt)
            self.activate = module.make_activate_layer(active_opt)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.liner(x)
            x = self.norm(x) if self.norm is not None else x
            x = self.activate(x) if self.activate is not None else x
            return x

    class _Attention():
        class __base(Module):
            def __init__(self, layer_opt: opt.attention) -> None:
                super().__init__()

                self.output_dim = layer_opt.output_dim
                self.head_count = layer_opt.head_count
                self.head_dim = layer_opt.get_head_dim()

                self.k_maker = module._Fc(opt.fc(layer_opt.input_dim, self.output_dim))
                init.xavier_uniform_(self.k_maker.liner.weight)
                self.k_maker.liner.bias.data.fill_(0)

                self.v_maker = module._Fc(opt.fc(layer_opt.input_dim, self.output_dim))
                init.xavier_uniform_(self.v_maker.liner.weight)
                self.v_maker.liner.bias.data.fill_(0)

                self.q_maker = module._Fc(opt.fc(layer_opt.input_dim, self.output_dim))
                init.xavier_uniform_(self.q_maker.liner.weight)
                self.q_maker.liner.bias.data.fill_(0)

                self.o_maker = module._Fc(opt.fc(layer_opt.output_dim, self.output_dim))
                init.xavier_uniform_(self.o_maker.liner.weight)
                self.o_maker.liner.bias.data.fill_(0)

            def dot_product(self, Q, K, V, mask=None):  # dot_product
                _logits = torch.matmul(Q, rearrange(K, 'batch head_num seq head_dim -> batch head_num head_dim seq'))  # -> batch head_num seq seq
                _logits /= sqrt(self.head_dim)

                if mask is not None:
                    _logits = _logits.masked_fill(mask == 0, -9e15)

                _attention = functional.softmax(_logits, dim=-1)

                return torch.matmul(_attention, V), _attention

        class _self_dot(__base):
            def __init__(self, layer_opt: opt.attention) -> None:
                super().__init__(layer_opt)

            def forward(self, QKV_source: torch.Tensor, mask: torch.Tensor = None, return_attention_map: bool = False) -> torch.Tensor:
                _q = self.q_maker(QKV_source)
                _q: torch.Tensor = rearrange(_q, 'batch seq (head_dim head_num) -> batch head_num seq head_dim', head_dim=self.head_dim, head_num=self.head_count)

                _k = self.k_maker(QKV_source)
                _k: torch.Tensor = rearrange(_k, 'batch seq (head_dim head_num) -> batch head_num seq head_dim', head_dim=self.head_dim, head_num=self.head_count)

                _v = self.v_maker(QKV_source)
                _v: torch.Tensor = rearrange(_v, 'batch seq (head_dim head_num) -> batch head_num seq head_dim', head_dim=self.head_dim, head_num=self.head_count)

                _value, _attention = self.dot_product(_q, _k, _v, mask)  # value -> batch head_num seq head_dim

                _value: torch.Tensor = rearrange(_value, 'batch head_num seq head_dim -> batch seq (head_dim head_num)')
                _outpot = self.o_maker(_value)
                return _outpot, _attention if return_attention_map else _outpot

        class _cross_dot(__base):
            def __init__(self, layer_opt: opt.attention) -> None:
                super().__init__(layer_opt)

            def forward(self, Q_source: torch.Tensor, KV_source: torch.Tensor, mask: torch.Tensor = None, return_attention_map: bool = False) -> torch.Tensor:
                _q = self.q_maker(Q_source)
                _q: torch.Tensor = rearrange(_q, 'batch seq (head_dim head_num) -> batch head_num seq head_dim', head_dim=self.head_dim, head_num=self.head_count)

                _k = self.k_maker(KV_source)
                _k: torch.Tensor = rearrange(_k, 'batch seq (head_dim head_num) -> batch head_num seq head_dim', head_dim=self.head_dim, head_num=self.head_count)

                _v = self.v_maker(KV_source)
                _v: torch.Tensor = rearrange(_v, 'batch seq (head_dim head_num) -> batch head_num seq head_dim', head_dim=self.head_dim, head_num=self.head_count)

                _value, _attention = self.dot_product(_q, _k, _v, mask)  # value -> batch head_num seq head_dim

                _value: torch.Tensor = rearrange(_value, 'batch head_num seq head_dim -> batch seq (head_dim head_num)')
                _outpot = self.o_maker(_value)
                return _outpot, _attention if return_attention_map else _outpot

    class _Position_encoding():
        class trigonometric(Module):
            def __init__(self, num_of_data, max_token_size=5000):
                super().__init__()
                pe = torch.zeros(max_token_size, num_of_data)
                position = torch.arange(0, max_token_size, dtype=torch.float).unsqueeze(1)
                div_term = torch.exp(torch.arange(0, num_of_data, 2).float() * (-log(10000.0) / num_of_data))
                pe[:, 0::2] = torch.sin(position * div_term)
                pe[:, 1::2] = torch.cos(position * div_term)
                pe = pe.unsqueeze(0)

                self.register_buffer("pe", pe, persistent=False)

            def forward(self, x):
                x = x + self.pe[:, : x.size(1)]
                return x

        class gaussian(Module):
            def __init__(self, num_of_data, max_token_size=5000):
                super().__init__()
                # in later fix it
                pe = torch.zeros(max_token_size, num_of_data)
                position = torch.arange(0, max_token_size, dtype=torch.float).unsqueeze(1)
                div_term = torch.exp(torch.arange(0, num_of_data, 2).float() * (-log(10000.0) / num_of_data))
                pe[:, 0::2] = torch.sin(position * div_term)
                pe[:, 1::2] = torch.cos(position * div_term)
                pe = pe.unsqueeze(0)

                self.register_buffer("pe", pe, persistent=False)

            def forward(self, x):
                x = x + self.pe[:, : x.size(1)]
                return x


class backbone():
    class resnet(Module):
        def __init__(self, option: opt.backbone):
            """
            args:
                type
                trained
                last_layer
            """
            super(backbone.resnet, self).__init__()
            self.option = option
            if self.option.type == 50:
                self._line = models.resnet50(pretrained=self.option.is_pretrained)
            elif self.option.type == 101:
                self._line = models.resnet101(pretrained=self.option.is_pretrained)
            else:
                _error.variable(
                    "backbone.resnet",
                    "Have some problem in parameter 'type'. use default value 50")
                self._line = models.resnet50(pretrained=not self.option.is_pretrained)

            # features parameters doesn't train
            for _parameters in self._line.parameters():
                _parameters.requires_grad = self.option.is_trainable

            # delete classfication module
            self._line.fc = None

        def forward(self, x):
            _x = self._line.conv1(x)
            _x = self._line.bn1(_x)
            out_conv1 = self._line.relu(_x)             # x shape : batch_size, 2048, h/2, w/2

            _x = self._line.maxpool(out_conv1)
            out_conv2 = self._line.layer1(_x)           # x shape : batch_size, 2048, h/4, w/4
            out_conv3 = self._line.layer2(out_conv2)    # x shape : batch_size, 2048, h/8, w/8
            out_conv4 = self._line.layer3(out_conv3)    # x shape : batch_size, 2048, h/16, w/16
            out_conv5 = self._line.layer4(out_conv4)    # x shape : batch_size, 2048, h/32, w/32

            if self.option.use_avg_pooling:
                _pooling: torch.Tensor = self._line.avgpool(out_conv5)  # x shape : batch_size, 2048, 1, 1
                return _pooling.view(_pooling.size(0), -1) if self.option.use_flat else _pooling

            else:
                return [out_conv1, out_conv2, out_conv3, out_conv4, out_conv5]

        def sumarry(self, input_shape):
            ModelSummary(self, input_shape)

    class vgg(Module):
        def __init__(self, option: opt.backbone):
            super(backbone.vgg, self).__init__()
            self.option = option
            if self.option.type == 11:
                _line = models.vgg11(pretrained=not self.option.is_pretrained)
            if self.option.type == 13:
                _line = models.vgg13(pretrained=not self.option.is_pretrained)
            elif self.option.type == 16:
                _line = models.vgg16(pretrained=not self.option.is_pretrained)
            elif self.option.type == 19:
                _line = models.vgg19(pretrained=not self.option.is_pretrained)
            else:
                _error.variable(
                    "backbone.vgg",
                    "Have some problem in parameter 'type'. use default value 19")
                self.option.type = 19
                _line = models.vgg19(pretrained=not self.option.is_pretrained)

            self._conv = _line.features
            self._avgpool = _line.avgpool

            for _parameter in self._conv.parameters():
                _parameter.requires_grad = self.option.is_trainable

        def forward(self, x):
            x = self._conv(x)  # x shape : batch_size, 512, 7, 7

            if self.option.use_avg_pooling:
                x = self._avgpool(x)  # x shape : batch_size, 512, 1, 1
            return x.view(x.size(0), -1) if self.opt.use_flat else x

        def sumarry(self, input_shape):
            ModelSummary(self, input_shape)

    class fcn(Module):
        def __init__(self, option: opt.backbone) -> None:
            super(backbone.fcn, self).__init__()
            self.option = option
            if self.option.type == 50:
                self._line = models.segmentation.fcn_resnet50(pretrained=self.option.is_pretrained)
            elif self.option.type == 101:
                self._line = models.segmentation.fcn_resnet101(pretrained=self.option.is_pretrained)
            else:
                _error.variable(
                    "backbone.resnet",
                    "Have some problem in parameter 'type'. use default value 50")
                self._line = models.segmentation.fcn_resnet50(pretrained=not self.option.is_pretrained)

            for _module in self._line.parameters():
                _module.requires_grad = self.option.is_trainable

        def forward(self, x):
            return self._line(x)


# class transformer():

#     # in later fix it
#     @staticmethod
#     class _attention(Module):
#         def __init__(
#                 self,
#                 input_shape: int,
#                 hidden_channel: list,
#                 k_size: int,
#                 is_self: bool = False):
#             super(transformer._attention, self).__init__()
#             self.is_self = is_self
#             if is_self:
#                 input_ch = input_shape[-1]
#                 data_size = input_shape[:2]
#                 Q_input, K_input, V_input = [input_ch, input_ch, input_ch]
#                 QKV_output = hidden_channel
#             else:
#                 E_ch, input_ch = input_shape[0][-1], input_shape[1][-1]
#                 data_size = input_shape[0][:2]
#                 Q_input, K_input, V_input = [input_ch, E_ch, E_ch]
#                 QKV_output = hidden_channel

#             Q_option = {
#                 "in_channels": Q_input,
#                 "out_channels": QKV_output,
#                 "kernel_size": 1}
#             K_option = {
#                 "in_channels": K_input,
#                 "out_channels": QKV_output,
#                 "kernel_size": 1}
#             V_option = {
#                 "in_channels": V_input,
#                 "out_channels": QKV_output,
#                 "kernel_size": 1}

#             self.W_q_conv = Conv2d(**Q_option)
#             self.W_k_conv = Conv2d(**K_option)
#             self.W_v_conv = Conv2d(**V_option)

#             self.pad = _torch_util.function.get_conv_pad(
#                 input_size=data_size,
#                 kernel_size=k_size)
#             S_option = {
#                 "in_channels": QKV_output,
#                 "out_channels": QKV_output,
#                 "kernel_size": k_size}
#             self.S_conv = Conv2d(**S_option)

#             self.softmax = Softmax(dim=1)

#         def forward(self, x):
#             if self.is_self:  # x -> [input, mask]
#                 W_q = self.W_q_conv(x)
#                 W_k = self.W_k_conv(x)
#                 W_v = self.W_v_conv(x)
#             else:             # x -> [from encoder, from decoder, mask]
#                 _D_array, _E_array = x
#                 W_q = self.W_q_conv(_D_array)
#                 W_k = self.W_k_conv(_E_array)
#                 W_v = self.W_v_conv(_E_array)

#             _pad_q = F.pad(W_q, self.pad) if self.pad is not None else W_q
#             W_qs = self.S_conv(_pad_q)
#             value_array = self.softmax(W_qs * W_k) * W_v

#             return value_array

#     @staticmethod
#     class _MHA(Module):
#         def __init__(
#                 self,
#                 multi_size: int,
#                 hidden_channel: int,
#                 input_shape: list,
#                 k_size: int,
#                 is_self: bool = False):
#             """
#             Args:
#                 frame        :
#                 multi_ct      :
#                 data_size
#                 input_chs
#                 kernel_size
#                 is_self      : If this module is self attention, set the "True"
#             Returns:
#                 return (np.uint8 array): image data
#             """
#             super(transformer._MHA, self).__init__()
#             self.multi_size = multi_size

#             if is_self:
#                 input_ch = input_shape[-1]
#             else:
#                 input_ch = input_shape[1][-1]
#             self.attentions = ModuleList(
#                 [transformer._attention(
#                     input_shape,
#                     hidden_channel,
#                     k_size,
#                     is_self) for _ct in range(multi_size)])

#             M_option = {
#                 "in_channels": hidden_channel * self.multi_size,
#                 "out_channels": input_ch,
#                 "kernel_size": 1}
#             self.M_conv = Conv2d(**M_option)

#             self.softmax = Softmax(dim=1)

#         def forward(self, x):
#             multi_holder = []
#             for attention in self.attentions:
#                 multi_holder.append(attention(x))

#             merge_torch.Tensor = _torch_util.layer._concat(multi_holder)

#             return self.M_conv(merge_torch.Tensor)

#     @staticmethod
#     class _FFNN(Module):
#         def __init__(self, input_ch, hidden_ch):
#             super(transformer._FFNN, self).__init__()
#             self.layer_1 = Conv2d(input_ch, hidden_ch, kernel_size=1)
#             self.activation = ReLU(inplace=True)
#             self.layer_2 = Conv2d(hidden_ch, input_ch, kernel_size=1)

#         def forward(self, x):
#             x = self.layer_1(x)
#             x = self.activation(x)
#             x = self.layer_2(x)
#             return x

#     @staticmethod
#     def segment_embed(batch, size, channel):
#         shape = (batch, channel, int(size[0]), int(size[1]))
#         return parameter.Parameter(randn(shape, requires_grad=True))

#     @staticmethod
#     class image_encoder(Module):
#         def __init__(self, multi_size, hidden_channel, input_shape, k_size):
#             super(transformer.image_encoder, self).__init__()

#             self.self_multi_head = parts.transformer._MHA(
#                 multi_size=multi_size,
#                 hidden_channel=hidden_channel,
#                 input_shape=input_shape,
#                 k_size=k_size,
#                 is_self=True)
#             self.batch_norm_01 = BatchNorm2d(input_shape[-1])
#             self.FFNN = parts.transformer._FFNN(input_shape[-1], input_shape[-1])
#             self.batch_norm_02 = BatchNorm2d(input_shape[-1])

#         def forward(self, x):
#             after_smh = self.self_multi_head(x) + x
#             after_smh = self.batch_norm_01(after_smh)
#             after_FFNN = self.FFNN(after_smh) + after_smh
#             after_FFNN = self.batch_norm_02(after_FFNN)

#             return after_FFNN

#     @staticmethod
#     class image_decoder(Module):
#         def __init__(self, multi_size, hidden_channel, E_data_shape, D_data_shape, k_size):
#             super(transformer.image_decoder, self).__init__()

#             self.self_multi_head = parts.transformer._MHA(
#                 multi_size=multi_size,
#                 hidden_channel=hidden_channel,
#                 input_shape=D_data_shape,
#                 k_size=k_size,
#                 is_self=True)
#             self.batch_norm_01 = BatchNorm2d(D_data_shape[-1])

#             self.multi_head = parts.transformer._MHA(
#                 multi_size=multi_size,
#                 hidden_channel=hidden_channel,
#                 input_shape=[E_data_shape, D_data_shape],
#                 k_size=k_size,
#                 is_self=False)
#             self.batch_norm_02 = BatchNorm2d(D_data_shape[-1])
#             self.FFNN = parts.transformer._FFNN(D_data_shape[-1], D_data_shape[-1])
#             self.batch_norm_03 = BatchNorm2d(D_data_shape[-1])

#         def forward(self, x):
#             E_data, D_data = x
#             after_smh = self.self_multi_head(D_data) + D_data
#             after_smh = self.batch_norm_01(after_smh)
#             after_mh = self.multi_head([after_smh, E_data]) + after_smh
#             after_mh = self.batch_norm_02(after_mh)
#             after_FFNN = self.FFNN(after_mh) + after_mh
#             after_FFNN = self.batch_norm_02(after_FFNN)

#             return after_FFNN
