from dataclasses import dataclass, asdict

from torch import save, load, mean, Tensor

# from torch.nn import ReLU, Softmax, parameter, ModuleList
from torch.nn import Module
from torch.nn import Linear, Conv2d, BatchNorm1d, BatchNorm2d, LayerNorm
from torch.nn import ReLU, LeakyReLU, Tanh, Sigmoid, GELU
from torch.nn import MSELoss, CrossEntropyLoss

from torch.optim import Optimizer

import torchvision.models as models
from torchsummary import summary as ModelSummary
from python_ex import _error as _e

_error = _e.Custom_error(
    module_name="torch_custom_utils_v 1.x",
    file_name="_model_part.py")


class opt():
    @dataclass
    class backbone_opt():
        use_flat: bool = True
        use_avg_pooling: bool = True

    class layer():
        @dataclass
        class fc_opt():
            in_features: int
            out_features: int
            bias: bool = True

        @dataclass
        class conv_opt():
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
            ...

        @dataclass
        class norm_opt():
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
        class active_opt():
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
                else:
                    return {}


class loss_function():
    @staticmethod
    def mse_loss(output, target) -> Tensor:
        """
        Args:
            output: [batch, c, h, w]
            target: [batch, c, h, w]
        Return:
            loss
        """
        return MSELoss()(output, target)

    @staticmethod
    def cross_entropy_loss(output, target, ignore_index=-100) -> Tensor:
        """
        Args:
            output: [batch, class_num, h, w]
            target: [batch, h, w]
        Return:
            loss
        """
        return CrossEntropyLoss(ignore_index=ignore_index)(output, target)

    @staticmethod
    def mean_loss(output, target) -> Tensor:
        return mean(output * target)


class custom_module(Module):
    def __init__(self, model_name):
        super(custom_module, self).__init__()
        self.model_name = model_name

    def sumarry(self, input_shape):
        ModelSummary(self, input_shape)

    def _save_to(self, save_dir, epoch, optim: Optimizer = None):
        save_dic = {'epoch': epoch,
                    'model_state_dict': self.state_dict(),
                    'optimizer_state_dict': optim.state_dict() if optim is not None else None}

        save(save_dic, save_dir + self.model_name + ".h5")

    def _load_from(self, model_file):
        checkpoint = load(model_file)
        self.load_state_dict(checkpoint["model_state_dict"])

        return checkpoint  # if restore train sasseion

    def forward(self, x):
        ...

    def set_layers(self, **parameters):
        ...


class layer():
    #
    @staticmethod
    def __make_norm_layer(out_features: int, dimension: int, opt: opt.layer.norm_opt) -> Module:
        # normalization setting
        if opt is None:
            return None
        elif opt.norm_type == "BatchNorm":
            if dimension == 1:
                return BatchNorm1d(out_features, **opt.to_parameters())

            elif dimension == 2:
                return BatchNorm2d(out_features, **opt.to_parameters())

    @staticmethod
    def __make_activate_layer(opt: opt.layer.active_opt) -> Module:
        # activation setting
        if opt is None:
            return None
        elif opt.active_type == "ReLU":
            return ReLU(**opt.to_parameters())
        elif opt.active_type == "LeakyReLU":
            return LeakyReLU(**opt.to_parameters())
        elif opt.active_type == "Tanh":
            return Tanh(**opt.to_parameters())
        elif opt.active_type == "Sigmoid":
            return Sigmoid(**opt.to_parameters())

    # layer block
    class _Fc(Module):
        def __init__(self, layer_opt: opt.layer.fc_opt, norm_opt: opt.layer.norm_opt = None, active_opt: opt.layer.active_opt = None):
            super(layer._Fc, self).__init__()

            self.liner = Linear(**asdict(layer_opt))
            self.norm = layer.__make_norm_layer(layer_opt.out_features, 1, norm_opt)
            self.activate = layer.__make_activate_layer(active_opt)

        def forward(self, x):
            x = self.liner(x)
            x = self.norm(x) if self.norm is not None else x
            x = self.activate(x) if self.activate is not None else x
            return x

    class _Conv2D(Module):
        def __init__(self, layer_opt: opt.layer.conv_opt, norm_opt: opt.layer.norm_opt = None, active_opt: opt.layer.active_opt = None):
            super(layer._Conv2D, self).__init__()

            self.liner = Conv2d(**asdict(layer_opt))
            self.norm = layer.__make_norm_layer(layer_opt.out_channels, 2, norm_opt)
            self.activate = layer.__make_activate_layer(active_opt)

        def forward(self, x):
            x = self.liner(x)
            x = self.norm(x) if self.norm is not None else x
            x = self.activate(x) if self.activate is not None else x
            return x

    class _Attention(Module):
        def __init__(
                self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                act_layer=GELU, norm_layer=LayerNorm) -> None:
            super().__init__()


class backbone():
    class resnet(Module):
        def __init__(self, type=50, train=False, option: opt.backbone_opt = opt.backbone_opt()):
            """
            args:
                type
                trained
                last_layer
            """
            super(backbone.resnet, self).__init__()
            self.opt = option
            if type == 50:
                self._line = models.resnet50(pretrained=not train)
            elif type == 101:
                self._line = models.resnet101(pretrained=not train)
            else:
                _error.variable(
                    "backbone.resnet",
                    "Have some problem in parameter 'type'. use default value 50")
                type = 50
                self._line = models.resnet50(pretrained=not train)

            # features parameters doesn't train
            for _parameters in self._line.conv1.parameters():
                _parameters.requires_grad = train
            for _parameters in self._line.bn1.parameters():
                _parameters.requires_grad = train
            for _parameters in self._line.relu.parameters():
                _parameters.requires_grad = train
            for _parameters in self._line.maxpool.parameters():
                _parameters.requires_grad = train
            for _parameters in self._line.layer1.parameters():
                _parameters.requires_grad = train
            for _parameters in self._line.layer2.parameters():
                _parameters.requires_grad = train
            for _parameters in self._line.layer3.parameters():
                _parameters.requires_grad = train
            for _parameters in self._line.layer4.parameters():
                _parameters.requires_grad = train

            for _parameters in self._line.avgpool.parameters():
                _parameters.requires_grad = train

            # delete classfication module
            self._line.fc = None

        def forward(self, x):
            x = self._line.conv1(x)
            x = self._line.bn1(x)
            x = self._line.relu(x)
            x = self._line.maxpool(x)

            x = self._line.layer1(x)
            x = self._line.layer2(x)
            x = self._line.layer3(x)
            x = self._line.layer4(x)  # x shape : batch_size, 2048, h/8, w/8

            if self.opt.use_avg_pooling:
                x = self._line.avgpool(x)  # x shape : batch_size, 2048, 1, 1
            return x.view(x.size(0), -1) if self.opt.use_flat else x

        def sumarry(self, input_shape):
            ModelSummary(self, input_shape)

    class vgg(Module):
        def __init__(self, type=19, train=False, option: opt.backbone_opt = opt.backbone_opt()):
            super(backbone.vgg, self).__init__()
            self.opt = option
            if type == 11:
                _line = models.vgg11(pretrained=not train)
            if type == 13:
                _line = models.vgg13(pretrained=not train)
            elif type == 16:
                _line = models.vgg16(pretrained=not train)
            elif type == 19:
                _line = models.vgg19(pretrained=not train)
            else:
                _error.variable(
                    "backbone.vgg",
                    "Have some problem in parameter 'type'. use default value 19")
                type = 19
                _line = models.vgg19(pretrained=not train)

            self._conv = _line.features
            self._avgpool = _line.avgpool

            for _parameter in self._conv.parameters():
                _parameter.requires_grad = train

        def forward(self, x):
            x = self._conv(x)  # x shape : batch_size, 512, 7, 7

            if self.opt.use_avg_pooling:
                x = self._avgpool(x)  # x shape : batch_size, 512, 1, 1
            return x.view(x.size(0), -1) if self.opt.use_flat else x

        def sumarry(self, input_shape):
            ModelSummary(self, input_shape)


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

#             merge_tensor = _torch_util.layer._concat(multi_holder)

#             return self.M_conv(merge_tensor)

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
