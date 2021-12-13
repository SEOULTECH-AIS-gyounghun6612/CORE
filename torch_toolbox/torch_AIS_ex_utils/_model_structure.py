from collections import namedtuple
from os import stat

from torch import save, load, cat
# from torch.nn import ReLU, Softmax, parameter, ModuleList
from torch.nn import Module
from torch.nn import Conv2d, BatchNorm2d, AdaptiveAvgPool2d
from torch.nn import ReLU, LeakyReLU, Tanh, Sigmoid

import torchvision.models as models
from torchsummary import summary as ModelSummary
from python_AIS_ex_utils import _error as _e

_error = _e.Custom_error(
    module_name="torch_custom_utils_v 1.x",
    file_name="_model_part.py")


class custom_module(Module):
    def __init__(self, model_name):
        super(custom_module, self).__init__()
        self.model_name = model_name

    def _save_to(self, save_dir, epoch, optim=None):
        save_dic = {'epoch': epoch,
                    'model_state_dict': self.state_dict(),
                    'optimizer_state_dict': optim.state_dict() if optim is not None else None}

        save(save_dic, save_dir + self.model_name + ".h5")

    def _load_from(self, model_file):
        checkpoint = load(model_file)
        self.load_state_dict(checkpoint["model_state_dict"])

        return checkpoint  # if restore train sasseion


class layer():
    conv_opt = namedtuple("conv_opt", ["stride", "padding", "groups"])
    norm_opt = namedtuple("norm_opt", ["normalization", "parameter"])
    active_opt = namedtuple("active_opt", ["activation", "parameter"])

    @staticmethod
    def concatenate(layers, dim=1):
        return cat(layers, dim=dim)

    @staticmethod
    class ConvBlock(Module):
        def __init__(self, in_ch, out_ch, k_size, conv_option, norm_option=None, activation_opt=None):
            super(layer.ConvBlock, self).__init__()

            _option = {} if conv_option is None \
                else {
                    "stride": conv_option.stride,
                    "padding": conv_option.padding,
                    "groups": conv_option.groups}

            self.conv_i = Conv2d(in_ch, out_ch, k_size, **_option)

            # batch normalization
            if norm_option is None:
                self.conv_i_bn = None
            elif norm_option.normalization == "BatchNorm2d":
                self.conv_i_bn = BatchNorm2d(out_ch, **norm_option.parameter)

            # activation setting
            if activation_opt is None:
                self.conv_i_act = None
            elif activation_opt.activation == "ReLU":
                self.conv_i_act = ReLU(**activation_opt.parameter)
            elif activation_opt.activation == "LeakyReLU":
                self.conv_i_act = LeakyReLU(**activation_opt.parameter)
            elif activation_opt.activation == "Tanh":
                self.conv_i_act = Tanh()
            elif activation_opt.activation == "Sigmoid":
                self.conv_i_act = Sigmoid()

        def forward(self, x):
            x = self.conv_i(x)
            x = self.conv_i_bn(x) if self.conv_i_bn is not None else x
            x = self.conv_i_act(x) if self.conv_i_act is not None else x
            return x

    @stat
    class attention(Module):
        def __init__(self) -> None:
            super().__init__()

    @staticmethod
    class kernel_attention(Module):
        def __init__(self, in_ch, out_ch, k_size, multi_head) -> None:
            super(layer.kernel_attention, self).__init__()

            if isinstance(k_size, int):
                k_size = [k_size, k_size]

            elif isinstance(k_size, list):
                k_size

            conv_option = layer.conv_opt(stride=1, padding=1, groups=in_ch)
            self.Q_conv = layer.ConvBlock(in_ch, in_ch, k_size)


class backbone():
    @staticmethod
    class resnet(Module):
        def __init__(self, type=50, trained=False, last_layer=None):
            """
            args:
                type
                trained
                last_layer
            """
            super(backbone.resnet, self).__init__()
            self._flat_option = last_layer
            if type == 50:
                self._line = models.resnet50(pretrained=trained)
            elif type == 101:
                self._line = models.resnet101(pretrained=trained)
            else:
                _error.variable(
                    "backbone.resnet",
                    "Have some problem in parameter 'type'. use default value 50")
                type = 50
                self._line = models.resnet50(pretrained=trained)

            # features parameters doesn't train
            for _parameters in self._line.conv1.parameters():
                _parameters.requires_grad = trained
            for _parameters in self._line.bn1.parameters():
                _parameters.requires_grad = trained
            for _parameters in self._line.relu.parameters():
                _parameters.requires_grad = trained
            for _parameters in self._line.maxpool.parameters():
                _parameters.requires_grad = trained
            for _parameters in self._line.layer1.parameters():
                _parameters.requires_grad = trained
            for _parameters in self._line.layer2.parameters():
                _parameters.requires_grad = trained
            for _parameters in self._line.layer3.parameters():
                _parameters.requires_grad = trained
            for _parameters in self._line.layer4.parameters():
                _parameters.requires_grad = trained

            for _parameters in self._line.avgpool.parameters():
                _parameters.requires_grad = trained

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
            x = self._line.layer4(x)

            if self._flat_option == "fc_flat":
                return x.view(x.size(0), -1)  # batch_size, 2048 * 32 * 32
            elif self._flat_option == "conv_flat":
                return self._line.avgpool(x)  # batch_size, 2048, 1, 1
            else:
                return x  # batch_size, 2048, 32, 32

        def sumarry(self, input_shape):
            ModelSummary(self, input_shape)

    @staticmethod
    class vgg(Module):
        def __init__(self, type=19, trained=False, last_layer=None):
            super(backbone.vgg, self).__init__()
            self._flat_option = last_layer
            if type == 11:
                _line = models.vgg11(pretrained=trained)
            if type == 13:
                _line = models.vgg13(pretrained=trained)
            elif type == 16:
                _line = models.vgg16(pretrained=trained)
            elif type == 19:
                _line = models.vgg19(pretrained=trained)
            else:
                _error.variable(
                    "backbone.vgg",
                    "Have some problem in parameter 'type'. use default value 19")
                type = 19
                _line = models.vgg19(pretrained=trained)

            self._conv = _line.features
            self._avgpool = _line.avgpool

            for _parameter in self._conv.parameters():
                _parameter.requires_grad = False

            if last_layer == "conv_flat":
                self._last_layer = AdaptiveAvgPool2d(1)

        def forward(self, x):
            x = self._conv(x)
            x = self._avgpool(x)

            if self._flat_option == "fc_flat":
                return x.view(x.size(0), -1)  # batch_size, 512 * 7 * 7
            elif self._flat_option == "conv_flat":
                return self._last_layer(x)  # batch_size, 512, 1, 1
            else:
                return x  # batch_size, 512, 7, 7

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
