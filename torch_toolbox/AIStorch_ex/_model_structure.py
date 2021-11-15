from torch import randn, save, load
from torch.nn import Conv2d
from torch.nn import ReLU, Softmax, parameter, ModuleList
from torch.nn import Module
from torch.nn import functional as F
from torch.nn import BatchNorm2d

import torchvision.models as models
from python_ex import _error as _e

if __package__ == "":
    # if this file in local project
    import _torch_util
else:
    # if this file in package folder
    from . import _torch_util

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
                    'optimizer_state_dict': optim.state_dict() if optim is not None else ""}

        save(save_dic, save_dir + self.model_name + ".h5")

    def _load_from(self, model_file):
        checkpoint = load(model_file)
        self.load_state_dict(checkpoint["model_state_dict"])

        return checkpoint  # if restore train sasseion


class backbone():
    @staticmethod
    class resnet(Module):
        def __init__(self, type=50, trained=True, flatten=False):
            super(backbone.resnet, self).__init__()
            self._flat = flatten
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
                _parameters.requires_grad = False
            for _parameters in self._line.bn1.parameters():
                _parameters.requires_grad = False
            for _parameters in self._line.relu.parameters():
                _parameters.requires_grad = False
            for _parameters in self._line.maxpool.parameters():
                _parameters.requires_grad = False
            for _parameters in self._line.layer1.parameters():
                _parameters.requires_grad = False
            for _parameters in self._line.layer2.parameters():
                _parameters.requires_grad = False
            for _parameters in self._line.layer3.parameters():
                _parameters.requires_grad = False
            for _parameters in self._line.layer4.parameters():
                _parameters.requires_grad = False

            for _parameters in self._line.avgpool.parameters():
                _parameters.requires_grad = False

            self._line.eval()
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

            x = self._line.avgpool(x)
            if self._flat:
                return _torch_util.layer._flatten(x)
            else:
                return x

    @staticmethod
    class vgg(Module):
        def __init__(self, type=19, trained=True, flatten=False):
            super(backbone.vgg, self).__init__()
            self._flat = flatten
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

        def forward(self, x):
            x = self._conv(x)
            x = self._avgpool(x)

            if self._flat:
                return _torch_util.layer._flatten(x)
            else:
                return x


class transformer():
    # in later fix it
    @staticmethod
    class _attention(Module):
        def __init__(
                self,
                input_shape: int,
                hidden_channel: list,
                k_size: int,
                is_self: bool = False):
            super(transformer._attention, self).__init__()
            self.is_self = is_self
            if is_self:
                input_ch = input_shape[-1]
                data_size = input_shape[:2]
                Q_input, K_input, V_input = [input_ch, input_ch, input_ch]
                QKV_output = hidden_channel
            else:
                E_ch, input_ch = input_shape[0][-1], input_shape[1][-1]
                data_size = input_shape[0][:2]
                Q_input, K_input, V_input = [input_ch, E_ch, E_ch]
                QKV_output = hidden_channel

            Q_option = {
                "in_channels": Q_input,
                "out_channels": QKV_output,
                "kernel_size": 1}
            K_option = {
                "in_channels": K_input,
                "out_channels": QKV_output,
                "kernel_size": 1}
            V_option = {
                "in_channels": V_input,
                "out_channels": QKV_output,
                "kernel_size": 1}

            self.W_q_conv = Conv2d(**Q_option)
            self.W_k_conv = Conv2d(**K_option)
            self.W_v_conv = Conv2d(**V_option)

            self.pad = _torch_util.function.get_conv_pad(
                input_size=data_size,
                kernel_size=k_size)
            S_option = {
                "in_channels": QKV_output,
                "out_channels": QKV_output,
                "kernel_size": k_size}
            self.S_conv = Conv2d(**S_option)

            self.softmax = Softmax(dim=1)

        def forward(self, x):
            if self.is_self:  # x -> [input, mask]
                W_q = self.W_q_conv(x)
                W_k = self.W_k_conv(x)
                W_v = self.W_v_conv(x)
            else:             # x -> [from encoder, from decoder, mask]
                _D_array, _E_array = x
                W_q = self.W_q_conv(_D_array)
                W_k = self.W_k_conv(_E_array)
                W_v = self.W_v_conv(_E_array)

            _pad_q = F.pad(W_q, self.pad) if self.pad is not None else W_q
            W_qs = self.S_conv(_pad_q)
            value_array = self.softmax(W_qs * W_k) * W_v

            return value_array

    @staticmethod
    class _MHA(Module):
        def __init__(
                self,
                multi_size: int,
                hidden_channel: int,
                input_shape: list,
                k_size: int,
                is_self: bool = False):
            """
            Args:
                frame        :
                multi_ct      :
                data_size
                input_chs
                kernel_size
                is_self      : If this module is self attention, set the "True"
            Returns:
                return (np.uint8 array): image data
            """
            super(transformer._MHA, self).__init__()
            self.multi_size = multi_size

            if is_self:
                input_ch = input_shape[-1]
            else:
                input_ch = input_shape[1][-1]
            self.attentions = ModuleList(
                [transformer._attention(
                    input_shape,
                    hidden_channel,
                    k_size,
                    is_self) for _ct in range(multi_size)])

            M_option = {
                "in_channels": hidden_channel * self.multi_size,
                "out_channels": input_ch,
                "kernel_size": 1}
            self.M_conv = Conv2d(**M_option)

            self.softmax = Softmax(dim=1)

        def forward(self, x):
            multi_holder = []
            for attention in self.attentions:
                multi_holder.append(attention(x))

            merge_tensor = _torch_util.layer._concat(multi_holder)

            return self.M_conv(merge_tensor)

    @staticmethod
    class _FFNN(Module):
        def __init__(self, input_ch, hidden_ch):
            super(transformer._FFNN, self).__init__()
            self.layer_1 = Conv2d(input_ch, hidden_ch, kernel_size=1)
            self.activation = ReLU(inplace=True)
            self.layer_2 = Conv2d(hidden_ch, input_ch, kernel_size=1)

        def forward(self, x):
            x = self.layer_1(x)
            x = self.activation(x)
            x = self.layer_2(x)
            return x

    @staticmethod
    def segment_embed(batch, size, channel):
        shape = (batch, channel, int(size[0]), int(size[1]))
        return parameter.Parameter(randn(shape, requires_grad=True))

    @staticmethod
    class image_encoder(Module):
        def __init__(self, multi_size, hidden_channel, input_shape, k_size):
            super(transformer.image_encoder, self).__init__()

            self.self_multi_head = parts.transformer._MHA(
                multi_size=multi_size,
                hidden_channel=hidden_channel,
                input_shape=input_shape,
                k_size=k_size,
                is_self=True)
            self.batch_norm_01 = BatchNorm2d(input_shape[-1])
            self.FFNN = parts.transformer._FFNN(input_shape[-1], input_shape[-1])
            self.batch_norm_02 = BatchNorm2d(input_shape[-1])

        def forward(self, x):
            after_smh = self.self_multi_head(x) + x
            after_smh = self.batch_norm_01(after_smh)
            after_FFNN = self.FFNN(after_smh) + after_smh
            after_FFNN = self.batch_norm_02(after_FFNN)

            return after_FFNN

    @staticmethod
    class image_decoder(Module):
        def __init__(self, multi_size, hidden_channel, E_data_shape, D_data_shape, k_size):
            super(transformer.image_decoder, self).__init__()

            self.self_multi_head = parts.transformer._MHA(
                multi_size=multi_size,
                hidden_channel=hidden_channel,
                input_shape=D_data_shape,
                k_size=k_size,
                is_self=True)
            self.batch_norm_01 = BatchNorm2d(D_data_shape[-1])

            self.multi_head = parts.transformer._MHA(
                multi_size=multi_size,
                hidden_channel=hidden_channel,
                input_shape=[E_data_shape, D_data_shape],
                k_size=k_size,
                is_self=False)
            self.batch_norm_02 = BatchNorm2d(D_data_shape[-1])
            self.FFNN = parts.transformer._FFNN(D_data_shape[-1], D_data_shape[-1])
            self.batch_norm_03 = BatchNorm2d(D_data_shape[-1])

        def forward(self, x):
            E_data, D_data = x
            after_smh = self.self_multi_head(D_data) + D_data
            after_smh = self.batch_norm_01(after_smh)
            after_mh = self.multi_head([after_smh, E_data]) + after_smh
            after_mh = self.batch_norm_02(after_mh)
            after_FFNN = self.FFNN(after_mh) + after_mh
            after_FFNN = self.batch_norm_02(after_FFNN)

            return after_FFNN
