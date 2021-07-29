from torch import randn
from torch.nn import Conv2d
from torch.nn import ReLU, Softmax, parameter, ModuleList
from torch.nn import Module
from torch.nn import functional as F

from . import _utils


class transformer():
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

            self.pad = _utils.layer.get_conv_pad(
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

            merge_tensor = _utils.layer._concat(multi_holder)

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


def load_check():
    print("!!! _model_part in custom torch utils load Success !!!")