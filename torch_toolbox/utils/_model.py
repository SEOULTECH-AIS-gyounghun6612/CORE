from torch.nn import BatchNorm2d
from torch.nn import Module

from . import _model_part as parts


class transformer():
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
