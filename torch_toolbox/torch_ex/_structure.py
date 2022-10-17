from dataclasses import asdict
from typing import List
from math import sqrt, log

# from torch.nn import ReLU, Softmax, parameter, ModuleList
from torch import Tensor, save, load, float
from torch import zeros, mean, matmul, sin, cos, exp, arange, argmax
from torch.nn import Module, init, functional, ModuleList, Sequential
from torch.nn import parameter
from torch.nn import Linear, Conv2d, ConvTranspose2d, BatchNorm1d, BatchNorm2d  # , LayerNorm
from torch.nn import ReLU, LeakyReLU, Tanh, Sigmoid  # , GELU
from torch.nn import MSELoss, CrossEntropyLoss
from torch.optim import Optimizer
from torchsummary import summary as ModelSummary

from einops import rearrange

from python_ex import _error as _e

import torchvision.models as models

if __package__ == "":
    # if this file in local project
    from _torch_base import opt

else:
    # if this file in package folder
    from ._torch_base import opt

_error = _e.Custom_error(
    module_name="torch_custom_utils_v 1.x",
    file_name="_structure.py")


class loss():
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
            target: [batch, class_num, h, w]
        Return:
            loss value
        """
        return CrossEntropyLoss(ignore_index=ignore_index)(output, argmax(target, dim=1))

    @staticmethod
    def mean_loss(output, target) -> Tensor:
        return mean(output * target)


class module():
    class custom_module(Module):
        def __init__(self, model_name: str):
            super(module.custom_module, self).__init__()
            self.model_name = model_name

        def sumarry(self, input_shape):
            ModelSummary(self, input_shape)

        def _save_to(self, save_dir, epoch, optim: Optimizer = None):
            save_dic = {"epoch": epoch,
                        "model_state_dict": self.state_dict(),
                        "optimizer_state_dict": optim.state_dict() if optim is not None else None}

            save(save_dic, save_dir + self.model_name + ".h5")

        def _load_from(self, model_file):
            checkpoint = load(model_file)
            self.load_state_dict(checkpoint["model_state_dict"])

            return checkpoint  # if restore train sasseion

        def forward(self, x):
            return x

        def set_layers(self, **parameters):
            ...

    @staticmethod
    def make_norm_layer(out_features: int, dimension: int, norm_opt: opt._layer_opt.norm2d) -> Module:
        # normalization setting
        if norm_opt is None:
            return None
        elif norm_opt.norm_type == "BatchNorm":
            if dimension == 1:
                return BatchNorm1d(out_features, **norm_opt.to_parameters())

            elif dimension == 2:
                return BatchNorm2d(out_features, **norm_opt.to_parameters())

    @staticmethod
    def make_activate_layer(active_opt: opt._layer_opt.active_function) -> Module:
        # activation setting
        if active_opt is None:
            return None
        elif active_opt.active_type == opt._layer_opt.active_name.ReLU:
            return ReLU(**active_opt.to_parameters())
        elif active_opt.active_type == opt._layer_opt.active_name.ReLU:
            return LeakyReLU(**active_opt.to_parameters())
        elif active_opt.active_type == opt._layer_opt.active_name.ReLU:
            return Tanh(**active_opt.to_parameters())
        elif active_opt.active_type == opt._layer_opt.active_name.ReLU:
            return Sigmoid(**active_opt.to_parameters())
        elif active_opt.active_type == opt._layer_opt.active_name.ReLU:
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
        def __init__(self, layer_opt: opt._layer_opt.fc, norm_opt: opt._layer_opt.norm2d = None, active_opt: opt._layer_opt.active_function = None):
            super(module._Fc, self).__init__()

            self.liner = Linear(**asdict(layer_opt))
            self.norm = module.make_norm_layer(layer_opt.out_features, 1, norm_opt)
            self.activate = module.make_activate_layer(active_opt)

        def forward(self, x: Tensor) -> Tensor:
            x = self.liner(x)
            x = self.norm(x) if self.norm is not None else x
            x = self.activate(x) if self.activate is not None else x
            return x

    class _Conv2D(Module):
        def __init__(self, layer_opt: opt._layer_opt.conv2d, norm_opt: opt._layer_opt.norm2d = None, active_opt: opt._layer_opt.active_function = None):
            super(module._Conv2D, self).__init__()

            self.liner = Conv2d(**asdict(layer_opt))
            self.norm = module.make_norm_layer(layer_opt.out_channels, 2, norm_opt)
            self.activate = module.make_activate_layer(active_opt)

        def forward(self, x: Tensor) -> Tensor:
            x = self.liner(x)
            x = self.norm(x) if self.norm is not None else x
            x = self.activate(x) if self.activate is not None else x
            return x

    class _UpConv2D(Module):
        def __init__(self, layer_opt: opt._layer_opt.conv2d, norm_opt: opt._layer_opt.norm2d = None, active_opt: opt._layer_opt.active_function = None):
            super(module._Conv2D, self).__init__()

            self.liner = ConvTranspose2d(**asdict(layer_opt))
            self.norm = module.make_norm_layer(layer_opt.out_channels, 2, norm_opt)
            self.activate = module.make_activate_layer(active_opt)

        def forward(self, x: Tensor) -> Tensor:
            x = self.liner(x)
            x = self.norm(x) if self.norm is not None else x
            x = self.activate(x) if self.activate is not None else x
            return x

    class _Attention():
        class __base(Module):
            def __init__(self, layer_opt: opt._layer_opt.attention) -> None:
                super().__init__()

                self.output_dim = layer_opt.output_dim
                self.head_count = layer_opt.head_count
                self.head_dim = layer_opt.get_head_dim()

                self.k_maker = module._Fc(opt._layer_opt.fc(layer_opt.input_dim, self.output_dim))
                init.xavier_uniform_(self.k_maker.liner.weight)
                self.k_maker.liner.bias.data.fill_(0)

                self.v_maker = module._Fc(opt._layer_opt.fc(layer_opt.input_dim, self.output_dim))
                init.xavier_uniform_(self.v_maker.liner.weight)
                self.v_maker.liner.bias.data.fill_(0)

                self.q_maker = module._Fc(opt._layer_opt.fc(layer_opt.input_dim, self.output_dim))
                init.xavier_uniform_(self.q_maker.liner.weight)
                self.q_maker.liner.bias.data.fill_(0)

                self.o_maker = module._Fc(opt._layer_opt.fc(layer_opt.output_dim, self.output_dim))
                init.xavier_uniform_(self.o_maker.liner.weight)
                self.o_maker.liner.bias.data.fill_(0)

            def dot_product(self, Q, K, V, mask=None):  # dot_product
                _logits = matmul(Q, rearrange(K, 'batch head_num seq head_dim -> batch head_num head_dim seq'))  # -> batch head_num seq seq
                _logits /= sqrt(self.head_dim)

                if mask is not None:
                    _logits = _logits.masked_fill(mask == 0, -9e15)

                _attention = functional.softmax(_logits, dim=-1)

                return matmul(_attention, V), _attention

        class _self_dot(__base):
            def __init__(self, layer_opt: opt._layer_opt.attention) -> None:
                super().__init__(layer_opt)

            def forward(self, QKV_source: Tensor, mask: Tensor = None, return_attention_map: bool = False) -> Tensor:
                _q = self.q_maker(QKV_source)
                _q: Tensor = rearrange(_q, 'batch seq (head_dim head_num) -> batch head_num seq head_dim', head_dim=self.head_dim, head_num=self.head_count)

                _k = self.k_maker(QKV_source)
                _k: Tensor = rearrange(_k, 'batch seq (head_dim head_num) -> batch head_num seq head_dim', head_dim=self.head_dim, head_num=self.head_count)

                _v = self.v_maker(QKV_source)
                _v: Tensor = rearrange(_v, 'batch seq (head_dim head_num) -> batch head_num seq head_dim', head_dim=self.head_dim, head_num=self.head_count)

                _value, _attention = self.dot_product(_q, _k, _v, mask)  # value -> batch head_num seq head_dim

                _value: Tensor = rearrange(_value, 'batch head_num seq head_dim -> batch seq (head_dim head_num)')
                _outpot = self.o_maker(_value)
                return _outpot, _attention if return_attention_map else _outpot

        class _cross_dot(__base):
            def __init__(self, layer_opt: opt._layer_opt.attention) -> None:
                super().__init__(layer_opt)

            def forward(self, Q_source: Tensor, KV_source: Tensor, mask: Tensor = None, return_attention_map: bool = False) -> Tensor:
                _q = self.q_maker(Q_source)
                _q: Tensor = rearrange(_q, 'batch seq (head_dim head_num) -> batch head_num seq head_dim', head_dim=self.head_dim, head_num=self.head_count)

                _k = self.k_maker(KV_source)
                _k: Tensor = rearrange(_k, 'batch seq (head_dim head_num) -> batch head_num seq head_dim', head_dim=self.head_dim, head_num=self.head_count)

                _v = self.v_maker(KV_source)
                _v: Tensor = rearrange(_v, 'batch seq (head_dim head_num) -> batch head_num seq head_dim', head_dim=self.head_dim, head_num=self.head_count)

                _value, _attention = self.dot_product(_q, _k, _v, mask)  # value -> batch head_num seq head_dim

                _value: Tensor = rearrange(_value, 'batch head_num seq head_dim -> batch seq (head_dim head_num)')
                _outpot = self.o_maker(_value)
                return _outpot, _attention if return_attention_map else _outpot

    class _Position_encoding():
        class trigonometric(Module):
            def __init__(self, num_of_data, max_token_size=5000):
                super().__init__()
                pe = zeros(max_token_size, num_of_data)
                position = arange(0, max_token_size, dtype=float).unsqueeze(1)
                div_term = exp(arange(0, num_of_data, 2).float() * (-log(10000.0) / num_of_data))
                pe[:, 0::2] = sin(position * div_term)
                pe[:, 1::2] = cos(position * div_term)
                pe = pe.unsqueeze(0)

                self.register_buffer("position_value", pe, persistent=False)

            def forward(self, x: Tensor):
                x = x + self.position_value[:, : x.size(1)]
                return x

        class gaussian(Module):
            def __init__(self, num_of_data, max_token_size=5000):
                super().__init__()
                # in later fix it
                pe = zeros(max_token_size, num_of_data)
                position = arange(0, max_token_size, dtype=float).unsqueeze(1)
                div_term = exp(arange(0, num_of_data, 2).float() * (-log(10000.0) / num_of_data))
                pe[:, 0::2] = sin(position * div_term)
                pe[:, 1::2] = cos(position * div_term)
                pe = pe.unsqueeze(0)

                self.register_buffer("position_value", pe, persistent=False)

            def forward(self, x: Tensor):
                x = x + self.position_value[:, : x.size(1)]
                return x


class backbone():
    class resnet(Module):
        def __init__(self, option: opt._layer_opt.backbone):
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
                _pooling: Tensor = self._line.avgpool(out_conv5)  # x shape : batch_size, 2048, 1, 1
                return _pooling.view(_pooling.size(0), -1) if self.option.use_flat else _pooling

            else:
                return [out_conv1, out_conv2, out_conv3, out_conv4, out_conv5]

        def sumarry(self, input_shape):
            ModelSummary(self, input_shape)

    class vgg(Module):
        def __init__(self, option: opt._layer_opt.backbone):
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

        def forward(self, x: Tensor):
            x = self._conv(x)  # x shape : batch_size, 512, 7, 7

            if self.option.use_avg_pooling:
                x = self._avgpool(x)  # x shape : batch_size, 512, 1, 1
            return x.view(x.size(0), -1) if self.option.use_flat else x

        def sumarry(self, input_shape):
            ModelSummary(self, input_shape)

    class fcn(Module):
        def __init__(self, option: opt._layer_opt.backbone) -> None:
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
