from torch import cuda
from torch import cat, save, sum
from torch import tensor, FloatTensor
from torch.nn import Conv2d
from torch.nn import ReLU, Tanh
from torch.nn import MSELoss, CrossEntropyLoss

from ais_utils import _numpy


class setting():
    SMALL_VALUE = 1E-5

    @staticmethod
    def is_cuda():
        return cuda.is_available()


class layer():
    @staticmethod
    def _concat(layers, dim=1):
        return cat(layers, dim=1)

    @staticmethod
    def _flatten(layer):
        return input.view(input.size(0), -1)

    @staticmethod
    def _make_edge(input, is_cuda):
        sobel_filter = tensor([[-1., -1., -1.], [-1., 8., -1.], [-1., -1., -1.]])
        if is_cuda:
            sobel_filter = sobel_filter.cuda()
        sobel = ReLU()(function.adept_kernel(input, sobel_filter, is_cuda))

        return Tanh()(sum(sobel, dim=1))


class function():
    @staticmethod
    def to_tensor(array):
        return FloatTensor(array)

    @staticmethod
    def get_conv_pad(input_size, kernel_size, interval=1, stride=1):
        if type(kernel_size) != list:
            kernel_size = [kernel_size, kernel_size]

        if stride != 1:
            size_h = input_size[0]
            size_w = input_size[1]

            pad_hs = (stride - 1) * (size_h - 1) + interval * (kernel_size[0] - 1)
            pad_ws = (stride - 1) * (size_w - 1) + interval * (kernel_size[1] - 1)
        else:
            pad_hs = interval * (kernel_size[0] - 1)
            pad_ws = interval * (kernel_size[1] - 1)

        pad_l = pad_hs // 2
        pad_t = pad_ws // 2

        return [pad_t, pad_ws - pad_t, pad_l, pad_hs - pad_l]

    @staticmethod
    def adept_kernel(input, fillter, is_cuda):
        _h, _w = fillter.shape
        _b, _c, _, _ = input.shape

        op_conv = Conv2d(_c, _c, kernel_size=[_h, _w], padding=1, groups=_c, bias=False).cuda() \
            if is_cuda else Conv2d(_c, _c, kernel_size=[_h, _w], padding=1, groups=_c, bias=False)

        # fillter = fillter.view(1, 1, _h, _w).repeat(_b, _c, 1, 1)
        fillter = fillter.reshape((1, 1, _h, _w))
        fillter = fillter.repeat(_c, 1, 1, 1)

        op_conv.weight.data = fillter

        return op_conv(input)


class loss():
    @staticmethod
    def mse(output, target):
        """
        Args:
            output: [batch, c, h, w]
            target: [batch, c, h, w]
        Return:
            loss
        """
        return MSELoss()(output, target)

    @staticmethod
    def cross_loss(output, target, ignore_index=-100):
        """
        Args:
            output: [batch, class_num, h, w]
            target: [batch, h, w]
        Return:
            loss
        """
        return CrossEntropyLoss(ignore_index=ignore_index)(output, target)

    @staticmethod
    def edge_loss(output, target, is_cuda):
        """
        Args:
            output: [batch, class_num, h, w]
            target: [batch, h, w]
        Return:
            loss
        """
        output_edge = layer._make_edge(output, is_cuda)
        target_edge = layer._make_edge(target, is_cuda)

        return loss.mse(output_edge, target_edge)


class File():
    @staticmethod
    def model_save_to(save_directory, model, epoch):
        save_dic = {'epoch': epoch,
                    'model_state_dict': model.state_dict()}

        save(save_dic, save_directory)

    @staticmethod
    def model_load_from(save_directory, model, epoch):
        pass

    @staticmethod
    def result_save_to(save_dir, result_type, save_format):
        pass


class evaluation():
    @staticmethod
    def iou(result, label, class_num):
        np_result = result.cpu().detach().numpy()
        np_label = label.cpu().detach().numpy()

        iou = _numpy.evaluation.iou(np_result, np_label, class_num)
        return iou

    @staticmethod
    def miou(result, label, class_num):
        iou = evaluation.iou(result, label, class_num)
        return iou.mean()


def load_check():
    print("!!! _utils in custom torch utils load Success !!!")
