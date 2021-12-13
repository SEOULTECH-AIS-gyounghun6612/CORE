import torch
from torch.nn import MSELoss, CrossEntropyLoss

from python_AIS_ex_utils import _base
from python_AIS_ex_utils import _numpy
# from python_AIS_ex_utils import _error as _e


class setting():
    SMALL_VALUE = 1E-5

    @staticmethod
    def is_cuda():
        return torch.cuda.is_available()


class layer():
    @staticmethod
    def get_conv_pad(kernel_size, input_size, interval=1, stride=1):
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


class tensor():
    torch_type = {
        "uint8": torch.uint8, "int32": torch.int32, "bool": torch.bool, "float32": torch.float32}

    @classmethod
    def from_numpy(self, np_array, type="float32"):
        return torch.tensor(np_array, dtype=self.torch_type[type])

    @staticmethod
    def to_numpy(tensor, type=None):
        return tensor.numpy()

    @staticmethod
    def make_tensor(size, shape_sample=None, norm_option=None, dtype="uint8", value=[0, 1]):
        _np_array = _numpy.base_process.get_array(size, shape_sample, norm_option, dtype, value)
        return tensor.from_numpy(_np_array, dtype)

    @staticmethod
    def range_cut(tensor, range_min, rage_max):
        return torch.clip(tensor, range_min, rage_max)


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
    def cross_entropy(output, target, ignore_index=-100):
        """
        Args:
            output: [batch, class_num, h, w]
            target: [batch, h, w]
        Return:
            loss
        """
        return CrossEntropyLoss(ignore_index=ignore_index)(output, target)


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


class log():
    log_data = {
        "train": {
            "loss": [],
            "eval": {}
        },
        "validation": {
            "loss": [],
            "eval": {}
        }
    }

    factor_list = ["loss", ]

    def __init__(self, factor_name, num_class) -> None:
        self.num_class = num_class
        if isinstance(factor_name, list):
            for _name in factor_name:
                self.log_data["train"]["eval"][_name] = []
                self.log_data["validation"]["eval"][_name] = []

                self.factor_list.append(_name)
        else:
            self.log_data["train"]["eval"][factor_name] = []
            self.log_data["validation"]["eval"][factor_name] = []
            self.factor_list.append(factor_name)

    def get_log_factor_list(self):
        return self.factor_list

    def update(self, squence, value):
        _active = self.log_data[squence]

        for _factor in list(value.keys()):
            if _factor in self.factor_list:
                _active[_factor].append(value[_factor]) if _factor == "loss" \
                    else _active["eval"][_factor].append(value[_factor])

    def log_save(self, log_info, save_dir, file_name="log.json"):
        save_pakage = {
            "info": log_info,
            "data": self.log_data}
        _base.file._json(save_dir, file_name, save_pakage, True)

    def get_last_log(self):
        pass

    def plot(self):
        pass
