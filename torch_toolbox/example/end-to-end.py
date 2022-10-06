from python_ex._base import directory, utils
from python_ex._label import Label_process

from torch_ex._torch_base import learing_mode, opt, Label_style, File_style

# directory
Data_root = "/home/ais-choi-01/DATA/Data/"
Result_root = directory._make_for_result(folder="../result_example/", root_dir=directory._relative_root())

# Learning_opt
learning_opt = opt._learning.base(
    # Infomation about train
    Learning_date=utils.time_stemp(is_text=True),
    Learning_detail="",
    Learning_mode=[learing_mode.TRAIN, learing_mode.VALIDATION],

    # About Learning type and style
    Max_epochs=30,
    This_epoch=0,

    # About logging
    Logging_parameters={learing_mode.TRAIN: ["loss", "acc_@1", "acc_@5"], learing_mode.VALIDATION: ["loss", "acc_@1", "acc_@5"]},
    Display_paramerters={learing_mode.TRAIN: ["loss", "acc_@1", "acc_@5"], learing_mode.VALIDATION: ["loss", "acc_@1", "acc_@5"]}
)

# Dataloader_opt
dataloader_opt = opt._dataloader(
    Batch_size=16,
    Num_workers=8,
    Data_process=Label_process.Imagenet_1k([Label_style.CLASSIFICATION, ], [224, 224], Data_root),
    Data_label_style=Label_style.CLASSIFICATION,
    Data_file_style=File_style.IMAGE_FILE)


# make custom model
from torch_ex._structure import opt, module, backbone
from torch_ex._torch_base import torch_utils


class PerceiverIO(module.custom):
    def __init__(self, model_name):
        super().__init__(model_name)

        self.latent = torch_utils._tensor.make_tensor()

        self.encode = module._Attention._cross_dot()
        self.decode = module._Attention._cross_dot()

        _process = []
        for _ct in range():
            _process.append(module._Attention._self_dot())

        self.process = module.make_module_list(_process)

    def forward(self, input, ouput_query):
        _output = self.encode(Q_source=self.latent, KV_source=input)

        for _p in self.process:
            _output = _p(_output)

        _output = self.decode(ouput_query, _output)
        return _output


class segmentation_type_one(module.custom):
    def __init__(self, model_name):
        super().__init__(model_name)

        self._fisrt = PerceiverIO()
        self._seceand = PerceiverIO()
        self._third = PerceiverIO()

        self.seg_model = backbone.fcn(opt.backbone(50))

    def forward(self, input):
        _output = self.seg_model(input)

        _fisrt_output = self._fisrt(_output)
        _seceand_output = self._seceand(_fisrt_output)
        _third_output = self._third(_seceand_output)

        return _third_output


class early_fusion(module.custom):
    def __init__(self, model_name):
        super().__init__(model_name)

        # fusion block
        self.back_bone = backbone.resnet(101, True, False, opt.backbone())  # resnet 101

        briges = []

        backbone_size = [64, 256, 512, 1024, 2048]
        for _ct in range(5):
            _ch = backbone_size[_ct]
            _b = pow(2, 1 + _ct)
            _front = module._UpConv2D(opt.conv2d(_ch, 64, _b, _b))
            _back = module._Conv2D(opt.conv2d(64, 3, 1))
            briges.append([_front, _back])
        self.briges = module.make_module_list(briges)

        self.position_encoder = module._Position_encoding.trigonometric()

        tr_list = []

        for _ct in range(12):
            tr_list.append(module._Attention._self_dot())

        self.tr_list = module.make_module_list(tr_list)

    def _fusion_block(self, x):
        _backbone_out = self.back_bone(x)

        _outputs = []
        for _ct in range(5):
            _tmp = _backbone_out[_ct]
            _tmp = self.briges[_ct][0](_tmp)
            _tmp = self.briges[_ct][1](_tmp)
            _outputs.append(_tmp)

        return torch_utils._layer.concatenate([x, ] + _outputs)

    def _position_embedding(self, x):
        self.position_encoder(x)
        return x

    def forward(self, x):
        x = self._fusion_block(x)
        x = self._position_embedding(x)
        for _ct in range():
            x = self.tr_list[_ct](x)

        return x


# make custom model trainer
from torch_ex._trainer import Learning_process


class custom_trainer(Learning_process.End_to_End):
    def __init__(self, result_dir: str = None, log_file: str = None, is_resotre: bool = False, **opts):
        super().__init__(result_dir, log_file, is_resotre, **opts)

    def fit(self, epoch: int, mode: str = "train", is_display: bool = False, save_root: str = None):
        return super().fit(epoch, mode, is_display, save_root)


learning_mode = custom_trainer(Result_root, learning_opt=learning_opt, dataloader_opt=dataloader_opt)
