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
    Debugging_paramerters={learing_mode.TRAIN: ["loss", "acc_@1", "acc_@5"], learing_mode.VALIDATION: ["loss", "acc_@1", "acc_@5"]}
)

# Dataloader_opt
dataloader_opt = opt._dataloader(
    Batch_size=16,
    Num_workers=8,
    Data_process=Label_process.Imagenet_1k([Label_style.CLASSIFICATION, ], [224, 224], Data_root),
    Data_label_style=Label_style.CLASSIFICATION,
    Data_file_style=File_style.IMAGE_FILE)


# make custom model
# from torch_ex._structure import custom_module


# make custom model trainer
from torch_ex._trainer import Learning_process


class custom_trainer(Learning_process.End_to_End):
    def __init__(self, result_dir: str = None, log_file: str = None, is_resotre: bool = False, **opts):
        super().__init__(result_dir, log_file, is_resotre, **opts)

    def fit(self, epoch: int, mode: str = "train", is_display: bool = False, save_root: str = None):
        return super().fit(epoch, mode, is_display, save_root)


learning_mode = custom_trainer(Result_root, learning_opt=learning_opt, dataloader_opt=dataloader_opt)