from python_ex._base import directory, utils
from python_ex._label import Label_process
from torch_ex._torch_base import learing_mode, opt, Label_style, File_style
from torch_ex._trainer import Learning_process
from torch_ex._structure import opt as structure_opt
from torch_ex._structure import backbone

# directory
Data_root = "/home/ais-choi-01/DATA/Data/"

# Learning_opt
learning_opt = opt._learning.E2E(
    # Infomation about train
    Learing_name="example",
    Learning_date=utils.time_stemp(is_text=True),
    Learning_detail="",

    Save_root=directory._relative_root(),

    Learning_mode=[learing_mode.VALIDATION],

    # About Learning type and style
    Max_epochs=30,
    This_epoch=0,

    # About logging
    Logging_parameters={learing_mode.TRAIN: ["loss", "IoU"], learing_mode.VALIDATION: ["loss", "IoU"]},
    Display_paramerters={learing_mode.TRAIN: ["loss", "IoU"], learing_mode.VALIDATION: ["loss", "IoU"]}
)

dataloader_opt = opt._dataloader(
    Batch_size=16,
    Num_workers=8,
    Data_process=Label_process.BDD_100k(import_style=[Label_style.SEM_SEG, ], data_size=[224, 224], root="/home/ais-choi-01/DATA/Data/"),
    Data_label_style=Label_style.SEM_SEG,
    Data_file_style=File_style.IMAGE_FILE)


class custom(Learning_process.End_to_End):
    def __init__(self, learning_opt: opt._learning.E2E, dataloader_opt: opt._dataloader, log_file: str = None, is_restore: bool = False) -> None:
        super().__init__(learning_opt, dataloader_opt, log_file, is_restore)

        learning_block = [backbone.fcn(structure_opt.backbone(101)), structure_opt.optim("Adam")]

        # set model n optim
        self.save_learning_moduel([learning_block, ], is_restore)

    def data_jump_to_gpu(self, data_list):
        return super().data_jump_to_gpu(data_list)

    def fit(self, epoch: int = 0, mode: learing_mode = learing_mode.TRAIN, is_display: bool = False, save_root: str = None):
        for _input, _label in self.dataloaders[mode]:
            if self.learning_opt.Use_cuda:
                ...
        return super().fit(epoch, mode, is_display, save_root)

    def result_save(self, mode: str, epoch: int, save_root: str):
        return super().result_save(mode, epoch, save_root)


process = custom(learning_opt=learning_opt, dataloader_opt=dataloader_opt)
process.fit(mode=learing_mode.VALIDATION)

print("end")
