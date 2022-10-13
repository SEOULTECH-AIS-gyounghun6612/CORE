from python_ex._base import directory, utils
from python_ex._label import Label_process
from torch_ex._torch_base import learing_mode, opt, Label_style, File_style
from torch_ex._trainer import Learning_process
from torch_ex._structure import opt as structure_opt
from torch_ex._structure import backbone

# directory
Data_root = "/home/ais-choi-01/DATA/Data/"
Result_root = directory._make_for_result(folder="../result_example/", root_dir=directory._relative_root())

# Learning_opt
learning_opt = opt._learning.E2E(
    # Infomation about train
    Learning_date=utils.time_stemp(is_text=True),
    Learning_detail="",
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
    Data_process=Label_process.BDD_100k(import_style=[Label_style.SEM_SEG, ], data_size=[224, 224], root=Data_root),
    Data_label_style=Label_style.SEM_SEG,
    Data_file_style=File_style.IMAGE_FILE)


class custom(Learning_process.End_to_End):
    def data_jump_to_gpu(self, data_list):
        return super().data_jump_to_gpu(data_list)

    def fit(self, epoch: int = 0, mode: str = "train", is_display: bool = False, save_root: str = None):
        return super().fit(epoch, mode, is_display, save_root)

    def result_save(self, mode: str, epoch: int, save_root: str):
        return super().result_save(mode, epoch, save_root)


process = custom(Result_root, learning_opt=learning_opt, dataloader_opt=dataloader_opt)
_model = backbone.fcn(structure_opt.backbone(101))
_optim = structure_opt.optim("Adam")

process.set_model_n_optim(_model, _optim)

print("end")
