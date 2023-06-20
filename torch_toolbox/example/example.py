# from .torch_ex import _Config


# _test_config


from torch_ex._Learning import End_to_End, Process_Name
from ..torch_ex._Dataset import Label, Data, Augment, Custom_Dataset_Process

_project_name = "test_learning"
_description = ""
_save_root = "./test/"
_mode_information = {
    Process_Name.TRAIN: {
        "amplification": 2,
        "augmentations": {
            "apply_mathod": Augment.Supported.ALBUIMIENTATIONS,
            "output_size": [512, 512]
        }
    },
    Process_Name.VALIDATION: {
        "amplification": 1,
        "augmentations": {
            "apply_mathod": Augment.Supported.ALBUIMIENTATIONS,
            "output_size": [512, 512]
        }
    }
}
_mode_list = list(_mode_information.keys())
_max_epoch = 10
_display_term = 0.1

_learing_process = End_to_End(_project_name, _description, _save_root, _mode_list, _max_epoch)


_data_info = [(None, Data.Format.image), (Label.Style.SEMENTIC, Data.Format.colormaps)]
_label_process = Label.Organization.BDD_100k()

_data_root = ""
_data_process = Data.Organize.BDD_100k(_data_root, _mode_list, _data_info, _label_process)

_augment = dict((_key, Augment.Plan(_options["amplification"], [Augment.Process._Build(**_options["augmentations"])])) for _key, _options in _mode_information.items())
_dataset = Custom_Dataset_Process(_data_process, _augment)
_learing_process._Set_dataloader_option(_dataset, 8, 1, 100)

_learing_process._Set_model_n_optim()