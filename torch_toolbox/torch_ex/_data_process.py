from torch.utils.data import DataLoader, Dataset

if __package__ == "":
    from _torch_base import learing_mode, opt, torch_utils

else:
    from ._torch_base import learing_mode, opt, torch_utils


class dataset():
    class basement(Dataset):
        def __init__(self, opt: opt._dataloader, mode: learing_mode):
            super().__init__()
            self.data_process = opt.Data_process
            self.data_process.set_learning_mode(mode.value)
            self.data_profile = self.data_process.get_data_profile(opt.Data_label_style, opt.Data_file_style)

        def __len__(self):
            return len(self.data_profile.Input)

        def __getitem__(self, index):
            data = self.data_process.work(self.data_profile, index)
            return [torch_utils._tensor.from_numpy(_data, "float32") for _data in data]


def make_dataloader(opt: opt._dataloader, mode: learing_mode, dataset_function: dataset.basement = dataset.basement):
    _dataset = dataset_function(opt, mode)
    return DataLoader(_dataset, batch_size=opt.Batch_size, num_workers=opt.Num_workers, shuffle=(mode.value == learing_mode.TRAIN.value))
