from torch.utils.data import DataLoader, Dataset

if __package__ == "":
    from _base import opt, torch_utils

else:
    from ._base import opt, torch_utils


class dataset():
    class basement(Dataset):
        def __init__(self, data_option: opt._data, learnig_style: str) -> None:
            super().__init__()
            self.data_size = data_option.Data_size
            self.data_style = data_option.Data_style

            self.data = data_option.Data_read_process
            self.data.make_datalist(self.data_style.get_data_directory(learnig_style))

        def __len__(self):
            return self.data.get_len()

        def __getitem__(self, index):
            data = self.data_style.make_data(self.data, index, self.data_size)
            return [torch_utils._tensor.from_numpy(_data, "float32") for _data in data]


class dataloader():
    @staticmethod
    def _make(dataset: dataset.basement, batch: int, worker: int, is_shuffle: bool):
        return DataLoader(dataset, batch_size=batch, num_workers=worker, shuffle=is_shuffle)
