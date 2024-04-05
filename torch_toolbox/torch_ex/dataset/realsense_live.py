from .basement import __Basement__


class Realsense_Dataset(__Basement__):
    def Make_datalist(self, data_root: str, data_category: str, **kwarg):
        ...

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        ...
