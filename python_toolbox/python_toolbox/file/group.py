"""Utility for copying files into sliding-window directory groups."""
from __future__ import annotations
from pathlib import Path
from shutil import copyfile


def Make_the_file_group(
    file_dir: Path, save_dir: Path,
    keyword: str, size: int, stride: int, overlap: int,
    drop_last: bool = False
) -> None:
    """Copies files into per-group directories using a sliding window.

    Args:
        file_dir: Source directory scanned with ``glob``.
        save_dir: Destination directory containing per-group subdirectories.
        keyword: Glob pattern used to filter source files.
        size: Number of files included in each group window.
        stride: Step between files selected inside one window.
        overlap: Number of items overlapped between adjacent groups.
        drop_last: Whether to drop the final incomplete group.
    """
    _range = size * stride
    _step = stride * (size - overlap)

    _file_list = sorted(file_dir.glob(keyword))
    save_dir.mkdir(exist_ok=True)

    if len(_file_list) % _step:
        _group_ct = (len(_file_list) // _step) + int(not drop_last)
    else:
        _group_ct = len(_file_list) // _step

    for _ct in range(_group_ct):
        _new_rgb_dir = save_dir / f"{_ct:0>5d}"
        _new_rgb_dir.mkdir(exist_ok=True)

        _st = _ct * _step
        _ed = _st + _range

        for _img_file in _file_list[_st:_ed:stride]:
            copyfile(_img_file, _new_rgb_dir / _img_file.name)
