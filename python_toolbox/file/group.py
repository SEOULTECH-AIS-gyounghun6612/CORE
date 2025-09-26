"""파일 그룹 디렉토리 생성 유틸 (데이터셋 패치 분할 용도)."""
from __future__ import annotations
from pathlib import Path
from shutil import copyfile


def Make_the_file_group(
    file_dir: Path, save_dir: Path,
    keyword: str, size: int, stride: int, overlap: int,
    drop_last: bool = False
) -> None:
    """지정된 크기와 간격으로 파일을 묶어 디렉토리 그룹을 생성함.

    슬라이딩 윈도우 방식으로 원본 디렉토리의 파일을 그룹화하여 각 그룹을
    별도 디렉토리로 복사함. 주로 시계열 데이터셋 패치 분할에 사용됨.

    Args:
        file_dir: 원본 파일 디렉토리.
        save_dir: 그룹별 저장 디렉토리 (자동 생성).
        keyword: glob 패턴 필터링 키워드.
        size: 그룹당 파일 개수.
        stride: 슬라이딩 윈도우 간격.
        overlap: 그룹 간 겹침 개수.
        drop_last: 마지막 부족 그룹 제거 여부.
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
