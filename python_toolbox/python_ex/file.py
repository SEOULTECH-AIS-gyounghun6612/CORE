from functools import wraps
from typing import TypeVar, Callable, cast, Any

from pathlib import Path
from shutil import copyfile

import json
import yaml

from .system import Path_utils


KEY = TypeVar(
    "KEY", bound=int | float | bool | str)
VALUE = TypeVar(
    "VALUE", bound=int | float | bool | str | tuple | list | dict | None)

F = TypeVar("F", bound=Callable)


def Suffix_check(path: Path, ext: list[str] | str, is_fix: bool = True):
    _ext = ext if isinstance(ext, list) else [ext]
    if path.suffix in _ext:
        return True, path

    return False, path.with_suffix(ext[0]) if is_fix else path


def Handle_exp(extra_exp: dict[type[Exception], str] | None = None):
    extra_exp = extra_exp or {}

    def Checker(func: F) -> F:

        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                _exp = type(e)
                message = extra_exp.get(_exp, "알 수 없는 파일 처리 오류 발생:")
                print(f"{message} -> {e}")
            return False, None
        return cast(F, wrapper)

    return Checker


BASIC_FILE_ERROR = {
    PermissionError: "파일 권한 부족"
}

JSON_FILE_READ_ERROR = {
    **BASIC_FILE_ERROR,
    json.JSONDecodeError: "JSON 파싱 오류 발생",
    TypeError: "JSON 변환 불가능한 데이터 포함"
}


class Process():
    class File_Process():
        @classmethod
        def Ensure_dir(cls, path: Path):
            path.parent.mkdir(parents=True, exist_ok=True)

        @classmethod
        def Read_from(
            cls, file: Path, enc: str = "UTF-8",
            **kwarg
        ) -> tuple[bool, Any]:
            raise NotImplementedError

        @classmethod
        def Write_to(
            cls, file: Path, data: Any, enc: str = "UTF-8",
            **kwarg
        ) -> bool:
            raise NotImplementedError

    class Text(File_Process):
        @classmethod
        @Handle_exp()
        def Read_from(
            cls, file: Path, enc: str = "UTF-8",
            start: int = 0, delim: str = "\n"
        ) -> tuple[bool, list]:
            _, _file = Suffix_check(file, ".txt")
            if _file.exists():
                return True, file.read_text(enc).split(delim)[start:]
            return False, []

        @classmethod
        @Handle_exp()
        def Write_to(
            cls, file: Path, data: str | list[str], enc: str = "UTF-8",
            anno: list[str] | str | None = None
        ) -> bool:
            cls.Ensure_dir(file)
            _, _path = Suffix_check(file, ".txt", True)

            _data = (
                [anno] if isinstance(anno, str) else anno
            ) if anno else []
            _data += [data] if isinstance(data, str) else data

            _path.write_text("\n".join(_data), enc)

            return True

    class Json(File_Process):
        @classmethod
        @Handle_exp(extra_exp=JSON_FILE_READ_ERROR)
        def Read_from(
            cls, file: Path, enc: str = "UTF-8"
        ) -> tuple[bool, dict[str, Any]]:
            _, _file = Suffix_check(file, ".json")

            if not _file.exists():
                return False, {}

            with file.open(encoding=enc) as _f:
                return True, json.load(_f)

        @classmethod
        @Handle_exp()
        def Write_to(
            cls, file: Path, data: dict[str, Any], enc: str = "UTF-8",
            indent: int = 4
        ) -> bool:
            cls.Ensure_dir(file)

            _, _path = Suffix_check(file, ".json", True)

            with _path.open(mode="w", encoding=enc) as _f:
                json.dump(data, _f, indent=indent)
            return True

    class Yaml(File_Process):
        loader = yaml.FullLoader

        @classmethod
        @Handle_exp()
        def Read_from(
            cls, file: Path, enc: str = "UTF-8",
            **kwarg
        ) -> tuple[bool, Any]:
            _, _file = Suffix_check(file, ".yaml")

            if not _file.exists():
                return False, {}

            with file.open(encoding=enc) as _f:
                return True, yaml.load(_f, cls.loader)

        @classmethod
        @Handle_exp()
        def Write_to(
            cls, file: Path, data: dict[str, Any], enc: str = "UTF-8",
            indent: int = 4
        ) -> bool:
            cls.Ensure_dir(file)

            _, _path = Suffix_check(file, ".yaml", True)

            with _path.open(mode="w", encoding=enc) as _f:
                yaml.dump(data, _f, indent=indent)
            return True


class Utils():
    @staticmethod
    def Read_from(file: Path, enc: str = "UTF-8", **kwarg):
        if Path.is_file(file):
            _suffix: str = file.suffix
            _name = _suffix[1:].capitalize()

            if hasattr(Process, _name):
                return getattr(Process, _name).Read_from(file, enc, **kwarg)

            raise ValueError(f"File extension '{_suffix}' is not supported")
        raise ValueError("!!! This path is not FILE !!!")

    @staticmethod
    def Write_to(
        file: Path,
        data: dict[str, VALUE] | list[str],
        enc: str = "UTF-8",
        **kwarg
    ):
        _suffix: str = file.suffix
        _name = _suffix[1:].capitalize()

        if isinstance(data, dict):
            if _name not in ["Yaml", "Json"]:
                raise ValueError((
                    "Format error:\n"
                    f"    Check data type(= {type(data)})"
                    f"and file format(= {_suffix[1:]}) before saving."
                ))
        else:
            if _name not in ["Text"]:
                raise ValueError((
                    "Format error:\n"
                    f"    Check data type(= {type(data)})"
                    f"and file format(= {_suffix[1:]}) before saving."
                ))

        getattr(Process, _name).Write_to(file, data, enc, **kwarg)

    @staticmethod
    def Make_the_file_group(
        file_dir: Path | str, save_dir: Path | str,
        keyword: str, size: int, stride: int, overlap: int,
        drop_last: bool = False
    ):
        _range = size * stride
        _step = stride * (size - overlap)

        _obj_dir = Path_utils.Path_check(file_dir)
        _save_dir = Path_utils.Path_check(save_dir)

        _file_list = Path_utils.Search_in(_obj_dir, keyword)

        if len(_file_list) % _step:
            _group_ct = (len(_file_list) // _step) + int(not drop_last)
        else:
            _group_ct = len(_file_list) // _step

        for _ct in range(_group_ct):
            _new_rgb_dir = _save_dir / f"{_ct:0>5d}"
            _new_rgb_dir.mkdir(exist_ok=True)

            _st = _ct * _step
            _ed = _st + _range

            for _img_file in _file_list[_st:_ed:stride]:
                copyfile(_img_file, _new_rgb_dir / _img_file.name)
