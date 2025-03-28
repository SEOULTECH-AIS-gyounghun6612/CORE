from functools import wraps
from typing import TypeVar, Callable, cast, Any

from pathlib import Path

import json
import yaml


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


class File_Process():
    @classmethod
    def Ensure_dir(cls, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)

    @classmethod
    def Read_from(
        cls, file: Path, enc: str = "UTF-8", **kw
    ) -> tuple[bool, Any]:
        raise NotImplementedError

    @classmethod
    def Write_to(
        cls, file: Path, data: Any, enc: str = "UTF-8", **kw
    ) -> bool:
        raise NotImplementedError


BASIC_FILE_ERROR = {
    PermissionError: "파일 권한 부족"
}


class Text(File_Process):
    @classmethod
    @Handle_exp()
    def Read_from(
        cls, file: Path, enc: str = "UTF-8", start: int = 0, delim: str = "\n"
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


JSON_FILE_ERROR = {
    **BASIC_FILE_ERROR,
    json.JSONDecodeError: "JSON 파싱 오류 발생",
    TypeError: "JSON 변환 불가능한 데이터 포함"
}


class Json(File_Process):
    @classmethod
    @Handle_exp(extra_exp=JSON_FILE_ERROR)
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

        with _path.open(encoding=enc) as _f:
            json.dump(data, _f, indent=indent)
        return True


class Yaml(File_Process):
    loader = yaml.FullLoader

    @classmethod
    @Handle_exp()
    def Read_from(
        cls, file: Path, enc: str = "UTF-8", **kw
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

        with _path.open(encoding=enc) as _f:
            yaml.dump(data, _f, indent=indent)
        return True


def Read_from(
    file: Path, enc: str = "UTF-8"
):
    if Path.is_file(file):
        _suffix: str = file.suffix

        if _suffix == ".json":
            return Json.Read_from(file, enc)

        if _suffix == ".txt":
            return Text.Read_from(file, enc)

        if _suffix == ".yaml":
            return Yaml.Read_from(file, enc)

        raise ValueError(f"File extension '{_suffix}' is not supported")
    raise ValueError("!!! This path is not FILE !!!")


def Write_to(
    file: Path,
    data: dict[str, VALUE] | list[str],
    enc: str = "UTF-8",
    **kw
):
    if isinstance(data, dict):
        _suffix: str = file.suffix
        if _suffix == ".json":
            return Json.Write_to(file, data, enc, **kw)

        if _suffix == ".txt":
            return Yaml.Write_to(file, data, enc, **kw)

    if isinstance(data, list):
        return Text.Write_to(file, data, enc, **kw)
