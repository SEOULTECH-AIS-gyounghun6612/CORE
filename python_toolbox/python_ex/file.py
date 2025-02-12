from typing import TypeVar

from pathlib import Path

import json

from python_ex.system import Path_utils

KEY = TypeVar(
    "KEY", bound=int | float | bool | str)
VALUE = TypeVar(
    "VALUE", bound=int | float | bool | str | tuple | list | dict | None)


def Read_from(file: Path, encoding_type: str = "UTF-8"):
    if Path.is_file(file):
        with file.open(encoding=encoding_type) as _file:
            _suffix = file.suffix[1:]
            if _suffix == "json":
                return _suffix, json.load(_file)
            if _suffix == "txt":
                return _suffix, _file.readlines()

            raise ValueError(f"File extension '{_suffix}' is not supported")
    else:
        raise ValueError("!!! This path is not FILE !!!")


def Write_to_file(
    file_name: str,
    data: dict[KEY, VALUE],
    save_dir: str | None = None,
    encoding_type: str = "UTF-8"
):
    _file_path = Path_utils.Join(file_name, save_dir)

    with _file_path.open("w", encoding=encoding_type) as _file:
        _suffix = _file_path.suffix[1:]
        if _suffix == "json":
            return json.dump(data, _file, indent=4)

        raise ValueError(f"File extension '{_suffix}' is not supported")


# class File():
#     class Support_Format(String.String_Enum):
#         TXT = auto()
#         JSON = auto()
#         CSV = auto()
#         YAML = auto()
#         XML = auto()

#     @staticmethod
#     def Extention_checker(file_name: str, file_format: File.Support_Format):
#         _format = str(file_format)
#         if "." in file_name:
#             _ext = file_name.split(".")[-1]
#             if _ext != _format:
#                 _file_name = file_name.replace(_ext, _format)
#             else:
#                 _file_name = file_name
#         else:
#             _file_name = f"{file_name}.{_format}"

#         return _file_name

#     class Text():
#         @staticmethod
#         def Read(
#             file_name: str,
#             file_dir: str,
#             encoding_type: str = "UTF-8"
#         ):
#             _file = Path_old.Join(
#                 File.Extention_checker(file_name, File.Support_Format.TXT),
#                 file_dir)

#             with open(_file, "r", encoding=encoding_type) as f:
#                 lines = f.read().splitlines()
#             return lines

#         @staticmethod
#         def Write(
#             file_name: str,
#             file_dir: str
#         ):
#             raise NotImplementedError

#     class Json():
#         KEYABLE = NUMBER | bool | str
#         VALUEABLE = KEYABLE | Tuple | list | dict | None
#         WRITEABLE = dict[KEYABLE, VALUEABLE]

#         @staticmethod
#         def Read(
#             file_name: str,
#             file_dir: str,
#             encoding_type: str = "UTF-8"
#         ) -> dict:
#             # make file path
#             _file = Path_old.Join(
#                 File.Extention_checker(file_name, File.Support_Format.JSON),
#                 file_dir)
#             _is_exist = Path_old.Exist_check(_file, Path_old.Type.FILE)

#             # read the file
#             if _is_exist:
#                 with open(_file, "r", encoding=encoding_type) as _file:
#                     _load_data = json.load(_file)
#                 return _load_data
#             print(f"file {file_name} is not exist in {file_dir}")
#             return {}

#         @staticmethod
#         def Write(
#             file_name: str,
#             file_dir: str,
#             data: WRITEABLE,
#             encoding_type: str = "UTF-8"
#         ):
#             # make file path
#             _file = Path_old.Join(
#                 File.Extention_checker(file_name, File.Support_Format.JSON),
#                 file_dir)

#             # dump to file
#             with open(_file, "w", encoding=encoding_type) as _file:
#                 try:
#                     json.dump(data, _file, indent="\t")
#                 except TypeError:
#                     return False
#             return True

#     class Csv():
#         @staticmethod
#         def Read_from_file(
#             file_name: str,
#             file_dir: str,
#             delimiter: str = "|",
#             encoding_type="UTF-8"
#         ) -> list[dict[str, Any]]:
#             """
#             """
#             # make file path
#             _file = Path_old.Join(
#                 File.Extention_checker(file_name, File.Support_Format.CSV),
#                 file_dir)
#             _is_exist = Path_old.Exist_check(_file, Path_old.Type.FILE)

#             if _is_exist:
#                 # read the file
#                 with open(_file, "r", encoding=encoding_type) as file:
#                     _raw_data = csv.DictReader(file, delimiter=delimiter)
#                     _read_data = [
#                         dict(
#                             (
#                                 _key.replace(" ", ""),
#                                 _value.replace(" ", "")
#                             ) for _key, _value in _line_dict.items()
#                         ) for _line_dict in _raw_data
#                     ]
#                 return _read_data
#             print(f"file {file_name} is not exist in {file_dir}")
#             return []

#         @staticmethod
#         def Write_to_file(
#             file_name: str,
#             file_dir: str,
#             data: list[dict],
#             feildnames: list[str],
#             delimiter: str = "|",
#             mode: Literal['a', 'w'] = 'w',
#             encoding_type="UTF-8"
#         ):
#             # make file path
#             _file = Path_old.Join(
#                 File.Extention_checker(file_name, File.Support_Format.CSV),
#                 file_dir)
#             _is_exist = Path_old.Exist_check(_file, Path_old.Type.FILE)

#             # dump to file
#             with open(
#                 _file,
#                 mode if not _is_exist else "w",
#                 encoding=encoding_type,
#                 newline=""
#             ) as _file:
#                 try:
#                     _dict_writer = csv.DictWriter(
#                         _file, fieldnames=feildnames, delimiter=delimiter)
#                     _dict_writer.writeheader()
#                     _dict_writer.writerows(data)
#                 except TypeError:
#                     return False
#             return True

#     class Yaml():
#         @staticmethod
#         def Read(
#             file_name: str,
#             file_dir: str,
#             encoding_type: str = "UTF-8"
#         ) -> dict:
#             # make file path
#             _file = Path_old.Join(
#                 File.Extention_checker(file_name, File.Support_Format.YAML),
#                 file_dir)
#             _is_exist = Path_old.Exist_check(_file, Path_old.Type.FILE)

#             # read the file
#             if _is_exist:
#                 with open(_file, "r", encoding=encoding_type) as _file:
#                     _load_data = yaml.load(_file, Loader=yaml.FullLoader)
#                 return _load_data
#             print(f"file {file_name} is not exist in {file_dir}")
#             return {}

#         @staticmethod
#         def Write(
#             file_name: str,
#             file_dir: str,
#             data,
#             encoding_type: str = "UTF-8"
#         ):
#             # make file path
#             _file = Path_old.Join(
#                 File.Extention_checker(file_name, File.Support_Format.YAML),
#                 file_dir)
#             # dump to file
#             with open(_file, "w", encoding=encoding_type) as _file:
#                 try:
#                     yaml.dump(data, _file, indent=4)
#                 except TypeError:
#                     return False
#             return True

#     class Xml():
#         @classmethod
#         def Xml_to_dict(cls, root_element: ET.Element):
#             _holder: dict[str, str | dict | None] = dict(
#                 (f"@{_att}", _v)for _att, _v in root_element.attrib.items()
#             )

#             for _child in root_element:
#                 _tag = _child.tag

#                 _data = cls.Xml_to_dict(_child) if (
#                     list(_child)
#                 ) else _child.text

#                 if _tag in _holder:
#                     _exist = _holder[_tag]
#                     if isinstance(_exist, list):
#                         _exist.append(_data)
#                     else:
#                         _exist = [_exist, _data]
#                 else:
#                     _holder[_tag] = _data

#             return _holder

#         @classmethod
#         def Read(
#             cls,
#             file_name: str,
#             file_dir: str,
#             cvt_func: Callable[[ET.Element], dict] | None = None
#         ) -> dict:
#             # make file path
#             _file = Path_old.Join(
#                 File.Extention_checker(file_name, File.Support_Format.XML),
#                 file_dir)
#             _is_exist = Path_old.Exist_check(_file, Path_old.Type.FILE)

#             # read the file
#             if _is_exist:
#                 _cvt = cvt_func if cvt_func else cls.Xml_to_dict
#                 return _cvt(ET.parse(_file).getroot())
#             print(f"file {file_name} is not exist in {file_dir}")
#             return {}
