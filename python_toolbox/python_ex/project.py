from __future__ import annotations
from typing import (
    Any, Literal, Type, TypeVar, Generic
)
from dataclasses import asdict, dataclass

from .system import Path, File, Time, String


@dataclass
class Config():
    """
    프로젝트에 사용되는 인자값을 관리하기 위한 객체(dataclass) 기본 구조

    --------------------------------------------------------------------
    """
    def Config_to_parameter(self) -> dict[str, Any]:
        """
        설정값을 사용가능한 인자값으로 변경하는 함수

        ----------------------------------------------------------------
        ### Parameters
        - None

        ### Return
        - dictionary : 파라메터로 활용하기 위하여 구성된 인자값

        ----------------------------------------------------------------
        """
        return asdict(self)

    def Write_to(self, file_name: str, file_dir: str):
        File.Json.Write(file_name, file_dir, asdict(self))  # type: ignore

    @staticmethod
    def Read_from(file_name: str, file_dir: str):
        return File.Json.Read(file_name, file_dir)


class Data_n_Block():
    @dataclass
    class Numbered_Data():
        id_num: int
        tag: list[str]

        def _Str_adjust(
            self,
            key: str,
            value: str,
            data_size: dict[str, int] | None = None,
            align: Literal["l", "c", "r"] = "r",
        ):
            if data_size is not None and key in data_size:
                return (
                    String.Str_adjust(key, data_size[key], mode=align)[-1],
                    String.Str_adjust(value, data_size[key], mode=align)[-1]
                )
            return (key, value)

        def Convert_data_from_csv(self, **kwarg: str):
            try:
                self.id_num = int(kwarg["id_num"])
                for _key in ["tag"]:
                    _data = File.CSV.Convert_from_str(kwarg[_key])
                    assert isinstance(_data, type(self.__dict__[_key]))
                    self.__dict__[_key] = _data
                return 0
            except ValueError:
                return 1
            except KeyError:
                return 2

        def Convert_data_to_csv(
            self,
            additional: dict[str, str] | None = None,
            data_size: dict[str, int] | None = None
        ) -> dict[str, str]:
            _data: dict[str, str] = dict((
                self._Str_adjust("id_num", str(self.id_num), data_size),
                self._Str_adjust(
                    "tag", File.CSV.Convert_to_str(self.tag), data_size)
            ))

            if additional is not None:
                _data.update(additional)

            return _data

        def __eq__(self, other):
            if isinstance(other, self.__class__):
                for _key, _value in self.__dict__.items():
                    if _key == "id_num":
                        continue

                    if _value == other.__dict__[_key]:
                        return False
                    return True
            return False

        def __ne__(self, other: Data_n_Block.Numbered_Data):
            return not self.__eq__(other)

    NUMBERED_DATA = TypeVar("NUMBERED_DATA", bound=Numbered_Data)

    class Block(Generic[NUMBERED_DATA]):
        def __init__(
            self,
            data_type: Type[Data_n_Block.NUMBERED_DATA],
            file_name: str = "data",
            file_dir: str = Path.WORK_SPACE,
        ) -> None:
            _file_path = Path.Join(file_name, file_dir)

            self.data_type = data_type

            if Path.Exist_check(_file_path):
                self.data_dict: dict[int, data_type] = self.Read_from_csv(
                    file_name, file_dir
                )
                self.next_id = max(self.data_dict) + 1
            else:
                self.data_dict: dict[int, data_type] = {}
                self.next_id = 0

        def Read_from_csv(
            self,
            file_name: str,
            file_dir: str
        ):
            _data_type = self.data_type
            _holder: dict[int, _data_type] = {}

            for _data in File.CSV.Read_from_file(file_name, file_dir):
                _id_num = int(_data["id_num"])

                _comp: Data_n_Block.NUMBERED_DATA = _data_type(
                    int(_data["id_num"]), [])
                _comp.Convert_data_from_csv(**_data)
                _holder[_id_num] = _comp

            return _holder

        def Write_to_csv(
            self,
            file_name: str,
            file_dir: str,
            data_socket_size: dict[str, int] | None = None
        ):
            _data_dict = self.data_dict

            if not _data_dict:
                return False

            return File.CSV.Write_to_file(
                file_name,
                file_dir,
                [
                    _data.Convert_data_to_csv(
                        data_size=data_socket_size
                    ) for _data in _data_dict.values()
                ],
                list(self.data_type.__annotations__.__dict__)
            )

        def Set_data(
            self,
            new_data: Data_n_Block.NUMBERED_DATA,
            is_override: bool = False
        ) -> bool:

            if isinstance(new_data, self.data_type):
                _data_id = new_data.id_num
                if is_override:
                    if _data_id in self.data_dict:  # override
                        self.data_dict.update({_data_id: new_data})
                        return True
                elif new_data not in self.data_dict.values():  # add
                    _this_id = self.next_id
                    new_data.id_num = self.next_id
                    self.data_dict[_this_id] = new_data
                    self.next_id += 1
                    return True
            return False

        def Get_data_from(self, id_num: int, is_pop: bool = False):
            if id_num in self.data_dict:
                if is_pop:
                    return True, self.data_dict.pop(id_num)
                else:
                    return True, self.data_dict[id_num]
            return False, None

        def Clear_data(self):
            self.data_dict = {}
            self.next_id = 0


class Template():
    """ ### 프로젝트 구성을 위한 기본 구조

    ---------------------------------------------------------------------
    ### Args
    - Super
        - None
    - This
        - `project_name`: 프로젝트 이름
        - `category`: 프로젝트 구분
        - `result_dir`: 프로젝트 결과 저장 최상위 경로를 생성하기 위한 경로

    ### Attributes
    - `project_name`: 프로젝트 이름
    - `result_root`: 프로젝트 결과 저장 최상위 경로

    ### Structure
    - Make_save_root: 프로젝트 결과 저장 최상위 경로 생성 함수

    """
    def __init__(
        self,
        project_name: str,
        category: str | None = None,
        result_dir: str | None = None
    ):
        self.project_name = project_name
        self.Make_save_root(category, result_dir)

    def Make_save_root(
        self,
        description: str | None = None,
        result_root: str | None = None
    ):
        """ ### 프로젝트 결과 저장 최상위 경로 생성 함수

        ------------------------------------------------------------------
        ### Args
            - arg_name: Description of the input argument

        ### Returns or Yields
            - data_format: Description of the output argument

        ### Raises
            - None

        """
        _this_time = Time.Stamp()
        _result_dir_dirs = Path.Join(
            [
                "result" if result_root is None else result_root,
                "default" if description is None else description,
                "default" if description is None else description,
                Time.Make_text_from(_this_time, "%Y-%m-%d_%H:%M:%S")
            ]
        )
        self.result_root = Path.Make_directory(_result_dir_dirs)
