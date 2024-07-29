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

        def Convert_save_format(
            self,
            additional: dict[str, str] | None = None,
            slot_length: dict[str, int] | None = None
        ) -> dict[str, str]:
            _data: dict[str, str] = {}

            if slot_length is None or "id_num" not in slot_length:
                _key = "id_num"
                _value = str(self.id_num)
            else:
                _key, _value = String.Str_adjust_with_key(
                    "id_num",
                    str(self.id_num),
                    slot_length["id_num"]
                )
            _data.update({_key: _value})

            if additional is not None:
                _data.update(additional)

            return _data

        def __eq__(self, other):
            if isinstance(other, self.__class__):
                _t_dict = self.__dict__
                _o_dict = other.__dict__
                if len(_t_dict) != len(_o_dict):
                    # option for "other come from child class"
                    # when is okay, change to "pass" or "..."
                    return False

                _t_hash = sum([
                    hash(_v) for _k, _v in _t_dict.items() if _k != "id_num"
                ])
                _o_hash = sum([
                    hash(_v) for _k, _v in _o_dict.items() if _k != "id_num"
                ])

                return _t_hash == _o_hash
            return False

        def __ne__(self, other):
            return not self.__eq__(other)


    NUMED_DATA = TypeVar(
        "NUMED_DATA",
        bound=Numbered_Data
    )

    @dataclass
    class Numbered_String_Data():
        _id_num: str

        def Convert_save_format(
            self,
            additional: dict[str, str] | None = None,
            slot_length: dict[str, int] | None = None
        ) -> dict[str, str]:
            _data: dict[str, str] = {}

            if slot_length is None or "_id_num" not in slot_length:
                _key = "_id_num"
                _value = self._id_num
            else:
                _key, _value = String.Str_adjust_with_key(
                    "_id_num",
                    self._id_num,
                    slot_length["_id_num"]
                )
            _data.update({_key: _value})

            if additional is not None:
                _data.update(additional)

            return _data

        def __eq__(self, other):
            if isinstance(other, self.__class__):
                _t_dict = self.__dict__
                _o_dict = other.__dict__
                if len(_t_dict) != len(_o_dict):
                    # option for "other come from child class"
                    # when is okay, change to "pass" or "..."
                    return False

                _t_hash = sum([
                    hash(_v) for _k, _v in _t_dict.items() if _k != "_id_num"
                ])
                _o_hash = sum([
                    hash(_v) for _k, _v in _o_dict.items() if _k != "_id_num"
                ])

                return _t_hash == _o_hash
            return False

        def __ne__(self, other):
            return not self.__eq__(other)

        @property
        def id_num(self):
            return int(self._id_num)

        @id_num.setter
        def id_num(self, id_num: int):
            self._id_num = str(id_num)

    NUMED_STRING_DATA = TypeVar(
        "NUMED_STRING_DATA",
        bound=Numbered_String_Data
    )

    class Block_Of_Numbered_String(Generic[NUMED_STRING_DATA]):
        def __init__(
            self,
            data_type: Type[Data_n_Block.NUMED_STRING_DATA],
            file_name: str = "data",
            file_dir: str = Path.WORK_SPACE,
        ) -> None:
            _file_path = Path.Join(file_name, file_dir)

            self.data_type = data_type

            self.data_dict: dict[int, data_type] = {}
            self.last_id = 0

            if Path.Exist_check(_file_path):
                self.Read_from_csv(file_name, file_dir)

            self.last_id = max(self.data_dict) if len(self.data_dict) else -1

        def Read_from_csv(
            self,
            file_name: str,
            file_dir: str
        ):
            _data_type = self.data_type
            _data_holder = self.data_dict

            for _data in File.CSV.Read_from_file(file_name, file_dir):
                _comp: Data_n_Block.NUMED_STRING_DATA = _data_type(**_data)
                _data_holder[int(_data["id_num"])] = _comp

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
                    _data.Convert_save_format(
                        slot_length=data_socket_size
                    ) for _data in _data_dict.values()
                ],
                list(self.data_type.__annotations__.__dict__)
            )

        def Update_data(
            self,
            new_data: Data_n_Block.NUMED_STRING_DATA,
            is_override: bool = False
        ) -> bool:
            if isinstance(new_data, self.data_type):
                _data_id = new_data.id_num
                if is_override:
                    self.data_dict[int(_data_id)] = new_data
                    return True
                elif new_data not in self.data_dict.values():  # add
                    _this_id = self.last_id + 1
                    new_data.id_num = _this_id
                    self.data_dict[_this_id] = new_data
                    self.last_id += 1
                    return True
                else:
                    raise ValueError(
                        f"This is already in {self.__class__.__name__} block")
            return False

        def Get_data_from(self, id_num: int, is_pop: bool = False):
            if id_num in self.data_dict:
                if is_pop:
                    return True, self.data_dict.pop(id_num)
                return True, self.data_dict[id_num]
            return False, None

        def Clear_data(self):
            self.data_dict = {}
            self.last_id = 0


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
                Time.Make_text_from(_this_time, "%Y-%m-%d_%H:%M:%S")
            ]
        )
        self.result_root = Path.Make_directory(_result_dir_dirs)
