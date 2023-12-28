from typing import Any, Dict, List
from dataclasses import asdict, dataclass

import time
from math import log10, floor

from ._System import Path, File


# -- DEFINE CONSTNAT -- #


# -- Mation Function -- #
@dataclass
class Config():
    """
    프로젝트에 사용되는 인자값을 관리하기 위한 객체(dataclass) 기본 구조

    -----------------------------------------------------------------------------------------------
    """

    def _Get_parameter(self) -> Dict[str, Any]:
        """
        json 파일에 기록된 정보를 인자값으로 변환하는 과정

        -------------------------------------------------------------------------------------------
        ### Parameters
        - None

        ### Return
        - parameter : 프로젝트에서 사용되는 인자값

        -------------------------------------------------------------------------------------------
        """
        return asdict(self)

    def _Convert_to_dict(self) -> Dict[str, File.Json.VALUEABLE]:
        """
        인자값을 json 파일에 기록 가능한 dictionary형 데이터로 변환하는 과정

        -------------------------------------------------------------------------------------------
        ### Parameters
        - None

        ### Return
        - dictionary : json 파일에 기록 하기 위한 dictionary형 데이터

        -------------------------------------------------------------------------------------------
        """
        return asdict(self)


class Project():
    def __init__(self, project_name: str, description: str, save_root: str):
        self.project_name = project_name
        self.description = description
        self.save_root = self._Make_save_root(save_root)

    def _Make_save_root(self, save_root: str):
        _working_day = Debuging.Time._Apply_text_form(Debuging.Time._Stemp(), True, "%Y-%m-%d")

        return Path._Make_directory(save_root, Path._Join(_working_day, self.project_name))


class Debuging():
    """
    프로젝트 진행에 따른 실행 내역 및 결과와 같은 주요 내용을 생성, 출력, 기록하기 위한 모듈

    -----------------------------------------------------------------------------------------------
    ### Module list
    - Time
    - Progress
    - Logging
    """

    class Time():
        @staticmethod
        def _Stemp(start_time: float | None = None):
            """
            현재 시간 정보를 생성하는 함수

            ---------------------------------------------------------------------------------------
            ### Parameters
            - start_time : 시간 측정을 위한 시작점

            ### Return
            - this_time : start_time 이후 흐른 시간 (start_time이 없는 경우 현재 시간)
            """
            return time.time() if start_time is None else time.time() - start_time

        @staticmethod
        def _Apply_text_form(source: float, is_local: bool = False, text_format: str = "%Y-%m-%d-%H:%M:%S"):
            """
            시간 정보를 텍스트로 변환하는 함수

            ---------------------------------------------------------------------------------------
            ### Parameters
            - source : 시간 정보
            - is_local : 프로그램이 작동하는 단말기의 지역시간 적용 여부
            - text_format : 생성하고자 하는 텍스트의 포멧. (https://docs.python.org/ko/3/library/time.html#time.strftime 참고)

            ### Return
            - time_text : 입력된 조건에 따라 시간 정보로 부터 생성된 텍스트
            """
            return time.strftime(text_format, time.localtime(source) if is_local else time.gmtime(source))

    class Progress():
        @staticmethod
        def _Count_auto_aligning(this_count: int, max_count: int):
            _string_ct = floor(log10(max_count)) + 1
            _this = f"{this_count}".rjust(_string_ct, "0")

            return f"{_this}/{max_count}"

        @staticmethod
        def _Progress_bar(iteration: int, total: int, prefix: str = '', suffix: str = '', decimals: int = 1, length: int = 100, fill: str = '█'):
            """
            Call in a loop to create terminal progress bar

            Parameters
            --------------------
            iteration
                current iteration
            total
                total iterations (Int)
            prefix
                prefix string (Str)
            suffix
                suffix string (Str)
            decimals
                positive number of decimals in percent complete (Int)
            length
                character length of bar (Int)
            fill
                bar fill character (Str)
            """
            percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
            filledLength = int(length * iteration // total)
            bar = fill * filledLength + '-' * (length - filledLength)
            print(f'\r{prefix} |{bar}| {percent}% {suffix}', end="\r")
            # Print New Line on Complete
            if iteration == total:
                print()

    class Logging():
        """
        프로젝트 진행에 따른 기록을 남기기 위한 객체 기본 구조

        -------------------------------------------------------------------------------------------
        ### Parameters
        - info : 프로젝트 세부 정보

        -------------------------------------------------------------------------------------------
        """
        _Annotation: File.Json.WRITEABLE = {}
        _Data: File.Json.WRITEABLE = {}

        def __init__(self, info: File.Json.WRITEABLE = {}):
            self._Insert(info, self._Annotation, True)

        def _Insert(self, data: File.Json.WRITEABLE, save_point: File.Json.WRITEABLE, is_overwrite: bool = False):
            for _key, _data in data.items():
                # overwrite or _key not exist in save_point
                if is_overwrite or _key not in save_point.keys():
                    save_point.update({_key: _data})
                else:
                    _slot = save_point[_key]
                    if isinstance(_slot, dict):
                        self._Insert(_data if isinstance(_data, dict) else {_key: _data}, _slot, is_overwrite)
                    elif isinstance(_data, dict):
                        save_point.update({_key: _data})
                    elif isinstance(_slot, list):
                        save_point[_key] = _slot + _data if isinstance(_data, list) else _slot + [_data, ]
                    else:
                        _slot = [save_point[_key], ]
                        save_point[_key] = _slot + _data if isinstance(_data, list) else _slot + [_data, ]

        def _Pick(self, pick: Dict[File.Json.KEYABLE, Dict | List | None], access_point: File.Json.WRITEABLE):
            _holder = {}

            for _key, _pick_info in pick.items():
                _pick_data = access_point[_key] if _key in access_point.keys() else None

                if _pick_info is None:
                    _holder.update({_key: _pick_data})

                elif isinstance(_pick_data, dict):
                    _holder.update(
                        {
                            _key: self._Pick(
                                _pick_info if isinstance(_pick_info, dict) else dict((_data_name, _pick_info) for _data_name in _pick_data.keys()),
                                _pick_data)
                        })
                elif isinstance(_pick_info, list):
                    _holder.update({_key: _pick_data[_pick_info[0]: _pick_info[-1]] if isinstance(_pick_data, list) else _pick_data})
                else:
                    _holder.update({_key: None})

            return _holder

        def _Load(self, file_dir: str, file_name: str):
            _save_pakage: dict[str, File.Json.WRITEABLE] = File.Json._Read(file_name, file_dir)

            if _save_pakage is not None:
                self._Insert(_save_pakage["annotation"], self._Annotation)
                self._Insert(_save_pakage["data"], self._Data)

        def _Save(self, file_dir: str, file_name: str):
            _save_pakage: File.Json.WRITEABLE = {
                "annotation": self._Annotation,
                "data": self._Data}

            File.Json._Write(file_name, file_dir, _save_pakage)

    class Visualize():
        ...
