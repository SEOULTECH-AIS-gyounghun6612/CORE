from typing import List, Dict, Union, Optional


if __package__ == "":
    from _base import Directory, File, JSON_WRITEABLE
    from _numpy import Array_Process, Np_Dtype, ndarray
else:
    from ._base import Directory, File, JSON_WRITEABLE
    from ._numpy import Array_Process, Np_Dtype, ndarray


# -- DEFINE CONSTNAT -- #


# -- Mation Function -- #
class Log():
    _Annotation: Dict[str, JSON_WRITEABLE] = {}
    _Data: Dict[str, JSON_WRITEABLE] = {}

    def __init__(self, info: Dict = {}, data: Dict = {}, file_dir: Optional[str] = None, file_name: str = "log.json"):
        if file_dir is None:
            self._insert(info, self._Annotation)
            self._insert(data, self._Data)

        else:
            self._load(file_dir, file_name) if File._exist_check(Directory._divider_check(f"{file_dir}{file_name}", True)) else ...

    def _insert(
            self,
            data_block: Dict[str, JSON_WRITEABLE],
            access_point: Dict[str, JSON_WRITEABLE],
            is_overwrite: bool = True):

        # pick data in search point
        for _key, _data in data_block.items():
            _key_exist = _key in access_point.keys()
            _slot = access_point[_key] if _key_exist else {}

            # truth table(row: _slot, col: _data, overwrite is False)
            #                               LOG_SUPORT_TYPE         list(LOG_SUPORT_TYPE),     dict
            #    LOG_SUPORT_TYPE      convert to list and append           append               x
            # list(LOG_SUPORT_TYPE)   convert to list and merge             merge               x
            #         dict                          x                         x            go to deep

            # overwrite
            if is_overwrite or not _key_exist:
                _slot = _data
            # go to deep
            elif isinstance(_data, dict) or isinstance(_slot, dict):
                self._insert(_data, _slot, is_overwrite) if isinstance(_data, dict) and isinstance(_slot, dict) else ...
            # add
            elif not isinstance(_slot, list):
                _slot = [_slot, ]

                if isinstance(_data, list):
                    _slot += _data
                else:
                    _slot.append(_data)

            access_point.update({_key: _slot})

    def _get_data(
            self,
            data_info: Dict[str, Optional[Union[str, List[str], Dict]]],
            access_point: Dict[str, JSON_WRITEABLE],
            is_pop: bool = False) -> Dict[str, JSON_WRITEABLE]:

        _holder = {}

        for _naem_tag, _tag_info in data_info.items():
            if _naem_tag not in access_point.keys():
                continue
            _selected_data = access_point[_naem_tag]

            # truth table(row: _picked_data, col: _tag_info, overwrite is False)
            #                 not dict                     dict
            #   None     pick_selected_data         pick_selected_data
            #   str              x            pick_data in selected_data[str]
            # list(str)          x            pick_data in selected_data[str]
            #   dict             x                       go to deep

            if _tag_info is not None:
                if not isinstance(_selected_data, dict):
                    continue
                elif isinstance(_tag_info, dict):
                    _holder = dict((
                        "_".join([_naem_tag, _getting_key]),
                        _getting_value) for _getting_key, _getting_value in self._get_data(_tag_info, _selected_data, is_pop).items())

                else:
                    if isinstance(_tag_info, str):
                        _holder.update(
                            {"_".join([_naem_tag, _tag_info]): _selected_data.pop(_tag_info) if is_pop else _selected_data[_tag_info]})
                    elif isinstance(_tag_info, list):
                        _holder.update(dict((
                            "_".join([_naem_tag, _call_key]),
                            _selected_data.pop(_call_key) if is_pop else _selected_data[_call_key]) for _call_key in _tag_info))
            else:
                _holder.update({_naem_tag: _selected_data})

                if is_pop:
                    del access_point[_naem_tag]

        return _holder
        # in later change this function => return dict or list

    def _get_length(
            self,
            data_info: Dict[str, Optional[Union[str, List[str], Dict]]],
            access_point: Dict[str, JSON_WRITEABLE]):

        _data = self._get_data(data_info, access_point)

        return dict((_key, len(value) if isinstance(value, (list, tuple, dict)) else 1) for _key, value in _data.items())

    def _load(self, file_dir: str, file_name: str):
        _save_pakage = File._json(file_dir, file_name)

        if _save_pakage is not None:
            self._insert(_save_pakage["annotation"], self._Annotation)
            self._insert(_save_pakage["data"], self._Data)

    def _save(self, file_dir: str, file_name: str):
        _save_pakage = {
            "annotation": self._Annotation,
            "data": self._Data}

        File._json(file_dir, file_name, is_save=True, data_dict=_save_pakage)


class ploter():
    ...
