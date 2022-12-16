from typing import List, Dict, Union, Optional


if __package__ == "":
    from _base import Directory, File, JSON_WRITEABLE
else:
    from ._base import Directory, File, JSON_WRITEABLE


# -- DEFINE CONSTNAT -- #


# -- Mation Function -- #
class Tracker():
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
            access_point.update({_key: {}}) if not _key_exist else ...
            _slot = access_point[_key]

            # truth table(row: _slot, col: _data, overwrite is False)
            #                               LOG_SUPORT_TYPE         list(LOG_SUPORT_TYPE),     dict
            #    LOG_SUPORT_TYPE      convert to list and append           append               x
            # list(LOG_SUPORT_TYPE)   convert to list and merge             merge               x
            #         dict                          x                         x            go to deep

            # overwrite
            if is_overwrite or not _key_exist:
                access_point[_key] = _data
                continue
            # go to deep
            elif isinstance(_data, dict) or isinstance(_slot, dict):
                self._insert(_data, _slot, is_overwrite) if isinstance(_data, dict) and isinstance(_slot, dict) else ...
                continue
            # add
            elif not isinstance(_slot, list):
                if isinstance(_data, list):
                    access_point[_key] = [_slot, ] + _data
                else:
                    access_point[_key] = [_slot, ] + [_data, ]
            else:
                if isinstance(_data, list):
                    _slot += _data
                else:
                    _slot.append(_data)

    def _get_data(
            self,
            data_info: Optional[Dict[str, Optional[Union[str, Dict]]]],
            access_point: Dict[str, JSON_WRITEABLE],
            is_pop: bool = False) -> Dict[str, JSON_WRITEABLE]:

        # truth table(row: _picked_data, col: _tag_info, overwrite is False)
        #                 not dict                     dict
        #   None     pick_selected_data         pick_selected_data
        #   str              x            pick_data in selected_data[str]
        #   dict             x                       go to deep

        _holder = {}

        if data_info is None:
            _holder.update(dict((_key, access_point.pop(_key) if is_pop else _data) for _key, _data in access_point.items()))
        else:
            for _data_name, _tag_info in data_info.items():
                if _data_name not in access_point.keys():
                    continue
                _access_point = access_point[_data_name]
                if _tag_info is None or not isinstance(_access_point, dict):
                    _pick = _access_point
                    if is_pop:
                        del _access_point
                    _holder.update({_data_name: _pick})
                elif isinstance(_tag_info, str):
                    _pick = _access_point.pop(_tag_info) if is_pop else _access_point[_tag_info]
                    _holder.update({_data_name: _pick})
                else:
                    _pick = self._get_data(_tag_info, _access_point, is_pop)
                    _holder.update(dict((f"{_data_name}_{_key}", _value) for _key, _value in _pick.items()))
        return _holder
        # in later change this function => return dict or list

    def _get_length(
            self,
            data_info: Optional[Dict[str, Optional[Union[str, Dict]]]],
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
