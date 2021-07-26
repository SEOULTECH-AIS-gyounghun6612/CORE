"""
File object
=====
    When write down python program, be used custom function.

Requirement
=====
    None
=====
"""
# Import module
import json
import platform

from glob import glob
from os import path, getcwd, mkdir

from . import _error as _e

# Set constant
DEBUG = False
_error = _e.Custom_error(
    module_name="ais_custom_utils_v 2.x",
    file_name="_base.py")


class directory():
    OS_THIS = platform.system()
    OS_WINDOW = "Windows"
    OS_UBUNTU = "Linux"
    SLASH = "/" if OS_THIS == OS_UBUNTU else "\\"
    RELARTION = "." + SLASH

    @classmethod
    def _slash_check(self, directory):
        _tmp_ct = -1 if self.OS_THIS == self.OS_UBUNTU else -2
        _tmp_dir = directory if directory[_tmp_ct:] == self.SLASH else directory + self.SLASH
        return _tmp_dir

    @classmethod
    def _devide(self, directory, point=-1):
        _dir = self._slash_check(directory)
        _comp = _dir.split(self.SLASH)[:-1]

        _front = ""
        _back = ""
        for _data in _comp[:point]:
            _front += _data + self.SLASH
        for _data in _comp[point:]:
            _back += _data + self.SLASH

        return _front, _back

    @classmethod
    def _exist_check(self, directory, is_file=False):
        return path.isfile(directory) if is_file\
            else path.isdir(self._slash_check(directory))

    @classmethod
    def _make(self, obj_dirs, root_dir=None):
        if isinstance(obj_dirs, str):
            obj_dirs = [obj_dirs, ]
        elif isinstance(obj_dirs, list):
            pass
        else:
            # !!!ERROR!!! wrong type entered
            _error.variable_stop(
                function_name="directory._make",
                variable_list=["obj_dirs", ],
                AA="Entered parameter 'obj_dirs' has unsuitable type data"
            )

        # root dir check
        if root_dir is not None:
            # not use relartion
            _root = self._slash_check(root_dir)
            if not self._exist_check(_root):
                _front, _back = self._devide(_root, -1)
                self._make(_back, _front)
        else:
            # use relartion
            _root = ""
            for _ct, _data in enumerate(obj_dirs):
                if _data.find(self.RELARTION):
                    obj_dirs[_ct] = self.RELARTION + _data

        # make directory
        for _ct, _data in enumerate(obj_dirs):
            _tem_dir = self._slash_check(_root + self.SLASH + _data)

            if not self._exist_check(_tem_dir):
                mkdir(_tem_dir)
            obj_dirs[_ct] = _tem_dir

        return obj_dirs

    @classmethod
    def _make_for_result(self, ):
        pass

    @classmethod
    def _inside_search(self, searched_dir, search_option="all", name="*", ext="*"):
        _dir = self._slash_check(searched_dir)
        _component_name = "*" if search_option == "all" else name
        if search_option == "all":
            _component_name = "*"
            _component_ext = ""
        else:
            _component_name = name
            _component_ext = \
                "" if ext == "*" else (ext if ext[0] == "." else "." + ext)

        _filter = _dir + _component_name + _component_ext

        return sorted(glob(_dir + _filter))

    @staticmethod  # Not yet
    def _compare():
        pass
        compare_obj = dir_checker(compare_obj, True)
        compare_data = compare_obj.split(SLASH)
        base_dir = dir_checker(base_dir, True)
        base_data = base_dir.split(SLASH)

        tmp_dir = "." + SLASH
        same_count = 0

        for _tmp_folder in base_data:
            if _tmp_folder in compare_data:
                same_count += 1
            else:
                break
        if len(base_data) - same_count:
            for _ct in range(len(base_data) - same_count):
                tmp_dir += ".." + SLASH

        for _folder in compare_data[same_count:]:
            tmp_dir += _folder + SLASH

    @classmethod
    def _get_main(self, just_name=True):
        return self._devide(getcwd())[-1] if just_name else self._slash_check(getcwd())

    @staticmethod
    def _del():
        pass

    @staticmethod
    def _copy():
        pass


class file():
    @staticmethod
    def _name_from_directory(dir):
        _last_component = dir.split(directory.SLASH)[-1]
        if _last_component.find(".") == -1:
            return None
        else:
            return _last_component

    @staticmethod
    def _extension_check(file_dir, exts, is_fix=False):
        file_name = file._name_from_directory(file_dir)
        is_positive = False

        if file_name is None:
            # !!!WARING!!! file directory not have file name
            _error.variable(
                function_name="file._extension_check",
                variable_list=["file_dir", ],
                AA="File name not exist in Entered Parameter 'file_dir'"
            ) if DEBUG else None

            # fix
            new_file_dir = file_dir + "." + exts[0] if is_fix else None

        else:
            file_ext = file_name.split(".")[-1]
            is_positive = file_ext in exts

            # fix
            if (not is_positive) and is_fix:
                _tem_ct = file_name.find(".")
                replace_file_name = file_name[:_tem_ct] + "." + exts[0]
                new_file_dir = file_dir.replace(file_name, replace_file_name)

        return [is_positive, new_file_dir] if is_fix else is_positive

    @staticmethod
    def _load():
        pass

    @staticmethod
    def _save():
        pass

    @staticmethod
    def _json(file_dir, file_name, data_dict=None, is_save=False):
        """
        Args:
            save_dir        :
            file_name       :
            data_dict       :
        Returns:
            return (dict)   :
        """
        # directory check
        if not directory._exist_check(file_dir):
            if is_save:
                # !!!WARING!!! save directory not exist
                _error.variable(
                    function_name="file.json_file",
                    variable_list=["file_dir", ],
                    AA="Entered directory '{}' not exist. In first make that".format(file_dir)
                ) if DEBUG else None

                directory._make(file_dir)

            else:
                # !!!ERROR!!! load directory not exist
                _error.variable_stop(
                    function_name="file.json_file",
                    variable_list=["file_dir", ],
                    AA="Entered directory '{}' not exist".format(file_dir)
                )

        # file_name check
        check_result, new_file_name = file._extension_check(file_name, ["json", ], True)
        if not check_result:
            # !!!WARING!!! extension not exist in file_name
            _error.variable(
                function_name="file.json_file",
                variable_list=["file_name", ],
                AA="Extension not exist in entered parameter 'file_name'."
            ) if DEBUG else None
            file_name = new_file_name

        # json file process load or save
        _tmp_dir = file_dir + file_name
        if is_save:
            # json file save
            _file = open(_tmp_dir, "w")
            json.dump(data_dict, _file, indent=4)
        else:
            # json file load
            if directory._exist_check(_tmp_dir, True):
                # json file exist
                _file = open(_tmp_dir, "r")
                return json.load(_file)

            else:
                # !!!ERROR!!! load json file not exist
                _error.variable_stop(
                    function_name="file.json_file",
                    variable_list=["file_dir", "file_name"],
                    AA="Load file '{}' not exist".format(_tmp_dir)
                )

    @staticmethod
    def _del():
        pass

    @staticmethod
    def _copy_to(dir):
        pass


class etc():
    @staticmethod
    def Progress_Bar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill='█'):
        """
        Call in a loop to create terminal progress bar
        Args:
            iteration   - Required  : current iteration (Int)
            total       - Required  : total iterations (Int)
            prefix      - Optional  : prefix string (Str)
            suffix      - Optional  : suffix string (Str)
            decimals    - Optional  : positive number of decimals in percent complete (Int)
            length      - Optional  : character length of bar (Int)
            fill        - Optional  : bar fill character (Str)
        Returns:
            Empty
        """
        percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
        filledLength = int(length * iteration // total)
        bar = fill * filledLength + '-' * (length - filledLength)
        print(f'\r{prefix} |{bar}| {percent}% {suffix}', end="\r")
        # Print New Line on Complete
        if iteration == total:
            print()


class server():
    @staticmethod
    def _connect():
        pass

    @staticmethod
    def _unconnect():
        pass


# FUNCTION
def load_check():
    print("!!! custom python module ais_utils _base load Success !!!")
