"""
File object
-----
    When write down python program, be used custom function.

Requirement
-----
    None
"""
# Import module
import json
import yaml
import platform
import time

from glob import glob
from os import path, system, getcwd, mkdir
from typing import Dict, List, Union


if __package__ == "":
    # if this file in local project
    import _error as _e

else:
    # if this file in package folder
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

    @classmethod  # fix it
    def _slash_check(self, directory: str, is_file: bool = False) -> str:
        # each os's directory divide slash fix
        if self.SLASH == "\\":
            from_dived = "/"
        else:
            from_dived = "\\"
        directory.replace(from_dived, self.SLASH)

        if not is_file:
            # if checked path is dir, check last slach exist in end of text
            return directory if directory[-len(self.SLASH):] == self.SLASH \
                else directory + self.SLASH

        return directory

    @classmethod
    def _exist_check(self, directory: str) -> bool:
        return path.isdir(directory)

    @classmethod
    def _devide(self, directory: str, point: int = -1) -> List[str]:
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
    def _relative_root(self, just_name: bool = False) -> str:
        return self._devide(getcwd())[-1] if just_name else self._slash_check(getcwd())

    @classmethod
    def _make(self, obj_dir: str, root_dir: str = None) -> str:
        if root_dir is not None:
            # use root directory
            # root directory check
            _dir = self._slash_check(root_dir)
            if not self._exist_check(_dir):
                # if root directory not exist, make it
                _front, _back = self._devide(_dir, -1)
                self._make(_back, _front)
        else:
            # use relative root directory (= cwd)
            _dir = self._relative_root()

        # make directory
        for _part in self._slash_check(obj_dir).split(self.SLASH):
            _dir = self._slash_check(_dir + _part)
            mkdir(_dir) if not self._exist_check(_dir) else None

        return _dir

    @classmethod
    def _make_for_result(self, folder: str = None, root_dir: str = None):
        if folder is None:
            _date = utils.time_stemp(True)
        else:
            _date = folder
        return self._make(f"result/{_date}/", root_dir)

    @classmethod
    def _inside_search(self, searched_dir: str, search_option: str = "all", name: str = "*", ext: str = "*"):
        serch_all = search_option == "all"
        _component_name = "*" if serch_all else "*" + name + "*"
        _component_ext = "" if serch_all else (ext if ext[0] == "." else "." + ext)

        search_list = sorted(glob(self._slash_check(searched_dir) + _component_name + _component_ext))

        if search_option in ["directory", "dir"]:
            search_list = [data for data in search_list if self._exist_check(data)]
        elif search_option in ["file", ]:
            search_list = [data for data in search_list if file._exist_check(data)]

        return sorted(search_list)

    @staticmethod  # Not yet
    def _compare():
        _error.not_yet("directory._compare")
        # compare_obj = dir_checker(compare_obj, True)
        # compare_data = compare_obj.split(SLASH)
        # base_dir = dir_checker(base_dir, True)
        # base_data = base_dir.split(SLASH)

        # tmp_dir = "." + SLASH
        # same_count = 0

        # for _tmp_folder in base_data:
        #     if _tmp_folder in compare_data:
        #         same_count += 1
        #     else:
        #         break
        # if len(base_data) - same_count:
        #     for _ct in range(len(base_data) - same_count):
        #         tmp_dir += ".." + SLASH

        # for _folder in compare_data[same_count:]:
        #     tmp_dir += _folder + SLASH

    @staticmethod
    def _del():
        _error.not_yet("directory._del")

    @staticmethod
    def _copy():
        _error.not_yet("directory._copy")

    class server():
        @staticmethod
        def local_connect(ip_num: str, user_id: str, password: str, root_dir: str, mount_dir: str, is_container: bool = False):
            if is_container:
                # in later make function for docker container
                _error.not_yet("not support for container system at function server.local_connect")
                pass
            else:
                if directory.OS_THIS == directory.OS_WINDOW:
                    _command = f"NET USE {mount_dir}: \\\\{ip_num}\\{root_dir} {password} /user:{user_id}"

                elif directory.OS_THIS == directory.OS_UBUNTU:
                    _command = "sudo -S mount -t cifs -o uid=1000,gid=1000,"
                    _command += f"username={user_id},password={password} //{ip_num}/{root_dir} {mount_dir}"

                system(_command)
                return mount_dir + ":" if directory.OS_THIS == directory.OS_WINDOW else mount_dir

        @staticmethod
        def _unconnect(mounted_dir: str):
            if directory.OS_THIS == directory.OS_WINDOW:
                system("NET USE {}: /DELETE".format(mounted_dir))
            elif directory.OS_THIS == directory.OS_UBUNTU:
                system("fuser -ck {}".format(mounted_dir))
                system("sudo umount {}".format(mounted_dir))


class file():
    @classmethod
    def _exist_check(self, file_path: str) -> bool:
        return path.isfile(file_path)

    @classmethod
    def _name_from_path(self, file_path: str) -> str:
        last_companant = directory._slash_check(file_path, is_file=True).split(directory.SLASH)[-1]
        if self._exist_check(file_path) or last_companant != "":
            return last_companant
        else:
            return None

    @staticmethod
    def _extension_check(file_dir, exts: List[str], is_fix: bool = False):
        file_name = file._name_from_path(file_dir)
        is_positive = False

        if file_name is None:
            # !!!WARING!!! file directory not have file name
            _error.variable(
                function_name="file._extension_check",
                variable_list=["file_dir", ],
                AA="File name not exist in Entered Parameter 'file_dir'"
            ) if DEBUG else None

            # fix
            file_name = file_dir + "." + exts[0] if is_fix else None

        else:
            file_ext = file_name.split(".")[-1]
            is_positive = file_ext in exts

            # fix
            if (not is_positive) and is_fix:
                _tem_ct = file_name.find(".")
                replace_file_name = file_name[:_tem_ct] + "." + exts[0]
                file_name = file_dir.replace(file_name, replace_file_name)

        return [is_positive, file_name] if is_fix else is_positive

    @classmethod
    def _json(self, file_dir: str, file_name: str, data_dict: Dict = None, is_save: bool = False):
        """
        Args:
            save_dir        :
            file_name       :
            data_dict       :
        Returns:
            return (dict)   :
        """
        # directory check
        file_dir = directory._slash_check(file_dir)
        if not directory._exist_check(file_dir):
            if is_save:
                # !!!WARING!!! save directory not exist
                _error.variable(
                    function_name="file.json_file",
                    variable_list=["file_dir", ],
                    AA="Entered directory '{}' not exist. In first make that".format(file_dir))
                directory._make(file_dir)

            else:
                # !!!ERROR!!! load directory not exist
                _error.variable_stop(
                    function_name="file.json_file",
                    variable_list=["file_dir", ],
                    AA="Entered directory '{}' not exist".format(file_dir)
                )

        # file_name check
        _, file_name = file._extension_check(file_name, ["json", ], True)

        # json file process load or save
        if is_save:
            # json file save
            _file = open(file_dir + file_name, "w")
            json.dump(data_dict, _file, indent=4)
        else:
            # json file load
            if self._exist_check(file_dir + file_name):
                # json file exist
                _file = open(file_dir + file_name, "r")
                return json.load(_file)

            else:
                # !!!ERROR!!! load json file not exist
                _error.variable_stop(
                    function_name="file.json_file",
                    variable_list=["file_dir", "file_name"],
                    AA="Load file '{}' not exist".format(file_dir + file_name)
                )

    @classmethod
    def _yaml(self, file_dir: str, file_name: str, data_dict: Dict = None, is_save: bool = False):
        """
        Args:
            save_dir        :
            file_name       :
            data_dict       :
        Returns:
            return (dict)   :
        """
        # directory check
        file_dir = directory._slash_check(file_dir)
        if not directory._exist_check(file_dir):
            if is_save:
                # !!!WARING!!! save directory not exist
                _error.variable(
                    function_name="file.yaml_file",
                    variable_list=["file_dir", ],
                    AA="Entered directory '{}' not exist. In first make that".format(file_dir))
                directory._make(file_dir)

            else:
                # !!!ERROR!!! load directory not exist
                _error.variable_stop(
                    function_name="file.yaml_file",
                    variable_list=["file_dir", ],
                    AA="Entered directory '{}' not exist".format(file_dir)
                )

        # file_name check
        _, file_name = file._extension_check(file_name, ["yml", 'yaml'], True)

        # yaml file process load or save
        if is_save:
            # json file save
            _file = open(file_dir + file_name, "w")
            yaml.dump(data_dict, _file, indent=4)
        else:
            # yaml file load
            if self._exist_check(file_dir + file_name):
                # yaml file exist
                _file = open(file_dir + file_name, "r")
                return yaml.load(_file)

            else:
                # !!!ERROR!!! load yaml file not exist
                _error.variable_stop(
                    function_name="file.yaml_file",
                    variable_list=["file_dir", "file_name"],
                    AA="Load file '{}' not exist".format(file_dir + file_name)
                )

    @staticmethod
    def _xml(file_dir, file_name, data_dict=None, is_save=False):
        pass

    @staticmethod
    def _del():
        _error.not_yet("file._del")

    @staticmethod
    def _copy_to(dir, file):
        _error.not_yet("file._copy_to")


class utils():
    @staticmethod
    def Progress_Bar(iteration: int, total: int, prefix: str = '', suffix: str = '', decimals: int = 1, length: int = 100, fill: str = '█'):
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

    @staticmethod
    def time_stemp(is_text=False):
        return time.strftime("%Y-%m-%d", time.localtime()) if is_text else time.time()


class tool_for():
    class _list():
        @staticmethod
        def is_num_over_range(target: list, obj_num: Union[int, list, range]) -> bool:
            """
            Arg:\n
                target (list) : \n
                obj_num (int, list[int], range) : \n
            """
            if isinstance(obj_num, list):
                _max = max(obj_num)
            elif isinstance(obj_num, int):
                _max = obj_num
                obj_num = [obj_num, ]
            elif isinstance(obj_num, range):
                _max = max(obj_num)
            else:
                # !!!ERROR!!! wrong type entered
                _error.data_type(
                    function_name="tool.list_tool.is_num_over_range",
                    variable_list=["obj_num", ],
                    AA="Error in parameter 'obj_num'.\n \
                        'obj_dirs' has unsuitable type data"
                )
                return True

            return _max > (len(target) + 1)

        @staticmethod
        def del_obj(target: list, obj_num: Union[int, list, range]):
            """
            Arg:
                target (list) : \n
                obj_num (int, list[int], range) : \n
            """
            if isinstance(obj_num, (list, range)):
                for _ct, _num in enumerate(obj_num):
                    del target[_num - _ct]
            elif isinstance(obj_num, int):
                del target[obj_num]

        @staticmethod
        def clear_list(target: list):
            del target[:]

    # class _dict():
    #     @classmethod
    #     def dict_to_labeling_holder(cls, dictionary: Dict[str, Union[Dict, str, list]], _root: Dict, Endpoint_holder: Any = None):
    #         for _node_name in dictionary.keys():
    #             _node_value = dictionary[_node_name]

    #         if isinstance(_node_value, dict):  # Nodes exist over 2 levels below.
    #             _root[_node_name] = cls.dict_to_labeling_holder(_node_value, {})

    #         elif isinstance(_node_value, list):  # Nodes exist in 1 level below.
    #             _root[_node_name] = {}
    #             for _name in _node_value:
    #                 _root[_node_name][_name] = Endpoint_holder

    #         elif isinstance(_node_value, str):  # end point node
    #             _root[_node_name] = {}
    #             _root[_node_name][_node_value] = Endpoint_holder

    #         else:
    #             pass

    #         return _root
