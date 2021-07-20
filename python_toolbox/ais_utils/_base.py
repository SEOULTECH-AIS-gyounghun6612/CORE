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
import datetime
import platform
import shutil

from glob import glob
from os import error, path, stat, system, getcwd, mkdir, remove

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
    def _devide(self, directory, point):
        _dir = self._slash_check(directory)
        _comp = _dir.split(self.SLASH)[:-1]

        _front = ""
        _back = ""
        for _data in _comp[:point]:
            _front += _data + self.SLASH
        for _data in _comp[point:]:
            _back += _data + self.SLASH

        return _front, _back

    @staticmethod
    def _exist_check(directory, is_file=False):
        return path.isfile(directory) if is_file\
            else path.isdir(directory._slash_check(directory))

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
            _root = directory._slash_check(root_dir)
            if not directory._exist_check(_root):
                _front, _back = directory._devide(_root, -1)
                directory._make(_back, _front)
        else:
            # use relartion
            _root = ""
            for _ct, _data in enumerate(obj_dirs):
                if _data.find(self.RELARTION):
                    obj_dirs[_ct] = self.RELARTION + _data

        # make directory
        for _ct, _data in enumerate(obj_dirs):
            _tem_dir = directory._slash_check(_root + self.SLASH + _data)

            if not directory._exist_check(_tem_dir):
                mkdir(_tem_dir)
            obj_dirs[_ct] = _tem_dir

        return obj_dirs

    @staticmethod
    def _inside_serch():
        pass

    @staticmethod
    def _compare():
        pass

    @staticmethod
    def _del():
        pass

    @staticmethod
    def _copy():
        pass


class file():
    @staticmethod
    def _from_directory(dir):
        _last_component = dir.split(directory.SLASH)[-1]
        if _last_component.find(".") == -1:
            return None
        else:
            return _last_component

    @staticmethod
    def _extension_check(file_dir, exts, is_fix=False):
        file_name = file._from_directory(file_dir)
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
    def _copy():
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

# ----- PROGRAM START WORKING IN BELOW LINE ----- #
"""
Custom function about directory
=====
"""
# CONSTANT
OS_THIS = platform.system()
OS_WINDOW = "Windows"
OS_UBUNTU = "Linux"
if OS_THIS == OS_WINDOW:
    SLASH = "\\"
elif OS_THIS == OS_UBUNTU:
    SLASH = "/"

SERVER_LOCAL_IP = "192.168.0.3"
DIR_CODE_N_RESULT = "Code_n_Result" + SLASH
DIR_DATASET = "Datasets" + SLASH
DIR_PUBLIC = "Public" + SLASH

COPY = 0
MOVE = 1
DLETET = 2


def connect_AIS_server(
        user_id: str, password: str, root_dir: str, mount_dir: str, is_local: bool = True) -> str:
    """
    Args:
        user_id     :   Your AIS server ID
        password    :   Your AIS server password
        root_dir    :   Mounted AIS server root dir. Read detail information in under block
            DIR_CODE_N_RESULT : Sector for save code n result
            DIR_DATASET : Sector for datasets load n save
            DIR_PUBLIC : Sector for else
        mount_dir   :   Where in your computer, for mounted the server. If use Ubuntu, directory,
                        else If use Window, drive letter
        is_local    :   If you connect in loacal this param is True. else, set False
    Returns:
        mount_dir : Mounted directory
    """
    if not is_local:
        raise _e.Not_yet_Error(
            loacation="connect_AIS_server",
            detail="About External IP connection connet"
        )

    _command = ""

    if OS_THIS == OS_WINDOW:
        _command += "NET USE "
        _command += "{MountDir}: ".format(MountDir=mount_dir)
        _command += "\\\\{ServerLocalIp}\\{RootDir} ".format(
            ServerLocalIp=SERVER_LOCAL_IP,
            RootDir=root_dir
        )
        _command += "{Password} ".format(Password=password)
        _command += "/user:{UserName}".format(UserName=user_id)

    elif OS_THIS == OS_UBUNTU:
        # when use Ubuntu, if want connect AIS server in local network
        _command += "sudo -S mount -t cifs -o username={UserName}".format(UserName=user_id)
        _command += ",password={Password}".format(Password=password)
        _command += ",uid=1000,gid=1000 "
        _command += "//{ServerLocalIp}/{RootDir} {MountDir}".format(
            ServerLocalIp=SERVER_LOCAL_IP,
            RootDir=root_dir,
            MountDir=mount_dir
        )
    system(_command)
    return mount_dir + ":" if OS_THIS == OS_WINDOW else mount_dir


def disconnect_AIS_server(mount_dir: str) -> None:
    """
    Args:
        mount_dir   :   Disconnected Dir
    Returns:
             None
    """
    if OS_THIS == OS_WINDOW:
        system("NET USE {}: /DELETE".format(mount_dir))
    elif OS_THIS == OS_UBUNTU:
        system("fuser -ck {MountDir}".format(MountDir=mount_dir))
        system("sudo umount {MountDir}".format(MountDir=mount_dir))


def dir_maker(obj_dirs: list or dict, root_dir: str = None) -> str:
    """
    Args:
        obj_dir     :   maked directory
        root_dir    :   root directory for maked directory
    Returns:
        maked_dir   :   maked folder's directory
    """

    _root = "./" if root_dir is None else (root_dir if root_dir[-1] == SLASH else root_dir + SLASH)

    if type(obj_dirs) == dict:
        _tmp_keys = obj_dirs.keys()
        for _key in _tmp_keys:
            maked_dir = _root + _key
            if not path.isdir(maked_dir):
                mkdir(maked_dir)
            dir_maker(obj_dirs[_key], maked_dir)

    elif type(obj_dirs) == list:
        for obj_dir in obj_dirs:
            maked_dir = _root + obj_dir
            if not path.isdir(maked_dir):
                mkdir(maked_dir)

    elif type(obj_dirs) == str:  # make the dir and return it
        maked_dir = _root + obj_dirs
        if not path.isdir(maked_dir):
            mkdir(maked_dir)

        return maked_dir
    else:
        _e.Custom_Variable_Error(
            loacation="ais_utils._base.dir_maker",
            parameters=["obj_dirs", ],
            detail="Parameter 'obj_dirs' type is must be in dict, list and str")


def result_dir_maker(result_dir: str or list = None, result_root: str = None) -> str:
    """
    Args:
        result_dir      :   maked result directory
        result_root     :   root directory for maked directory
    Returns:
        return          :   maked folder's directory
    """
    _root = dir_maker(root_dir=result_root, obj_dirs="") if result_root is not None \
        else dir_maker(obj_dirs="Result")

    _date_str = datetime.datetime.now().strftime('%Y_%m_%d')

    if type(result_dir) == list:
        for _dir in result_dir:
            _result_dir = _dir if _dir is not None else "{}{}".format(_date_str, SLASH)
            _root = dir_maker(_result_dir, _root)

    elif type(result_dir) == str:
        _result_dir = _dir if _dir is not None else "{}{}".format(_date_str, SLASH)
        _root = dir_maker(_result_dir, _root)

    return _root


def dir_work(obj_dir: str, mode: int, dst_dir: str = None):
    if mode == DLETET:
        shutil.rmtree(obj_dir)
    else:
        if dst_dir is None:
            _e.Custom_Variable_Error(
                loacation="ais_utils._base.dir_work",
                parameters="dst_dir")
        shutil.copytree(obj_dir, dst_dir)

        if mode == MOVE:
            shutil.rmtree(obj_dir)


def get_work_folder(is_last_dir=True):
    """
    Args:
        None
    Returns:
        return  :   Project folder Name
    """
    if is_last_dir:
        return getcwd().split(SLASH)[-1] + SLASH
    else:
        return getcwd() + SLASH


def dir_compare(base_dir, compare_obj, is_dir=True):
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

    return dir_checker(tmp_dir, not is_dir)


def dir_compare_to_work_folder(compare_dir):
    return dir_compare(get_work_folder(False), compare_dir)


def dir_checker(checked_dir, is_reverse):
    if is_reverse:
        return checked_dir if checked_dir[-1] != SLASH else checked_dir[:-1]
    else:
        return checked_dir if checked_dir[-1] == SLASH else checked_dir + SLASH


def get_dir_list(obj_dir: str, dir_str: str = "*", is_last_dir: bool = False) -> list:
    """
    Args:
        obj_dir             :
        dir_str             :
        is_last_dir         :
    Returns:
        _dir_list (list)   :   Project folder Name
    """
    _dir_list = sorted(glob(obj_dir + dir_str))
    if is_last_dir:
        _dir_list = [_temp.split(SLASH)[-1] for _temp in _dir_list]
    return _dir_list


"""
Custom function about file R/W
=====
"""
# CONSTANT
#    EMPTY


# FUNCTION
def get_file_list(obj_dir: str, file_str: str = "*", ext: str = ".*", just_file_name: bool = False) -> list:
    """
    Args:
        obj_dir             :
        file_str            :
        ext                 :
        just_file_name      :
    Returns:
        _file_list (list)   :   Project folder Name
    """
    _ext = "." + ext if ext[0] != "." else ext
    _file_list = sorted(glob(obj_dir + file_str + _ext))
    if just_file_name:
        _file_list = [_temp.split(SLASH)[-1] for _temp in _file_list]
    return _file_list


def json_file(save_dir: str, file_name: str = None, data_dict: dict = None, is_save: bool = False) -> None:
    """
    Args:
        save_dir        :
        file_name       :
        data_dict       :
    Returns:
        return (dict)   :
    """

    if file_name is None:
        if save_dir.split(SLASH)[-1].split(".")[-1] != "json":
            _e.Custom_Variable_Error(
                loacation="ais_utils._base.json_file",
                parameters=["save_dir", "file_name"],
                details="When 'file_name' set the None, setted data in 'save_dir',\
                    must be have JSON file name."
            )
        else:
            _file_dir = save_dir
    else:
        _file_dir = save_dir + file_name

    if is_save:
        _file = open(_file_dir, "w")
        json.dump(data_dict, _file, indent=4)
    else:
        _file = open(_file_dir, "r")
        return json.load(_file)


def file_work(obj_file: str, mode: int, dst_dir: str = None):
    if mode == DLETET:
        remove(obj_file)
    else:
        if dst_dir is None:
            _e.Custom_Variable_Error(
                loacation="ais_utils._base.file_work",
                parameters="dst_dir")
        shutil.copyfile(obj_file, dst_dir)
        if mode == MOVE:
            remove(obj_file)


"""
Custom function about precess debug
=====
"""
# CONSTANT
#    EMPTY


# FUNCTION
def load_success():
    print("!!! custom python module ais_utils _base load Success !!!")
