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
from os import path, system, getcwd, mkdir, remove

from . import _error as _e
# import _error as _e

# Set constant
DEBUG = False


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


# FUNCTION
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
            File_name="base_utils",
            Function="connect_AIS_server",
            Detail="About External IP connection connet"
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


def dir_maker(obj_dirs: list, root_dir: str = None) -> str:
    """
    Args:
        obj_dir     :   maked directory
        root_dir    :   root directory for maked directory
    Returns:
        maked_dir   :   maked folder's directory
    """

    if type(obj_dirs) != list:
        _obj_dirs = [obj_dirs]
    else:
        _obj_dirs = obj_dirs

    for _obj_dir in _obj_dirs:
        # slash check
        _obj_dir = _obj_dir if _obj_dir[-1] == SLASH else _obj_dir + SLASH

        # new_dir
        maked_dir = _obj_dir if root_dir is None\
            else (root_dir + _obj_dir if root_dir[-1] == SLASH else root_dir + SLASH + _obj_dir)
        if not path.isdir(maked_dir):
            mkdir(maked_dir)

        # use mkdir after dir check
        if not path.isdir(maked_dir):
            mkdir(maked_dir)

        root_dir = maked_dir
    return maked_dir


def result_dir_maker(result_dir: str = None, result_root: str = None) -> str:
    """
    Args:
        result_dir      :   maked result directory
        result_root     :   root directory for maked directory
    Returns:
        return          :   maked folder's directory
    """
    _result_root = dir_maker(obj_dir=result_root) if result_root is not None \
        else dir_maker(root_dir=".", obj_dir="Result")
    _result_dir = result_dir if result_dir is not None \
        else "{}{}".format(datetime.datetime.now().strftime('%Y_%m_%d'), SLASH)
    return dir_maker(_result_dir, _result_root)


def dir_work(obj_dir: str, mode: int, dst_dir: str = None):
    if mode == DLETET:
        shutil.rmtree(obj_dir)
    else:
        if dst_dir is None:
            _e.Variable_Error(" Function - {}, Varialbe - {}. Ckech it Again".format("dir_work", "dst_dir"))
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
        return getcwd().split("/")[-1] + SLASH
    else:
        return getcwd()


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


def json_file(save_dir: str, file_name: str, data_dict: dict, is_save: bool) -> None:
    """
    Args:
        save_dir        :
        file_name       :
        data_dict       :
    Returns:
        return (dict)   :
    """
    if is_save:
        _file = open(save_dir + file_name, "w")
        json.dump(data_dict, _file, indent=4)
    else:
        _file = open(save_dir + file_name, "r")
        return json.load(_file)


def file_work(obj_file: str, mode: int, dst_dir: str = None):
    if mode == DLETET:
        remove(obj_file)
    else:
        if dst_dir is None:
            _e.Variable_Error(" Function - {}, Varialbe - {}. Ckech it Again".format("file_work", "dst_dir"))
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


def load_success():
    print("!!! custom python module AIS_utils_base load Success !!!")
