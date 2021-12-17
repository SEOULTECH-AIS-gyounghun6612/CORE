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
import platform

from glob import glob
from os import path, system, getcwd, mkdir

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
    RELARTION = "." + SLASH

    @classmethod  # fix it
    def _slash_check(self, directory, is_file=False):
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
    def _exist_check(self, directory, is_file=False):
        return path.isfile(directory) if is_file else path.isdir(directory)

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
                AA="Error in parameter 'obj_dirs'.\n \
                    'obj_dirs' has unsuitable type data"
            )

        # root dir check
        if root_dir is not None:
            # _root check
            _root = self._slash_check(root_dir)
            if not self._exist_check(_root):
                _front, _back = self._devide(_root, -1)
                self._make(_back, _front)
        else:
            # use relartion from __file__
            _root = self.RELARTION
            for _ct, _data in enumerate(obj_dirs):
                if not _data.find(self.RELARTION):
                    obj_dirs[_ct] = _data[len(self.RELARTION):]

        # make directory
        for _ct, _data in enumerate(obj_dirs):
            _dir_componant = self._slash_check(_data).split(self.SLASH)

            _tem_dir = _root
            for _componant in _dir_componant:
                _tem_dir = self._slash_check(_tem_dir + _componant)

                if not self._exist_check(_tem_dir):
                    mkdir(_tem_dir)
            obj_dirs[_ct] = _tem_dir

        return obj_dirs

    @classmethod
    def _make_for_result(self, ):
        _error.not_yet("directory._make_for_result")

    @classmethod
    def _inside_search(self, searched_dir, search_option="file", name="*", ext="*"):
        _dir = self._slash_check(searched_dir)

        serch_all = search_option == "all"
        _component_name = "*" if serch_all else "*" + name + "*"
        _component_ext = "" if (serch_all or ext == "*") else (ext if ext[0] == "." else "." + ext)

        _filter = _dir + _component_name + _component_ext

        search_list = sorted(glob(_filter))

        if search_option == "directory":
            search_list = [data for data in search_list if self._exist_check(data)]

        elif search_option == "file":
            search_list = [data for data in search_list if self._exist_check(data, True)]

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

    @classmethod
    def _get_main(self, just_name=True):
        return self._devide(getcwd())[-1] if just_name else self._slash_check(getcwd())

    @staticmethod
    def _del():
        _error.not_yet("directory._del")

    @staticmethod
    def _copy():
        _error.not_yet("directory._copy")


class file():
    @staticmethod
    def _name_from_directory(dir):
        last_companant = directory._slash_check(dir, is_file=True).split(directory.SLASH)[-1]
        if directory._exist_check(dir, True) or last_companant != "":
            return last_companant
        else:
            return None

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
            if directory._exist_check(file_dir + file_name, True):
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

    @staticmethod
    def _xml(file_dir, file_name, data_dict=None, is_save=False):
        pass

    @staticmethod
    def _del():
        _error.not_yet("file._del")

    @staticmethod
    def _copy_to(dir, file):
        _error.not_yet("file._copy_to")


class process_in_code():
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
    def local_connect(loacal_ip, user_id, password, root_dir, mount_dir, is_container=False):
        if is_container:
            # in later make function for docker container
            _error.not_yet("not support for container system at function server.local_connect")
            pass
        else:
            _command = ""
            if directory.OS_THIS == directory.OS_WINDOW:
                _command += "NET USE "
                _command += "{MountDir}: ".format(MountDir=mount_dir)
                _command += "\\\\{ServerLocalIp}\\{RootDir} ".format(
                    ServerLocalIp=loacal_ip,
                    RootDir=root_dir
                )
                _command += "{Password} ".format(Password=password)
                _command += "/user:{UserName}".format(UserName=user_id)

            elif directory.OS_THIS == directory.OS_UBUNTU:
                # when use Ubuntu, if want connect AIS server in local network
                _command += "sudo -S mount -t cifs -o username={UserName}".format(UserName=user_id)
                _command += ",password={Password}".format(Password=password)
                _command += ",uid=1000,gid=1000 "
                _command += "//{ServerLocalIp}/{RootDir} {MountDir}".format(
                    ServerLocalIp=loacal_ip,
                    RootDir=root_dir,
                    MountDir=mount_dir
                )
            system(_command)
            return mount_dir + ":" if directory.OS_THIS == directory.OS_WINDOW else mount_dir

    @staticmethod
    def _unconnect(mounted_dir):
        if directory.OS_THIS == directory.OS_WINDOW:
            system("NET USE {}: /DELETE".format(mounted_dir))
        elif directory.OS_THIS == directory.OS_UBUNTU:
            system("fuser -ck {}".format(mounted_dir))
            system("sudo umount {}".format(mounted_dir))


# FUNCTION
def load_check():
    print("!!! custom python module ais_utils _base load Success !!!")
