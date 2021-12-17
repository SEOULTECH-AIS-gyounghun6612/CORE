import json

if __package__ == "":
    # if this file in local project
    import _base
    import _error as _e

else:
    # if this file in package folder
    from . import _base
    from . import _error as _e


# Set constant
DEBUG = False
_error = _e.Custom_error(
    module_name="ais_custom_utils_v 2.x",
    file_name="_json.py")


class file():
    @staticmethod
    def read(file_dir, file_name):
        if not _base.directory._exist_check(file_dir):
            # !!!WARING!!! save directory not exist
            _error.variable(
                function_name="file.json_file",
                variable_list=["file_dir", ],
                AA="Entered directory '{}' not exist. In first make that".format(file_dir))

            _base.directory._make(file_dir)

        _, file_name = _base.file._extension_check(file_name, ["json", ], True)

        # json file load
        if _base.directory._exist_check(file_dir + file_name, True):
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
    def write(file_dir, file_name, data):
        if not _base.directory._exist_check(file_dir):
            # !!!WARING!!! save directory not exist
            _error.variable(
                function_name="file.json_file",
                variable_list=["file_dir", ],
                AA="Entered directory '{}' not exist. In first make that".format(file_dir))

            _base.directory._make(file_dir)

        _, file_name = _base.file._extension_check(file_name, ["json", ], True)

        _file = open(file_dir + file_name, "w")
        json.dump(file_dir + file_name, _file, indent=4)


class data_process():
    @staticmethod
    def get_keys(diction):
        pass