"""
File object
=====
    Some custom error folder for make python custom utils.

Requirement
=====
    None

"""
# Import module
# stand alone module


class Custom_error():
    def __init__(self, module_name, file_name) -> None:
        self.error_text =\
            "!!!!!ERROR!!!!!\n" +\
            "Module: {}\n".format(module_name) +\
            "File: {}\n".format(file_name) +\
            "Function: {}\n" +\
            "Error type: {}\n" +\
            "Detail: {}\n" +\
            "!!!!!ERROR!!!!!\n"

        self.warring_text =\
            "!!!!!WARRING!!!!!\n" +\
            "Module: {}\n".format(module_name) +\
            "File: {}\n".format(file_name) +\
            "Function: {}\n" +\
            "WARRING type: {}\n" +\
            "Detail: {}\n" +\
            "!!!!!WARRING!!!!!\n"

    def variable(self, function_name, detail):
        print(self.warring_text.format(function_name, "variable", detail))

    def variable_stop(self, function_name, detail):
        assert False, self.warring_text.format(function_name, "variable", detail)


def Custom_Variable_Error(loacation: str, parameters: list, detail: str = None):
    """
    Args:
        Funtion_name:
        parameters:
        details:
    Returns:
        return  :   Project folder Name
    """
    _is_detail = detail is not None
    _paras = [parameters, ] if type(parameters) != list else parameters

    printed_line = "Sorry. in this {} code, about the {} have some problem\n\t Detail : {}" if _is_detail\
        else "Sorry. in this {} code, about the {} have some problem\n"

    _error_paras_text = ""
    for parameter in _paras:
        _error_paras_text += parameter + ", "
    _error_paras_text = _error_paras_text[:-2]

    printed_line = printed_line.format(loacation, _error_paras_text, detail) if _is_detail\
        else printed_line.format(loacation, _error_paras_text)

    assert False, "Custom_Variable_Error\n" + printed_line


def Not_yet_Error(loacation, detail):
    assert False, "Sorry. in this {} code, about the {} is Not yet. m(__)m".format(loacation, detail)


def No_model_Error(Model_dir):
    assert False, "There was a problem with {} model.\n Please check again.".format(Model_dir)


def load_success():
    print("!!! custom python module ais_utils _error load Success !!!")
