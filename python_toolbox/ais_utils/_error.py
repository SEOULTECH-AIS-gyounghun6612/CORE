"""
File object
=====
    Some custom error folder for make python custom utils.

Requirement
=====
    None

"""
# Import module
# Empty


def Custom_Variable_Error(loacation: str, parameters: list, details: list = None):
    """
    Args:
        Funtion_name:
        parameters:
        details:
    Returns:
        return  :   Project folder Name
    """
    _is_detail = details is not None
    _paras = [parameters, ] if type(parameters) != list else parameters
    _details = ([details, ] if type(details) != list else details) if _is_detail else None

    assert len(_paras) == len(_details)
    printed_datas = zip(_paras, _details) if _is_detail\
        else parameters

    printed_line = "Sorry. in this {} code, about the {} have some problem\n" if _is_detail\
        else "Sorry. in this {} code, about the {} have some problem\n\t Detail : {}"

    error_text = ""
    for parameter in printed_datas:
        error_text += printed_line.format(loacation, parameter) if _is_detail\
            else printed_line.format(loacation, parameter[0], parameter[1])
    assert False, "Custom_Variable_Error\n" + error_text


def Not_yet_Error(loacation, detail):
    assert False, "Sorry. in this {} code, about the {} is Not yet. m(__)m".format(loacation, detail)


def No_model_Error(Model_dir):
    assert False, "There was a problem with {} model.\n Please check again.".format(Model_dir)


def load_success():
    print("!!! custom python module AIS_utils_error load Success !!!")
