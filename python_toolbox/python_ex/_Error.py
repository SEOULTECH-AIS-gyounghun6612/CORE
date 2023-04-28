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
# import traceback
# update error message test use to "traceback.format_stack()"


class Custom_error():
    # Detail_text
    NOT_YET = \
        "For now, this part is not yet complete."

    VARIABLE = \
        "The variable causing the problem: \n \
        {}\n \
        Anotation: \n \
        {}"

    DATA_TYPE = \
        "The variable causing the problem: \n \
        {}\n \
        Anotation: \n \
        {}"

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

    def not_yet(self, function_name):
        assert False, self.error_text.format(function_name, "Not yet", self.NOT_YET)

    def variable(self, function_name, variable_list, AA):
        """
        Args:
            function_name   : (No narrative)
            variable_list   : (No narrative)
            AA              : Additional annotation
        Returns:
            None
        """
        print_text = self.VARIABLE.format(variable_list, AA)
        print(self.warring_text.format(function_name, "variable", print_text))

    def variable_stop(self, function_name, variable_list, AA):
        """
        Args:
            function_name   : (No narrative)
            variable_list   : (No narrative)
            AA              : Additional annotation
        Returns:
            None
        """
        print_text = self.VARIABLE.format(variable_list, AA)
        assert False, self.error_text.format(function_name, "variable", print_text)

    # fix it later
    def data_type(self, function_name, variable_list, AA):
        """
        Args:
            function_name   : (No narrative)
            variable_list   : (No narrative)
            AA              : Additional annotation
        Returns:
            None
        """
        print_text = self.DATA_TYPE.format(variable_list, AA)
        print(self.warring_text.format(function_name, "variable", print_text))

    # fix it later
    def data_type_stop(self, function_name, variable_list, AA):
        """
        Args:
            function_name   : (No narrative)
            variable_list   : (No narrative)
            AA              : Additional annotation
        Returns:
            None
        """
        print_text = self.DATA_TYPE.format(variable_list, AA)
        assert False, self.error_text.format(function_name, "variable", print_text)


def load_check():
    pass
