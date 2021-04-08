from PyQt5.QtWidgets import\
    QWidget, QDialog, QMessageBox, QGroupBox, QFileDialog,\
    QFrame, QSizePolicy, QHeaderView

from PyQt5.QtGui import QImage, QPixmap

from ais_utils import _base
from ais_utils import _cv2

MESSAGE_ICON_FLAG = {
    "no": QMessageBox.NoIcon,
    "question": QMessageBox.Question,
    "info": QMessageBox.Information,
    "warning": QMessageBox.Warning,
    "critical": QMessageBox.Critical}
MESSAGE_BUTTON = {
    "OK": QMessageBox.Ok,
    "NO": QMessageBox.No,
    "RE": QMessageBox.Retry}


"""
CUSTOM WIDGET
====================
"""


class page(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        print("please make the init function")


class subpage(QDialog):
    def __init__(self):
        super().__init__()
        self.initUI()

    def onOKButtonClicked(self):
        self.accept()

    def onCancelButtonClicked(self):
        self.reject()

    def showModal(self):
        return super().exec_()

    def initUI(self):
        print("please make the init function")


class sub_section():
    def __init__(self, name, default_check_option=None):
        self.cover = QGroupBox(name)
        if default_check_option is not None:
            self.cover.setCheckable(True)
            self.cover.setChecked(default_check_option)
        self.layout = None

    def set_the_gui(self):
        self.cover.setLayout(self.layout)
        self.cover.setFlat(True)


class h_line(QFrame):
    def __init__(self):
        super().__init__()
        self.setMinimumWidth(1)
        self.setFixedHeight(20)
        self.setFrameShape(QFrame.HLine)
        self.setFrameShadow(QFrame.Sunken)
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum)
        return


class v_line(QFrame):
    def __init__(self):
        super().__init__()
        self.setFixedWidth(20)
        self.setMinimumHeight(1)
        self.setFrameShape(QFrame.VLine)
        self.setFrameShadow(QFrame.Sunken)
        self.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Preferred)
        return


"""
CUSTOM FUNCTION
====================
"""


def opencv_img_converter(img_file):
    img = _cv2.read_img(
        file_dir=img_file,
        color_type=_cv2.COLOR_RGB)
    _h, _w, _c = img.shape
    qImg = QImage(img.data, _w, _h, _w * _c, QImage.Format_RGB888)
    return QPixmap.fromImage(qImg)


def table_init(table_widget, row, H_header=None):
    if H_header is not None:
        # when use table init
        table_widget.clear()
        table_widget.setColumnCount(len(H_header))
        table_widget.setHorizontalHeaderLabels(H_header)
        table_widget.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
    else:
        # when use table clear
        table_widget.clearContents()

    table_widget.setRowCount(row)


def file_n_dir_dialog(parent_widget, dialog_title, default_dir, ext_filter, error_massage):
    # extract file list
    if default_dir is not None:
        default_dir = default_dir if default_dir[-1] == _base.SLASH else default_dir + _base.SLASH
    else:
        default_dir = "." + _base.SLASH

    if ext_filter == "dir":
        _get_data = QFileDialog.getExistingDirectory(
            parent=parent_widget,
            caption=dialog_title,
            directory=default_dir)
    else:
        _get_data = QFileDialog.getOpenFileNames(
            parent=parent_widget,
            caption=dialog_title,
            directory=default_dir,
            filter=ext_filter)[0]

    if not len(_get_data):
        _answer = make_message_box(
            title="Get File List Error" if ext_filter != "dir" else "Get Directory Error",
            message=error_massage,
            icon_flag="warning",
            bt_flags=["OK", "RE"]
        )
        if _answer == QMessageBox.Retry:
            default_dir, _get_data = file_n_dir_dialog(
                parent_widget,
                dialog_title,
                default_dir,
                ext_filter,
                error_massage)

    return default_dir, _get_data


"""
CUSTOM DIALOG
====================
"""


def make_message_box(title, message, icon_flag, bt_flags):

    assert icon_flag in MESSAGE_ICON_FLAG.keys()
    assert all([_tmp in MESSAGE_BUTTON.keys() for _tmp in bt_flags])

    msg = QMessageBox()
    msg.setIcon(MESSAGE_ICON_FLAG[icon_flag])
    msg.setWindowTitle(title)
    msg.setText(message)

    bt_flag = 0
    for _tmp_bt_flag in bt_flags:
        bt_flag = bt_flag | MESSAGE_BUTTON[_tmp_bt_flag]

    msg.setStandardButtons(bt_flag)
    return msg.exec_()


# def shorcut_letter_add(label_text, shorcut_latter):
#     _latter = shorcut_latter.split("+")[-1] if "+" in shorcut_latter\
#         else shorcut_latter


def load_success():
    print("!!! custom python module ais_gui _qt__utils load Success !!!")
