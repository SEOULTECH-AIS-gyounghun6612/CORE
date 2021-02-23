"""
File object
=====
    When use CDnet-2014 dataset, be used custom function.

Requirement
=====
    numpy
    cv2_utils (custom)

File information
=====
    first edit date   : 2020/05/21
    first editer      : choi keonghun
    last edit date    : 2020/12/16
    last editer       : choi keonghun

Import module
=====
"""

import numpy as np
import cv2_utils as cv_etc

DATA_CATEGOEY = [
    "badWeather", "baseline", "cameraJitter", "dynamicBackground", "intermittentObjectMotion",
    "lowFramerate", "nightVideos", "PTZ", "shadow", "thermal", "turbulence"]
DATA_SUB_CATEGOEY = {
    "badWeather": [
        "blizzard", "skating", "snowFall", "wetSnow"],
    "baseline": [
        "highway", "office", "pedestrians", "PETS2006"],
    "cameraJitter": [
        "badminton", "boulevard", "sidewalk", "traffic"],
    "dynamicBackground": [
        "boats", "canoe", "fall", "fountain01", "fountain02", "overpass"],
    "intermittentObjectMotion": [
        "abandonedBox", "parking", "sofa", "streetLight", "tramstop", "winterDriveway"],
    "lowFramerate": [
        "port_0_17fps", "tramCrossroad_1fps", "tunnelExit_0_35fps", "turnpike_0_5fps"],
    "nightVideos": [
        "bridgeEntry", "busyBoulvard", "fluidHighway", "streetCornerAtNight", "tramStation",
        "winterStreet"],
    "PTZ": [
        "continuousPan", "intermittentPan", "twoPositionPTZCam", "zoomInZoomOut"],
    "shadow": [
        "backdoor", "bungalows", "busStation", "copyMachine", "cubicle", "peopleInShade"],
    "thermal": [
        "corridor", "diningRoom", "lakeSide", "library", "park"],
    "turbulence": [
        "turbulence0", "turbulence1", "turbulence2", "turbulence3"]}
CDNET_LABEL = {
    "static": 0,
    "Hard shadow": 50,  # BACK
    "Outside region of interest": 85,
    "Unknown motion": 170,  # IG
    "Motion": 255  # FORE
}


def extrect_CDnet_train_number(root_dir: str, package_size: int = 0) -> list:
    """
    Args:
        root_dir     : Each data diretory
        package_size : Number of input images to be matched per label image
    Returns:
        _start_num   : train image start number
        _end_num     : train image end number
    """
    _temporalROI = open(root_dir + "temporalROI.txt", "r")
    _start_end_txt = _temporalROI.readline().split(" ")
    _start_num = int(_start_end_txt[0]) - 1
    _end_num = int(_start_end_txt[-1])

    if package_size != 0:
        if _start_num - (package_size - 1) >= 0:
            _start_num -= package_size - 1
        else:
            _start_num = 0

    return [_start_num, _end_num]


def label_converter(
        label_img: np.array,
        convert_list: list,
        image_size: list):
    """
    Args:
        label_img       : Each data diretory
        convert_list    : Number of input images to be matched per label image
        image_size      :
        is_last_channel :
    Returns:
        _start_num   : train image start number
        _end_num     : train image end number
    """
    # list check
    assert(type(convert_list) == list)

    _convert_labels = []

    for _convert in convert_list:
        _convert_label_img = np.zeros_like(label_img, np.float)
        for _convert_text in _convert:
            _convert_label_img += np.where(label_img == CDNET_LABEL[_convert_text], 1., 0.)
        _convert_labels.append(_convert_label_img)

    if image_size is not None:
        for _ct, _cvt_label in enumerate(_convert_labels):
            _convert_labels[_ct] = np.round(
                cv_etc.img_resize(_cvt_label[0], image_size))[np.newaxis, :, :]

    return _convert_labels
