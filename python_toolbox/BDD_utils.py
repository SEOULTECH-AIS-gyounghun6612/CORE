import numpy as np
import cv2_utils as cv_etc


BDD_100K_LABELS = [
    [(0x00, 0x00, 0x00), 'unlabeled or statick'],
    [(0x00, 0x4A, 0x6F), 'dynamic'],
    [(0x51, 0x00, 0x51), 'ground'],
    [(0xA0, 0xAA, 0xFA), 'parking'],
    [(0x8C, 0x96, 0xE6), 'rail track'],
    [(0x80, 0x40, 0x80), 'road'],
    [(0xE8, 0x23, 0xF4), 'sidewalk'],
    [(0x64, 0x64, 0x96), 'bridge'],
    [(0x46, 0x46, 0x46), 'building'],
    [(0x99, 0x99, 0xBE), 'fence'],
    [(0xB4, 0x64, 0xB4), 'garage'],
    [(0xB4, 0xA5, 0xB4), 'guard rail'],
    [(0x5A, 0x78, 0x96), 'tunnel'],
    [(0x9C, 0x66, 0x66), 'wall'],
    [(0x64, 0xAA, 0xFA), 'banner'],
    [(0xFA, 0xDC, 0xDC), 'billboard'],
    [(0x00, 0xA5, 0xFF), 'lane divider'],
    [(0x99, 0x99, 0x99), 'pole'],
    [(0x64, 0xDC, 0xDC), 'street light'],
    [(0x00, 0x46, 0xFF), 'traffic cone'],
    [(0xDC, 0xDC, 0xDC), 'traffic device'],
    [(0x1E, 0xAA, 0xFA), 'traffic light'],
    [(0x00, 0xDC, 0xDC), 'traffic sign'],
    [(0xFA, 0xAA, 0xFA), 'traffic sign frame'],
    [(0x98, 0xFB, 0x98), 'terrain'],
    [(0x23, 0x8E, 0x6B), 'vegetation'],
    [(0xB4, 0x82, 0x46), 'sky'],
    [(0x3C, 0x14, 0xDC), 'person or parking sign'],
    [(0x00, 0x00, 0xFF), 'rider'],
    [(0x20, 0x0B, 0x77), 'bicycle'],
    [(0x64, 0x3C, 0x00), 'bus'],
    [(0x8E, 0x00, 0x00), 'car'],
    [(0x5A, 0x00, 0x00), 'caravan'],
    [(0xE6, 0x00, 0x00), 'motorcycle'],
    [(0x6E, 0x00, 0x00), 'trailer'],
    [(0x64, 0x50, 0x00), 'train'],
    [(0x46, 0x00, 0x00), 'truck'],
]


def make_img_2_seg_onehot(img):
    _h, _w, _c = np.shape(img)
    seg_onehot = np.zeros([_h, _w, len(BDD_100K_LABELS)], np.uint8)
    seg_edge = np.zeros([_h, _w, len(BDD_100K_LABELS)], np.uint8)

    for _count, _label in enumerate(BDD_100K_LABELS):
        _this_train_id = _count
        _this_train_label_color = _label[0]
        _masked_label = np.where(np.all(img == _this_train_label_color, 2), 1, 0)

        seg_onehot[:, :, _count] = _masked_label

        _lt = _masked_label[: -2, : -2]
        _ct = _masked_label[: -2, 1: -1]
        _rt = _masked_label[: -2, 2:]
        _lm = _masked_label[1: -1, : -2]
        _cm = _masked_label[1: -1, 1: -1]
        _rm = _masked_label[1: -1, 2:]
        _lb = _masked_label[2:, : -2]
        _cb = _masked_label[2:, 1: -1]
        _rb = _masked_label[2:, 2:]

        _temp = _lt + _ct + _rt + _lm - (8 * _cm) + _rm + _lb + _cb + _rb
        seg_edge[1: -1, 1: -1, _this_train_id] = \
            np.logical_or(seg_edge[1: -1, 1: -1, _this_train_id], np.where(_temp != 0, 1, 0))

    edge_percentage = np.sum(np.sqrt(seg_edge * seg_edge), 2)
    edge_percentage = np.array(
        cv_etc.PIXEL_MAX * (edge_percentage / np.max(edge_percentage)), np.uint8)

    return seg_onehot, edge_percentage


def make_seg_image(seg_data, is_display=True, is_last_channel=False):
    if is_last_channel:
        _ch_num = 2
    else:
        _ch_num = 0
    if len(np.shape(seg_data)) == 4:
        _argmax_output = np.argmax(seg_data, axis=_ch_num + 1)[0]
    elif len(np.shape(seg_data)) == 3:
        _argmax_output = np.argmax(seg_data, axis=_ch_num)

    _h, _w = np.shape(_argmax_output)
    _temp_labels_color = np.array(list(np.array(BDD_100K_LABELS)[:, 0]), dtype=np.uint8)
    output_img = _temp_labels_color[_argmax_output]
    # debug
    if is_display:
        cv_etc.img_render(output_img, "test", False, False)
    return output_img


def cal_acc(output, target):
    _argmax_output = np.argmax(output, axis=1)
    _argmax_target = np.argmax(target, axis=1)
    correct = (_argmax_output == _argmax_target)[0]

    return np.mean(correct)
