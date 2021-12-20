from python_ex import _cv2


class canny_processed(_cv2.image_process):
    def __init__(self, origianl_image):
        super().__init__(origianl_image)
        self.pick_th = [0, 0]

    def doing_process(self, parameters):
        _temp_img = _cv2.cv2.blur(self.original_img, (3, 3))
        self.pick_th[0] = parameters[0] * 5
        self.pick_th[1] = parameters[1] * 5

        self.processed_img = _cv2.make_canny(
            img=_temp_img,
            high=parameters[0] * 5,
            low=parameters[1] * 5,
            is_First_ch=False,
            is_zero2one=False
        )
        return False, False


file_name = "./002cd38e-c7defded.png"

img_data = _cv2.read_img(
    file_dir=file_name,
    color_type=_cv2.COLOR_BGR)

process_block = canny_processed(img_data)
process_block_2 = canny_processed(img_data)


trackbar_high = _cv2.trackbar("high", [0, 100])
trackbar_low = _cv2.trackbar("low", [0, 100])

window = _cv2.trackbar_window(
    window_name=["prs_1", "prs_2"],
    trackbars=[trackbar_high, trackbar_low],
    image_process=[process_block, process_block_2]
)
