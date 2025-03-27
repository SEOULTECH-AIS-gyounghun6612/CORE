from . import  Parser, PARSER, Custom_Dataset


from python_ex.system import Path, File


class Kitii_Parser(Parser):
    def __init__(self, data_dir: str, **kwarg) -> None:
        super().__init__(data_dir, **kwarg)

    def Get_data_from(self, data_dir: str, **kwarg):
        _rgb_forlder = Path.Join()
        self.input_folder = input_folder
        self.start_idx = start_idx





        return super().Get_data_from(data_dir, **kwarg)




        _rgb_folder = os.path.join(
            f"{self.input_folder}", "sequences", f"{data_id:0>2d}")
        self.color_paths = sorted(
            glob.glob(os.path.join(_rgb_folder, "image_2", "*.png"))
        )
        self.color_paths_r = sorted(
            glob.glob(os.path.join(_rgb_folder, "image_3", "*.png"))
        )
        assert len(self.color_paths) == len(self.color_paths_r)
        self.color_paths = self.color_paths[start_idx:]
        self.color_paths_r = self.color_paths_r[start_idx:]
        self.n_img = len(self.color_paths)

        self.poses = []
        self.load_poses(
            os.path.join(
                f"{self.input_folder}", "poses", f"{data_id:0>2d}.txt"
            )
        )
        self.poses = self.poses[start_idx:]

        self.instrict = {}
        self.load_instrict(
            os.path.join(
                _rgb_folder, "calib.txt"
            )
        )



class KittiDataset(Custom_Dataset):
    def __init__(self, args, path, config):
        super().__init__(args, path, config)
        _parser = Kitii_Parser(
            config["Dataset"]["dataset_path"],
            config["Dataset"]["dataset_id"],
            config["Dataset"]["start_idx"]
        )
        self.num_imgs = _parser.n_img
        self.color_paths = _parser.color_paths
        self.color_paths_r = _parser.color_paths_r
        self.poses = _parser.poses

        self.is_stero = config["Dataset"]["sensor_type"] == "stereo"

        self.width = config["Dataset"]["Calibration"]["width"]
        self.height = config["Dataset"]["Calibration"]["height"]
        self.fx = _parser.instrict["fx"]
        self.fy = _parser.instrict["fy"]
        self.cx = _parser.instrict["cx"]
        self.cy = _parser.instrict["cy"]
        self.baseline = _parser.instrict["baseline"]

        self.fovx = focal2fov(self.fx, self.width)
        self.fovy = focal2fov(self.fy, self.height)

        self.K = np.array(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]]
        )

    def __getitem__(self, idx):
        color_path = self.color_paths[idx]
        image = cv2.imread(color_path, cv2.IMREAD_UNCHANGED)

        depth = None
        if self.is_stero:
            color_path_r = self.color_paths_r[idx]
            _gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image_r = cv2.imread(color_path_r, 0)

            stereo = cv2.StereoSGBM_create(
                minDisparity=0,
                numDisparities=64,
                blockSize=20
            )
            stereo.setUniquenessRatio(40)
            disparity = stereo.compute(image, image_r) / 16.0
            disparity[disparity == 0] = 1e10
            depth = self.fx * self.baseline / (
                disparity
            )  # Following ORB-SLAM2 config, baseline*fx
            depth[depth < 0] = 0

        pose = self.poses[idx]

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = (
            torch.from_numpy(image / 255.0)
            .clamp(0.0, 1.0)
            .permute(2, 0, 1)
            .to(device=self.device, dtype=self.dtype)
        )
        pose = torch.from_numpy(pose).to(device=self.device)

        return image, depth, pose
