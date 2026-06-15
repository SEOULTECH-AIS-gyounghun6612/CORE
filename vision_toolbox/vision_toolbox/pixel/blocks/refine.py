import cv2
import numpy as np

from vision_toolbox.pipeline.core import Base_Step
from vision_toolbox.pixel.state import ImageVisionState
from vision_toolbox.pixel.functional.morphology import morphology_ex, fill_light_area
from vision_toolbox.pixel.functional.measure import find_contours

class MorphologyClean(Base_Step):
    """닫힘(Closing) 및 열림(Opening) 연산을 순차적으로 적용하여 노이즈 제거 및 영역을 정돈합니다."""
    def __init__(
        self, 
        close_kernel_size: int = 5, 
        close_iterations: int = 1,
        open_kernel_size: int = 5,
        open_iterations: int = 1,
        order: str = "close_first",
        name: str = "MorphologyClean"
    ):
        super().__init__(name=name)
        self.close_ksize = close_kernel_size
        self.close_iters = close_iterations
        self.open_ksize = open_kernel_size
        self.open_iters = open_iterations
        self.order = order

    def forward(self, state: ImageVisionState, *args, **kwargs) -> ImageVisionState:
        if state.mask is None or np.count_nonzero(state.mask) == 0:
            return state

        curr_mask = state.mask
        if self.order == "close_first":
            curr_mask = morphology_ex(curr_mask, cv2.MORPH_CLOSE, self.close_ksize, self.close_iters)
            curr_mask = morphology_ex(curr_mask, cv2.MORPH_OPEN, self.open_ksize, self.open_iters)
        else:
            curr_mask = morphology_ex(curr_mask, cv2.MORPH_OPEN, self.open_ksize, self.open_iters)
            curr_mask = morphology_ex(curr_mask, cv2.MORPH_CLOSE, self.close_ksize, self.close_iters)
            
        state.mask = curr_mask
        return state

class KeepLargestContour(Base_Step):
    """마스크 내 여러 덩어리 중 가장 면적이 큰 외곽선 영역만 남깁니다."""
    def forward(self, state: ImageVisionState, *args, **kwargs) -> ImageVisionState:
        if state.mask is None or np.count_nonzero(state.mask) == 0:
            return state

        contours = find_contours(state.mask, mode="max")
        if not contours:
            state.mask = np.zeros_like(state.mask)
            return state

        res_mask = np.zeros_like(state.mask)
        cv2.drawContours(res_mask, contours, -1, 255, thickness=cv2.FILLED)
        state.mask = res_mask
        return state

class KMeansCluster(Base_Step):
    """마스크 내 포인트들을 거리 기반으로 군집화(K-Means)하고 가장 큰 군집만 남깁니다."""
    def __init__(self, initial_k: int = 5, spatial_merge_ratio: float = 10.0, name: str = "KMeansCluster"):
        super().__init__(name=name)
        self.initial_k = initial_k
        self.spatial_merge_ratio = spatial_merge_ratio

    def forward(self, state: ImageVisionState, *args, **kwargs) -> ImageVisionState:
        if state.mask is None or np.count_nonzero(state.mask) == 0:
            return state

        mask = state.mask
        points = cv2.findNonZero(mask)
        if points is None:
            state.mask = np.zeros_like(mask)
            return state
            
        pts_data = points.reshape(-1, 2).astype(np.float32)
        
        if len(pts_data) < self.initial_k:
            return state

        MAX_POINTS = 5000
        use_sampling = len(pts_data) > MAX_POINTS
        
        if use_sampling:
            indices = np.random.choice(len(pts_data), MAX_POINTS, replace=False)
            pts_for_kmeans = pts_data[indices]
        else:
            pts_for_kmeans = pts_data

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, labels, centers = cv2.kmeans(pts_for_kmeans, self.initial_k, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
        
        h, w = state.image.shape[:2]
        diagonal = np.sqrt(h**2 + w**2)
        merge_threshold = diagonal / self.spatial_merge_ratio
        
        group_map = {i: i for i in range(self.initial_k)}
        for i in range(self.initial_k):
            for j in range(i + 1, self.initial_k):
                dist = np.linalg.norm(centers[i] - centers[j])
                if dist < merge_threshold:
                    root_i = group_map[i]
                    root_j = group_map[j]
                    if root_i != root_j:
                        target_group = min(root_i, root_j)
                        source_group = max(root_i, root_j)
                        for k in group_map:
                            if group_map[k] == source_group:
                                group_map[k] = target_group

        labels_flatten = labels.flatten()
        group_labels = np.array([group_map[l] for l in labels_flatten])
        counts = np.bincount(group_labels)
        largest_group_id = np.argmax(counts)
        
        valid_center_indices = {k for k, v in group_map.items() if v == largest_group_id}
        final_mask = np.zeros_like(mask)
        
        if use_sampling:
            dists = np.linalg.norm(pts_data[:, None, :] - centers[None, :, :], axis=2)
            nearest_center_idxs = np.argmin(dists, axis=1)
            mask_indices = np.isin(nearest_center_idxs, list(valid_center_indices))
            target_points = points[mask_indices]
        else:
            target_indices = (group_labels == largest_group_id)
            target_points = points[target_indices]
        
        for pt in target_points:
            final_mask[pt[0, 1], pt[0, 0]] = 255
            
        state.mask = final_mask
        return state

class RefineLightArea(Base_Step):
    """이진화된 마스크를 축소하여 조명 영역의 밀도를 파악하고, 모폴로지 연산을 통해 줄무늬 사이의 간격을 메운 뒤 복원합니다."""
    def __init__(self, target_size: int = 64, kernel_size: int = 5, name: str = "RefineLightArea"):
        super().__init__(name=name)
        self.target_size = target_size
        self.kernel_size = kernel_size

    def forward(self, state: ImageVisionState, *args, **kwargs) -> ImageVisionState:
        if state.mask is not None:
            state.mask = fill_light_area(state.mask, self.target_size, self.kernel_size)
        return state
