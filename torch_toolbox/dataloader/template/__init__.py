from .classification._base import (
    Classification_Dataset_Config,
    Classification_Dataset,
)
from .classification.image import (
    Classification_Image_Dataset_Config,
    Classification_Image_Dataset,
)
from .detection.coco import COCO_Dataset_Config, COCO_Dataset

CLASSIFICATION_DATASET = Classification_Dataset | Classification_Image_Dataset
CLASSIFICATION_DATASET_CONFIG = Classification_Dataset_Config | Classification_Image_Dataset_Config
