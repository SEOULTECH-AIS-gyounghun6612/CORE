import cv2
from pathlib import Path
from typing import Optional
from vision_toolbox.typing.images import IMG_1C

def Read(img_file: Path, scale: float | None = 0.5) -> Optional[IMG_1C]:
    """
    이미지를 파일에서 읽어와 그레이스케일로 변환하고, 선택적으로 크기를 조절합니다.
    """
    _img: IMG_1C = cv2.imread(str(img_file), cv2.IMREAD_GRAYSCALE)
    if _img is None:
        return None
        
    if scale is not None:
        return cv2.resize(_img, dsize=None, fx=scale, fy=scale)
    return _img
