from torch import float32
from torchvision import transforms


TRANSFORM_IMAGE_NET = transforms.Compose([
    transforms.Resize((224, 224), antialias=True),
    transforms.ConvertImageDtype(float32),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def Get_transform(name: str | None = None):
    if name is None:
        return None
    if name == "ImageNet":
        return TRANSFORM_IMAGE_NET
    raise ValueError(f"지원하지 않는 transform: {name}")