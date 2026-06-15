"""Color encoding helpers for OpenGL segmentation IDs."""
def encode_id_to_color(idx: int) -> tuple[int, int, int]:
    """Encodes an integer ID into an RGB byte triplet."""
    return (idx & 0xFF, (idx >> 8) & 0xFF, (idx >> 16) & 0xFF)
