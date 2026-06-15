"""Camera conversion helpers for Blender scene setup."""
from __future__ import annotations

from typing import Any

from ....scene.node.type.camera import Camera


def Camera_object_name(label: str) -> str:
    """Builds the Blender object name for a shared camera label."""
    return f"focus_camera__{label}"


def Get_or_create_camera_object(bpy: Any, camera_node: Camera):
    """Returns an existing Blender camera object or creates one."""
    _name = Camera_object_name(camera_node.label)
    _obj = bpy.data.objects.get(_name)
    if _obj is not None:
        return _obj
    _camera_data = bpy.data.cameras.new(name=_name)
    _obj = bpy.data.objects.new(_name, _camera_data)
    bpy.context.scene.collection.objects.link(_obj)
    return _obj


def Apply_camera_intrinsic(camera_data: Any, intrinsic: Any) -> None:
    """Applies shared camera intrinsics to a Blender camera datablock."""
    _width = int(intrinsic.width)
    _height = int(intrinsic.height)
    _fx = float(intrinsic.fx)
    _cx = float(intrinsic.cx)
    _cy = float(intrinsic.cy)
    _sensor_width = 36.0
    _sensor_height = _sensor_width * (_height / _width)
    camera_data.type = "PERSP"
    camera_data.sensor_fit = "HORIZONTAL"
    camera_data.sensor_width = _sensor_width
    camera_data.sensor_height = _sensor_height
    camera_data.lens = _fx * _sensor_width / _width
    camera_data.shift_x = -((_cx - (_width * 0.5)) / _width)
    camera_data.shift_y = ((_height * 0.5) - _cy) / _height


def Configure_camera(bpy: Any, camera_node: Camera, unit_length: float) -> None:
    """Configures the active Blender scene from a shared camera node.

    Args:
        bpy: Blender Python module handle.
        camera_node: Shared camera node used as the source of truth.
        unit_length: Scene unit scale used to normalize clip distances.
    """
    import math
    from mathutils import Matrix

    if camera_node.intrinsic is None:
        raise ValueError(f"Camera '{camera_node.label}' intrinsic is None.")
    _scene = bpy.context.scene
    _camera_obj = Get_or_create_camera_object(bpy, camera_node)
    _camera_data = _camera_obj.data
    Apply_camera_intrinsic(_camera_data, camera_node.intrinsic)
    _camera_data.clip_start = float(camera_node.intrinsic.near_clip / unit_length)
    _camera_data.clip_end = float(camera_node.intrinsic.far_clip / unit_length)
    # USD stage is Y-up; Blender auto-applies +90° X rotation to imported objects.
    # Apply the same correction so focus_camera aligns with the imported scene.
    _yup_to_zup = Matrix.Rotation(math.pi / 2, 4, 'X')
    _camera_obj.matrix_world = _yup_to_zup @ Matrix(camera_node.world_matrix.astype(float).tolist())
    _scene.render.resolution_x = int(camera_node.intrinsic.width)
    _scene.render.resolution_y = int(camera_node.intrinsic.height)
    _scene.render.resolution_percentage = 100
    _scene.camera = _camera_obj
