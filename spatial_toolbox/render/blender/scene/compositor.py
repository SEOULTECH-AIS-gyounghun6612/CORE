"""Blender compositor helpers for channel output wiring."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from ...core.channel import DEPTH, NORMAL, RGB, SEGMENTATION
from ...core.renderer import Render_Request


def Configure_passes(bpy: Any, request: Render_Request) -> None:
    """Enables Blender view-layer passes required by the request."""
    _view_layer = bpy.context.view_layer
    _view_layer.use_pass_z = DEPTH in request.channels
    _view_layer.use_pass_normal = NORMAL in request.channels
    _view_layer.use_pass_object_index = False
    _view_layer.use_pass_cryptomatte_object = SEGMENTATION in request.channels


def Build_output_path(temp_dir: Path, camera_label: str, channel: str, suffix: str) -> Path:
    """Builds the expected output file path for one channel render."""
    return temp_dir / f"{camera_label}_{channel}_0001.{suffix}"


def Get_output_socket(node: Any, names: tuple[str, ...]) -> Any:
    """Returns the first existing output socket among candidate names."""
    for _name in names:
        for _socket in node.outputs:
            if _socket.name == _name or getattr(_socket, "identifier", None) == _name:
                return _socket
    raise KeyError(f"Render layer socket not found. expected one of: {names}")



def Resolve_output_path(path: Path) -> Path:
    """Resolves Blender's final output path for a compositor file node."""
    if path.exists():
        return path
    _stem_prefix = path.stem[:-4]
    _matched = sorted(path.parent.glob(f"{_stem_prefix}*.{path.suffix.lstrip('.')}"))
    if _matched:
        return _matched[-1]
    raise FileNotFoundError(f"Blender render output not found: {path}")


def Configure_compositor_outputs(
    bpy: Any,
    temp_dir: Path,
    request: Render_Request,
    camera_label: str,
    segmentation_object_name: str | None = None,
    existing_crypto_node: Any | None = None,
) -> tuple[dict[str, Path], Any | None]:
    """Builds compositor file-output nodes for the requested channels.

    Returns (output_paths, crypto_node). On subsequent calls, pass the returned
    crypto_node back as existing_crypto_node to skip full rebuild and only
    update matte_id.
    """
    _scene = bpy.context.scene
    _scene.use_nodes = True
    _tree = _scene.node_tree
    _nodes = _tree.nodes
    _links = _tree.links

    _crypto_node: Any | None = None

    # Only update matte_id if compositor tree already built for this layout
    if existing_crypto_node is not None and SEGMENTATION in request.channels:
        if segmentation_object_name:
            existing_crypto_node.matte_id = segmentation_object_name
        _outputs: dict[str, Path] = {}
        for _channel in request.channels:
            _suffix = "png" if _channel == RGB else "exr"
            _out = Build_output_path(temp_dir, camera_label, _channel, _suffix)
            if _out.exists():
                _out.unlink()
            _outputs[_channel] = _out
        return _outputs, existing_crypto_node

    _nodes.clear()
    _render_layers = _nodes.new("CompositorNodeRLayers")
    _render_layers.location = (0.0, 0.0)

    _outputs = {}
    _config = {
        RGB: (("Image",), "PNG", "RGB", "png"),
        DEPTH: (("Depth", "Z"), "OPEN_EXR", "BW", "exr"),
        NORMAL: (("Normal",), "OPEN_EXR", "RGB", "exr"),
    }

    for _index, _channel in enumerate(request.channels):
        if _channel == SEGMENTATION:
            _file_format, _color_mode, _suffix = "OPEN_EXR", "BW", "exr"
        else:
            _socket_names, _file_format, _color_mode, _suffix = _config[_channel]
        _node = _nodes.new("CompositorNodeOutputFile")
        _node.base_path = str(temp_dir)
        _node.location = (360.0, -220.0 * _index)
        _node.format.file_format = _file_format
        _node.format.color_mode = _color_mode
        if _file_format == "OPEN_EXR":
            _node.format.color_depth = "32"
        _slot = _node.file_slots[0]
        _slot.path = f"{camera_label}_{_channel}_"
        if _channel == SEGMENTATION:
            if not segmentation_object_name:
                raise ValueError("segmentation_object_name is required for segmentation output")
            _crypto_node = _nodes.new("CompositorNodeCryptomatteV2")
            _crypto_node.location = (180.0, -220.0 * _index)
            _crypto_node.source = "RENDER"
            _crypto_node.scene = _scene
            _crypto_node.layer_name = f"{bpy.context.view_layer.name}.CryptoObject"
            _crypto_node.matte_id = segmentation_object_name
            _links.new(_render_layers.outputs["Image"], _crypto_node.inputs["Image"])
            _links.new(_crypto_node.outputs["Matte"], _node.inputs[0])
        else:
            _links.new(Get_output_socket(_render_layers, _socket_names), _node.inputs[0])
        _out = Build_output_path(temp_dir, camera_label, _channel, _suffix)
        if _out.exists():
            _out.unlink()
        _outputs[_channel] = _out
    return _outputs, _crypto_node
