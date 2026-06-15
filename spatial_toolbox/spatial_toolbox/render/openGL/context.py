"""OpenGL context adapters used by the renderer."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Literal

Context_Type = Literal["auto", "egl", "embedded"]


class Base_GL_Context(ABC):
    """Abstract OpenGL context lifecycle."""

    @abstractmethod
    def Setup(self) -> None:
        """Initializes the backing context."""
        ...

    @abstractmethod
    def Teardown(self) -> None:
        """Releases the backing context."""
        ...


class Embedded_Context(Base_GL_Context):
    """No-op context used when rendering is already embedded."""

    def Setup(self) -> None:
        """Keeps setup as a no-op."""
        return None

    def Teardown(self) -> None:
        """Keeps teardown as a no-op."""
        return None


class EGL_Context(Base_GL_Context):
    """EGL pbuffer context for headless OpenGL rendering."""

    def __init__(self, width: int, height: int) -> None:
        try:
            from OpenGL import EGL as _egl_mod  # fail early if EGL unavailable
            self._egl_mod = _egl_mod
        except Exception as exc:
            raise RuntimeError("EGL is unavailable in this environment.") from exc
        self.width = width
        self.height = height
        self._display = None
        self._surface = None
        self._ctx = None

    def Setup(self) -> None:
        """Creates EGL display, pbuffer surface, and OpenGL context, then makes them current."""
        import ctypes
        _egl = self._egl_mod

        _display = _egl.eglGetDisplay(_egl.EGL_DEFAULT_DISPLAY)
        if _display == _egl.EGL_NO_DISPLAY:
            raise RuntimeError("eglGetDisplay returned EGL_NO_DISPLAY")
        if not _egl.eglInitialize(_display, None, None):
            raise RuntimeError("eglInitialize failed")

        _cfg_attribs = (ctypes.c_int * 13)(
            _egl.EGL_SURFACE_TYPE,    _egl.EGL_PBUFFER_BIT,
            _egl.EGL_RED_SIZE,        8,
            _egl.EGL_GREEN_SIZE,      8,
            _egl.EGL_BLUE_SIZE,       8,
            _egl.EGL_DEPTH_SIZE,      24,
            _egl.EGL_RENDERABLE_TYPE, _egl.EGL_OPENGL_BIT,
            _egl.EGL_NONE,
        )
        _configs = (_egl.EGLConfig * 1)()
        _num_cfg = ctypes.c_int(0)
        if not _egl.eglChooseConfig(_display, _cfg_attribs, _configs, 1, _num_cfg) or _num_cfg.value == 0:
            raise RuntimeError("eglChooseConfig failed — no matching EGL config")
        _config = _configs[0]

        if not _egl.eglBindAPI(_egl.EGL_OPENGL_API):
            raise RuntimeError("eglBindAPI(EGL_OPENGL_API) failed")

        _pb_attribs = (ctypes.c_int * 5)(
            _egl.EGL_WIDTH,  self.width,
            _egl.EGL_HEIGHT, self.height,
            _egl.EGL_NONE,
        )
        _surface = _egl.eglCreatePbufferSurface(_display, _config, _pb_attribs)
        if _surface == _egl.EGL_NO_SURFACE:
            raise RuntimeError("eglCreatePbufferSurface failed")

        _ctx = _egl.eglCreateContext(_display, _config, _egl.EGL_NO_CONTEXT, None)
        if _ctx == _egl.EGL_NO_CONTEXT:
            raise RuntimeError("eglCreateContext failed")

        if not _egl.eglMakeCurrent(_display, _surface, _surface, _ctx):
            raise RuntimeError("eglMakeCurrent failed")

        self._display = _display
        self._surface = _surface
        self._ctx = _ctx

    def Teardown(self) -> None:
        """Releases EGL context, surface, and display."""
        _egl = self._egl_mod
        if self._display is not None:
            _egl.eglMakeCurrent(
                self._display,
                _egl.EGL_NO_SURFACE,
                _egl.EGL_NO_SURFACE,
                _egl.EGL_NO_CONTEXT,
            )
            if self._ctx is not None:
                _egl.eglDestroyContext(self._display, self._ctx)
            if self._surface is not None:
                _egl.eglDestroySurface(self._display, self._surface)
            _egl.eglTerminate(self._display)
        self._display = self._surface = self._ctx = None


def Build_GL_Context(
    width: int,
    height: int,
    context_type: Context_Type = "auto",
) -> Base_GL_Context:
    """Builds the preferred OpenGL context adapter for the environment."""
    if context_type == "egl":
        return EGL_Context(width, height)
    if context_type == "embedded":
        return Embedded_Context()
    # "auto": EGL 우선, 실패 시 Embedded 폴백
    try:
        return EGL_Context(width, height)
    except Exception:
        return Embedded_Context()
